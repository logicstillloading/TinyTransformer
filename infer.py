"""
Interactive and CLI Inference Script for TinyTransformer (~230M parameters).
Features:
- Sliding-window repetition penalty (vectorized on GPU, prevents syntax decay in code)
- No-repeat n-gram blocking (eliminates repetitive phrase attractor states)
- Min-P sampling (filters low-probability hallucinations)
- Smart Chat Templating with automatic 'User: ...\nAssistant: ' wrapping
- Stop sequence truncation (halts immediately before hallucinated next-turn 'User:' headers)
- Clean, stylized Markdown CLI output with generation metrics (latency, tok/s)
"""

import argparse
import os
import sys

# Configure UTF-8 stdout/stderr on Windows to avoid charmap UnicodeEncodeErrors
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import time
from typing import List, Optional, Set
import torch
import torch.nn.functional as F

from model.transformer import TinyTransformer, TransformerConfig
from data.tokenizer import get_or_create_tokenizer, Tokenizer


def format_prompt(raw_text: str, is_raw: bool = False) -> str:
    """Intelligently wraps chat prompts while preserving code completions."""
    text = raw_text.strip()
    if is_raw:
        return raw_text
    # If already formatted or code completion, leave as is
    if (
        text.startswith("User:")
        or text.startswith("###")
        or text.startswith("def ")
        or text.startswith("class ")
        or text.startswith("import ")
        or text.startswith("from ")
        or text.endswith("=")
        or text.endswith(":")
    ):
        return raw_text
    return f"User: {text}\nAssistant:"


@torch.inference_mode()
def generate_streaming(
    model: TinyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repetition_penalty: float = 1.15,
    repetition_window: int = 64,
    no_repeat_ngram_size: int = 3,
    suppress_raw_bytes: bool = False,
    device: str = "cuda",
) -> str:
    """Generates text from prompt, streaming tokens live to stdout with modern sampling."""
    model.eval()
    encoded = tokenizer.encode(prompt, add_bos=True)
    curr_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    # Prefill prompt with wiped fresh KV cache
    t0 = time.perf_counter()
    autocast_device = "cuda" if "cuda" in str(device) else "cpu"
    with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
        logits, kv_caches, _, _ = model(curr_ids, use_cache=True)
        next_token_logits = logits[:, -1, :].clone()

    generated_tokens: List[int] = []
    all_token_ids: List[int] = list(encoded)
    full_output_chars: str = ""

    print("\n" + "─" * 70)
    print(f"❯ Prompt:\n{prompt.strip()}\n")
    print("❯ Completion:")

    for step in range(max_new_tokens):
        # 1. No-repeat n-gram blocking (prevents phrase loops)
        if no_repeat_ngram_size > 0 and len(all_token_ids) >= no_repeat_ngram_size:
            ngram_prefix = tuple(all_token_ids[-(no_repeat_ngram_size - 1):])
            banned_tokens = set()
            for i in range(len(all_token_ids) - no_repeat_ngram_size + 1):
                if tuple(all_token_ids[i : i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned_tokens.add(all_token_ids[i + no_repeat_ngram_size - 1])
            for b_tok in banned_tokens:
                next_token_logits[0, b_tok] = -float("Inf")

        # 2. Sliding-window repetition penalty (vectorized on GPU)
        if repetition_penalty != 1.0 and len(all_token_ids) > 0:
            window = all_token_ids[-repetition_window:] if repetition_window > 0 else all_token_ids
            unique_window_tokens = list(set(window))
            indices = torch.tensor(unique_window_tokens, dtype=torch.long, device=device)
            vals = next_token_logits[0, indices]
            neg_mask = vals < 0
            vals[neg_mask] *= repetition_penalty
            vals[~neg_mask] /= repetition_penalty
            next_token_logits[0, indices] = vals

        # 3. Sampling
        if temperature <= 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / max(temperature, 1e-5)

            # 4. Min-P Filtering (filter out low probability noise)
            if min_p > 0.0:
                probs = F.softmax(scaled_logits, dim=-1)
                max_prob = torch.max(probs, dim=-1, keepdim=True).values
                min_thresh = min_p * max_prob
                scaled_logits[probs < min_thresh] = -float("Inf")

            # 5. Top-K Cutoff
            if top_k > 0:
                topk_vals, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < topk_vals[:, [-1]]] = -float("Inf")

            # 6. Top-P Nucleus Cutoff
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(scaled_logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum().item() == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        generated_tokens.append(token_id)
        all_token_ids.append(token_id)

        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        full_output_chars += token_text

        # Stop conditions: EOS token
        if token_id == tokenizer.eos_token_id:
            break

        # Stop conditions: Stop sequences (prevent multi-turn self-talk hallucination)
        if "User:" in prompt:
            if full_output_chars.endswith("\nUser:") or full_output_chars.endswith("\nUser :") or full_output_chars.endswith("\nAssistant:"):
                # Clean prompt stop
                break

        print(token_text, end="", flush=True)

        with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
            logits, kv_caches, _, _ = model(next_token, kv_caches=kv_caches, use_cache=True)
            next_token_logits = logits[:, -1, :].clone()

    t1 = time.perf_counter()
    total_time = t1 - t0
    tok_count = len(generated_tokens)
    speed = tok_count / max(total_time, 1e-5)
    print("\n" + "─" * 70)
    print(f"📊 Stats: {tok_count} tokens generated in {total_time:.2f}s ({speed:.1f} tok/s)")
    print("─" * 70 + "\n")
    return tokenizer.decode(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="TinyTransformer Inference CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k cutoff")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus cutoff")
    parser.add_argument("--min-p", type=float, default=0.05, help="Min-p dynamic noise cutoff (0.05 = 5%% of max prob)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Sliding-window repetition penalty")
    parser.add_argument("--repetition-window", type=int, default=64, help="Sliding window size for repetition penalty (0 = all)")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Block repeating n-grams of this size (0 = disabled)")
    parser.add_argument("--raw", action="store_true", help="Disable automatic User:/Assistant: chat wrapping")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = get_or_create_tokenizer(args.tokenizer)

    config = TransformerConfig(vocab_size=tokenizer.vocab_size)
    print("Initializing TinyTransformer (~230M params)...")
    model = TinyTransformer(config).to(args.device)

    # Maintain precision matching pretraining (FP32 norms/gates, BF16 2D projections)
    for name, p in model.named_parameters():
        is_fp32 = (
            "norm" in name
            or name.endswith(".gate")
            or "bias" in name
            or "tok_embeddings" in name
            or "lm_head" in name
            or p.ndim < 2
        )
        if is_fp32:
            p.data = p.data.to(torch.float32)
        else:
            p.data = p.data.to(torch.bfloat16)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path) and os.path.exists("checkpoints/checkpoint_latest.pt"):
        ckpt_path = "checkpoints/checkpoint_latest.pt"

    if os.path.exists(ckpt_path):
        print(f"Loading weights from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[*] Checkpoint loaded cleanly! Resumed from Step {ckpt.get('step', 'unknown')} ({ckpt.get('tokens_processed', 0):,} tokens)")
    else:
        print("Note: Running with initialized weights (no checkpoint loaded).")

    if args.prompt:
        formatted = format_prompt(args.prompt, is_raw=args.raw)
        generate_streaming(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            device=args.device,
        )
    else:
        print("\n=======================================================")
        print("  TinyTransformer Interactive Mode")
        print("  Type your prompt below (Ctrl+C or 'exit' to quit)")
        print("=======================================================")
        while True:
            try:
                user_prompt = input("\n❯ ")
                if not user_prompt.strip():
                    continue
                if user_prompt.strip().lower() in ("exit", "quit"):
                    break
                formatted = format_prompt(user_prompt, is_raw=args.raw)
                generate_streaming(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                    repetition_window=args.repetition_window,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    device=args.device,
                )
            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
