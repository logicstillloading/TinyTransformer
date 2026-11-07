"""
Non-Destructive CPU Factual Probe and Text Generation Suite for TinyTransformer.
Evaluates model factual entity binding, domain grounding, and text generation
strictly on CPU without consuming GPU VRAM or interrupting ongoing pretraining.

Usage:
    python test.py
    python test.py --checkpoint checkpoints/checkpoint_latest.pt
    python test.py --probes-only
    python test.py --generate-only
"""

import argparse
import io
import os
import sys
import torch

# Configure UTF-8 stdout/stderr on Windows to avoid charmap UnicodeEncodeErrors during generation of math/Greek symbols
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
from data.tokenizer import get_or_create_tokenizer
from model.transformer import TinyTransformer, TransformerConfig

FACTUAL_PROBES = [
    {
        "domain": "Geography",
        "prompt": "The capital city of France is",
        "target": " Paris",
    },
    {
        "domain": "Chemistry",
        "prompt": "The chemical formula for water is",
        "target": " H",
    },
    {
        "domain": "History",
        "prompt": "The first president of the United States was",
        "target": " George",
    },
    {
        "domain": "Biology",
        "prompt": "Photosynthesis takes place inside the",
        "target": " chlor",
    },
    {
        "domain": "Physics",
        "prompt": "Albert Einstein developed the theory of",
        "target": " relat",
    },
]

GEN_PROMPTS = [
    "The capital city of France is",
    "User: Explain how photosynthesis works in plants.\nAssistant:",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "The fundamental theorem of calculus states that",
    "In machine learning, gradient descent is an optimization algorithm used to",
]


def run_factual_probes(m: TinyTransformer, tok, step: int, tokens_seen: int, ema_loss: float):
    print("\n" + "=" * 80)
    print(f"  [>] FACTUAL PROBE SUITE | Step: {step:,} | Tokens: {tokens_seen:,} ({tokens_seen/1e6:.1f}M) | EMA Loss: {ema_loss:.3f}")
    print("=" * 80)
    print(f"{'Domain':<12} | {'Prompt':<42} | {'Target':<9} | {'Prob':<10} | {'Rank / Total':<18}")
    print("-" * 80)

    total_vocab = tok.vocab_size
    device = next(m.parameters()).device

    for probe in FACTUAL_PROBES:
        prompt = probe["prompt"]
        target = probe["target"]
        domain = probe["domain"]

        ids = torch.tensor([tok.encode(prompt)], device=device)
        with torch.no_grad():
            p = m(ids)[0][0, -1].softmax(-1)

        t_id = tok.encode(target)[0]
        prob = p[t_id].item()
        rank = (p > p[t_id]).sum().item() + 1
        pct = (rank / total_vocab) * 100.0

        top1_id = torch.argmax(p).item()
        top1_str = tok.decode([top1_id]).strip()

        print(f"{domain:<12} | {prompt[:40]:<42} | {target.strip():<9} | {prob:.4e} | #{rank:<5d} ({pct:4.1f}%) [top: '{top1_str}']")

    print("=" * 80 + "\n", flush=True)


def run_generation_previews(
    m: TinyTransformer,
    tok,
    step: int,
    tokens_seen: int,
    max_new_tokens: int = 50,
    temperature: float = 0.60,
    top_k: int = 40,
    top_p: float = 0.90,
    min_p: float = 0.03,
    no_repeat_ngram_size: int = 0,
    greedy: bool = False,
):
    sampling_desc = "GREEDY (argmax)" if greedy else f"Temp={temperature:.2f}, Top-K={top_k}, Top-P={top_p:.2f}, Min-P={min_p:.2f}, RepPen=1.05"
    print("=" * 80)
    print(f"  [>] TEXT GENERATION PREVIEWS | Step: {step:,} | Tokens: {tokens_seen/1e6:.1f}M | {sampling_desc}")
    print("=" * 80)

    device = next(m.parameters()).device

    for idx, prompt in enumerate(GEN_PROMPTS, 1):
        encoded = tok.encode(prompt, add_bos=True)
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

        # Universal sampling parameters applied uniformly to every prompt
        p_temp = 0.0 if greedy else temperature
        p_top_k = 0 if greedy else top_k
        p_top_p = 1.0 if greedy else top_p
        p_min_p = 0.0 if greedy else min_p
        p_rep = 1.0 if greedy else 1.05

        with torch.no_grad():
            out_ids = m.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=p_temp,
                top_k=p_top_k,
                top_p=p_top_p,
                min_p=p_min_p,
                repetition_penalty=p_rep,
                repetition_window=64,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=tok.eos_token_id,
            )
        gen_text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
        completion = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text.strip()

        print(f"\n--- [Prompt {idx}] {prompt.strip()} ---")
        print(f"{completion}")

    print("\n" + "=" * 80 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyTransformer on CPU or GPU")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu)")
    parser.add_argument("--probes-only", action="store_true", help="Run only the logit factual probes")
    parser.add_argument("--generate-only", action="store_true", help="Run only text generation previews")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.60, help="Sampling temperature (default: 0.60)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K cutoff (default: 40)")
    parser.add_argument("--top-p", type=float, default=0.90, help="Top-P nucleus cutoff (default: 0.90)")
    parser.add_argument("--min-p", type=float, default=0.03, help="Min-P probability threshold (default: 0.03)")
    parser.add_argument("--no-repeat-ngram", type=int, default=0, help="No-repeat n-gram size (default: 0 = disabled)")
    parser.add_argument("--greedy", action="store_true", help="Force greedy argmax decoding (temperature=0.0)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[!] Checkpoint not found at: {args.checkpoint}")
        sys.exit(1)

    tok = get_or_create_tokenizer("tokenizer.json")
    cfg = TransformerConfig(vocab_size=tok.vocab_size)
    m = TinyTransformer(cfg)

    print(f"[*] Loading checkpoint into {args.device.upper()} memory: {args.checkpoint}...", flush=True)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.to(args.device).float().eval()

    step = ck.get("step", 0)
    tokens_seen = ck.get("tokens_processed", 0)
    metrics = ck.get("metrics", {})
    ema_loss = metrics.get("ema_loss", 0.0)

    if not args.generate_only:
        run_factual_probes(m, tok, step, tokens_seen, ema_loss)

    if not args.probes_only:
        run_generation_previews(
            m=m,
            tok=tok,
            step=step,
            tokens_seen=tokens_seen,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            no_repeat_ngram_size=args.no_repeat_ngram,
            greedy=args.greedy,
        )


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()