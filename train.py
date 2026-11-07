"""
Production Pretraining Script for TinyTransformer (~238M Parameters).
Features:
- Keller Jordan's Muon optimizer (Newton-Schulz iteration) for 2D weights + AdamW for embeddings/RMSNorms/Engram tables.
- Auxiliary Critic RL model ('bsing' detector) with warmup schedule and one-sided label smoothing.
- Dynamic 6-stream dataset mixture (28% cosmopedia, 28% fineweb, 15% finemath, 12% python, 10% wikipedia, 7% instruction) with deterministic resumption.
- Corruption-proof atomic checkpointing (latest and previous only).
- Live Goated CLI telemetry with step, tokens, tok/s, loss, EMA loss, step PPL, EMA PPL, Critic status, VRAM, ETA.
- Mini-inference generation tests every 250 steps with clean CLI display and multi-format session logging (CSV and JSONL).
"""

import argparse
from collections import deque
import csv
from datetime import datetime, timedelta
import glob
import json
import math
import numpy as np
import os
import shutil
import sys

# Configure unbuffered line-buffered stdout/stderr for real-time telemetry streaming in Colab / Linux / Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import gc
import psutil
import threading
import time
from typing import Dict, Any, Optional, List, Tuple

# Enable virtual memory expandable segments (Linux only; ignored on Windows CUDA allocator)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F

from model.transformer import TinyTransformer, TransformerConfig
from model.critic import CriticModel, compute_critic_loss, compute_generator_adversarial_loss
from model.hardware import detect_hardware_profile, HardwareProfile, XLA_AVAILABLE, xm
from optimizers.muon import create_dual_optimizers, create_critic_optimizer
from data.tokenizer import get_or_create_tokenizer, Tokenizer
from data.dataset import StreamingDatasetMixture, STREAM_CONFIGS
from data.cloud_sync import sync_cloud_checkpoint, load_cloud_config, auto_sync_on_interrupt, verify_checkpoint_integrity


INFER_PROMPTS = [
    # Q&A / Instruction
    "User: Explain how photosynthesis works in plants.\nAssistant:",
    "User: What is a binary search tree in computer science?\nAssistant:",
    "User: Solve the equation 2x + 5 = 15 for x.\nAssistant:",
    # Direct Autocomplete & Science
    "The fundamental theorem of calculus states that",
    "In machine learning, gradient descent is an optimization algorithm used to",
    "The capital city of France is",
    "Albert Einstein developed the theory of",
    "The chemical formula for water is",
    "The Roman Empire was characterized by",
    "DNA is a double-stranded molecule that encodes",
    # Code Autocomplete & Synthesis
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "def is_prime(num):\n    \"\"\"Check if a number is prime.\"\"\"\n",
    "def binary_search(arr, target):\n    \"\"\"Search for target in sorted array.\"\"\"\n",
]


def get_lr_multiplier(step: int, warmup_steps: int = 0, max_steps: int = 100_000, min_ratio: float = 0.1) -> float:
    """
    If warmup_steps > 0: linear warmup for step < warmup_steps.
    Followed by cosine decay down to min_ratio.
    If warmup_steps == 0: starts at 1.0 immediately for maximum early learning velocity.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-6, float(step) / float(max(1, warmup_steps)))
    if step >= max_steps:
        return min_ratio
    decay_steps = max(1, max_steps - warmup_steps)
    progress = float(max(0, step - warmup_steps)) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine_factor


def format_time(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))


def cast_model_dtypes(model: torch.nn.Module, is_cuda_dev: bool, dtype: torch.dtype):
    """
    Maintains FP32 precision for RMSNorm weights, tied embeddings, and sparse Engram memory tables,
    while 2D Transformer weights and linear projections (attention, SwiGLU FFN, Engram gating & projections)
    run in BF16 or FP16 for maximum Tensor Core and TPU MXU throughput.
    """
    if is_cuda_dev and dtype in (torch.bfloat16, torch.float16):
        for name, p in model.named_parameters():
            is_fp32 = (
                "norm" in name
                or "tok_embeddings" in name
                or "lm_head" in name
                or "memory_table" in name
                or "centroids" in name
            )
            if is_fp32:
                p.data = p.data.to(torch.float32)
            else:
                p.data = p.data.to(dtype)
    elif is_cuda_dev:
        model.to(dtype=dtype)


def _state_to_cpu(obj):
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    elif isinstance(obj, list):
        return [_state_to_cpu(v) for v in obj]
    else:
        return obj


def _async_save_worker(state: Dict[str, Any], checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    prev_path = os.path.join(checkpoint_dir, "checkpoint_prev.pt")
    tmp_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt.tmp")
    hp_path = os.path.join(checkpoint_dir, "hyperparameters.json")

    # 1. Write to temporary file
    torch.save(state, tmp_path)

    # 2. Flush and force sync to disk
    with open(tmp_path, "a+b") as f:
        f.flush()
        os.fsync(f.fileno())

    # 3. Rotate existing latest checkpoint to prev
    if os.path.exists(latest_path):
        try:
            if os.path.exists(prev_path):
                os.remove(prev_path)
            os.replace(latest_path, prev_path)
        except Exception as e:
            print(f"[Checkpoint Warning] Error rotating previous checkpoint: {e}")

    # 4. Atomic rename tmp to latest
    os.replace(tmp_path, latest_path)

    # 5. Write human-readable hyperparameters JSON
    hp = state.get("hyperparameters", {})
    if hp:
        try:
            with open(hp_path, "w", encoding="utf-8") as f:
                json.dump(hp, f, indent=2)
        except Exception:
            pass

    # 6. Write human-readable checkpoint manifest JSON with full reconstructibility lineage
    manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
    try:
        manifest_data = {
            "step": state.get("step"),
            "tokens_processed": state.get("tokens_processed"),
            "metrics": state.get("metrics"),
            "session_info": state.get("session_info"),
            "session_history": state.get("session_history"),
            "associated_files": state.get("session_info", {}).get("associated_files", {}),
            "hyperparameters": hp,
            "system_metadata": state.get("system_metadata"),
            "manifest_generated_at": datetime.now().isoformat(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
    except Exception as e:
        print(f"[Manifest Warning] Could not write manifest: {e}")

    print(f"[*] [Async Save] Checkpoint at step {state['step']} successfully persisted to disk in background.")
    del state
    gc.collect()


_save_thread = None
_save_lock = threading.Lock()


def wait_for_async_save():
    global _save_thread
    with _save_lock:
        if _save_thread is not None and _save_thread.is_alive():
            _save_thread.join()
            _save_thread = None


def save_atomic_checkpoint(
    checkpoint_dir: str,
    step: int,
    tokens_processed: int,
    model: TinyTransformer,
    critic: CriticModel,
    optimizers: Dict[str, Any],
    dataset: StreamingDatasetMixture,
    metrics: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]] = None,
    session_info: Optional[Dict[str, Any]] = None,
    session_history: Optional[List[Dict[str, Any]]] = None,
    sync: bool = True,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    wait_for_async_save()

    # Fast non-blocking CPU snapshot
    muon_sd = optimizers["muon"].state_dict() if "muon" in optimizers else None
    adam_sd = optimizers["adam"].state_dict() if "adam" in optimizers else None
    critic_sd = optimizers["critic"].state_dict() if "critic" in optimizers else None
    ds_sd = dataset.state_dict()

    # Default session metadata if not passed
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    arch_summary = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": cfg.vocab_size if cfg else 16384,
        "hidden_dim": cfg.hidden_dim if cfg else 640,
        "num_layers": cfg.num_layers if cfg else 24,
        "num_heads": getattr(cfg, "num_heads", 10) if cfg else 10,
        "intermediate_dim": cfg.intermediate_dim if cfg else 1920,
        "engram_layers": cfg.engram_layers if cfg else (5, 11),
        "global_engram_layer": 16,
    }

    raw_state = {
        "step": step,
        "tokens_processed": tokens_processed,
        "model_state_dict": model.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "opt_muon_state_dict": muon_sd,
        "opt_adam_state_dict": adam_sd,
        "opt_critic_state_dict": critic_sd,
        "muon_state_dict": muon_sd,
        "adam_state_dict": adam_sd,
        "dataloader_state": ds_sd,
        "dataset_state": ds_sd,
        "metrics": metrics,
        "hyperparameters": hyperparameters or {},
        "session_info": session_info or {},
        "session_history": session_history or [],
        "system_metadata": {
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python_version": sys.version,
            "architecture": arch_summary,
            "saved_at": datetime.now().isoformat(),
        },
    }
    cpu_state = _state_to_cpu(raw_state)
    del raw_state

    if sync:
        _async_save_worker(cpu_state, checkpoint_dir)
    else:
        # Launch disk save in background thread so training continues without stalling GPU
        global _save_thread
        with _save_lock:
            _save_thread = threading.Thread(target=_async_save_worker, args=(cpu_state, checkpoint_dir), daemon=False)
            _save_thread.start()


def archive_milestone_checkpoint(
    checkpoint_dir: str,
    step: int,
    tokens_processed: int,
    metrics: Dict[str, Any],
    recent_logs: List[Dict[str, Any]],
    infer_outputs: Optional[Dict[str, str]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Archives checkpoint_latest.pt into checkpoints/checkpoint(step_<STEP>)
    and generates companion .md and .json reports containing recent loss/PPL
    progression (last 10 steps) and mini-inference preview generations every 1,000 steps.
    """
    wait_for_async_save()
    latest_pt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")

    if not os.path.exists(latest_pt):
        print(f"[Archive Warning] No checkpoint found at {latest_pt}")
        return ""

    archive_name = f"checkpoint(step_{step})"
    dest_dir = os.path.join(checkpoint_dir, archive_name)
    os.makedirs(dest_dir, exist_ok=True)

    dest_pt = os.path.join(dest_dir, f"{archive_name}.pt")
    dest_manifest = os.path.join(dest_dir, "checkpoint_manifest.json")
    dest_md = os.path.join(dest_dir, f"{archive_name}.md")
    dest_json = os.path.join(dest_dir, f"{archive_name}.json")

    # 1. Copy weights and manifest
    shutil.copy2(latest_pt, dest_pt)
    if os.path.exists(manifest_path):
        shutil.copy2(manifest_path, dest_manifest)

    curr_loss = metrics.get("current_loss") or (recent_logs[-1]["loss"] if recent_logs else 0.0)
    ema_loss = metrics.get("ema_loss") or (recent_logs[-1]["avg_loss"] if recent_logs else 0.0)
    curr_ppl = metrics.get("current_ppl") or (recent_logs[-1]["ppl"] if recent_logs else 0.0)
    ema_ppl = metrics.get("ema_ppl") or (recent_logs[-1]["avg_ppl"] if recent_logs else 0.0)

    # 2. Build structured JSON report
    report_data = {
        "step": step,
        "tokens_processed": tokens_processed,
        "timestamp": datetime.now().isoformat(),
        "latest_metrics": {
            "loss": curr_loss,
            "ema_loss": ema_loss,
            "ppl": curr_ppl,
            "ema_ppl": ema_ppl,
        },
        "last_10_logged_steps": recent_logs,
        "mini_inference_preview": infer_outputs or {},
        "hyperparameters": hyperparameters or {},
    }

    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 3. Build Markdown report
    md_lines = [
        f"# TinyTransformer Checkpoint Report — Step {step:,}",
        "",
        f"- **Global Step**: `{step:,}`",
        f"- **Tokens Processed**: `{tokens_processed:,}` ({tokens_processed/1e6:.2f}M)",
        f"- **Latest Step Loss**: `{curr_loss:.4f}` (EMA Avg: `{ema_loss:.4f}`)",
        f"- **Latest Step PPL**: `{curr_ppl:.2f}` (EMA Avg: `{ema_ppl:.2f}`)",
        f"- **Archive Directory**: `{dest_dir}`",
        f"- **Archive File**: `{dest_pt}`",
        f"- **Report Generated**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "---",
        "",
        "## Recent Metrics Progression (Last 10 Logged Steps)",
        "",
        "| Step | Loss | EMA Loss | Step PPL | EMA PPL | Speed (tok/s) | VRAM (MB) | Timestamp |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for entry in recent_logs:
        s = entry.get("step", 0)
        l = entry.get("loss", 0.0)
        al = entry.get("avg_loss", 0.0)
        p = entry.get("ppl", 0.0)
        ap = entry.get("avg_ppl", 0.0)
        spd = entry.get("tokens_per_sec", 0.0)
        vr = entry.get("vram_mb", 0.0)
        ts = entry.get("timestamp", "").split("T")[-1][:8]
        md_lines.append(f"| {s:,} | {l:.4f} | {al:.4f} | {p:.2f} | {ap:.2f} | {spd:.0f} | {vr:.0f} | {ts} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Mini-Inference Preview Generations",
        "",
    ])

    if infer_outputs:
        for p_idx, (p_text, completion) in enumerate(infer_outputs.items(), 1):
            md_lines.append(f"### Prompt {p_idx}")
            md_lines.append("```text")
            md_lines.append(p_text.strip())
            md_lines.append("```")
            clean_comp = completion.strip() if completion else "*(empty)*"
            md_lines.append(f"> {clean_comp.replace(chr(10), chr(10) + '> ')}")
            md_lines.append("")
    else:
        md_lines.append("*No mini-inference preview recorded at this step.*")
        md_lines.append("")

    with open(dest_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Also copy to root of checkpoints directory
    root_md = os.path.join(checkpoint_dir, f"{archive_name}.md")
    root_json = os.path.join(checkpoint_dir, f"{archive_name}.json")
    shutil.copy2(dest_md, root_md)
    shutil.copy2(dest_json, root_json)

    file_size_gb = os.path.getsize(dest_pt) / (1024 ** 3)
    print(f"[+] [Archive] Step {step} archived to {dest_dir} ({file_size_gb:.2f} GB) with {archive_name}.md and {archive_name}.json")
    return dest_dir


@torch.inference_mode()
def run_mini_inference(
    model: TinyTransformer,
    tokenizer: Tokenizer,
    step: int,
    device: str,
    max_new_tokens: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, str]:
    """Fast non-blocking completions for standard benchmark prompts."""
    is_tpu = "tpu" in str(device).lower() or "xla" in str(device).lower()
    if max_new_tokens is None:
        max_new_tokens = 35 if is_tpu else (100 if step >= 1000 else 50)
    model.eval()
    results = {}
    print("\n" + "=" * 70)
    print(f"| MINI-INFERENCE GENERATION TEST AT STEP {step:06d} (max_new_tokens={max_new_tokens})")
    print("=" * 70)

    is_xla = "xla" in str(device)
    autocast_device = "cuda" if "cuda" in str(device) else "cpu"
    autocast_dtype = dtype if autocast_device == "cuda" else torch.bfloat16
    for p_idx, prompt in enumerate(INFER_PROMPTS, 1):
        encoded = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        try:
            with torch.amp.autocast(autocast_device, dtype=autocast_dtype):
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.80,
                    top_k=50,
                    top_p=0.95,
                    min_p=0.05,
                    repetition_penalty=1.15,
                    repetition_window=128,
                    no_repeat_ngram_size=0 if prompt.lstrip().startswith("def ") else 4,
                    eos_token_id=tokenizer.eos_token_id,
                    suppress_raw_bytes=False,
                )
            gen_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
            if gen_text.startswith(prompt):
                completion = gen_text[len(prompt):].strip()
            else:
                completion = gen_text.strip()

            # Intelligent stop sequence truncation
            if "User:" in prompt:
                # Instruction chat: stop on next simulated user/assistant turns
                for stop_marker in ["\nUser:", "\nUser :", "\nAssistant:", "\nHuman:"]:
                    if stop_marker in completion:
                        completion = completion.split(stop_marker)[0].strip()
            elif not prompt.strip().startswith("def ") and not prompt.strip().startswith("class "):
                # Conceptual/Science: halt if young model drifts into unprompted code basins or multi-newlines
                for stop_marker in ["\n```python", "\n```", "\ndef ", "\nclass ", "\n\n\n"]:
                    if stop_marker in completion:
                        completion = completion.split(stop_marker)[0].strip()
            else:
                # Code completion: halt if model attempts to switch to chat or repetitive blocks
                for stop_marker in ["\nUser:", "\n###", "\n\n\n\n"]:
                    if stop_marker in completion:
                        completion = completion.split(stop_marker)[0].strip()
        except Exception as e:
            completion = f"[Generation Error: {e}]"

        results[prompt] = completion
        print(f"\n┌─ [Evaluation Prompt {p_idx}/{len(INFER_PROMPTS)}]")
        print(f"│ ❯ Prompt: {prompt.strip()}")
        print(f"│ ❯ Completion:\n│   {completion.replace(chr(10), chr(10) + '│   ')}")
        print("└" + "─" * 68)

    print("=" * 70 + "\n")
    model.train()
    return results


@torch.no_grad()
def eval_fixed(model, shard_paths, device, n_windows=16, seq_len=1024):
    model.eval()
    out = {}
    for name, path in shard_paths.items():
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        tot = 0.0
        for i in range(n_windows):
            s = i * (seq_len + 1)
            w = torch.from_numpy(arr[s:s + seq_len + 1].astype(np.int64)).to(device)
            x, y = w[:-1].unsqueeze(0), w[1:].unsqueeze(0)
            with torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
                logits = model(x)[0]
            tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
        out[name] = tot / n_windows
    model.train()
    return out


def _train_worker(rank: int, args):
    # Hardware acceleration flags (Ampere Tensor Core TF32 & cuDNN auto-tuner)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Multi-core rank and world-size detection
    is_master = (rank == 0)
    world_size = 1
    if XLA_AVAILABLE and xm is not None:
        try:
            world_size = xm.xrt_world_size()
            rank = xm.get_ordinal()
            is_master = xm.is_master_ordinal()
        except Exception:
            pass

    # 0. Checkpoint Auto-Discovery across Cloud & Kaggle directories
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    if is_master and args.resume and not os.path.exists(latest_ckpt_path):
        candidate_paths = [
            "/kaggle/input/tinytransformer-checkpoint/checkpoint_latest.pt",
            "/kaggle/input/tinytransformer/checkpoint_latest.pt",
            "/kaggle/input/checkpoint_latest.pt",
            "/kaggle/working/checkpoints/checkpoint_latest.pt",
            "/content/drive/MyDrive/TinyTransformer/checkpoints/checkpoint_latest.pt",
        ]
        if os.path.exists("/kaggle/input"):
            candidate_paths.extend(glob.glob("/kaggle/input/**/checkpoint_latest.pt", recursive=True))

        for cand in candidate_paths:
            if os.path.exists(cand):
                print(f"[+] [Kaggle/Cloud Discovery] Found checkpoint at: {cand}")
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                try:
                    shutil.copy2(cand, latest_ckpt_path)
                    print(f"[+] Successfully staged to {latest_ckpt_path}")
                except Exception:
                    try:
                        os.symlink(cand, latest_ckpt_path)
                    except Exception:
                        pass
                break

    if getattr(args, "drive_url", None):
        args.cloud_url = args.cloud_url or args.drive_url

    # Cloud Checkpoint Auto-Sync (Google Drive / Hugging Face)
    if is_master and ((args.resume and not os.path.exists(latest_ckpt_path)) or getattr(args, "cloud_sync", False)):
        print(f"[*] [CloudSync] Checking cloud for latest checkpoint ({latest_ckpt_path} missing or sync requested)...")
        synced = sync_cloud_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            hf_repo=args.hf_repo,
            drive_id=args.drive_id,
            url=args.cloud_url,
            force_pull=getattr(args, "cloud_sync", False)
        )
        if synced and os.path.exists(synced):
            print(f"[+] [CloudSync] Ready to resume from: {synced}")

    if world_size > 1 and XLA_AVAILABLE and xm is not None:
        xm.rendezvous("checkpoint_staged")

    # Auto-restore hyperparameters from checkpoint if resuming
    resumed_ckpt = None
    if args.resume and os.path.exists(latest_ckpt_path):
        if is_master:
            print(f"[*] Inspecting checkpoint for saved hyperparameters: {latest_ckpt_path}")
        resumed_ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        saved_hp = resumed_ckpt.get("hyperparameters", {})
        if saved_hp:
            if is_master:
                print("[*] Automatically restoring saved hyperparameters:")
            for hp_key, hp_val in saved_hp.items():
                cli_flag_1 = f"--{hp_key.replace('_', '-')}"
                cli_flag_2 = f"--{hp_key}"
                # If user did NOT explicitly override flag on CLI, restore saved value
                if (cli_flag_1 not in sys.argv and cli_flag_2 not in sys.argv) and hasattr(args, hp_key):
                    setattr(args, hp_key, hp_val)
                    if is_master:
                        print(f"    • {hp_key}: {hp_val}")

    # Dynamic Hardware Profile Auto-Tuning (RTX 3050 Ti, Tesla T4, TPU v5e, TPU v3-8, etc.)
    hw = detect_hardware_profile(
        user_batch_size=args.batch_size,
        user_grad_accum=args.grad_accum,
        user_dtype=args.dtype,
        user_device=args.device,
        seq_len=args.seq_len,
    )
    args.batch_size = hw.batch_size
    args.grad_accum = hw.grad_accum
    dtype = hw.dtype
    target_device = hw.device
    args.device = hw.device_str
    is_cuda_dev = hw.is_cuda
    is_tpu_dev = hw.is_tpu
    if is_master:
        hw.print_banner()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 1. Tokenizer
    tokenizer = get_or_create_tokenizer("tokenizer.json")
    vocab_size = tokenizer.vocab_size

    # 2. Models
    cfg = TransformerConfig(vocab_size=vocab_size, max_seq_len=args.seq_len)
    model = TinyTransformer(cfg).to(target_device)
    model.tie_weights()
    cast_model_dtypes(model, is_cuda_dev or is_tpu_dev, dtype)

    # Keep Critic on CPU unless enable_critic=True to save VRAM and avoid CUDA overhead
    critic_device = target_device if args.enable_critic else "cpu"
    critic = CriticModel(in_dim=cfg.hidden_dim, max_seq_len=args.seq_len).to(critic_device)
    if is_cuda_dev or is_tpu_dev:
        critic = critic.to(dtype=dtype)

    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()
    else:
        model.disable_gradient_checkpointing()

    breakdown = model.get_parameter_breakdown()

    # 3. Optimizers
    optimizers = create_dual_optimizers(
        model=model,
        critic=critic,
        lr_muon=args.lr_muon,
        lr_adam=args.lr_adam,
        critic_lr=args.critic_lr,
    )

    # 3b. Parameter groups for separate gradient clipping
    # Prevents spikes in tied embeddings from starving Muon momentum and Engram memory
    muon_clip_params = []
    if "muon" in optimizers:
        for pg in optimizers["muon"].param_groups:
            muon_clip_params.extend(pg["params"])
    muon_param_set = set(muon_clip_params)

    engram_clip_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and ("memory_table" in name or "centroids" in name)
    ]
    engram_param_set = set(engram_clip_params)

    adam_dense_clip_params = [
        p for p in model.parameters()
        if p.requires_grad and p not in muon_param_set and p not in engram_param_set
    ]

    # 4. Checkpoint Resumption
    start_step = 0
    total_tokens_processed = 0
    ema_loss = None
    ema_ppl = None
    dataset_resume_state = None

    if resumed_ckpt is not None:
        missing, unexpected = model.load_state_dict(resumed_ckpt["model_state_dict"], strict=False)
        print(f"[resume] missing={missing} unexpected={unexpected}")
        assert not missing, "checkpoint doesn't match model"
        model.tie_weights()
        cast_model_dtypes(model, is_cuda_dev or is_tpu_dev, dtype)
        critic.load_state_dict(resumed_ckpt["critic_state_dict"], strict=False)
        muon_sd = resumed_ckpt.get("opt_muon_state_dict") or resumed_ckpt.get("muon_state_dict")
        adam_sd = resumed_ckpt.get("opt_adam_state_dict") or resumed_ckpt.get("adam_state_dict")
        critic_sd = resumed_ckpt.get("opt_critic_state_dict")
        if muon_sd and "muon" in optimizers:
            try:
                optimizers["muon"].load_state_dict(muon_sd)
            except Exception as e:
                print(f"[Optimizer Warning] Could not load Muon optimizer state ({e}). Resuming with fresh momentum.")
        if adam_sd and "adam" in optimizers:
            try:
                optimizers["adam"].load_state_dict(adam_sd)
            except Exception as e:
                print(f"[Optimizer Warning] Could not load AdamW optimizer state ({e}). Resuming with fresh moments.")
        if critic_sd and "critic" in optimizers:
            try:
                optimizers["critic"].load_state_dict(critic_sd)
            except Exception as e:
                print(f"[Optimizer Warning] Could not load Critic optimizer state ({e}). Resuming with fresh moments.")

        start_step = resumed_ckpt["step"]
        total_tokens_processed = resumed_ckpt.get("tokens_processed", 0)
        metrics = resumed_ckpt.get("metrics") or {}
        ema_loss = metrics.get("ema_loss")
        ema_ppl = metrics.get("ema_ppl")
        dataset_resume_state = resumed_ckpt.get("dataset_state") or resumed_ckpt.get("dataloader_state")
        session_history = list(resumed_ckpt.get("session_history") or [])
        print(f"[*] Successfully resumed at step {start_step:,} with {total_tokens_processed:,} tokens seen.")
        # Free checkpoint memory immediately
        del resumed_ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif args.resume:
        print(f"[!] No checkpoint found at {latest_ckpt_path}. Starting clean pretraining.")
        session_history = []
    else:
        session_history = []

    if args.compile and hasattr(torch, "compile"):
        print("[*] Compiling model with torch.compile() for max kernel fusion...")
        try:
            model = torch.compile(model)
            print("[+] Model successfully compiled.")
        except Exception as e:
            print(f"[!] torch.compile failed ({e}), continuing in eager mode.")

    # 5. Dataset
    dataset = StreamingDatasetMixture(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=getattr(args, "seed", 42) + rank,
        resume_state=dataset_resume_state,
        effective_batch_size=args.batch_size * args.grad_accum,
    )
    data_iter = iter(dataset)

    # 6. Session Logger (Separate timestamped CSV & JSONL with lineage tracking)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_resumed" if (args.resume and start_step > 0) else ""
    session_full_id = f"session_{session_id}{suffix}"
    session_start_tokens = total_tokens_processed
    session_order = len(session_history) + 1

    jsonl_log_path = os.path.join(args.log_dir, f"{session_full_id}.jsonl")
    csv_log_path = os.path.join(args.log_dir, f"{session_full_id}.csv")
    hp_log_path = os.path.join(args.log_dir, f"{session_full_id}_hyperparameters.json")

    if is_master:
        csv_file = open(csv_log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step", "tokens_processed", "loss", "avg_loss", "ppl", "avg_ppl",
            "lr_muon", "lr_adam", "critic_loss", "real_acc", "fake_acc", "adv_loss", "reward",
            "lambda_adv", "tokens_per_sec", "ram_mb", "vram_mb", "elapsed_sec", "timestamp"
        ])
        csv_file.flush()
        jsonl_file = open(jsonl_log_path, "w", encoding="utf-8")
    else:
        csv_file = None
        csv_writer = None
        jsonl_file = None

    CHINCHILLA_TARGET = 2_600_000_000  # 2.6B tokens for 130M dense backbone (20x scaling)

    current_hyperparameters = {
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "max_steps": args.max_steps,
        "lr_muon": args.lr_muon,
        "lr_adam": args.lr_adam,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "critic_lr": args.critic_lr,
        "critic_start_tokens": args.critic_start_tokens,
        "critic_ramp_tokens": args.critic_ramp_tokens,
        "critic_start_step": args.critic_start_step,
        "critic_loss_target": args.critic_loss_target,
        "critic_slice_len": args.critic_slice_len,
        "critic_accum_steps": args.critic_accum_steps,
        "enable_critic": args.enable_critic,
        "adv_weight": args.adv_weight,
        "device": args.device,
        "dtype": str(dtype),
        "chinchilla_target_tokens": CHINCHILLA_TARGET,
    }

    # Save human-readable hyperparameters to session logs
    if is_master:
        with open(hp_log_path, "w", encoding="utf-8") as f:
            json.dump(current_hyperparameters, f, indent=2)

    def get_session_metadata(curr_step: int, curr_tokens: int):
        s_info = {
            "current_session_id": session_full_id,
            "session_order": session_order,
            "session_start_step": start_step,
            "current_step": curr_step,
            "session_start_tokens": session_start_tokens,
            "tokens_processed": curr_tokens,
            "associated_files": {
                "log_csv": os.path.abspath(csv_log_path),
                "log_jsonl": os.path.abspath(jsonl_log_path),
                "hyperparameters_json": os.path.abspath(hp_log_path),
                "tokenizer_file": os.path.abspath("tokenizer.json"),
                "checkpoint_latest": os.path.abspath(os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")),
                "checkpoint_prev": os.path.abspath(os.path.join(args.checkpoint_dir, "checkpoint_prev.pt")),
                "checkpoint_manifest": os.path.abspath(os.path.join(args.checkpoint_dir, "checkpoint_manifest.json")),
            },
            "relative_files": {
                "log_csv": os.path.normpath(csv_log_path),
                "log_jsonl": os.path.normpath(jsonl_log_path),
                "hyperparameters_json": os.path.normpath(hp_log_path),
                "tokenizer_file": "tokenizer.json",
            },
            "is_resumed": bool(args.resume and start_step > 0),
            "timestamp": datetime.now().isoformat(),
        }
        current_record = {
            "order": session_order,
            "session_id": session_full_id,
            "step_range": [start_step, curr_step],
            "tokens_range": [session_start_tokens, curr_tokens],
            "log_csv": os.path.normpath(csv_log_path),
            "log_jsonl": os.path.normpath(jsonl_log_path),
            "hyperparameters_json": os.path.normpath(hp_log_path),
            "resumed": bool(args.resume and start_step > 0),
            "timestamp": datetime.now().isoformat(),
        }
        full_history = session_history + [current_record]
        return s_info, full_history

    # CLI Banner
    if is_master:
        print("\n" + "=" * 80)
        print("  [>] TINYTRANSFORMER PRETRAINING (GOATED CLI)")
        print("=" * 80)
        print(f"* Total Parameters:  {breakdown['total_params']/1e6:.2f}M ({breakdown['unique_params']:,} unique, ~237.8M params)")
        print(f"* Backbone Layers:   {cfg.num_layers} layers, hidden={cfg.hidden_dim}, heads={cfg.num_heads}, ffn_mult={cfg.ffn_mult} ({breakdown['backbone_params']/1e6:.2f}M params)")
        print(f"* Engram Tables:     80M global (post-L16) + two 10M at L{cfg.engram_layers[0]}/L{cfg.engram_layers[1]} ({breakdown['engram_params']/1e6:.2f}M params)")
        print(f"* Critic Model:      Auxiliary Transformer BS-Detector ({critic.num_parameters/1e6:.2f}M params)")
        print(f"* Weights Footprint: {breakdown['bf16_weight_size_mb']:.1f} MB in BF16")
        print(f"* Optimizers:        Keller Jordan Muon (2D internal) + AdamW (Embeds/Engram/Norms)")
        print(f"* LR Schedule:       Warmup {args.lr_warmup_steps} st, Cosine to {args.lr_decay_steps:,} st (min {args.lr_min_ratio:.1f}x) | Base Muon: {args.lr_muon:.4f}, AdamW: {args.lr_adam:.6f}")
        print(f"* Critic Schedule:   Start: {args.critic_start_tokens/1e6:.0f}M tokens, Ramp: {args.critic_ramp_tokens/1e6:.0f}M tokens, Subsample: {args.critic_accum_steps}/{args.grad_accum} batches ({args.critic_slice_len} tok slice), Freeze: <0.20")
        mix_desc = ", ".join(f"{int(c['weight']*100)}% {c['name'].capitalize()}" for c in STREAM_CONFIGS)
        print(f"* Mixture Ratio:     {mix_desc}")
        print(f"* Device / Dtype:    {args.device.upper()} / {args.dtype} (Cores: {world_size})")
        print(f"* Batch Config:      Micro-batch {args.batch_size} x Accum {args.grad_accum} x {world_size} Cores (Effective: {args.batch_size * args.grad_accum * args.seq_len * world_size:,} tokens/step)")
        print(f"* Chinchilla Target: 2.60 Billion tokens ({total_tokens_processed/1e6:.2f}M seen, {total_tokens_processed / CHINCHILLA_TARGET * 100:.3f}% reached)")
        print(f"* Session Log:       {csv_log_path}")
        print("=" * 80 + "\n")

    step = start_step
    start_time = time.time()
    last_log_time = time.time()
    last_tokens_count = total_tokens_processed

    model.train()
    critic.train()

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and is_cuda_dev)) if is_cuda_dev else torch.amp.GradScaler("cpu", enabled=False)
    if scaler.is_enabled():
        print("[*] FP16 GradScaler enabled for numerical stability on Turing/T4 Tensor Cores.")

    # Zero all gradients
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    accum_loss_lm = 0.0
    accum_loss_critic = 0.0
    accum_loss_adv = 0.0
    accum_reward = 0.0
    accum_real_acc = 0.0
    accum_fake_acc = 0.0

    recent_log_history = deque(maxlen=10)
    current_loss = 0.0
    current_ppl = 0.0

    VAL_SHARDS = {os.path.basename(p).split("_")[1]: p for p in sorted(glob.glob("data/shards/val_*.bin"))}

    while step < args.max_steps:
        step += 1
        step_tokens = 0
        accum_loss_lm_tensor = torch.zeros(1, device=target_device)
        accum_aux_loss_tensor = torch.zeros(1, device=target_device)
        accum_loss_critic_tensor = torch.zeros(1, device=target_device)
        accum_loss_adv_tensor = torch.zeros(1, device=target_device)
        accum_reward_tensor = torch.zeros(1, device=target_device)
        accum_real_acc_tensor = torch.zeros(1, device=target_device)
        accum_fake_acc_tensor = torch.zeros(1, device=target_device)
        accum_loss_adv = 0.0
        accum_reward = 0.0
        accum_real_acc = 0.0
        accum_fake_acc = 0.0

        # Update dynamic learning rates via linear warmup + cosine decay schedule
        lr_multiplier = get_lr_multiplier(
            step,
            warmup_steps=args.lr_warmup_steps,
            max_steps=min(args.max_steps, args.lr_decay_steps),
            min_ratio=args.lr_min_ratio,
        )
        curr_lr_muon = args.lr_muon * lr_multiplier
        curr_lr_adam = args.lr_adam * lr_multiplier
        if "muon" in optimizers:
            for pg in optimizers["muon"].param_groups:
                pg["lr"] = curr_lr_muon
        if "adam" in optimizers:
            for pg in optimizers["adam"].param_groups:
                pg["lr"] = curr_lr_adam

        # Determine Critic activation via token-based schedule
        loss_gated = (ema_loss is None or ema_loss <= args.critic_loss_target)
        token_warmed = (total_tokens_processed >= args.critic_start_tokens)
        critic_active = args.enable_critic and token_warmed and loss_gated

        if not critic_active:
            alpha = 1.0
            beta = 0.0
            lambda_adv = 0.0
            if not args.enable_critic:
                critic_status = "Disabled (Pure Foundation Pretraining)"
            else:
                critic_status = f"Dormant (tokens<{args.critic_start_tokens/1e6:.1f}M or loss>{args.critic_loss_target})"
            num_critic_micro = 1
            critic_micro_steps = set()
        else:
            active_tokens = max(0, total_tokens_processed - args.critic_start_tokens)
            ramp = min(1.0, float(active_tokens) / float(max(1, args.critic_ramp_tokens)))
            # Alpha / Beta convex blending (1.0 -> 1.0 - adv_weight, 0.0 -> adv_weight)
            alpha = 1.0 - args.adv_weight * ramp
            beta = args.adv_weight * ramp
            lambda_adv = beta
            critic_status = f"Active (α={alpha:.2f}, β={beta:.2f})"
            # Subsample micro-batches for Critic evaluation to eliminate PCIe memory thrashing & maximize throughput
            critic_stride = max(1, args.grad_accum // max(1, args.critic_accum_steps))
            critic_micro_steps = set(m for m in range(args.grad_accum) if m % critic_stride == 0)
            num_critic_micro = max(1, len(critic_micro_steps))

        # Gradient accumulation loop
        for micro_step in range(args.grad_accum):
            # Fetch micro-batch directly to device
            if args.batch_size == 1:
                x, y = next(data_iter)
                bx = x.unsqueeze(0)
                by = y.unsqueeze(0)
            elif args.batch_size == 2:
                x1, y1 = next(data_iter)
                x2, y2 = next(data_iter)
                bx = torch.stack([x1, x2])
                by = torch.stack([y1, y2])
            else:
                batch_x_list = []
                batch_y_list = []
                for _ in range(args.batch_size):
                    x, y = next(data_iter)
                    batch_x_list.append(x)
                    batch_y_list.append(y)
                bx = torch.stack(batch_x_list)
                by = torch.stack(batch_y_list)

            if is_cuda_dev:
                bx = bx.pin_memory().to(target_device, non_blocking=True)
                by = by.pin_memory().to(target_device, non_blocking=True)
            else:
                bx = bx.to(target_device)
                by = by.to(target_device)
            step_tokens += bx.numel() * world_size

            # 1. Generator Forward (Clean Next-Token Prediction under autocast)
            if is_cuda_dev:
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits, _, _, aux_loss = model(bx, return_hidden=False)
            elif is_tpu_dev:
                logits, _, _, aux_loss = model(bx, return_hidden=False)
            else:
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    logits, _, _, aux_loss = model(bx, return_hidden=False)

            # Vectorized cross-entropy loss computation in FP32 precision to prevent precision loss & underflow
            loss_lm = F.cross_entropy(
                logits.float().view(-1, vocab_size),
                by.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            # Base next-token generator loss scaled over all grad_accum steps
            loss_lm_scaled = (alpha * loss_lm + 0.01 * aux_loss) / args.grad_accum

            run_critic_now = critic_active and (micro_step in critic_micro_steps)
            if run_critic_now:
                # Sequence slicing to keep peak VRAM strictly <3.3 GB and eliminate PCIe memory thrashing
                slice_len = min(args.critic_slice_len, bx.size(1))
                logits_slice = logits[:, :slice_len, :]
                by_slice = by[:, :slice_len]

                critic.requires_grad_(False)
                real_embeds = model.get_token_embeddings(by_slice).detach()
                fake_embeds = model.sample_embeddings_gumbel(logits_slice, tau=1.0)

                if beta > 0.0:
                    loss_adv, reward = compute_generator_adversarial_loss(critic, fake_embeds)
                    loss_adv_scaled = (beta * loss_adv) / num_critic_micro
                    if is_tpu_dev:
                        reward_val = reward.mean().detach()
                        adv_loss_val = loss_adv.detach()
                    else:
                        reward_val = reward.mean().item()
                        adv_loss_val = loss_adv.item()
                else:
                    loss_adv_scaled = 0.0
                    reward_val = torch.tensor(0.5, device=target_device) if is_tpu_dev else 0.5
                    adv_loss_val = torch.tensor(0.0, device=target_device) if is_tpu_dev else 0.0

                loss_total = loss_lm_scaled + loss_adv_scaled
                scaler.scale(loss_total).backward()

                # Critic Step (re-enable critic grads; detached fake_embeds prevents double backprop into generator)
                critic.requires_grad_(True)
                loss_critic, real_acc, fake_acc = compute_critic_loss(
                    critic, real_embeds, fake_embeds.detach(), label_smoothing=True
                )
                scaler.scale(loss_critic / num_critic_micro).backward()

                accum_loss_critic_tensor.add_(loss_critic.detach().to(target_device) / num_critic_micro)
                if is_tpu_dev:
                    accum_loss_adv_tensor.add_(adv_loss_val.to(target_device) / num_critic_micro)
                    accum_reward_tensor.add_(reward_val.to(target_device) / num_critic_micro)
                    accum_real_acc_tensor.add_(real_acc.to(target_device) / num_critic_micro)
                    accum_fake_acc_tensor.add_(fake_acc.to(target_device) / num_critic_micro)
                else:
                    accum_loss_adv += adv_loss_val / num_critic_micro
                    accum_reward += reward_val / num_critic_micro
                    accum_real_acc += real_acc / num_critic_micro
                    accum_fake_acc += fake_acc / num_critic_micro
            else:
                scaler.scale(loss_lm_scaled).backward()

            # Async non-blocking accumulation of generator loss on GPU
            accum_loss_lm_tensor.add_(loss_lm.detach())
            accum_aux_loss_tensor.add_(aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else aux_loss)

        # Resolve accumulated losses with single sync
        accum_loss_lm = accum_loss_lm_tensor.item() / args.grad_accum
        accum_aux_loss = accum_aux_loss_tensor.item() / args.grad_accum
        if critic_active:
            accum_loss_critic = accum_loss_critic_tensor.item()
            if is_tpu_dev:
                accum_loss_adv = accum_loss_adv_tensor.item()
                accum_reward = accum_reward_tensor.item()
                accum_real_acc = accum_real_acc_tensor.item()
                accum_fake_acc = accum_fake_acc_tensor.item()
        else:
            accum_loss_critic = 0.0
            accum_real_acc = 0.5
            accum_fake_acc = 0.5
            accum_reward = 0.5
            accum_loss_adv = 0.0

        # Unscale before gradient clipping if scaler is active
        if scaler.is_enabled():
            if muon_clip_params:
                scaler.unscale_(optimizers["muon"])
            if "adam" in optimizers:
                scaler.unscale_(optimizers["adam"])
            if critic_active and "critic" in optimizers:
                scaler.unscale_(optimizers["critic"])

        # Separate gradient clipping for Muon parameters, AdamW dense parameters, and Engram memory parameters
        # Prevents tied embedding spikes from starving Muon momentum and Engram updates
        use_foreach = is_cuda_dev
        gn = {}
        if muon_clip_params:
            gn["muon"] = torch.nn.utils.clip_grad_norm_(muon_clip_params, max_norm=1.0, foreach=use_foreach)
        if adam_dense_clip_params:
            gn["dense"] = torch.nn.utils.clip_grad_norm_(adam_dense_clip_params, max_norm=1.0, foreach=use_foreach)
        if engram_clip_params:
            gn["engram"] = torch.nn.utils.clip_grad_norm_(engram_clip_params, max_norm=1.0, foreach=use_foreach)
        if critic_active:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0, foreach=use_foreach)

        if (not math.isfinite(accum_loss_lm)) or not all(torch.isfinite(v) for v in gn.values()):
            print(f"[!] step {step}: non-finite loss/grad, skipping update")
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.update()
            continue

        if is_tpu_dev and XLA_AVAILABLE and xm is not None:
            xm.optimizer_step(optimizers["muon"])
            xm.optimizer_step(optimizers["adam"])
            if critic_active and accum_loss_critic >= 0.20:
                xm.optimizer_step(optimizers["critic"])
            xm.mark_step()
        elif scaler.is_enabled():
            scaler.step(optimizers["muon"])
            scaler.step(optimizers["adam"])
            # Dynamic Critic freeze: only step critic optimizer if loss_critic >= 0.20
            if critic_active and accum_loss_critic >= 0.20:
                scaler.step(optimizers["critic"])
            scaler.update()
        else:
            optimizers["muon"].step()
            optimizers["adam"].step()
            # Dynamic Critic freeze: only step critic optimizer if loss_critic >= 0.20
            if critic_active and accum_loss_critic >= 0.20:
                optimizers["critic"].step()

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        total_tokens_processed += step_tokens

        # Moving averages
        current_loss = accum_loss_lm
        current_ppl = math.exp(min(current_loss, 20.0))

        if ema_loss is None:
            ema_loss = current_loss
            ema_ppl = current_ppl
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * current_loss
            # Exponential moving average of perplexity calculated from ema_loss to avoid Jensen's inequality bias
            ema_ppl = math.exp(min(ema_loss, 20.0))

        # Telemetry & Logging
        if is_master and (step % args.log_interval == 0 or step == 1 or step == args.max_steps):
            now = time.time()
            elapsed_interval = max(now - last_log_time, 1e-5)
            interval_tokens = total_tokens_processed - last_tokens_count
            tok_per_sec = interval_tokens / elapsed_interval

            total_elapsed = max(now - start_time, 1e-5)
            steps_done = step - start_step
            sec_per_step = total_elapsed / max(steps_done, 1)

            # Milestones: 4:1 (520M tok) and 8:1 (1.0B tok) calculated dynamically from tokens_per_step
            tokens_per_step = args.batch_size * args.grad_accum * args.seq_len * world_size
            target_4_1_steps = 520_000_000 // max(1, tokens_per_step)
            target_8_1_steps = 1_000_000_000 // max(1, tokens_per_step)
            eta_4_1 = max(0.0, sec_per_step * (target_4_1_steps - step))
            eta_8_1 = max(0.0, sec_per_step * (target_8_1_steps - step))

            # RAM & VRAM (resting allocated vs true peak during step)
            process = psutil.Process()
            mi = process.memory_full_info()
            ram_mb = mi.rss / (1024 * 1024)
            uss_mb = getattr(mi, "uss", mi.rss) / (1024 * 1024)
            vram_alloc_mb = 0.0
            vram_peak_mb = 0.0
            if torch.cuda.is_available():
                vram_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats()
            vram_mb = vram_alloc_mb

            # Critic CLI indicator
            if not critic_active:
                if not args.enable_critic:
                    critic_cli = "Critic: [Disabled (Pure Foundation)]"
                else:
                    critic_cli = f"Critic: [Dormant: tok<{args.critic_start_tokens/1e6:.0f}M or loss>{args.critic_loss_target}]"
            else:
                if accum_loss_critic < 0.20:
                    critic_cli = f"Critic: [Frozen: Closs={accum_loss_critic:.2f}<0.20, α={alpha:.2f}, β={beta:.2f}]"
                else:
                    critic_cli = f"Critic: [Act: α={alpha:.2f}, β={beta:.2f}, Closs={accum_loss_critic:.2f}, Acc={0.5*(accum_real_acc+accum_fake_acc):.2f}]"

            # Goated Terminal Print
            pct_chinchilla = (total_tokens_processed / CHINCHILLA_TARGET) * 100.0
            gn_str = " ".join(f"{k}:{float(v):.2f}" for k, v in gn.items())
            log_line = (
                f"[{step:05d}/{args.max_steps:05d}] "
                f"Loss: {current_loss:6.3f} (avg: {ema_loss:6.3f}) | "
                f"PPL: {current_ppl:6.1f} (avg: {ema_ppl:6.1f}) | "
                f"LR: [M:{curr_lr_muon:.5f}|A:{curr_lr_adam:.2e}] | "
                f"GN: [{gn_str}] | "
                f"{critic_cli} | "
                f"Tokens: {total_tokens_processed:,} ({total_tokens_processed/1e6:5.2f}M | {tok_per_sec:5.0f} t/s | {pct_chinchilla:.3f}%) | "
                f"RAM: {ram_mb:4.0f}MB (priv {uss_mb:4.0f}) | "
                f"VRAM: {vram_alloc_mb:4.0f}MB (pk: {vram_peak_mb:4.0f}MB) | "
                f"Elapsed: {format_time(total_elapsed)} | "
                f"ETA(4:1): {format_time(eta_4_1)} | ETA(8:1): {format_time(eta_8_1)}"
            )
            if sys.stdout.isatty():
                # Clear active in-place line before printing the permanent block
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
            print(log_line)

            # Write to CSV
            if csv_writer:
                csv_writer.writerow([
                    step, total_tokens_processed, round(current_loss, 4), round(ema_loss, 4),
                    round(current_ppl, 2), round(ema_ppl, 2), round(curr_lr_muon, 6), round(curr_lr_adam, 8),
                    round(accum_loss_critic, 4), round(accum_real_acc, 3), round(accum_fake_acc, 3),
                    round(accum_loss_adv, 4), round(accum_reward, 3), round(lambda_adv, 3),
                    round(tok_per_sec, 1), round(ram_mb, 1), round(vram_mb, 1), round(total_elapsed, 1),
                    datetime.now().isoformat()
                ])
                csv_file.flush()
            recent_log_history.append({
                "step": step,
                "tokens_processed": total_tokens_processed,
                "loss": round(current_loss, 4),
                "avg_loss": round(ema_loss, 4),
                "ppl": round(current_ppl, 2),
                "avg_ppl": round(ema_ppl, 2),
                "tokens_per_sec": round(tok_per_sec, 1),
                "vram_mb": round(vram_mb, 1),
                "timestamp": datetime.now().isoformat(),
            })
            last_log_time = now
            last_tokens_count = total_tokens_processed

            # Write to JSONL
            if jsonl_file:
                json_record = {
                    "step": step,
                    "tokens_processed": total_tokens_processed,
                    "loss": current_loss,
                    "avg_loss": ema_loss,
                    "ppl": current_ppl,
                    "avg_ppl": ema_ppl,
                    "lr_multiplier": lr_multiplier,
                    "lr_muon": curr_lr_muon,
                    "lr_adam": curr_lr_adam,
                    "critic_loss": accum_loss_critic,
                    "real_acc": accum_real_acc,
                    "fake_acc": accum_fake_acc,
                    "adv_loss": accum_loss_adv,
                    "reward": accum_reward,
                    "lambda_adv": lambda_adv,
                    "tokens_per_sec": tok_per_sec,
                    "vram_mb": vram_mb,
                    "timestamp": datetime.now().isoformat(),
                }
                jsonl_file.write(json.dumps(json_record) + "\n")
                jsonl_file.flush()
        elif is_master:
            # Active in-place line: updates live figures on the SAME single line in interactive terminals
            # Non-TTY file logs strictly receive clean 10-step permanent blocks
            if sys.stdout.isatty():
                now = time.time()
                total_elapsed = max(now - start_time, 1e-5)
                vram_cur = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                step_in_block = step % args.log_interval if (step % args.log_interval) != 0 else args.log_interval
                active_line = (
                    f"\r  >> [{step:05d}/{args.max_steps:05d}] ({step_in_block}/{args.log_interval}) | "
                    f"Loss: {current_loss:5.3f} (avg: {ema_loss:5.3f}) | "
                    f"PPL: {current_ppl:4.1f} | "
                    f"{total_tokens_processed/1e6:5.2f}M tok | "
                    f"{vram_cur:4.0f}MB"
                )
                sys.stdout.write(active_line[:80].ljust(80))
                sys.stdout.flush()

        # Periodic memory cleanup
        if step % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                import pyarrow as pa
                pa.default_memory_pool().release_unused()
            except Exception:
                pass

        # Dynamic check for infer_interval override (5x increased by default)
        for check_path in ["infer_interval.txt", os.path.join(args.checkpoint_dir, "infer_interval.txt")]:
            if os.path.exists(check_path):
                try:
                    with open(check_path, "r") as f:
                        val = int(f.read().strip())
                        if val > 0 and val != args.infer_interval:
                            args.infer_interval = val
                            print(f"[*] Dynamically updated infer_interval to {args.infer_interval} steps.")
                except Exception:
                    pass

        # Mini-Inference Generation Test
        infer_outputs = None
        if is_master and step % args.infer_interval == 0:
            infer_outputs = run_mini_inference(model, tokenizer, step, args.device, dtype=dtype)
            if VAL_SHARDS:
                val = eval_fixed(model, VAL_SHARDS, args.device)
                W = {c["name"]: c["weight"] for c in STREAM_CONFIGS}
                wmean = sum(v * W[k] for k, v in val.items()) / sum(W[k] for k in val)
                print("[VAL] " + " | ".join(f"{k}: {v:.3f}" for k, v in val.items()) + f" | wmean: {wmean:.3f}")
            # Immediately reclaim generation buffers and reset fragmented CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Log mini-inference to JSONL session
            if jsonl_file:
                infer_record = {
                    "step": step,
                    "type": "mini_inference",
                    "generations": infer_outputs,
                    "timestamp": datetime.now().isoformat(),
                }
                jsonl_file.write(json.dumps(infer_record) + "\n")
                jsonl_file.flush()

        # Checkpoint Saving
        if is_master and (step % args.save_interval == 0 or step == args.max_steps):
            s_info, full_hist = get_session_metadata(step, total_tokens_processed)
            save_metrics = {
                "ema_loss": ema_loss,
                "ema_ppl": ema_ppl,
                "current_loss": current_loss,
                "current_ppl": current_ppl,
            }
            save_atomic_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                step=step,
                tokens_processed=total_tokens_processed,
                model=model,
                critic=critic,
                optimizers=optimizers,
                dataset=dataset,
                metrics=save_metrics,
                hyperparameters=current_hyperparameters,
                session_info=s_info,
                session_history=full_hist,
                sync=(step == args.max_steps),
            )
            print(f"[*] Checkpoint atomically saved at step {step} to {args.checkpoint_dir}/checkpoint_latest.pt")

            # Milestone Checkpoint Archiving with .md and .json reports (every 1,000 steps by default)
            if args.archive_interval > 0 and step % args.archive_interval == 0 and step > 0:
                archive_milestone_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    step=step,
                    tokens_processed=total_tokens_processed,
                    metrics=save_metrics,
                    recent_logs=list(recent_log_history),
                    infer_outputs=infer_outputs,
                    hyperparameters=current_hyperparameters,
                )

        # Check remote control flags for safe runtime interaction
        if is_master and (os.path.exists("stop.flag") or os.path.exists("save.flag")):
            s_info, full_hist = get_session_metadata(step, total_tokens_processed)
            save_metrics = {
                "ema_loss": ema_loss,
                "ema_ppl": ema_ppl,
                "current_loss": current_loss,
            }
            save_atomic_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                step=step,
                tokens_processed=total_tokens_processed,
                model=model,
                critic=critic,
                optimizers=optimizers,
                dataset=dataset,
                metrics=save_metrics,
                hyperparameters=current_hyperparameters,
                session_info=s_info,
                session_history=full_hist,
                sync=True,
            )
            print(f"[*] Checkpoint explicitly saved via flag at step {step}.")
            latest_saved_pt = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
            is_valid, msg, cinfo = verify_checkpoint_integrity(latest_saved_pt)
            if is_valid:
                print(f"[+] [Anti-Corruption Guard] Checkpoint verified healthy at step {step} (0 NaNs, 0 Infs).")
                if os.path.exists("stop.flag") or getattr(args, "auto_push_cloud", False):
                    auto_sync_on_interrupt(args.checkpoint_dir, hf_repo=args.hf_repo)
            else:
                print(f"[!] [CRITICAL ALERT] Checkpoint failed integrity check: {msg}")

            if os.path.exists("save.flag"):
                try: os.remove("save.flag")
                except Exception: pass
            if os.path.exists("stop.flag"):
                try: os.remove("stop.flag")
                except Exception: pass
                print("[*] stop.flag detected. Exiting training cleanly.")
                break

        # Dynamic critic activation flag
        if os.path.exists("critic.flag"):
            args.enable_critic = True
            args.adv_weight = 0.3
            args.critic_start_tokens = total_tokens_processed
            args.critic_start_step = step
            args.critic_loss_target = 20.0
            critic_is_cuda = next(critic.parameters()).is_cuda
            if not critic_is_cuda and torch.cuda.is_available():
                critic = critic.to("cuda", dtype=dtype)
                optimizers["critic"] = create_critic_optimizer(critic, critic_lr=args.critic_lr)
            try: os.remove("critic.flag")
            except Exception: pass
            if is_master:
                print(f"[*] critic.flag detected! Activated Critic co-training dynamically at step {step}.")

        accum_loss_lm = 0.0
        accum_loss_critic = 0.0
        accum_loss_adv = 0.0
        accum_reward = 0.0
        accum_real_acc = 0.0
        accum_fake_acc = 0.0

    if is_master:
        wait_for_async_save()
        if csv_file:
            csv_file.close()
        if jsonl_file:
            jsonl_file.close()
        print("\n[OK] Pretraining completed successfully!")


def train():
    parser = argparse.ArgumentParser(description="TinyTransformer Pretraining Script")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device micro-batch size (auto-tuned if not set)")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (auto-tuned if not set)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length in tokens")
    parser.add_argument("--max-steps", type=int, default=100_000_000, help="Maximum pretraining steps")
    parser.add_argument("--lr-muon", type=float, default=0.006, help="Learning rate for Muon optimizer (default: 0.006)")
    parser.add_argument("--lr-adam", type=float, default=3e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--lr-warmup-steps", type=int, default=50, help="Linear warmup steps (default: 50 for smooth early learning)")
    parser.add_argument("--lr-decay-steps", type=int, default=79_000, help="Cosine decay target steps matching 2.6B token Chinchilla target (79k steps at 32k tokens/step)")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1, help="Minimum learning rate ratio at the end of decay")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True, help="Enable activation checkpointing for maximum VRAM headroom")
    parser.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint", help="Disable activation checkpointing")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Learning rate for Critic optimizer")
    parser.add_argument("--critic-start-tokens", type=int, default=250_000_000, help="Cumulative token threshold before activating adversarial RL loss")
    parser.add_argument("--critic-ramp-tokens", type=int, default=100_000_000, help="Token count over which alpha/beta convex blending ramps")
    parser.add_argument("--critic-start-step", "--critic-warmup-steps", type=int, default=7_500, help="Legacy step threshold before activating adversarial RL loss")
    parser.add_argument("--critic-loss-target", type=float, default=2.4, help="Loss target threshold below which critic co-training activates (default: 2.4)")
    parser.add_argument("--adv-weight", type=float, default=0.3, help="Max adversarial loss weight for generator (default: 0.3)")
    parser.add_argument("--enable-critic", action="store_true", help="Enable co-training of the auxiliary critic model")
    parser.add_argument("--critic-slice-len", type=int, default=256, help="Sequence length slice for Critic Gumbel-Softmax evaluation to reduce VRAM (default: 256)")
    parser.add_argument("--critic-accum-steps", type=int, default=2, help="Number of micro-batches per optimizer step to evaluate Critic on (default: 2)")
    parser.add_argument("--save-interval", type=int, default=250, help="Checkpoint save interval")
    parser.add_argument("--infer-interval", type=int, default=1250, help="Mini-inference interval (5x increased: 1250 steps)")
    parser.add_argument("--archive-interval", type=int, default=2500, help="Checkpoint archive interval with .md and .json reports (default: 2500 steps)")
    parser.add_argument("--log-interval", type=int, default=10, help="Console logging interval")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Session logs directory")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, xla, or cpu - auto-detected if None)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                        help="Compute dtype (auto selects bfloat16 on Ampere/Ada/Hopper or float16 on T4/V100 for maximum Tensor Core throughput)")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile for maximum kernel fusion speed")
    parser.add_argument("--auto-profile", action="store_true", default=True, help="Auto-tune for RTX 3050 Ti, Tesla T4, or TPU v5e")
    parser.add_argument("--cloud-sync", "--pull-cloud", action="store_true", help="Check and download latest checkpoint from cloud before resuming")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID for cloud checkpoint sync (e.g. username/repo)")
    parser.add_argument("--drive-id", type=str, default=None, help="Google Drive file ID for checkpoint sync")
    parser.add_argument("--drive-url", type=str, default=None, help="Google Drive share link to checkpoint_latest.pt")
    parser.add_argument("--cloud-url", type=str, default=None, help="Direct HTTP/HTTPS checkpoint URL")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for dataset sampling")
    parser.add_argument("--tpu-cores", type=int, default=None, choices=[1, 8], help="Number of TPU cores (1 or 8 for Kaggle TPU v3-8)")
    args = parser.parse_args()

    # Check for TPU multi-core distributed launch
    is_multi_proc = (
        bool(os.environ.get("WORLD_SIZE") and int(os.environ.get("WORLD_SIZE", "1")) > 1)
        or bool(os.environ.get("TPU_NUM_DEVICES") and int(os.environ.get("TPU_NUM_DEVICES", "1")) > 1)
    )

    is_kaggle_tpu = bool(os.environ.get("TPU_NAME")) or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    target_cores = args.tpu_cores or (8 if (is_kaggle_tpu and not is_multi_proc and XLA_AVAILABLE and xm is not None) else 1)

    if target_cores == 8 and not is_multi_proc and XLA_AVAILABLE and xm is not None:
        import torch_xla.distributed.xla_multiprocessing as xmp
        print("\n" + "=" * 80)
        print("  [>] SPAWNING 8-CORE DISTRIBUTED PRETRAINING ON KAGGLE TPU v3-8")
        print("=" * 80 + "\n", flush=True)
        xmp.spawn(_train_worker, args=(args,), nprocs=8, start_method="fork")
    else:
        _train_worker(0, args)


if __name__ == "__main__":
    train()
