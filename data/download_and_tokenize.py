"""
Chinchilla Target Data Downloader & NVMe Pre-Tokenization Pipeline.
Streams one dataset at a time to keep host RAM under 150MB, writing contiguous
uint16 binary shards to disk for instant zero-RAM memory-mapped training.

Mixture Ratio (SOTA SmolLM2-135M / MobileLLM Rebalanced):
- 45% FineWeb-Edu Dedup (HuggingFaceTB/smollm-corpus, fineweb-edu-dedup)
- 20% Cosmopedia v2 (HuggingFaceTB/smollm-corpus, cosmopedia-v2)
- 15% Python-Edu (Avelina/python-edu-cleaned)
- 15% FineMath-4plus (HuggingFaceTB/finemath, finemath-4plus)
- 5% Instruction (HuggingFaceTB/smol-smoltalk)
"""

import argparse
import gc
import glob
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Any, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import Tokenizer, get_or_create_tokenizer
from data.dataset import STREAM_CONFIGS, INSTRUCTION_SUBSETS, format_instruction_item

CJK_REGEX = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]')

CHINCHILLA_TARGET_TOKENS = 2_600_000_000  # 2.6 Billion tokens (20x parameter scaling)
DEFAULT_SHARD_SIZE = 5_000_000            # 5 Million tokens per shard (~10 MB uint16)


def get_shard_stats(shard_dir: str = "data/shards") -> Dict[str, Any]:
    """Scans local shards directory and calculates total tokens per stream."""
    if not os.path.exists(shard_dir):
        return {"total_tokens": 0, "by_stream": {}, "total_mb": 0.0}

    stats = {"total_tokens": 0, "by_stream": {}, "total_mb": 0.0}
    total_bytes = 0

    for cfg in STREAM_CONFIGS:
        name = cfg["name"]
        shards = sorted(glob.glob(os.path.join(shard_dir, f"{name}_*.bin")))
        stream_bytes = sum(os.path.getsize(f) for f in shards)
        stream_tokens = stream_bytes // 2
        total_bytes += stream_bytes
        stats["by_stream"][name] = {
            "tokens": stream_tokens,
            "shards": len(shards),
            "mb": stream_bytes / (1024 * 1024),
        }
        stats["total_tokens"] += stream_tokens

    stats["total_mb"] = total_bytes / (1024 * 1024)
    return stats


def download_and_tokenize_stream(
    cfg: Dict[str, Any],
    target_tokens: int,
    shard_size: int,
    shard_dir: str,
    tokenizer: Tokenizer,
):
    name = cfg["name"]
    from datasets import load_dataset

    existing_shards = sorted(glob.glob(os.path.join(shard_dir, f"{name}_*.bin")))
    existing_tokens = sum(os.path.getsize(f) // 2 for f in existing_shards)
    tokens_needed = max(0, target_tokens - existing_tokens)

    print(f"\n[*] Processing Stream: {name.upper()} (Target: {target_tokens:,} tokens, Existing: {existing_tokens:,})")
    if tokens_needed <= 0:
        print(f"    [OK] Stream '{name}' already satisfied on disk.")
        return

    existing_indices = []
    for f in existing_shards:
        base = os.path.splitext(os.path.basename(f))[0]
        parts = base.split("_")
        if parts[-1].isdigit():
            existing_indices.append(int(parts[-1]))
    shard_idx = max(existing_indices, default=-1) + 1

    seen_docs = set()
    for f in existing_shards + sorted(glob.glob(os.path.join(shard_dir, f"val_{name}_*.bin"))):
        a = np.fromfile(f, dtype=np.uint16)
        bos = np.flatnonzero(a == tokenizer.bos_token_id)
        eos = np.flatnonzero(a == tokenizer.eos_token_id)
        for b in bos:
            j = np.searchsorted(eos, b)
            if j < len(eos) and eos[j] - b > 128:
                seen_docs.add(hash(a[b + 1:eos[j]].tobytes()))

    tokens_in_buffer: List[int] = []
    tokens_written = 0
    t0 = time.time()
    last_print = t0
    epoch = 1

    progress_path = os.path.join(shard_dir, f"{name}_progress.json")
    docs_seen = 0
    if os.path.exists(progress_path):
        with open(progress_path) as pf:
            docs_seen = json.load(pf).get("docs_seen", 0)

    def _get_dataset(cfg, skip=0):
        if cfg["config"] is not None:
            ds = load_dataset(cfg["ds"], cfg["config"], split="train", streaming=True)
        else:
            ds = load_dataset(cfg["ds"], split="train", streaming=True)
        return ds.skip(skip) if skip > 0 else ds

    print(f"    Connecting to Hugging Face: {cfg['ds']} ({cfg['config']})...", flush=True)
    ds = _get_dataset(cfg, docs_seen)
    it = iter(ds)

    while tokens_written < tokens_needed:
        try:
            sample = next(it)
            docs_seen += 1
            if "messages" in sample or "instruction" in sample or "prompt" in sample:
                text = format_instruction_item(sample)
            else:
                text = sample.get("text", "")
        except StopIteration:
            epoch += 1
            print(f"    [*] Epoch {epoch} completed for '{name}'. Resetting deduplication cache for diverse multi-epoch tokenization...", flush=True)
            seen_docs.clear()
            ds = _get_dataset(cfg, 0)
            docs_seen = 0
            it = iter(ds)
            sample = next(it)
            docs_seen += 1
            if "messages" in sample or "instruction" in sample or "prompt" in sample:
                text = format_instruction_item(sample)
            else:
                text = sample.get("text", "")
        except Exception as e:
            print(f"    [Warning] Network blip on '{name}': {e}. Retrying with backoff...", flush=True)
            connected = False
            for attempt in range(1, 11):
                wait_time = min(30.0, 2.0 * attempt)
                time.sleep(wait_time)
                try:
                    ds = _get_dataset(cfg, docs_seen)
                    it = iter(ds)
                    connected = True
                    print(f"    [OK] Successfully reconnected to '{name}' (attempt {attempt}).", flush=True)
                    break
                except Exception as retry_e:
                    print(f"    [Warning] Reconnect attempt {attempt}/10 failed: {retry_e}.", flush=True)
            if not connected:
                print(f"    [Error] Could not reconnect to '{name}' after 10 attempts. Moving to next stream.", flush=True)
                break
            continue

        if not text or len(text.strip()) < 15:
            continue

        # CJK and non-ASCII filter for code to completely eliminate byte-token fragmentation
        if name == "python":
            if CJK_REGEX.search(text):
                continue
            non_ascii = sum(1 for c in text if ord(c) >= 128)
            if non_ascii / max(len(text), 1) > 0.005:
                continue

        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(token_ids) > 130:
            h = hash(np.array(token_ids[1:-1], dtype=np.uint16).tobytes())
            if h in seen_docs:
                continue
            seen_docs.add(h)
        tokens_in_buffer.extend(token_ids)

        now = time.time()
        if now - last_print >= 5.0:
            current_total = tokens_written + len(tokens_in_buffer)
            speed = current_total / max(now - t0, 1e-5)
            pct = (current_total / tokens_needed) * 100.0
            print(f"    [{name.upper()}] Buffer: {len(tokens_in_buffer):,} / {shard_size:,} | Total: {current_total:,} ({pct:.1f}%) | Speed: {speed:,.0f} tok/s", flush=True)
            last_print = now

        while len(tokens_in_buffer) >= shard_size:
            shard_path = os.path.join(shard_dir, f"{name}_{shard_idx:05d}.bin")
            arr = np.array(tokens_in_buffer[:shard_size], dtype=np.uint16)
            with open(shard_path, "wb") as f:
                f.write(arr.tobytes())

            tokens_written += shard_size
            tokens_in_buffer = tokens_in_buffer[shard_size:]
            shard_idx += 1
            json.dump({"docs_seen": docs_seen}, open(progress_path, "w"))

            now = time.time()
            speed = tokens_written / max(now - t0, 1e-5)
            pct = (tokens_written / tokens_needed) * 100.0
            print(f"    [*] Saved: {os.path.basename(shard_path)} ({shard_size/1e6:.1f}M tok) | Progress: {pct:.1f}% | Speed: {speed:,.0f} tok/s", flush=True)

    # Flush any remaining tokens to a final shard if substantial (>= 100k tokens)
    if len(tokens_in_buffer) >= 100_000:
        shard_path = os.path.join(shard_dir, f"{name}_{shard_idx:05d}.bin")
        arr = np.array(tokens_in_buffer, dtype=np.uint16)
        with open(shard_path, "wb") as f:
            f.write(arr.tobytes())
        tokens_written += len(arr)
        print(f"    [*] Saved final shard: {os.path.basename(shard_path)} ({len(arr):,} tokens)", flush=True)
        json.dump({"docs_seen": docs_seen}, open(progress_path, "w"))

    # Clean up single stream to completely free RAM
    del it, ds, tokens_in_buffer
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def download_and_tokenize(
    target_tokens: int = 10_000_000,
    shard_size: int = DEFAULT_SHARD_SIZE,
    shard_dir: str = "data/shards",
    tokenizer_path: str = "tokenizer.json",
    stream_filter: Optional[str] = None,
    phase: str = "all",
):
    os.makedirs(shard_dir, exist_ok=True)
    tokenizer = get_or_create_tokenizer(tokenizer_path)

    print("=" * 70)
    print("  [>] NVMe PRE-TOKENIZER & DATASET SHARD GENERATOR")
    print("=" * 70)
    print(f"* Target Total Tokens: {target_tokens:,} ({target_tokens / 1e6:.1f} Million)")
    print(f"* Shard Directory:     {shard_dir}")
    print(f"* Shard Size:          {shard_size:,} tokens (~{shard_size * 2 / (1024*1024):.1f} MB in uint16)")
    print(f"* Phase:               {phase.upper()}")
    if stream_filter:
        print(f"* Stream Filter:       {stream_filter}")
    print("=" * 70)

    # Filter by phase
    if phase == "instruct":
        active_configs = [c for c in STREAM_CONFIGS if c["name"] == "instruction"]
    elif phase == "foundation":
        active_configs = [c for c in STREAM_CONFIGS if c["name"] != "instruction"]
    else:
        active_configs = list(STREAM_CONFIGS)

    # Re-normalize weights if subset
    tot_weight = sum(c["weight"] for c in active_configs)
    normalized_configs = [{**c, "weight": c["weight"] / tot_weight} for c in active_configs]

    for cfg in normalized_configs:
        if stream_filter and cfg["name"] != stream_filter:
            continue
        stream_target = target_tokens if (stream_filter or len(normalized_configs) == 1) else int(target_tokens * cfg["weight"])
        if cfg["name"] == "instruction":
            print(f"\n[+] Processing Multi-Faceted Instruction Mixture across {len(INSTRUCTION_SUBSETS)} specialized domains...")
            for sub in INSTRUCTION_SUBSETS:
                sub_target = max(200_000, int(stream_target * sub["weight"]))
                sub_cfg = {
                    "name": f"instruction_{sub['name']}",
                    "ds": sub["ds"],
                    "config": sub["config"],
                    "weight": sub["weight"],
                }
                download_and_tokenize_stream(
                    cfg=sub_cfg,
                    target_tokens=sub_target,
                    shard_size=shard_size,
                    shard_dir=shard_dir,
                    tokenizer=tokenizer,
                )
        else:
            download_and_tokenize_stream(
                cfg=cfg,
                target_tokens=stream_target,
                shard_size=shard_size,
                shard_dir=shard_dir,
                tokenizer=tokenizer,
            )

    final_stats = get_shard_stats(shard_dir)
    print("\n" + "=" * 70)
    print("  [SUCCESS] NVMe Pre-Tokenization Finished!")
    print("=" * 70)
    print(f"* Total Tokens on NVMe: {final_stats['total_tokens']:,}")
    print(f"* Total Disk Footprint: {final_stats['total_mb']:.1f} MB")
    for name, s in final_stats["by_stream"].items():
        print(f"  • {name:12s}: {s['tokens']:,} tokens across {s['shards']} shards ({s['mb']:.1f} MB)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and Pre-Tokenize into NVMe Shards")
    parser.add_argument("--target-tokens", "--tokens", type=int, default=10_000_000, dest="target_tokens", help="Target total tokens (default: 10M)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Tokens per shard file (default: 5M)")
    parser.add_argument("--shard-dir", type=str, default="data/shards", help="Directory to save shards")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--stream", type=str, default=None, help="Filter to specific stream (e.g. python, cosmopedia)")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "foundation", "instruct"], help="Tokenize foundation, instruct, or all streams")
    args = parser.parse_args()

    download_and_tokenize(
        target_tokens=args.target_tokens,
        shard_size=args.shard_size,
        shard_dir=args.shard_dir,
        tokenizer_path=args.tokenizer,
        stream_filter=args.stream,
        phase=args.phase,
    )
