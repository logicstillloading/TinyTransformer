"""
Standalone CPU script to download and pre-tokenize fresh Wikipedia shards (wikimedia/wikipedia, 20231101.en)
into data/shards_staging/ without interrupting active GPU training.
"""

import os
import sys
import time
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import Tokenizer

def main():
    target_tokens = 80_000_000   # 80 Million tokens (~16 shards of 5M tokens, ~160 MB)
    shard_size = 5_000_000      # 5 Million tokens per uint16 shard
    staging_dir = "data/shards_staging"
    os.makedirs(staging_dir, exist_ok=True)

    print("[*] Loading tokenizer from tokenizer.json...", flush=True)
    tokenizer = Tokenizer.from_file("tokenizer.json")
    print(f"[*] Tokenizer loaded (vocab: {tokenizer.vocab_size:,})", flush=True)

    print("[*] Connecting to Hugging Face: wikimedia/wikipedia (20231101.en)...", flush=True)
    for attempt in range(1, 6):
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            it = iter(ds)
            print("[*] Connection established successfully!", flush=True)
            break
        except Exception as e:
            print(f"[!] Connection attempt {attempt}/5 failed: {e}. Retrying...", flush=True)
            time.sleep(3.0)

    tokens_in_buffer = []
    tokens_written = 0
    shard_idx = 0
    t0 = time.time()
    last_print = t0

    print(f"[*] Starting tokenization: Target = {target_tokens:,} tokens ({target_tokens/1e6:.0f}M)", flush=True)

    while tokens_written < target_tokens:
        try:
            sample = next(it)
            text = sample.get("text", "")
        except StopIteration:
            print("[!] Stream reached end, looping back...", flush=True)
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            it = iter(ds)
            continue
        except Exception as e:
            print(f"[!] Read error: {e}. Retrying in 2s...", flush=True)
            time.sleep(2.0)
            continue

        if not text or len(text.strip()) < 50:
            continue

        # Clean next-token sequence with <bos> and <eos>
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        tokens_in_buffer.extend(token_ids)

        now = time.time()
        if now - last_print >= 5.0:
            current_total = tokens_written + len(tokens_in_buffer)
            speed = current_total / max(now - t0, 1e-5)
            pct = (current_total / target_tokens) * 100.0
            print(f"    [Wikipedia] Buffer: {len(tokens_in_buffer):,} / {shard_size:,} | Total: {current_total:,} ({pct:.1f}%) | Speed: {speed:,.0f} tok/s", flush=True)
            last_print = now

        while len(tokens_in_buffer) >= shard_size and tokens_written < target_tokens:
            shard_path = os.path.join(staging_dir, f"wikipedia_{shard_idx:05d}.bin")
            arr = np.array(tokens_in_buffer[:shard_size], dtype=np.uint16)
            with open(shard_path, "wb") as f:
                f.write(arr.tobytes())

            tokens_written += shard_size
            tokens_in_buffer = tokens_in_buffer[shard_size:]
            shard_idx += 1

            now = time.time()
            speed = tokens_written / max(now - t0, 1e-5)
            pct = (tokens_written / target_tokens) * 100.0
            print(f"    [*] Staged: {os.path.basename(shard_path)} ({shard_size/1e6:.1f}M tok) | Progress: {pct:.1f}% ({tokens_written:,}/{target_tokens:,}) | Speed: {speed:,.0f} tok/s", flush=True)

    # Flush remaining if >= 500k
    if len(tokens_in_buffer) >= 500_000 and tokens_written < target_tokens:
        shard_path = os.path.join(staging_dir, f"wikipedia_{shard_idx:05d}.bin")
        arr = np.array(tokens_in_buffer, dtype=np.uint16)
        with open(shard_path, "wb") as f:
            f.write(arr.tobytes())
        tokens_written += len(arr)
        print(f"    [*] Staged final shard: {os.path.basename(shard_path)} ({len(arr):,} tokens)", flush=True)

    elapsed = time.time() - t0
    print(f"\n[OK] Wikipedia Shard Staging Complete: {tokens_written:,} tokens written across {shard_idx} shards in {elapsed/60:.1f} min ({tokens_written/elapsed:,.0f} tok/s)", flush=True)

if __name__ == "__main__":
    main()
