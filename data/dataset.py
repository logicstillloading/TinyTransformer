"""
Dynamic Streaming Mixture Dataloader with Deterministic Resumption.
Dataset Mixture:
- 28% Cosmopedia v2 (HuggingFaceTB/smollm-corpus, cosmopedia-v2)
- 28% FineWeb-Edu Dedup (HuggingFaceTB/smollm-corpus, fineweb-edu-dedup)
- 15% FineMath-4plus (HuggingFaceTB/finemath, finemath-4plus)
- 12% Python-Edu (Avelina/python-edu-cleaned)
- 10% Wikipedia (wikimedia/wikipedia, 20231101.en)
- 7% Instruction (HuggingFaceTB/smol-smoltalk)

Key Features:
- Dual Engine: NVMe np.memmap acceleration (zero RAM, 50k+ tok/s) with seamless bounded-stream fallback.
- Interleaved Mixture: Every cycle guarantees exact proportions across all 6 sources in randomized chunks.
- Resumption Fidelity: Tracks document offsets and buffers for zero duplicated data processing.
"""

from collections import deque
import gc
import glob
import os
import random
from typing import Dict, List, Optional, Tuple, Iterator, Any
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .tokenizer import Tokenizer


STREAM_CONFIGS = [
    {"name": "cosmopedia", "ds": "HuggingFaceTB/smollm-corpus", "config": "cosmopedia-v2", "weight": 0.28},
    {"name": "fineweb", "ds": "HuggingFaceTB/smollm-corpus", "config": "fineweb-edu-dedup", "weight": 0.28},
    {"name": "finemath", "ds": "HuggingFaceTB/finemath", "config": "finemath-4plus", "weight": 0.15},
    {"name": "python", "ds": "Avelina/python-edu-cleaned", "config": None, "weight": 0.12},
    {"name": "wikipedia", "ds": "wikimedia/wikipedia", "config": "20231101.en", "weight": 0.10},
    {"name": "instruction", "ds": "HuggingFaceTB/smol-smoltalk", "config": None, "weight": 0.07},
]


def release_system_memory():
    """Forces PyArrow and Python garbage collectors to release cached pages back to the OS."""
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    gc.collect()


class StreamingDatasetMixture(IterableDataset):
    """
    Hybrid iterable dataset mixing 4 specialized corpora in exact proportions.
    - If local binary shards exist in shard_dir, reads via fast np.memmap with chunk shuffling (NVMe mode, zero RAM).
    - Otherwise, streams concurrently from all 4 Hugging Face Hub datasets with strict RAM limits.
    - Guarantees per-batch randomized interleaved chunking so every batch contains all 4 datasets.
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int = 1024,
        seed: int = 42,
        resume_state: Optional[Dict[str, Any]] = None,
        shard_dir: str = "data/shards",
        effective_batch_size: int = 8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.shard_dir = shard_dir
        self.effective_batch_size = max(4, effective_batch_size)
        self.weights = [cfg["weight"] for cfg in STREAM_CONFIGS]
        self.names = [cfg["name"] for cfg in STREAM_CONFIGS]
        self.phase: str = "foundation"
        self.instruct_ratio: float = 0.50
        self.stream_epochs: Dict[str, int] = {cfg["name"]: 0 for cfg in STREAM_CONFIGS}

        # Scan for NVMe memory-mapped shards
        self.stream_memmaps: Dict[str, List[np.memmap]] = {}
        total_nvme_tokens = 0

        if os.path.exists(shard_dir):
            for cfg in STREAM_CONFIGS:
                name = cfg["name"]
                shards = sorted(glob.glob(os.path.join(shard_dir, f"{name}_*.bin")))
                if shards:
                    self.stream_memmaps[name] = [np.memmap(s, dtype=np.uint16, mode="r") for s in shards]
                    total_nvme_tokens += sum(len(m) for m in self.stream_memmaps[name])

        self.has_local_shards = any(len(v) > 0 for v in self.stream_memmaps.values())
        if self.has_local_shards:
            print(f"[Dataset] Found NVMe binary shards with {total_nvme_tokens:,} tokens (zero-RAM mode active)")

        # Tracking state
        self.samples_consumed: Dict[str, int] = {cfg["name"]: 0 for cfg in STREAM_CONFIGS}
        self.tokens_processed: int = 0
        self.rng = random.Random(seed)

        # Chunk offset tracking for sampling NVMe shards without replacement
        self.stream_chunk_indices: Dict[str, List[Tuple[int, int]]] = {}
        self.stream_chunk_ptrs: Dict[str, int] = {}
        self._init_chunk_indices()

        # Ultra-efficient bounded deques to prevent memory bloat
        self.stream_buffers: Dict[str, deque] = {cfg["name"]: deque() for cfg in STREAM_CONFIGS}
        self.stream_queue: deque = deque()

        self.iterators: Optional[List[Optional[Iterator]]] = None
        self._chunks_yielded: int = 0

        if resume_state is not None:
            self.load_state_dict(resume_state)

    def _init_chunk_indices(self):
        for name, memmaps in self.stream_memmaps.items():
            valid_chunks: List[Tuple[int, int]] = []
            for mm_idx, mm in enumerate(memmaps):
                max_offset = len(mm) - self.seq_len - 1
                if max_offset >= 0:
                    for offset in range(0, max_offset + 1, self.seq_len):
                        valid_chunks.append((mm_idx, offset))
            if valid_chunks:
                self.rng.shuffle(valid_chunks)
                self.stream_chunk_indices[name] = valid_chunks
                self.stream_chunk_ptrs[name] = 0

    def set_phase(self, phase: str = "foundation", instruct_ratio: float = 0.50):
        """
        Dynamically adjusts stream weights between Foundation Pretraining and
        Instruction + Foundation Annealing Mix.

        - 'foundation': Standard pretraining mixture (7% instruction, 93% foundation).
        - 'instruct': Rebalances to instruct_ratio (e.g. 50% instruction, 50% foundation replay).
        """
        self.phase = phase
        self.instruct_ratio = instruct_ratio
        if phase == "instruct":
            foundation_total = sum(cfg["weight"] for cfg in STREAM_CONFIGS if cfg["name"] != "instruction")
            scale = (1.0 - instruct_ratio) / max(foundation_total, 1e-6)
            new_weights = []
            for cfg in STREAM_CONFIGS:
                if cfg["name"] == "instruction":
                    new_weights.append(instruct_ratio)
                else:
                    new_weights.append(cfg["weight"] * scale)
            self.weights = new_weights
            mix_breakdown = ", ".join(f"{cfg['name']}: {w*100:.1f}%" for cfg, w in zip(STREAM_CONFIGS, self.weights))
            print(f"[Dataset] Transitioned to INSTRUCT-MIX phase: {mix_breakdown}")
        else:
            self.weights = [cfg["weight"] for cfg in STREAM_CONFIGS]
            print(f"[Dataset] Set to FOUNDATION phase: Default pretraining mixture active.")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "samples_consumed": dict(self.samples_consumed),
            "dataset_indices": dict(self.samples_consumed),  # GEMINI.md schema compatibility
            "tokens_processed": self.tokens_processed,
            "phase": getattr(self, "phase", "foundation"),
            "instruct_ratio": getattr(self, "instruct_ratio", 0.50),
            "stream_epochs": dict(getattr(self, "stream_epochs", {})),
            "rng_state": self.rng.getstate(),
            "stream_buffers": {k: list(v) for k, v in self.stream_buffers.items()},
            "stream_queue": list(self.stream_queue),
            "stream_chunk_ptrs": dict(self.stream_chunk_ptrs),
            "stream_chunk_indices": {k: list(v) for k, v in self.stream_chunk_indices.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if "phase" in state:
            p = state["phase"]
            ir = state.get("instruct_ratio", 0.50)
            self.set_phase(p, instruct_ratio=ir)
        if "stream_epochs" in state:
            self.stream_epochs.update(state["stream_epochs"])
        if "samples_consumed" in state:
            self.samples_consumed.update(state["samples_consumed"])
        elif "dataset_indices" in state:
            self.samples_consumed.update(state["dataset_indices"])
        if "tokens_processed" in state:
            self.tokens_processed = state["tokens_processed"]
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
        if "stream_buffers" in state:
            for k, v in state["stream_buffers"].items():
                if k in self.stream_buffers:
                    self.stream_buffers[k] = deque(v)
        if "stream_queue" in state:
            self.stream_queue = deque(state["stream_queue"])
        if "stream_chunk_indices" in state:
            saved_ptrs = state.get("stream_chunk_ptrs", {})
            for k, v in state["stream_chunk_indices"].items():
                old = [tuple(c) for c in v]
                ptr = saved_ptrs.get(k, 0)
                if k in self.stream_memmaps and len(self.stream_memmaps[k]) > 0:
                    n = len(self.stream_memmaps[k])
                    cur = self.stream_chunk_indices.get(k, [])
                    if len(old) == len(cur) and all(c[0] < n for c in old):
                        self.stream_chunk_indices[k] = old
                        self.stream_chunk_ptrs[k] = ptr
                    else:
                        old_set = set(old)
                        unseen = [c for c in old[ptr:] if c[0] < n] + [c for c in cur if c not in old_set]
                        seen = [c for c in old[:ptr] if c[0] < n]
                        if unseen:
                            self.rng.shuffle(unseen)
                            self.stream_chunk_indices[k] = unseen + seen
                        self.stream_chunk_ptrs[k] = 0
                        print(f"[Dataset] '{k}': shards changed -> {len(unseen):,} unseen chunks first, {len(seen):,} seen chunks last.")
                else:
                    self.stream_chunk_indices[k] = old
                    self.stream_chunk_ptrs[k] = ptr
        print(f"[Dataset] Resumed state: {self.samples_consumed}, tokens: {self.tokens_processed:,}")

    def _init_stream_iterators(self) -> List[Optional[Iterator]]:
        from datasets import load_dataset

        iterators = []
        for cfg in STREAM_CONFIGS:
            name = cfg["name"]
            # If stream is already satisfied by local NVMe shards, skip live network stream
            if name in self.stream_memmaps and self.stream_memmaps[name]:
                iterators.append(None)
                continue

            skip_count = self.samples_consumed.get(name, 0)
            try:
                if cfg["config"] is not None:
                    ds = load_dataset(cfg["ds"], cfg["config"], split="train", streaming=True)
                else:
                    ds = load_dataset(cfg["ds"], split="train", streaming=True)
                if skip_count > 0:
                    print(f"[Dataset] Fast-forwarding stream '{name}' by {skip_count:,} samples...")
                    ds = ds.skip(skip_count)
                it = iter(ds)
                iterators.append(it)
            except Exception as e:
                print(f"[Dataset Warning] Could not load stream '{name}' ({cfg['ds']}): {e}")
                iterators.append(None)
        return iterators

    def _synthetic_text_generator(self, stream_idx: int) -> str:
        """Fallback generator when network or dataset streaming is interrupted."""
        stream_name = self.names[stream_idx]
        if stream_name == "cosmopedia":
            topics = [
                "The solar system consists of eight planets orbiting the Sun in elliptical paths with varying orbital periods.",
                "Cellular respiration is the biochemical process that converts glucose and oxygen into ATP within mitochondria.",
                "Photosynthesis takes place within the chloroplasts of plant cells using chlorophyll pigments to capture light.",
                "Quantum electrodynamics describes the relativistic quantum field theory of electrodynamics and photon interactions.",
            ]
        elif stream_name == "finemath":
            topics = [
                "Theorem: If f is continuous on [a, b] and differentiable on (a, b), then there exists c in (a, b) where f'(c) = (f(b)-f(a))/(b-a).",
                "The eigenvalues of a symmetric real matrix are always real, and eigenvectors corresponding to distinct eigenvalues are orthogonal.",
                "Let G be a group. A subgroup H is normal if and only if g H g^{-1} = H for all g in G. This implies the quotient group G/H is well-defined.",
                "Proof by induction: For the base case n=1, the equality 1 = 1(2)/2 holds. Assuming true for k, the summation simplifies to (k+1)(k+2)/2.",
            ]
        elif stream_name == "fineweb":
            topics = [
                "Education systems around the world emphasize critical thinking, analytical reasoning, and interdisciplinary collaboration.",
                "The history of literature reflects changing cultural norms, philosophical movements, and major historical developments.",
                "Modern macroeconomics analyzes aggregate indicators such as GDP, inflation rates, and national income distribution.",
                "Cognitive psychology examines internal mental processes including memory acquisition, linguistic cognition, and problem solving.",
            ]
        elif stream_name == "instruction":
            topics = [
                "User: Can you explain the difference between a list and a tuple in Python?\nAssistant: In Python, a list is mutable, meaning its elements can be modified after creation, and is defined using square brackets []. A tuple is immutable, meaning its elements cannot be changed once created, and is defined using parentheses (). Tuples are typically faster and consume less memory than lists.",
                "User: What are three tips for effective time management?\nAssistant: Here are three key strategies for managing your time effectively:\n1. Prioritize tasks using the Eisenhower Matrix (urgent vs important).\n2. Use time-blocking or the Pomodoro technique to maintain focus.\n3. Eliminate multitasking to increase depth of work and efficiency.",
                "User: How do airplanes stay in the air?\nAssistant: Airplanes fly by generating aerodynamic lift. As the plane moves forward, the curved shape of the wings (airfoil) causes air to move faster over the top surface than the bottom. According to Bernoulli's principle and Newton's third law, this pressure difference produces an upward lifting force that counteracts gravity.",
            ]
        elif stream_name == "wikipedia":
            topics = [
                "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1840.",
                "DNA, or deoxyribonucleic acid, is a polymer composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development and functioning of all known organisms.",
                "The Roman Empire was the post-Republican period of ancient Rome, featuring an autocratic form of government and large territorial holdings around the Mediterranean Sea in Europe, Northern Africa, and Western Asia.",
                "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 and the current description of gravitation in modern physics.",
            ]
        else:  # python
            topics = [
                "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
                "class LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.cache = {}\n    def get(self, key):\n        if key not in self.cache: return -1\n        val = self.cache.pop(key)\n        self.cache[key] = val\n        return val",
                "import torch\nimport torch.nn as nn\nclass SwiGLU(nn.Module):\n    def __init__(self, d_in, d_out):\n        super().__init__()\n        self.w1 = nn.Linear(d_in, d_out, bias=False)\n        self.w2 = nn.Linear(d_in, d_out, bias=False)\n    def forward(self, x):\n        return self.w1(x) * torch.sigmoid(self.w1(x)) * self.w2(x)",
            ]
        return self.rng.choice(topics)

    def _generate_batch_stream_indices(self, total_items: int) -> List[int]:
        """
        Generates randomized stream assignments for `total_items` sequences,
        distributed proportionally according to STREAM_CONFIGS weights using the
        largest remainder (Hamilton / Hare-Niemeyer) method.
        """
        exact_shares = [total_items * w for w in self.weights]
        counts = [int(s) for s in exact_shares]
        remainder_slots = total_items - sum(counts)

        fractions = [(exact_shares[i] - counts[i], i) for i in range(len(self.weights))]
        fractions.sort(key=lambda x: x[0], reverse=True)
        for _, idx in fractions[:remainder_slots]:
            counts[idx] += 1

        indices = []
        for stream_idx, count in enumerate(counts):
            indices.extend([stream_idx] * count)

        self.rng.shuffle(indices)
        return indices

    def _get_chunk_from_stream(self, stream_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches a sequence-length chunk. Prioritizes zero-RAM NVMe memmaps with streaming fallback."""
        name = self.names[stream_idx]

        # 1. Fast Path: NVMe memory-mapped shards (Zero RAM overhead, 50k+ tok/s)
        # Stepping through shuffled chunk offsets without replacement across shards
        if (
            name in self.stream_chunk_indices
            and self.stream_chunk_indices[name]
            and name in self.stream_memmaps
            and self.stream_memmaps[name]
        ):
            indices = self.stream_chunk_indices[name]
            ptr = self.stream_chunk_ptrs.get(name, 0)
            if ptr >= len(indices):
                # Reshuffle chunk offsets without replacement on epoch completion
                self.rng.shuffle(indices)
                ptr = 0
                self.stream_epochs[name] = self.stream_epochs.get(name, 0) + 1
                print(f"[Dataset] '{name}': Epoch {self.stream_epochs[name]} complete. Reshuffled {len(indices):,} chunks for next epoch.")

            mm_idx, offset = indices[ptr]
            self.stream_chunk_ptrs[name] = ptr + 1

            # Guard against mismatched shard indices after reloading or shard regeneration
            mm_list = self.stream_memmaps[name]
            mm_idx = int(mm_idx) % len(mm_list)
            mm = mm_list[mm_idx]
            max_offset = len(mm) - self.seq_len - 1

            if max_offset >= 0:
                offset = min(max(0, int(offset)), max_offset)

                # Instruction alignment: snap to nearest <bos> so conversations cleanly start with <bos>User:
                if name == "instruction":
                    search_window = mm[offset : min(len(mm), offset + self.seq_len)]
                    bos_matches = np.where(search_window == self.tokenizer.bos_token_id)[0]
                    if len(bos_matches) > 0:
                        cand = offset + int(bos_matches[0])
                        if cand <= max_offset:
                            offset = cand

                chunk = torch.from_numpy(mm[offset : offset + self.seq_len + 1].astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                self.tokens_processed += self.seq_len
                self._chunks_yielded += 1
                return x, y

        # 2. Streaming Fallback Path: Bounded deque memory buffers
        buf = self.stream_buffers[name]

        # Instruction alignment: snap to nearest <bos> so conversations cleanly start with <bos>User:
        if name == "instruction":
            while buf and buf[0] != self.tokenizer.bos_token_id:
                buf.popleft()

        while len(buf) < self.seq_len + 1:
            it = self.iterators[stream_idx] if self.iterators is not None else None
            text = None
            if it is not None:
                try:
                    item = next(it)
                    if "messages" in item:
                        text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in item["messages"]])
                    else:
                        text = item.get("text", "")
                    self.samples_consumed[name] += 1
                except StopIteration:
                    try:
                        cfg = STREAM_CONFIGS[stream_idx]
                        from datasets import load_dataset
                        if cfg["config"] is not None:
                            ds = load_dataset(cfg["ds"], cfg["config"], split="train", streaming=True)
                        else:
                            ds = load_dataset(cfg["ds"], split="train", streaming=True)
                        it = iter(ds)
                        self.iterators[stream_idx] = it
                        item = next(it)
                        if "messages" in item:
                            text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in item["messages"]])
                        else:
                            text = item.get("text", "")
                        self.samples_consumed[name] += 1
                    except Exception as e:
                        print(f"[Dataset Warning] '{name}' stream reload error: {e}")
                        self.iterators[stream_idx] = None
                        text = None
                except Exception as e:
                    print(f"[Dataset Warning] '{name}' read error: {e}")
                    text = None

            if not text or len(text.strip()) < 10:
                self._fails = getattr(self, "_fails", 0) + 1
                if it is None or self._fails > 100:
                    raise RuntimeError(f"[Dataset] '{name}': no shards and stream unavailable. Refusing synthetic fallback.")
                continue
            self._fails = 0

            # Cap max tokens per sample to 2 * seq_len while preserving <eos> boundary
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(tokens) > self.seq_len * 2:
                tokens = tokens[: self.seq_len * 2 - 1] + [self.tokenizer.eos_token_id]
            buf.extend(tokens)

        chunk_in = []
        for _ in range(self.seq_len):
            if buf:
                chunk_in.append(buf.popleft())
            else:
                chunk_in.append(self.tokenizer.pad_token_id)

        target_token = buf[0] if buf else self.tokenizer.pad_token_id
        chunk_target = chunk_in[1:] + [target_token]

        self.tokens_processed += self.seq_len
        self._chunks_yielded += 1

        if self._chunks_yielded % 512 == 0:
            release_system_memory()

        x = torch.tensor(chunk_in, dtype=torch.long)
        y = torch.tensor(chunk_target, dtype=torch.long)
        return x, y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.iterators is None:
            self.iterators = self._init_stream_iterators()

        # Replenish queue in cycles of at least 100 chunks so that 3% (0.03) is an exact integer (3 chunks)
        cycle_size = max(self.effective_batch_size, 100)
        if cycle_size % self.effective_batch_size != 0:
            cycle_size = ((cycle_size // self.effective_batch_size) + 1) * self.effective_batch_size

        while True:
            if not self.stream_queue:
                self.stream_queue = deque(self._generate_batch_stream_indices(cycle_size))

            stream_idx = self.stream_queue.popleft()
            yield self._get_chunk_from_stream(stream_idx)
