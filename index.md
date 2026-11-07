# TinyTransformer Project Index & Codebase Directory

Welcome to **TinyTransformer**, a high-efficiency ~220M parameter language model training engine featuring 100M Engram retrieval tables, Hybrid Dual-Resolution Attention, an Adversarial RL Critic, Muon + AdamW dual optimization, and atomic checkpointing.

---

## 1. Directory Tree & Architecture Layout

```
d:\TinyTransformer\
├── architecture.md             # In-depth architectural equations, parameter tables & analysis
├── gemini.md                   # Multi-agent concurrency, session continuity & workflow rules
├── index.md                    # This document: Project directory, file index & quickstart
├── train.py                    # Main training engine with live CLI telemetry & atomic checkpointing
├── infer.py                    # Interactive inference CLI (greedy, top-k, top-p, temperature)
├── plot_metrics.py             # Session metric visualizer (generates loss/PPL/throughput graphs)
├── model\
│   ├── __init__.py             # Model package exports
│   ├── attention.py            # Hybrid Dual-Resolution Attention (8x32 + 4x64 heads)
│   ├── engram.py               # 2-Step Hierarchical Engram Memory Tables (80M + 2x 10M)
│   ├── critic.py               # Adversarial RL Critic ("anti-bsing" hallucination discriminator)
│   └── transformer.py          # Full ~220M TinyTransformer backbone assembly & tied embeddings
├── optimizers\
│   ├── __init__.py             # Optimizer exports
│   └── muon.py                 # Muon Newton-Schulz matrix optimizer & dual AdamW router
├── data\
│   ├── __init__.py             # Data package exports
│   ├── tokenizer.py            # Custom 16k Byte-Pair Encoding (BPE) tokenizer trainer & codec
│   └── dataset.py              # Multi-source streaming loader (40% Cosmo2, 30% Math, 20% Edu, 10% Code)
├── checkpoints\                # Checkpoint directory (retains strictly latest.pt and prev.pt)
│   ├── checkpoint_latest.pt    # Current valid checkpoint
│   └── checkpoint_prev.pt      # Previous rollback checkpoint
└── logs\                       # Training session logs
    ├── session_*.csv           # Tabular per-step metrics (loss, PPL, tokens/s, LR, reward)
    └── session_*.jsonl         # Detailed JSONL records including mini-inference texts
```

---

## 2. File Index & Content Descriptions

### Root Files
| File | Role & Contents |
| :--- | :--- |
| `train.py` | **Primary Training Harness**. Implements the complete training loop, dual Muon/AdamW step, mixed-precision BF16 autocast, adversarial critic loss blending, live telemetry CLI (Step, Tok/s, Loss, PPL, ETA), mini-inference prompt testing every 250 steps, auto-plotting every 500 steps, and atomic 2-slot checkpoint rotation (`checkpoint_latest.pt` $\leftrightarrow$ `checkpoint_prev.pt`). |
| `infer.py` | **Interactive Generation Script**. Loads any saved checkpoint and generates completions for arbitrary user prompts with configurable temperature, top-$k$, and nucleus top-$p$ filtering. |
| `plot_metrics.py` | **Telemetry Visualizer**. Reads session CSV logs and renders publication-ready plots of Loss, Perplexity, Throughput, and Adversarial Critic Rewards over time (`loss_ppl_curve.png`). |
| `architecture.md` | Detailed architectural breakdown containing exact parameter mathematical derivations, tensor layouts, operational dynamics, strengths, and failure modes. |
| `gemini.md` | Protocol guidelines for multi-agent concurrency, state tracking, and session resumption across AI pair-programming interactions. |
| `index.md` | This file; project inventory, directory map, and quickstart commands. |

### `model/` Subpackage
| File | Role & Contents |
| :--- | :--- |
| `model/attention.py` | Implements `MultiHeadAttention`. 10 uniform heads of dim 64 (640 hidden dimension). Integrates shared Rotary Positional Embeddings (RoPE) and causal FlashAttention (SDPA). |
| `model/engram.py` | Implements `NGramHashEngramTable` (`HierarchicalEngramTable`). Holds the 80M Global memory table (post-layer 16) and two 10M tables (layers 5 and 11). Executes O(1) polynomial coprime hashing for bigram/trigram context with dynamic context-aware gating. |
| `model/critic.py` | Implements `CriticModel`. A 4-layer transformer trained to discriminate between authentic ground-truth text and corrupted/hallucinated text ("bsing" detector), producing an adversarial reinforcement learning reward. |
| `model/transformer.py` | Glues all components into the cohesive `TinyTransformer` class (~238M params) with tied input/output embeddings, Pre-RMSNorm, SwiGLU FFN ($3.0\times$ expansion), and integrated Engram layers. |

### `optimizers/` Subpackage
| File | Role & Contents |
| :--- | :--- |
| `optimizers/muon.py` | Implements the **Muon** optimizer with 5th-order Newton-Schulz polynomial matrix orthogonalization for all 2D linear weight matrices ($W_q, W_k, W_v, W_o, W_{gate}, W_{up}, W_{down}$, ~130.25M params) and Keller Jordan aspect ratio scaling. Partitions embeddings, stationary engrams, and RMSNorms to AdamW (~107.6M params). |

### `data/` Subpackage
| File | Role & Contents |
| :--- | :--- |
| `data/tokenizer.py` | Trains and manages a custom 16,384 vocabulary Byte-Pair Encoding (BPE) tokenizer trained directly on the balanced pretraining data mixture. Saves to `tokenizer.json`. |
| `data/dataset.py` | Dual-mode dataset loader: reads fast local memory-mapped shards from `data/shards/` if present (50k+ tok/s), or falls back to live streaming across 6 interleaved corpora (Cosmopedia v2, FineWeb-Edu, FineMath, Python-Edu, Wikipedia, Smoltalk). |
| `test.py` | **Non-Destructive CPU Factual Probe**. Evaluates target token probabilities and vocabulary rank (e.g. `' Paris'` rank progression) on CPU without interrupting GPU training. |
| `QUICKSTART.md` | **Operational QuickStart Guide**. Complete guide for running training, resumption, remote control flags, and evaluation. |
| `data/download_and_tokenize.py` | Pre-downloads and tokenizes the full 2.6 Billion Chinchilla token target into compact binary shards (`shard_*.bin`) for offline zero-network training. |

---

## 3. Quickstart Commands

### 1. Launch Pretraining from Step 0
```powershell
python train.py --grad-accum 32 --lr-muon 0.006 --lr-decay-steps 79000 --archive-interval 2500
```

### 2. Resume Pretraining from Latest Checkpoint
```powershell
# Automatically restores saved hyperparameters, weights, optimizers, and dataset state!
python train.py --resume
```

### 3. Run Non-Destructive CPU Evaluation Probe (Zero VRAM impact)
```powershell
# Runs alongside training on CPU to track factual association rank
python test.py
```

### 4. Interactive Generation CLI
```powershell
python infer.py --checkpoint checkpoints/checkpoint_latest.pt --prompt "def quicksort(arr):" --temperature 0.7 --top-k 40
```

### 5. Generate Metric Visualizations
```powershell
python plot_metrics.py
```

See [QUICKSTART.md](file:///d:/TinyTransformer/QUICKSTART.md) for full operational details.
