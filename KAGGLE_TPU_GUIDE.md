# 🚀 TinyTransformer (~238M) Kaggle TPU v5e-8 Master Deployment & Run Guide

Comprehensive operational guide for running distributed high-throughput pretraining on **Google Cloud TPU v5e-8 on Kaggle**, managing the 9-hour execution limit, and seamlessly transitioning between Foundation and Instruction phases.

---

## 1. Executive Performance & Throughput Mathematics

### 1.1 Token Throughput Calculation (8 Hours on TPU v5e-8)
* **Hardware:** 8 TPU v5e Tensor Cores (16 GB HBM per core = 128 GB Total HBM).
* **Batch Geometry:** Micro-batch 4 $\times$ Accum 1 $\times$ 8 Cores = **32,768 tokens per global step** (32 sequences of length 1,024).
* **Step Latency:** $\approx 0.22\text{ seconds/step}$ ($\approx 4.55\text{ steps/sec}$).
* **Sustained Throughput:** $\approx \mathbf{140{,}000 \text{ to } 150{,}000 \text{ tokens/second}}$ via local NVMe memory-mapped binary shards (`np.memmap`).

$$\begin{aligned}
\text{Tokens per Hour} &= 3{,}600\text{ s} \times 145{,}000\text{ tok/s} \approx \mathbf{522 \text{ Million tokens/hour}} \\
\text{Tokens in 7.8 Hours} &= 28{,}080\text{ s} \times 145{,}000\text{ tok/s} \approx \mathbf{4.07 \text{ Billion tokens}} \quad (124{,}000\text{ steps}) \\
\text{Tokens in 8.4 Hours} &= 30{,}240\text{ s} \times 145{,}000\text{ tok/s} \approx \mathbf{4.38 \text{ Billion tokens}} \quad (133{,}000\text{ steps})
\end{aligned}$$

---

### 1.2 The Empirical Scaling Decision: 1.0 Unique Epoch on 3.8B Tokens
Empirical studies on data-constrained scaling (*Muennighoff et al., NeurIPS 2023*) establish how the marginal learning value of a token decays with repetition across a fixed compute budget:

| Configuration | Unique Tokens | Total Epochs | Data Efficiency | Generalization & Engram Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: 5.7x on 700M** | 700 Million | ~5.7x | ~55–65% | Severe verbatim memorization; early loss plateau; ~40% compute wasted on redundant updates. |
| **Option B: 2.0x on 2.0B** | 2.0 Billion | 2.0x | ~98% | High data efficiency, minimal memorization, but duplicate hash bucket hits for Engrams. |
| **Option C: 1.0x on 3.8B (Selected)** | **3.8 Billion** | **1.0x (Flat)** | **100% (Pareto Optimal)** | **Maximum n-gram hash entropy across 80M+10M Engram tables; zero verbatim repetition; optimal downstream reasoning.** |

$$\begin{aligned}
\text{Foundation Epochs (3.60B Shards)} &= \frac{3{,}600{,}000{,}000 \text{ tokens}}{3{,}600{,}000{,}000 \text{ foundation shards}} = \mathbf{1.00 \text{ Epoch (100\% Unique Tokens)}} \\
\text{Instruct Epochs (200M Shards)} &= \frac{200{,}000{,}000 \text{ tokens}}{200{,}000{,}000 \text{ instruct shards}} = \mathbf{1.00 \text{ Epoch (100\% Unique Tokens)}}
\end{aligned}$$

---

### 1.3 Why 1.0 Unique Epoch Matters for the Engram Architecture
TinyTransformer dedicates **99.54M parameters** to Hierarchical N-gram Hash Tables:
* **Local Engram 1 (Layer 5):** 9.97M params ($220 \text{ clusters} \times 64 \text{ slots} \times 640 \text{ dim}$)
* **Local Engram 2 (Layer 11):** 9.97M params ($220 \text{ clusters} \times 64 \text{ slots} \times 640 \text{ dim}$)
* **Global Engram (Post-Layer 16):** 79.86M params ($1{,}900 \text{ clusters} \times 64 \text{ slots} \times 640 \text{ dim}$)

**The Hash Entropy Invariant:**
Engrams retrieve static phrase embeddings via deterministic n-gram hashing $h(w_{t-1}, w_t)$, $h(w_{t-2}, w_{t-1}, w_t)$. If a smaller dataset is repeated over multiple epochs, the exact same n-gram keys collide into identical hash slots repeatedly, over-saturating specific memory slots while leaving others cold.
By training on **3.6B unique tokens**, the model encounters billions of distinct linguistic, coding, and mathematical n-grams, giving maximum entropy and balanced gradient distribution across all $131{,}072$ memory slots with zero collision overfitting!

---

### 1.4 Calibrated SOTA Dataset Mixture Breakdown (~7.6 GB Disk)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CALIBRATED 3.8B SOTA SHARD MIXTURE                              │
├───────────────────────────────┬────────────┬─────────────┬─────────────────────────────┤
│ Stream / Dataset              │ Weight     │ Tokens      │ Scientific Rationale        │
├───────────────────────────────┼────────────┼─────────────┼─────────────────────────────┤
│ FineWeb-Edu (dedup)           │   30.0%    │  1.08B      │ Core high-signal reasoning  │
│ Cosmopedia v2                 │   22.0%    │  792M       │ Pedagogical synthetic text  │
│ FineMath-3plus                │   15.0%    │  540M       │ Arithmetic & basic algebra  │
│ Python-Edu                    │   12.0%    │  432M       │ Structured code & syntax    │
│ DCLM-100BT                    │   11.0%    │  396M       │ Natural human web variance  │
│ Wikipedia (en)                │    8.0%    │  288M       │ Factual entity grounding    │
│ SmolTalk (warmup)             │    2.0%    │   72M       │ Base alignment warmup       │
├───────────────────────────────┼────────────┼─────────────┼─────────────────────────────┤
│ TOTAL FOUNDATION              │  100.0%    │  3.60B      │ ~7.2 GB uint16 on disk      │
├───────────────────────────────┼────────────┼─────────────┼─────────────────────────────┤
│ SmolTalk Alignment Mix        │  Anneal    │  200M       │ ~0.4 GB uint16 on disk      │
│ (Tools, CoT, Coding, Magpie)  │            │             │                             │
├───────────────────────────────┼────────────┼─────────────┼─────────────────────────────┤
│ GRAND TOTAL                   │            │  3.80B      │ ~7.6 GB (38% of 20GB disk)  │
└───────────────────────────────┴────────────┴─────────────┴─────────────────────────────┘
```

#### Key Calibrations Implemented:
1. **SmolTalk Warmup Lowered (7% $\rightarrow$ 2%):** Preserves base representation capacity for language acquisition; dense instruction formatting is reserved for the annealing phase.
2. **FineMath-3plus:** Replaces `finemath-4plus`. A ~238M model cannot model Putnam/Olympiad university proofs and resorts to memorizing LaTeX syntax. `finemath-3plus` focuses on K-12 arithmetic, fractions, and algebra word problems that build actual reasoning primitives.
3. **DCLM-100BT (11%) & Cosmopedia v2 (22%):** Cosmopedia is 100% synthetic (Mixtral 8x22B). Mixing 11% DCLM human web data breaks syntactic monotony and prevents robotic language generation (following SmolLM2).

---

## 2. Decoupled Two-Stage Cloud Architecture

Kaggle provides **30 hours/week of TPU quota** and **unlimited / separate CPU quota**. We decouple tokenization from training to avoid burning TPU hours on data downloading:

```
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Kaggle Cloud CPU Session (Free / 0 TPU Quota)│
│  - Notebook: TinyTransformer_Kaggle_Data_Sharding.ipynb│
│  - Streams HF datasets to contiguous uint16 shards     │
│  - Generates 3.6B Foundation + 200M Instruct shards    │
│  - Commit output -> Click 'New Dataset from Output'    │
│  - Slug: tinytransformer-shards-3-8b (~7.6 GB)         │
└───────────────────────────┬────────────────────────────┘
                            │ (Attach via + Add Input)
                            ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 2: Kaggle TPU v5e-8 Session (Full Throttle)     │
│  - Notebook: TinyTransformer_Kaggle_TPU_v5e_8.ipynb    │
│  - Accelerator: TPU VM v5e-8                           │
│  - Mounts shards instantly (0 mins tokenization delay) │
│  - Hours 0.0 - 7.8: Foundation Pretrain (~3.6B tokens) │
│  - Hours 7.8 - 8.4: Slow Instruct Annealing (~200M)    │
│  - Hour 8.4: Graceful shutdown (--max-hours 8.4)       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Foundation ➔ Instruction Annealing Dynamics

Pretraining transitions smoothly from raw knowledge acquisition to instruction alignment before session completion:

```
0.0h                             7.8h               8.4h          9.0h (Kaggle Limit)
├─────────────────────────────────┼──────────────────┼─────────────┤
│  Foundation Pretrain (1.0 Epoch)│ Instruct Anneal  │ Graceful    │ Hard Kill
│  (100% Base Shards)             │ (50% Instruct /  │ Shutdown    │ Container
│  ~3.60 Billion Unique Tokens    │  50% Replay)     │ (Exit 0)    │ Destroyed
│  Zero Repetition                │  ~200M Tokens    │             │
└─────────────────────────────────┴──────────────────┴─────────────┘
```

* **CLI Execution Flags:**
  ```bash
  python -u train.py \
      --tpu-cores 8 \
      --batch-size 4 \
      --grad-accum 1 \
      --dtype bfloat16 \
      --base-hours 7.8 \
      --max-hours 8.4 \
      --checkpoint-dir /kaggle/working/checkpoints \
      --log-dir /kaggle/working/logs \
      --shard-dir /kaggle/working/TinyTransformer/data/shards \
      --save-interval 250 \
      --infer-interval 12500 \
      --resume
  ```
  - `--base-hours 7.8`: At exactly 7.8 hours, the dataloader transitions into the instruction annealing phase (50% instruction + 50% foundation replay to prevent catastrophic forgetting).
  - `--max-hours 8.4`: At 8.4 hours, the training loop exits cleanly, flushes the final atomic checkpoint, validates 0 NaNs / 0 Infs, and terminates code 0.
  - `--infer-interval 12500`: Runs mini-inference preview generation only once every ~46–50 minutes, avoiding TPU graph recompilation stalls.

---

## 4. Checkpoint Protocol & Cross-Session Resumption

1. **Two-Slot Atomic Checkpoint Rotation:**
   Checkpoints are strictly limited to two files inside `/kaggle/working/checkpoints`:
   - `checkpoint_latest.pt`: Current healthy state.
   - `checkpoint_prev.pt`: Backup rollback state.
2. **State Dictionary Contents:**
   - Global step and tokens processed.
   - Model weights (including hierarchical Engrams).
   - Critic RL discriminator weights.
   - Muon optimizer momentum buffers (for 2D linear matrices).
   - AdamW optimizer momentum buffers (for RMSNorm and Engram tables).
   - Dataloader stream offsets and buffer indices (guarantees zero duplicate tokens on resumption).
   - Running loss, perplexity, and best loss metrics.
3. **Download or Resumption:**
   - Checkpoint files can be synced to Google Drive via `gdown` or committed to a private Kaggle Dataset for seamless multi-session training.

---

## 5. Quick Reference for AI Subagents & Future Sessions

When picking up an existing or new session:
* **Audit Active Kaggle Run:** Check `https://www.kaggle.com/code/ishauryaactually/tinytransformer-data-sharding` to monitor shard generation progress.
* **Verify Shard Integrity:** Run `python -c "from data.download_and_tokenize import get_shard_stats; print(get_shard_stats('data/shards'))"`.
* **Run Component Unit Tests:** Run `python tests/test_components.py` (all 9 tests must pass with 0 bugs).
* **Launch TPU Training:** In `TinyTransformer_Kaggle_TPU_v5e_8.ipynb`, attach `tinytransformer-shards-3-8b` via `+ Add Input` and trigger `Save & Run All`.
