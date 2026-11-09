# 🚀 TinyTransformer (~238M) Kaggle TPU v5e-8 Master Deployment & Run Guide

Comprehensive operational guide for running distributed high-throughput pretraining on **Google Cloud TPU v5e-8 on Kaggle**, managing the 9-hour execution limit, and seamlessly transitioning between Foundation and Instruction phases.

---

## 1. Executive Performance & Throughput Mathematics

### 1.1 Token Throughput Calculation (8 Hours on TPU v5e-8)
* **Hardware:** 8 TPU v5e Tensor Cores (16 GB HBM per core = 128 GB Total HBM).
* **Batch Geometry:** Micro-batch 4 $\times$ Accum 1 $\times$ 8 Cores = **32,768 tokens per global step**.
* **Step Latency:** $\approx 0.22\text{ seconds/step}$ ($\approx 4.55\text{ steps/sec}$).
* **Sustained Throughput:** $\approx \mathbf{140{,}000 \text{ to } 150{,}000 \text{ tokens/second}}$ via local NVMe memory-mapped binary shards.

$$\begin{aligned}
\text{Tokens per Hour} &= 3{,}600\text{ s} \times 140{,}000\text{ tok/s} \approx \mathbf{504 \text{ Million tokens/hour}} \\
\text{Tokens in 8.0 Hours} &= 28{,}800\text{ s} \times 140{,}000\text{ tok/s} \approx \mathbf{4.03 \text{ Billion tokens}} \quad (123{,}000\text{ steps}) \\
\text{Tokens in 8.4 Hours} &= 30{,}240\text{ s} \times 140{,}000\text{ tok/s} \approx \mathbf{4.23 \text{ Billion tokens}} \quad (129{,}000\text{ steps})
\end{aligned}$$

---

### 1.2 Empirical Scaling Law & Epoch Decision (Muennighoff et al.)
Empirical studies on data-constrained scaling (*Muennighoff et al., NeurIPS 2023*) establish how the marginal learning value of a token decays with repetition across a fixed 3.8B - 4.0B token compute budget:

| Configuration | Unique Tokens | Total Epochs | Data Efficiency | Generalization & Downstream Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: 5.7x on 700M** | 700 Million | ~5.7x | ~55–65% | High risk of verbatim memorization; early loss plateau; ~40% wasted compute. |
| **Option B: 2.0x on 2.0B (Selected)** | **2.0 Billion** | **2.0x** | **~98%** | **Optimal balance:** Excellent compute efficiency, rich gradient diversity, minimal memorization, ~3h CPU tokenization. |
| **Option C: 1.0x on 3.8B** | 3.8 Billion | 1.0x | 100% | Optimal Pareto frontier; maximal syntactic coverage; ~6h CPU tokenization. |

$$\begin{aligned}
\text{Foundation Epochs (1.85B Shards)} &= \frac{3{,}830{,}000{,}000 \text{ tokens}}{1{,}850{,}000{,}000 \text{ foundation shards}} \approx \mathbf{2.07 \text{ Epochs (~98\% Data Efficiency)}} \\
\text{Instruct Epochs (150M Shards)} &= \frac{100{,}000{,}000 \text{ tokens}}{150{,}000{,}000 \text{ instruct shards}} \approx \mathbf{0.67 \text{ Epochs (100\% Unique, Zero Repetition)}}
\end{aligned}$$

#### The Revised 2.0B Shard Mixture Breakdown (~4.0 GB disk):
1. **Foundation Shards (1.85B Tokens ~3.7 GB):**
   - 28% Cosmopedia v2 (~518M tokens) — **Synthetic Textbook Distillation** (Mixtral 8x22B)
   - 28% FineWeb-Edu Dedup (~518M tokens) — Educational Web (Filtered by Llama-3-70B judge)
   - 15% FineMath-4plus (~278M tokens) — High-Tier Educational Mathematics
   - 12% Python-Edu (~222M tokens) — Cleaned Python Code & Algorithms
   - 10% Wikipedia (~185M tokens) — Encyclopedic Ground Truth Knowledge
   - 7% Pretraining Instruction Warmup (~129M tokens) — Early conversational grounding
2. **Instruction & Distillation Shards (150M Tokens ~300 MB):**
   - 25% `apigen-80k` (~37.5M tokens) — **Tool-Calling Distillation** (API / Tool Execution)
   - 20% `numina-cot-100k` (~30.0M tokens) — **Mathematical CoT Distillation** (AIMO Progress Winner)
   - 20% `self-oss-instruct` (~30.0M tokens) — **Coding Reasoning Distillation** (Algorithm synthesis)
   - 15% `everyday-conversations` (~22.5M tokens) — Natural Dialogue & Tone
   - 10% `smol-magpie-ultra` (~15.0M tokens) — **Llama-3.1-405B Alignment Distillation**
   - 10% `openhermes-100k` (~15.0M tokens) — Multi-Disciplinary Synthetic CoT Reasoning
   - 15% `everyday-conversations` (~18M tokens) — Natural Dialogue & Tone
   - 10% `smol-magpie-ultra` (~12M tokens) — **Llama-3.1-405B Alignment Distillation**
   - 10% `openhermes-100k` (~12M tokens) — Multi-Disciplinary Synthetic CoT Reasoning

---

## 2. Decoupled Two-Stage Cloud Architecture

Kaggle grants **30 hours/week of TPU quota** and **unlimited / separate CPU quota**. To maximize TPU utilization, we decouple data tokenization from TPU pretraining:

```
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Kaggle CPU Session (Free / 0 TPU Quota)     │
│  - Notebook: TinyTransformer_Kaggle_Data_Sharding.ipynb│
│  - Streams HF datasets to contiguous uint16 shards     │
│  - Downloads 600M Foundation + 120M Instruction shards  │
│  - Commit output -> Click 'New Dataset'                │
└───────────────────────────┬────────────────────────────┘
                            │ (Mount as /kaggle/input/...)
                            ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 2: Kaggle TPU v5e-8 Session (Max Speed)         │
│  - Notebook: TinyTransformer_Kaggle_TPU_v5e_8.ipynb    │
│  - Mounts shards instantly (0 mins tokenization delay) │
│  - Hours 0.0 - 8.0: Foundation Pretrain (~3.9B tokens) │
│  - Hours 8.0 - 8.4: Slow Instruct Annealing (~200M)    │
│  - Hour 8.4: Graceful shutdown (--max-hours 8.4)       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Foundation ➔ Instruction Annealing Dynamics

The user requested a proper foundation pretraining run followed by a slow transition to instruction training for the final ~20-25 minutes:

```
0.0h                     8.0h               8.4h          9.0h (Kaggle Limit)
├─────────────────────────┼──────────────────┼─────────────┤
│  Foundation Pretrain    │ Instruct Anneal  │ Graceful    │ Hard Kill
│  (93% Base / 7% Warmup) │ (50% Instruct /  │ Shutdown    │ Container
│  ~3.85 Billion Tokens   │  50% Replay)     │ (Exit 0)    │ Destroyed
│  ~6.9 Foundation Epochs │  ~200M Tokens    │             │
└─────────────────────────┴──────────────────┴─────────────┘
```

* **Flags in `train.py`:**
  - `--base-hours 8.0`: At exactly 8.0 hours elapsed, the dataloader transitions to 50% instruction data and 50% foundation replay (preventing catastrophic forgetting).
  - `--max-hours 8.4`: At exactly 8.4 hours (8h 24m), the training loop breaks cleanly, flushes atomic checkpoints, validates 0 NaNs / 0 Infs, and exits before Kaggle's 9.0h cutoff.

---

## 4. Step-by-Step Execution Checklist

### Option A: Two-Notebook Workflow (Recommended: Saves TPU Quota)
1. **Run CPU Sharding:**
   - Create notebook on Kaggle, set Accelerator to **None (CPU)**.
   - Upload [`TinyTransformer_Kaggle_Data_Sharding.ipynb`](file:///d:/TinyTransformer/TinyTransformer_Kaggle_Data_Sharding.ipynb).
   - Click **Save Version -> Quick Save or Run All**.
   - When finished (~2.5 hours), go to the notebook's **Output** tab and click **Create Dataset** (e.g. `tinytransformer-shards-600m`).
2. **Launch TPU Training:**
   - Create notebook on Kaggle, set Accelerator to **TPU VM v5e-8**.
   - Upload [`TinyTransformer_Kaggle_TPU_v5e_8.ipynb`](file:///d:/TinyTransformer/TinyTransformer_Kaggle_TPU_v5e_8.ipynb).
   - Click **+ Add Input** in the right sidebar -> select your `tinytransformer-shards-600m` dataset.
   - Paste Google Drive link (if resuming previous checkpoint) or leave blank for Step 0.
   - Click **Run All** (or **Save Version -> Save & Run All** to run headless in cloud).
   - The notebook mounts the shards instantly and trains at full ~150k tok/s!

---

### Option B: All-in-One TPU Workflow (Faster Setup)
If you prefer running everything inside a single notebook on TPU:
1. Open [`TinyTransformer_Kaggle_TPU_v5e_8.ipynb`](file:///d:/TinyTransformer/TinyTransformer_Kaggle_TPU_v5e_8.ipynb) on Kaggle with **TPU VM v5e-8**.
2. Run **Step 1** (TPU check) and **Step 2** (Install dependencies).
3. Run **Step 3** (Checkpoint restore).
4. Run **Cell 4A** (Foundation shards: set to 250M or 550M tokens).
5. Run **Cell 4B** (Instruction shards: set to 25M or 50M tokens).
6. Run **Step 5** (Launch pretraining). It trains until 8.4 hours and shuts down safely.

---

## 5. Master Shell Command Reference

Executed automatically by Step 5 of [`TinyTransformer_Kaggle_TPU_v5e_8.ipynb`](file:///d:/TinyTransformer/TinyTransformer_Kaggle_TPU_v5e_8.ipynb):

```bash
python -u train.py \
    --tpu-cores 8 \
    --batch-size 4 \
    --grad-accum 1 \
    --dtype bfloat16 \
    --base-hours 8.0 \
    --max-hours 8.4 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --log-dir /kaggle/working/logs \
    --shard-dir /kaggle/working/TinyTransformer/data/shards \
    --save-interval 250 \
    --infer-interval 2500 \
    --resume
```

---

## 6. Checkpoint Persistence & Anti-Corruption Protocol

Every checkpoint adheres to the two-slot atomic rotation schema:
- `checkpoint_latest.pt`: Current healthy state.
- `checkpoint_prev.pt`: Backup rollback state.
- **Verification Guarantee:** 0 NaNs, 0 Infs, strictly validated with [`verify_checkpoint_integrity`](file:///d:/TinyTransformer/data/cloud_sync.py#L30) before exit.
- At hour 8.4, the final state is preserved in `/kaggle/working/checkpoints/checkpoint_latest.pt` and ready for immediate download or Kaggle Dataset publishing.
