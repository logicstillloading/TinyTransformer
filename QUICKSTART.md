# TinyTransformer: QuickStart & Command Cheat Sheet

Comprehensive operational guide and copy-paste cheat sheet for training, evaluating, and monitoring the **TinyTransformer (~234M parameter)** architecture on consumer NVIDIA hardware (RTX 3050 Ti Laptop GPU / 4.3 GB VRAM).

---

## ⚡ 1. Copy-Paste Command Cheat Sheet

All commands below are ready to copy directly into **PowerShell** or **Command Prompt (CMD)** inside `d:\TinyTransformer`.

### 1.1 Training Commands

* **Resume Pretraining with Critic Co-Training (Phase 2 - Active Adversarial RL)**:
  Resumes from `checkpoints/checkpoint_latest.pt` and engages the auxiliary ~9.2M parameter Critic discriminator. Active when cumulative tokens $\ge 250\text{M}$ and EMA loss $\le 2.7$:
  ```powershell
  python train.py --resume --enable-critic --critic-loss-target 2.7
  ```

* **Resume Foundation Pretraining (Pure Next-Token Prediction / Critic Dormant)**:
  Resumes automatically from `checkpoints/checkpoint_latest.pt` with all learning rate schedules, optimizer momentum, and dataset streams intact:
  ```powershell
  python train.py --resume --grad-accum 32 --lr-muon 0.006 --lr-decay-steps 79000 --archive-interval 2500
  ```
  *(Short alias: `python train.py --resume`)*

* **Start Fresh Pretraining from Step 0**:
  ```powershell
  python train.py --grad-accum 32 --lr-muon 0.006 --lr-decay-steps 79000 --archive-interval 2500
  ```

---

### 1.2 Safe Remote Control Flags (No Crash / Zero Lost Tokens)

Control the running training process non-destructively by creating empty flag files in the root folder:

* **Graceful Stop & Save** (Finishes current step, saves `checkpoint_latest.pt`, exits cleanly):
  * PowerShell:
    ```powershell
    New-Item stop.flag -ItemType File
    ```
  * Command Prompt (CMD):
    ```cmd
    type nul > stop.flag
    ```

* **Immediate Save Without Stopping** (Saves atomic backup and continues training):
  * PowerShell:
    ```powershell
    New-Item save.flag -ItemType File
    ```
  * Command Prompt (CMD):
    ```cmd
    type nul > save.flag
    ```

* **Hot-Activate Critic Live** (Dynamically activates the auxiliary Critic discriminator without stopping training):
  * PowerShell:
    ```powershell
    New-Item critic.flag -ItemType File
    ```
  * Command Prompt (CMD):
    ```cmd
    type nul > critic.flag
    ```

* **Clean / Cancel Any Active Flags**:
  * PowerShell:
    ```powershell
    Remove-Item -Force -ErrorAction SilentlyContinue stop.flag, save.flag, critic.flag
    ```
  * Command Prompt (CMD):
    ```cmd
    del /f /q stop.flag save.flag critic.flag 2>nul
    ```

---

### 1.3 Running Model Evaluations & Generation Tests

> [!NOTE]
> `test.py` runs on **CPU by default**, meaning you can run it **at any time while GPU training is actively running** without causing GPU out-of-memory errors!

* **Run Full Evaluation Suite** (Both factual probe rankings + generation samples):
  ```powershell
  python test.py
  ```

* **Run Generation Previews Only** (Code, math, essays, reasoning):
  ```powershell
  python test.py --generate-only --max-new-tokens 60
  ```

* **Run Factual Entity Probes Only** (Tests top token predictions for Einstein, Washington, Paris, H2O, Chloroplast):
  ```powershell
  python test.py --probes-only
  ```

* **Run Deterministic Greedy Generation** (No temperature / random sampling):
  ```powershell
  python test.py --greedy --generate-only
  ```

* **Evaluate an Archived Milestone Checkpoint**:
  ```powershell
  python test.py --checkpoint "checkpoints/checkpoint(step_2500)/checkpoint(step_2500).pt"
  ```

---

### 1.4 Interactive Inference & Prompting (`infer.py`)

* **Single Prompt Generation (Safe on CPU while GPU trains)**:
  ```powershell
  python infer.py --device cpu --prompt "def fibonacci(n):"
  ```

* **Single Prompt on GPU (Fast, use when training is stopped)**:
  ```powershell
  python infer.py --device cuda --prompt "Explain the theory of relativity:"
  ```

* **Interactive Multi-Turn Chat (Safe on CPU)**:
  ```powershell
  python infer.py --device cpu
  ```

---

### 1.5 Live Monitoring & Visualization

* **Check GPU VRAM, Utilization, and Temperature**:
  ```powershell
  nvidia-smi
  ```

* **Watch Live Training Logs (Follows newest entries in real-time)**:
  * PowerShell:
    ```powershell
    Get-Content (Get-ChildItem logs\session_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 20 -Wait
    ```
  * CMD / Git Bash:
    ```bash
    tail -f $(ls -t logs/session_*.csv | head -1)
    ```

* **Generate Training Curves Plot** (Saves to `logs/metrics_plot.png`):
  ```powershell
  python plot_metrics.py
  ```

* **Verify Latest Checkpoint Timestamp & Size**:
  * PowerShell:
    ```powershell
    Get-Item checkpoints\checkpoint_latest.pt | Select-Object Name, Length, LastWriteTime
    ```
  * Command Prompt (CMD):
    ```cmd
    dir checkpoints\checkpoint_latest.pt
    ```

* **Emergency Rescue (Kill Stalled Python Process)**:
  * PowerShell:
    ```powershell
    Stop-Process -Name python -Force
    ```
  * Command Prompt (CMD):
    ```cmd
    taskkill /F /IM python.exe
    ```

---

## 2. Architecture & Hyperparameter Summary

| Component | Specification |
| :--- | :--- |
| **Model Size** | **~237.8M total parameters** (~138.3M dense backbone + 99.5M Engram tables) |
| **Dense Backbone** | 24 layers, hidden dim $d=640$, 10 heads $\times$ 64 head_dim, SwiGLU ($3.0\times$, intermediate dim $1920$) |
| **Attention Stability** | **QK-RMSNorm** ($d_k=64$) to eliminate attention logit growth; Rotary Position Embeddings (RoPE) |
| **Engram Tables** | 80M Global table (post-layer 16) + two 10M Local tables (Layer 5 & 11) using $O(1)$ coprime polynomial hashing |
| **Critic Model** | **~9.20M parameters** (Transformer binary discriminator) co-training via Gumbel-Softmax straight-through embeddings; activates at 250M tokens when EMA loss $\le 2.7$; convex blend ($\alpha: 1.0 \to 0.7$, $\beta: 0.0 \to 0.3$); dynamic freeze if critic loss $< 0.20$ |
| **Dual Optimizers** | **Keller Jordan Muon** ($\text{lr}=0.006$, 5th-order Newton-Schulz) for 2D internal linear weights + **AdamW** ($\text{lr}=3\times 10^{-4}$) for embeddings, engrams, and RMSNorms |
| **Batch Configuration** | Micro-batch size $1$, Sequence length $1,024$, Gradient accumulation $32$ (**Effective: 32,768 tokens/step**) |
| **Mixture Ratio** | **28% Cosmopedia v2, 28% FineWeb-Edu, 15% FineMath-4plus, 12% Python-Edu, 10% Wikipedia, 7% Smol-SmolTalk** |
| **Hardware Footprint** | **1,785 MB allocated / 3,270 MB peak VRAM** (safely below 3,650 MB PyTorch limit on 4.3 GB GPU) |
| **Chinchilla Target** | **2.60 Billion tokens** (~79,345 steps at 32k tokens/step) |

---

## 3. Universal Sampling Configuration

Both `train.py` (live preview) and `test.py` / `infer.py` utilize a uniform, non-cheating sampling configuration across all prompt categories:

| Parameter | Value | Purpose |
| :--- | :---: | :--- |
| `temperature` | `0.60` | Balanced entropy: sharp enough for valid syntax/math, creative enough for essays. |
| `top_k` | `40` | Eliminates irrelevant tail tokens. |
| `top_p` | `0.90` | Standard nucleus cutoff for semantic consistency. |
| `min_p` | `0.03` | Filters out low-probability hallucinations relative to the top logit. |
| `repetition_penalty` | `1.05` | Gentle penalty to discourage loops without breaking recursive calls or variable reuse. |
| `no_repeat_ngram_size` | `0` | Disabled (`0`). Aggressive n-gram blocking breaks code recursion (`fibonacci` calling `fibonacci`) and standard English syntax. |

---

## 4. Factual Knowledge Progression

Tracking target token logit rankings out of the **16,384 vocabulary** on CPU evaluation:

| Domain | Prompt | Target | Step 750 (24.6M) | Step 1,500 (49.1M) | Step 2,500 (81.9M) | Step 3,500 (114.7M) | Step 3,750 (122.9M) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physics** | "Albert Einstein developed the theory of" | ` relat` | #3,875 | #14 (Top 0.1%) | #12 (Top 0.07%) | #9 (Top 0.05%) | **#1 (Top 0.01%, 8.0% prob)** |
| **History** | "The first president of the United States was" | ` George` | #2,140 | #51 (Top 0.3%) | #18 (Top 0.11%) | #8 (Top 0.05%) | **#5 (Top 0.03%, 2.0% prob)** |
| **Chemistry** | "The chemical formula for water is" | ` H` | #890 | #135 (Top 0.8%) | #185 (Top 1.1%) | #178 (Top 1.1%) | **#289 (Top 1.8%)** |
| **Geography** | "The capital city of France is" | ` Paris` | #3,438 | #935 (Top 5.7%) | #412 (Top 2.5%) | #243 (Top 1.5%) | **#37 (Top 0.23%)** |
| **Biology** | "Photosynthesis takes place inside the" | ` chlor` | #4,200 | #1,866 (Top 11.4%) | #820 (Top 5.0%) | #340 (Top 2.1%) | **#106 (Top 0.65%)** |

---

## 5. Checkpoint & Milestone Storage

1. **Working Checkpoints (`checkpoints/`)**:
   - `checkpoint_latest.pt`: Current healthy state (written via atomic file rename).
   - `checkpoint_prev.pt`: Prior rollback state.
   - Rotates every `--save-interval` (default: 250 steps / ~8.2M tokens).
2. **Milestone Archives (`checkpoints/checkpoint(step_XXXX)/`)**:
   - Created every `--archive-interval` (default: 2,500 steps / ~81.9M tokens).
   - Preserves `checkpoint(step_XXXX).pt` accompanied by `.json` and `.md` reports detailing loss, perplexity, and mini-inference generation samples.
