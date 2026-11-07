# Gemini & Multi-Agent Concurrency Protocol: TinyTransformer

Operational guidelines, state preservation conventions, and concurrency safety rules for Gemini and autonomous subagents operating across training sessions in `d:\TinyTransformer`.

---

## 1. Concurrency Model & Work Partitioning

When multiple agents or subagents (e.g. `DeepCoder`, `DeepInvestigator`, or background monitors) operate on this codebase simultaneously, they must adhere to the following partition boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│  - Plans execution strategy                                 │
│  - Routes coding tasks to DeepCoder                         │
│  - Routes debugging/telemetry tasks to DeepInvestigator     │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────┐┌──────────────────────────────┐
│         DeepCoder           ││       DeepInvestigator       │
│  - Edits model/ & train.py  ││  - Audits logs/ & checkpoints│
│  - Implements optimizers    ││  - Evaluates metric curves   │
│  - Writes unit tests        ││  - Runs non-modifying probes │
└─────────────────────────────┘└──────────────────────────────┘
```

### Safety Rules:
1. **Single Implementer Rule**: Only one subagent may modify active Python source code (`model/`, `optimizers/`, `train.py`) at any given time to eliminate race conditions and git-like merge conflicts.
2. **GPU Single-Process Exclusivity**: The physical host possesses one NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4.3 GB VRAM. Subagents must never spawn parallel GPU training runs simultaneously. Before launching any CUDA training or evaluation script, check running processes or task status.

---

## 2. Cross-Session State Persistence

To ensure seamless resumption across sessions without re-processing data or corrupting states:

### 2.1 Two-Slot Atomic Checkpoint Protocol
Checkpoints are strictly limited to two files inside `d:\TinyTransformer\checkpoints\`:
- `checkpoint_latest.pt`: The current healthy state.
- `checkpoint_prev.pt`: The backup rollback state.

**Atomic Write Procedure**:
1. Serialize complete state to temporary file: `checkpoint_temp.pt`.
2. Flush to disk via `torch.save(...)` and filesystem sync.
3. If `checkpoint_latest.pt` exists, atomically rename it to `checkpoint_prev.pt`.
4. Atomically rename `checkpoint_temp.pt` to `checkpoint_latest.pt`.
*Result*: An unexpected interrupt or power failure can never leave a half-written corrupt checkpoint.

### 2.2 Checkpoint State Dictionary Schema
Every saved checkpoint must contain the following keys:
```python
{
    "step": int,  # Current global training step
    "tokens_processed": int,  # Cumulative tokens consumed
    "model_state_dict": dict,  # Generator weights (including Engrams)
    "critic_state_dict": dict,  # Adversarial critic weights
    "muon_state_dict": dict,  # Muon optimizer momentum & buffers
    "adam_state_dict": dict,  # AdamW optimizer moments & buffers
    "dataloader_state": {
        "dataset_indices": dict,  # Current document offset per source
        "buffer_position": int,  # Exact token offset within buffer
    },
    "metrics": {
        "running_loss": float,
        "running_ppl": float,
        "best_loss": float,
    },
    "hyperparameters": {
        "batch_size": int,
        "grad_accum": int,
        "seq_len": int,
        "lr_muon": float,
        "lr_adam": float,
        "critic_lr": float,
        "critic_warmup_steps": int,
        "critic_loss_target": float,
        "adv_weight": float,
        "chinchilla_target_tokens": int,
    },
}
```

---

## 3. Session Logging & Non-Destructive Continuity

1. **Unique Session Identifiers**: Every training run generates a unique timestamped session ID:
   `session_YYYYMMDD_HHMMSS`.
2. **Tabular CSV Metrics**: Appended on every single step to `logs/session_<id>.csv`.
3. **Structured Event JSONL**: Mini-inference test completions, hyperparameter changes, and critic losses are appended to `logs/session_<id>.jsonl`.
4. **Resumption Logging**: When running with `--resume`, the logger detects the existing session or appends a `session_<id>_resumed.csv` while maintaining global step and cumulative token continuity.

---

## 4. Agent Guidelines for Code Modifications

1. **Parameter Budget Invariant**: The total parameter count must strictly remain within $\approx 220\text{M} - 238\text{M}$ BF16. Do not alter hidden dimension ($640$), SwiGLU multiplier ($3.0$), layer count ($24$), or Engram table sizes ($80\text{M} + 2 \times 10\text{M}$) without explicit authorization.
2. **Dependency Integrity**: Avoid introducing heavy external frameworks (e.g. Deepspeed, Megatron). The codebase relies strictly on standard PyTorch, Hugging Face `tokenizers`, `datasets`, and `matplotlib`.
3. **Lint & Smoke Tests**: After making changes, always run quick verification checks (`python -c "import torch..."`) before updating the user or declaring milestones complete.
