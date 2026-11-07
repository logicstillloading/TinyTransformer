import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 TinyTransformer (~238M) TPU v5e-1 Pretraining\n",
                "\n",
                "**High-Performance Cloud Pretraining Engine for Google Cloud TPU v5e-1 / Google Colab**\n",
                "\n",
                "### 📐 Model Architecture & Specifications\n",
                "- **Parameters:** ~237.85 Million (BF16 native)\n",
                "- **Backbone:** 24 Transformer layers, hidden dim 640, 10 attention heads, SwiGLU 3.0x multiplier\n",
                "- **Engram Memory:** 80M global (post-L16) + two 10M tables at L5 & L11 (99.5M params)\n",
                "- **Critic BS-Detector:** 4-layer auxiliary discriminator (9.2M params) for adversarial alignment\n",
                "- **Optimizers:** Dual Keller Jordan Muon (2D internal weights) + AdamW (Embeds, Norms, Engram tables)\n",
                "- **Effective Batch Size:** 32,768 tokens/step (Micro-batch 8 x Accum 4 x 1024 seq length)\n",
                "- **Target Acceleration:** Google Cloud TPU v5e-1 via PyTorch/XLA with non-blocking graph execution\n",
                "- **Anti-Corruption Guard:** Rigorous NaN/Inf detection, mmap deserialization, two-slot atomic backup\n",
                "- **Repository:** `https://github.com/logicstillloading/TinyTransformer`"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## ⚙️ Step 1: TPU Hardware & PyTorch / XLA Verification\n",
                "Verifies that the Google Colab runtime is set to **TPU v5e-1** (or TPU v2/v3) and initializes PyTorch XLA."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check hardware and runtime configuration\n",
                "import os\n",
                "import sys\n",
                "import torch\n",
                "\n",
                "print(f\"[*] PyTorch Version: {torch.__version__}\")\n",
                "print(f\"[*] CUDA Available: {torch.cuda.is_available()}\")\n",
                "\n",
                "# Check TPU / XLA environment\n",
                "try:\n",
                "    import torch_xla\n",
                "    import torch_xla.core.xla_model as xm\n",
                "    xla_dev = xm.xla_device()\n",
                "    hw_type = xm.xla_device_hw(xla_dev)\n",
                "    print(f\"[+] TPU / XLA is active!\")\n",
                "    print(f\"[+] Target Device: {xla_dev} (Hardware Type: {hw_type})\")\n",
                "except ImportError:\n",
                "    print(\"[!] torch_xla not installed in this environment.\")\n",
                "    print(\"[*] Installing torch_xla for TPU v5e...\")\n",
                "    !pip install -q torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html\n",
                "    import torch_xla\n",
                "    import torch_xla.core.xla_model as xm\n",
                "    print(f\"[+] TPU / XLA initialized: {xm.xla_device()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## 📂 Step 2: Mount Google Drive (Persistent Checkpoints & Logs)\n",
                "Mounts Google Drive to ensure your trained checkpoints and logs persist across Colab sessions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Create persistent checkpoints and logs directories in Google Drive\n",
                "DRIVE_DIR = \"/content/drive/MyDrive/TinyTransformer\"\n",
                "DRIVE_CKPT_DIR = os.path.join(DRIVE_DIR, \"checkpoints\")\n",
                "DRIVE_LOGS_DIR = os.path.join(DRIVE_DIR, \"logs\")\n",
                "\n",
                "os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)\n",
                "os.makedirs(DRIVE_LOGS_DIR, exist_ok=True)\n",
                "\n",
                "print(f\"[+] Google Drive mounted successfully!\")\n",
                "print(f\"[+] Checkpoint Directory: {DRIVE_CKPT_DIR}\")\n",
                "print(f\"[+] Checkpoints in Drive: {os.listdir(DRIVE_CKPT_DIR)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## 📥 Step 3: Clone or Update TinyTransformer Codebase\n",
                "Clones the official GitHub repository or pulls the latest updates."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone or pull latest repository\n",
                "%cd /content\n",
                "if not os.path.exists(\"/content/TinyTransformer\"):\n",
                "    !git clone https://github.com/logicstillloading/TinyTransformer.git\n",
                "    %cd /content/TinyTransformer\n",
                "else:\n",
                "    %cd /content/TinyTransformer\n",
                "    !git pull origin main\n",
                "\n",
                "# Install required dependencies\n",
                "!pip install -q tokenizers datasets matplotlib requests huggingface_hub"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## 🛡️ Step 4: Anti-Corruption Checkpoint Verification\n",
                "Exhaustively verifies checkpoint integrity (0 NaNs, 0 Infs, schema integrity, mmap memory safety)\n",
                "before starting pretraining to ensure corrupted weights are never loaded."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from data.cloud_sync import verify_checkpoint_integrity\n",
                "\n",
                "latest_ckpt = os.path.join(DRIVE_CKPT_DIR, \"checkpoint_latest.pt\")\n",
                "if os.path.exists(latest_ckpt):\n",
                "    print(f\"[*] Verifying Drive Checkpoint: {latest_ckpt}...\")\n",
                "    valid, msg, info = verify_checkpoint_integrity(latest_ckpt)\n",
                "    if valid:\n",
                "        print(f\"[+] [VERIFIED HEALTHY] Checkpoint passed all anti-corruption guards!\")\n",
                "        print(f\"    Global Step: {info['step']:,}\")\n",
                "        print(f\"    Tokens Processed: {info['tokens_processed']:,} ({info['tokens_processed']/1e6:.2f}M tokens)\")\n",
                "        print(f\"    File Size: {info['size_mb']:.1f} MB\")\n",
                "        print(f\"    Metrics: {info['metrics']}\")\n",
                "    else:\n",
                "        print(f\"[!] [CORRUPTION ALERT] {msg}\")\n",
                "        prev_ckpt = os.path.join(DRIVE_CKPT_DIR, \"checkpoint_prev.pt\")\n",
                "        if os.path.exists(prev_ckpt):\n",
                "            print(f\"[*] Attempting rollback to backup: {prev_ckpt}\")\n",
                "            v_prev, m_prev, i_prev = verify_checkpoint_integrity(prev_ckpt)\n",
                "            if v_prev:\n",
                "                print(f\"[+] Backup verified healthy at step {i_prev['step']:,}. Rolling back...\")\n",
                "                import shutil\n",
                "                shutil.copy2(prev_ckpt, latest_ckpt)\n",
                "else:\n",
                "    print(f\"[*] No checkpoint found in Drive yet. Starting fresh from Step 0.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## ⚡ Step 5: Launch Pretraining on TPU v5e-1 (Max Speed)\n",
                "Launches training with TPU v5e-1 optimization flags:\n",
                "- `--device xla`: Direct hardware dispatch to TPU v5e MMUs\n",
                "- `--batch-size 8 --grad-accum 4`: Optimal 32,768 effective tokens per step\n",
                "- `--dtype bfloat16`: Native hardware BF16 precision\n",
                "- `--checkpoint-dir`: Saves directly to persistent Google Drive with atomic two-slot rotation\n",
                "- `--resume`: Resumes seamlessly from the latest healthy step"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python train.py \\\n",
                "  --device xla \\\n",
                "  --batch-size 8 \\\n",
                "  --grad-accum 4 \\\n",
                "  --dtype bfloat16 \\\n",
                "  --checkpoint-dir /content/drive/MyDrive/TinyTransformer/checkpoints \\\n",
                "  --save-interval 250 \\\n",
                "  --resume"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## 📊 Step 6: Loss & PPL Curves / Telemetry Audit\n",
                "Generates live loss, perplexity, and learning rate telemetry curves directly from session logs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import glob\n",
                "import matplotlib.pyplot as plt\n",
                "import pandas as pd\n",
                "\n",
                "log_files = sorted(glob.glob(\"/content/drive/MyDrive/TinyTransformer/logs/session_*.csv\"))\n",
                "if log_files:\n",
                "    latest_log = log_files[-1]\n",
                "    print(f\"[*] Auditing latest session log: {latest_log}\")\n",
                "    df = pd.read_csv(latest_log)\n",
                "    print(f\"    Logged Steps: {len(df)}\")\n",
                "    print(df.tail())\n",
                "    \n",
                "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "    axes[0].plot(df[\"step\"], df[\"loss\"], color=\"#2563eb\", label=\"Cross Entropy Loss\")\n",
                "    axes[0].set_title(\"Training Loss Curve\")\n",
                "    axes[0].set_xlabel(\"Step\")\n",
                "    axes[0].set_ylabel(\"Loss\")\n",
                "    axes[0].grid(True, alpha=0.3)\n",
                "    axes[0].legend()\n",
                "    \n",
                "    axes[1].plot(df[\"step\"], df[\"ppl\"], color=\"#10b981\", label=\"Perplexity\")\n",
                "    axes[1].set_title(\"Perplexity Curve\")\n",
                "    axes[1].set_xlabel(\"Step\")\n",
                "    axes[1].set_ylabel(\"PPL\")\n",
                "    axes[1].grid(True, alpha=0.3)\n",
                "    axes[1].legend()\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "else:\n",
                "    print(\"[*] No session logs found in Google Drive yet.\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "TPU",
        "colab": {
            "gpuType": "v5e-1",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("TinyTransformer_Colab_TPU_v5e.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("[+] Created TinyTransformer_Colab_TPU_v5e.ipynb successfully!")
