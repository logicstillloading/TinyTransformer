"""
Metrics Plotting Tool for TinyTransformer Pretraining Sessions.
Visualizes:
1. Generator Loss & Exponential Moving Average (EMA) Loss
2. Step Perplexity (PPL) & Average PPL
3. Critic Discriminator Loss & Generator Adversarial Reward
4. Training Throughput (Tokens per Second)
"""

import argparse
import glob
import json
import os
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt


def load_session_logs(log_path: str) -> List[Dict[str, Any]]:
    records = []
    if log_path.endswith(".jsonl"):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if "loss" in item:
                            records.append(item)
                    except json.JSONDecodeError:
                        pass
    elif log_path.endswith(".csv"):
        import csv
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = {
                        "step": int(row["step"]),
                        "loss": float(row["loss"]),
                        "avg_loss": float(row["avg_loss"]),
                        "ppl": float(row["ppl"]),
                        "avg_ppl": float(row["avg_ppl"]),
                        "critic_loss": float(row.get("critic_loss", 0.0)),
                        "reward": float(row.get("reward", 0.5)),
                        "tokens_per_sec": float(row.get("tokens_per_sec", 0.0)),
                        "tokens_processed": int(row.get("tokens_processed", 0)),
                    }
                    records.append(record)
                except (ValueError, KeyError):
                    pass
    return records


def find_latest_log_file(log_dir: str = "logs") -> Optional[str]:
    files = glob.glob(os.path.join(log_dir, "*.jsonl")) + glob.glob(os.path.join(log_dir, "*.csv"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_metrics(records: List[Dict[str, Any]], output_path: str):
    if not records:
        print("[Plot] No records found to plot.")
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    avg_losses = [r["avg_loss"] for r in records]
    ppls = [min(r["ppl"], 500.0) for r in records]  # Cap extreme initial spikes for clean plotting
    avg_ppls = [min(r["avg_ppl"], 500.0) for r in records]
    critic_losses = [r.get("critic_loss", 0.0) for r in records]
    rewards = [r.get("reward", 0.5) for r in records]
    tok_per_sec = [r.get("tokens_per_sec", 0.0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TinyTransformer Pretraining Metrics", fontsize=16, fontweight="bold")

    # 1. Loss Curve
    ax0 = axes[0, 0]
    ax0.plot(steps, losses, label="Step Loss", alpha=0.4, color="#3498db")
    ax0.plot(steps, avg_losses, label="Moving Avg Loss", color="#1f77b4", linewidth=2)
    ax0.set_title("Generator Loss")
    ax0.set_xlabel("Step")
    ax0.set_ylabel("Cross Entropy Loss")
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend()

    # 2. Perplexity Curve
    ax1 = axes[0, 1]
    ax1.plot(steps, ppls, label="Step PPL", alpha=0.4, color="#e67e22")
    ax1.plot(steps, avg_ppls, label="Moving Avg PPL", color="#d35400", linewidth=2)
    ax1.set_title("Perplexity (PPL)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("PPL")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 3. Critic & RL Adversarial Dynamics
    ax2 = axes[1, 0]
    ax2.plot(steps, critic_losses, label="Critic BS-Detector Loss", color="#9b59b6", linewidth=1.5)
    ax2.plot(steps, rewards, label="Generator Adversarial Reward", color="#2ecc71", linewidth=1.5)
    ax2.set_title("Adversarial Critic Dynamics")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Score / Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 4. Training Throughput
    ax3 = axes[1, 1]
    ax3.plot(steps, tok_per_sec, label="Tokens / Sec", color="#e74c3c", linewidth=1.5)
    ax3.set_title("Training Throughput")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Tokens / Sec")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Training metrics graph successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot TinyTransformer training metrics")
    parser.add_argument("--log-file", type=str, default=None, help="Path to .jsonl or .csv log file")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing logs")
    parser.add_argument("--output", type=str, default="logs/metrics_plot.png", help="Output image file path")
    args = parser.parse_args()

    log_file = args.log_file
    if not log_file:
        log_file = find_latest_log_file(args.log_dir)
        if not log_file:
            print(f"No log files found in '{args.log_dir}'.")
            return

    print(f"Reading logs from: {log_file}")
    records = load_session_logs(log_file)
    plot_metrics(records, args.output)


if __name__ == "__main__":
    main()
