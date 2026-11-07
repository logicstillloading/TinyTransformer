"""
Archive Utility for TinyTransformer Checkpoints.
Safely archives checkpoint_latest.pt into checkpoints/checkpoint(step_<STEP>)
and generates companion .md and .json reports containing recent loss/PPL
progression (last 10 steps) and the mini-inference preview generations.
"""

import os
import shutil
import json
import csv
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional


def find_latest_session_file(log_dir: str = "logs", ext: str = ".jsonl") -> Optional[str]:
    files = glob.glob(os.path.join(log_dir, f"*{ext}"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def get_recent_logs_from_csv(csv_path: str, max_rows: int = 10) -> List[Dict[str, Any]]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    records = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    records.append({
                        "step": int(row["step"]),
                        "tokens_processed": int(row.get("tokens_processed", 0)),
                        "loss": float(row["loss"]),
                        "avg_loss": float(row["avg_loss"]),
                        "ppl": float(row["ppl"]),
                        "avg_ppl": float(row["avg_ppl"]),
                        "tokens_per_sec": float(row.get("tokens_per_sec", 0.0)),
                        "vram_mb": float(row.get("vram_mb", 0.0)),
                        "timestamp": row.get("timestamp", ""),
                    })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"[!] Error reading CSV log {csv_path}: {e}")
    return records[-max_rows:]


def get_preview_from_jsonl(jsonl_path: str, target_step: int) -> Dict[str, str]:
    if not jsonl_path or not os.path.exists(jsonl_path):
        return {}
    preview_generations = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "mini_inference":
                        if rec.get("step") == target_step or not preview_generations:
                            preview_generations = rec.get("generations", {})
                            if rec.get("step") == target_step:
                                break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[!] Error reading JSONL log {jsonl_path}: {e}")
    return preview_generations


def archive_latest_checkpoint(
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    step_override: Optional[int] = None,
    recent_logs: Optional[List[Dict[str, Any]]] = None,
    infer_outputs: Optional[Dict[str, str]] = None,
) -> str:
    manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
    latest_pt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    if not os.path.exists(latest_pt):
        print(f"[!] No latest checkpoint found at {latest_pt}")
        return ""

    step = step_override
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                if step is None:
                    step = manifest.get("step")
        except Exception as e:
            print(f"[!] Error reading manifest: {e}")

    if step is None:
        import torch
        ckpt = torch.load(latest_pt, map_location="cpu", weights_only=False)
        step = ckpt.get("step", 0)

    archive_name = f"checkpoint(step_{step})"
    dest_dir = os.path.join(checkpoint_dir, archive_name)
    os.makedirs(dest_dir, exist_ok=True)

    dest_pt = os.path.join(dest_dir, f"{archive_name}.pt")
    dest_manifest = os.path.join(dest_dir, "checkpoint_manifest.json")
    dest_md = os.path.join(dest_dir, f"{archive_name}.md")
    dest_json = os.path.join(dest_dir, f"{archive_name}.json")

    # 1. Copy weights and manifest
    if not os.path.exists(dest_pt) or os.path.getsize(dest_pt) == 0:
        print(f"[*] Archiving step {step} weights to {dest_pt}...")
        shutil.copy2(latest_pt, dest_pt)
    else:
        print(f"[*] Weights already present at {dest_pt}")

    if os.path.exists(manifest_path):
        shutil.copy2(manifest_path, dest_manifest)

    # 2. Extract recent logs if not passed
    if recent_logs is None:
        latest_csv = find_latest_session_file(log_dir, ".csv")
        recent_logs = get_recent_logs_from_csv(latest_csv, max_rows=10)
        # If latest csv had fewer than 10, try falling back to previous csv
        if len(recent_logs) < 10:
            csv_files = sorted(glob.glob(os.path.join(log_dir, "*.csv")), key=os.path.getmtime, reverse=True)
            all_records = []
            for cf in csv_files[:2]:
                all_records = get_recent_logs_from_csv(cf, max_rows=10) + all_records
            recent_logs = all_records[-10:]

    # 3. Extract mini-inference preview if not passed
    if infer_outputs is None:
        latest_jsonl = find_latest_session_file(log_dir, ".jsonl")
        infer_outputs = get_preview_from_jsonl(latest_jsonl, target_step=step)
        if not infer_outputs:
            jsonl_files = sorted(glob.glob(os.path.join(log_dir, "*.jsonl")), key=os.path.getmtime, reverse=True)
            for jf in jsonl_files:
                infer_outputs = get_preview_from_jsonl(jf, target_step=step)
                if infer_outputs:
                    break

    tokens_processed = manifest.get("tokens_processed", step * 8192)
    metrics = manifest.get("metrics", {})
    curr_loss = metrics.get("current_loss") or (recent_logs[-1]["loss"] if recent_logs else 0.0)
    ema_loss = metrics.get("ema_loss") or (recent_logs[-1]["avg_loss"] if recent_logs else 0.0)
    curr_ppl = metrics.get("current_ppl") or (recent_logs[-1]["ppl"] if recent_logs else 0.0)
    ema_ppl = metrics.get("ema_ppl") or (recent_logs[-1]["avg_ppl"] if recent_logs else 0.0)

    # 4. Generate structured JSON report
    report_data = {
        "step": step,
        "tokens_processed": tokens_processed,
        "timestamp": datetime.now().isoformat(),
        "latest_metrics": {
            "loss": curr_loss,
            "ema_loss": ema_loss,
            "ppl": curr_ppl,
            "ema_ppl": ema_ppl,
        },
        "last_10_logged_steps": recent_logs,
        "mini_inference_preview": infer_outputs,
        "hyperparameters": manifest.get("hyperparameters", {}),
    }

    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 5. Generate Markdown report
    md_lines = [
        f"# TinyTransformer Checkpoint Report — Step {step:,}",
        "",
        f"- **Global Step**: `{step:,}`",
        f"- **Tokens Processed**: `{tokens_processed:,}` ({tokens_processed/1e6:.2f}M)",
        f"- **Latest Step Loss**: `{curr_loss:.4f}` (EMA Avg: `{ema_loss:.4f}`)",
        f"- **Latest Step PPL**: `{curr_ppl:.2f}` (EMA Avg: `{ema_ppl:.2f}`)",
        f"- **Archive Directory**: `{dest_dir}`",
        f"- **Archive File**: `{dest_pt}`",
        f"- **Report Generated**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "---",
        "",
        "## Recent Metrics Progression (Last 10 Logged Steps)",
        "",
        "| Step | Loss | EMA Loss | Step PPL | EMA PPL | Speed (tok/s) | VRAM (MB) | Timestamp |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for entry in recent_logs:
        s = entry.get("step", 0)
        l = entry.get("loss", 0.0)
        al = entry.get("avg_loss", 0.0)
        p = entry.get("ppl", 0.0)
        ap = entry.get("avg_ppl", 0.0)
        spd = entry.get("tokens_per_sec", 0.0)
        vr = entry.get("vram_mb", 0.0)
        ts = entry.get("timestamp", "").split("T")[-1][:8]
        md_lines.append(f"| {s:,} | {l:.4f} | {al:.4f} | {p:.2f} | {ap:.2f} | {spd:.0f} | {vr:.0f} | {ts} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Mini-Inference Preview Generations",
        "",
    ])

    if infer_outputs:
        for p_idx, (p_text, completion) in enumerate(infer_outputs.items(), 1):
            md_lines.append(f"### Prompt {p_idx}")
            md_lines.append("```text")
            md_lines.append(p_text.strip())
            md_lines.append("```")
            clean_comp = completion.strip() if completion else "*(empty)*"
            md_lines.append(f"> {clean_comp.replace(chr(10), chr(10) + '> ')}")
            md_lines.append("")
    else:
        md_lines.append("*No mini-inference preview recorded at this step.*")
        md_lines.append("")

    with open(dest_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Also copy to root of checkpoints directory
    root_md = os.path.join(checkpoint_dir, f"{archive_name}.md")
    root_json = os.path.join(checkpoint_dir, f"{archive_name}.json")
    shutil.copy2(dest_md, root_md)
    shutil.copy2(dest_json, root_json)

    file_size_gb = os.path.getsize(dest_pt) / (1024 ** 3)
    print(f"[+] Successfully archived {dest_pt} ({file_size_gb:.2f} GB) with {archive_name}.md and {archive_name}.json")
    return dest_dir


if __name__ == "__main__":
    archive_latest_checkpoint()
