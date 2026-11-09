"""
Automated Kaggle Dataset Packaging & Deployment Pipeline for TinyTransformer.

Handles:
1. Shard integrity scanning & token audit (3.6B Foundation + 200M Instruction).
2. Generates Kaggle-compliant dataset-metadata.json.
3. Automatically stages binary shards with zero-copy hardlinks (os.link) to keep disk space under 7.6GB.
4. Triggers Kaggle CLI push or provides 1-click UI instructions.

Tutorials & Documentation:
- Master Kaggle TPU v5e-8 & Sharding Guide: KAGGLE_TPU_GUIDE.md
- Sharding Notebook: TinyTransformer_Kaggle_Data_Sharding.ipynb
- TPU Training Notebook: TinyTransformer_Kaggle_TPU_v5e_8.ipynb
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.download_and_tokenize import get_shard_stats


def generate_kaggle_dataset_metadata(
    output_dir: str,
    slug: str = "tinytransformer-pretraining-shards",
    title: str = "TinyTransformer Pretraining Shards",
    username: str = None,
) -> str:
    """Generates official Kaggle dataset-metadata.json specification."""
    os.makedirs(output_dir, exist_ok=True)

    if not username:
        # Check Kaggle API config or environment
        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_json):
            try:
                with open(kaggle_json) as f:
                    username = json.load(f).get("username")
            except Exception:
                pass
        if not username:
            username = os.environ.get("KAGGLE_USERNAME", "your-kaggle-username")

    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "apache-2.0"}],
        "keywords": ["nlp", "transformers", "pretraining", "synthetic-data", "distillation"],
        "description": (
            "Pre-tokenized binary shards (uint16) for TinyTransformer (~238M parameters). "
            "Includes Cosmopedia v2 (synthetic textbook distillation), FineWeb-Edu Dedup, "
            "FineMath-4plus, Python-Edu, Wikipedia, and multi-faceted SmolTalk instruction/CoT distillation."
        ),
    }

    metadata_path = os.path.join(output_dir, "dataset-metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def package_and_export(
    shard_dir: str = "data/shards",
    export_dir: str = "/kaggle/working/shards_export",
    slug: str = "tinytransformer-shards",
    title: str = "TinyTransformer Pretraining Shards (Foundation + Instruct)",
    username: str = None,
    push: bool = False,
):
    print("\n" + "=" * 75)
    print("  [>] PACKAGING TINYTRANSFORMER SHARDS FOR KAGGLE DATASET")
    print("=" * 75)

    stats = get_shard_stats(shard_dir)
    total_tokens = stats["total_tokens"]
    total_mb = stats["total_mb"]

    if total_tokens == 0:
        print(f"[!] Warning: No binary shards found in '{shard_dir}'.")
        return

    print(f"* Source Shards:     {shard_dir}")
    print(f"* Total Tokens:      {total_tokens:,} ({total_tokens/1e6:.1f}M tokens)")
    print(f"* Total Footprint:   {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
    print(f"* Target Export:     {export_dir}")
    print("=" * 75)

    os.makedirs(export_dir, exist_ok=True)

    # 1. Generate Kaggle dataset-metadata.json
    meta_file = generate_kaggle_dataset_metadata(
        output_dir=export_dir,
        slug=slug,
        title=title,
        username=username,
    )
    print(f"[+] Kaggle metadata generated: {meta_file}")

    # 2. Stage binary shards into export_dir (symlink or copy)
    shards = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
    staged = 0
    for s in shards:
        dst = os.path.join(export_dir, os.path.basename(s))
        if not os.path.exists(dst):
            try:
                os.link(s, dst)  # Hardlink on Linux/Kaggle (0 space, 0 copy delay)
            except Exception:
                try:
                    os.symlink(s, dst)
                except Exception:
                    shutil.copy2(s, dst)
        staged += 1

    print(f"[+] Staged {staged} shards to export directory ({export_dir}).")

    # 3. Check for Kaggle CLI deployment
    has_kaggle_cli = shutil.which("kaggle") is not None
    has_creds = (
        os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )

    if push and has_kaggle_cli and has_creds:
        print(f"[*] Kaggle CLI authenticated. Publishing dataset to Kaggle Cloud...")
        try:
            res = subprocess.run(
                ["kaggle", "datasets", "create", "-p", export_dir, "--dir-mode", "zip"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                print(f"[+] [SUCCESS] Kaggle Dataset published successfully!\n{res.stdout}")
            else:
                # If dataset already exists, push a new version
                print(f"[*] Dataset may already exist. Attempting to push new version...")
                res_ver = subprocess.run(
                    ["kaggle", "datasets", "version", "-p", export_dir, "-m", f"Automated shard update ({total_tokens/1e6:.0f}M tokens)", "--dir-mode", "zip"],
                    capture_output=True,
                    text=True,
                )
                print(res_ver.stdout or res_ver.stderr)
        except Exception as e:
            print(f"[!] Kaggle CLI upload error: {e}")
    else:
        print("\n" + "-" * 75)
        print("  💡 HOW TO PUBLISH THIS AS A KAGGLE DATASET IN 3 CLICKS (NO CLI NEEDED):")
        print("-" * 75)
        print("  1. In your Kaggle Notebook top right, click 'Save Version' (Commit).")
        print("  2. Select 'Save & Run All (Commit)' -> click 'Save'.")
        print("  3. Once the run completes, click on the Notebook's 'Output' tab.")
        print("  4. Look for the folder 'shards_export' and click '+ New Dataset'.")
        print(f"  5. Title it '{title}' and click 'Create'.")
        print("  6. Now you can attach this dataset in any TPU notebook via '+ Add Input'!")
        print("-" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Shards for Kaggle Dataset")
    parser.add_argument("--shard-dir", type=str, default="data/shards", help="Source shards directory")
    parser.add_argument("--export-dir", type=str, default="/kaggle/working/shards_export", help="Output directory to stage dataset")
    parser.add_argument("--slug", type=str, default="tinytransformer-shards-2b", help="Dataset URL slug")
    parser.add_argument("--title", type=str, default="TinyTransformer Pretraining Shards (1.85B Foundation + 150M Instruct)", help="Dataset title")
    parser.add_argument("--username", type=str, default=None, help="Kaggle username")
    parser.add_argument("--push", action="store_true", help="Push directly via Kaggle CLI if authenticated")
    args = parser.parse_args()

    package_and_export(
        shard_dir=args.shard_dir,
        export_dir=args.export_dir,
        slug=args.slug,
        title=args.title,
        username=args.username,
        push=args.push,
    )
