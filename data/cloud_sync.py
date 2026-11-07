"""
Plug-and-Play Cloud Checkpoint Synchronization Engine for TinyTransformer.
Supports:
1. Hugging Face Hub (fast, multi-threaded, authenticated or public model repo)
2. Google Drive (via direct export link or file ID)
3. Direct HTTP/HTTPS URLs (resumable chunks with progress bar)
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Optional, Dict, Any
import requests

CONFIG_FILE = "cloud_config.json"


def load_cloud_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cloud_config(config: Dict[str, Any]):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[CloudSync Warning] Could not save {CONFIG_FILE}: {e}")


def download_from_hf(repo_id: str, filename: str = "checkpoint_latest.pt", local_dir: str = "checkpoints", token: Optional[str] = None) -> Optional[str]:
    """Downloads checkpoint from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"[*] [CloudSync] Pulling '{filename}' from Hugging Face Hub repository: '{repo_id}'...")
        os.makedirs(local_dir, exist_ok=True)
        dest_path = os.path.join(local_dir, filename)
        tmp_path = dest_path + ".tmp"

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=token or os.environ.get("HF_TOKEN"),
        )
        print(f"[+] [CloudSync] Successfully downloaded checkpoint to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[!] [CloudSync] Hugging Face Hub download failed: {e}")
        return None


def download_from_drive(file_id: str, dest_path: str) -> bool:
    """Downloads a public or shared Google Drive file directly without external tools."""
    print(f"[*] [CloudSync] Pulling checkpoint from Google Drive (ID: {file_id})...")
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    params = {"id": file_id, "confirm": "t"}
    
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    tmp_path = dest_path + ".tmp"

    try:
        response = session.get(url, params=params, stream=True, timeout=30)
        # Handle Drive confirmation token for files >100MB
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                params["confirm"] = value
                response = session.get(url, params=params, stream=True, timeout=30)
                break

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        start_time = time.time()

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):  # 8MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        speed = mb / max(1e-5, time.time() - start_time)
                        sys.stdout.write(f"\r    >> Downloaded {mb:.1f} MB ({pct:.1f}%) | Speed: {speed:.1f} MB/s")
                        sys.stdout.flush()

        print()
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 10000:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.replace(tmp_path, dest_path)
            print(f"[+] [CloudSync] Checkpoint downloaded successfully to: {dest_path}")
            return True
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print("[!] [CloudSync] Downloaded file appears too small or corrupt.")
            return False
    except Exception as e:
        print(f"[!] [CloudSync] Google Drive download failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def download_from_url(url: str, dest_path: str) -> bool:
    """Streams a checkpoint from any direct HTTP/HTTPS URL."""
    print(f"[*] [CloudSync] Streaming checkpoint from direct URL: {url}...")
    tmp_path = dest_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        start_time = time.time()

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        speed = mb / max(1e-5, time.time() - start_time)
                        sys.stdout.write(f"\r    >> Downloaded {mb:.1f} MB ({pct:.1f}%) | Speed: {speed:.1f} MB/s")
                        sys.stdout.flush()

        print()
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.replace(tmp_path, dest_path)
        print(f"[+] [CloudSync] Successfully saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"[!] [CloudSync] URL download failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def upload_to_hf(src_path: str, repo_id: str, filename: str = "checkpoint_latest.pt", token: Optional[str] = None) -> bool:
    """Uploads checkpoint to a Hugging Face Hub repository."""
    if not os.path.exists(src_path):
        print(f"[!] [CloudSync] Source file not found: {src_path}")
        return False
    try:
        from huggingface_hub import upload_file, HfApi
        token = token or os.environ.get("HF_TOKEN")
        print(f"[*] [CloudSync] Uploading '{src_path}' ({os.path.getsize(src_path) / (1024**3):.2f} GB) to HF Hub: '{repo_id}'...")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        upload_file(
            path_or_fileobj=src_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
        )
        print(f"[+] [CloudSync] Successfully published checkpoint to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"[!] [CloudSync] Hugging Face Hub upload failed: {e}")
        return False


def sync_cloud_checkpoint(
    checkpoint_dir: str = "checkpoints",
    hf_repo: Optional[str] = None,
    drive_id: Optional[str] = None,
    url: Optional[str] = None,
    force_pull: bool = False,
) -> Optional[str]:
    """
    Main plug-and-play entrypoint:
    Ensures 'checkpoint_dir/checkpoint_latest.pt' is available by pulling from cloud
    if missing locally or if force_pull is True.
    """
    dest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if os.path.exists(dest_path) and not force_pull:
        return dest_path

    cfg = load_cloud_config()
    hf_repo = hf_repo or cfg.get("hf_repo") or os.environ.get("HF_CHECKPOINT_REPO")
    drive_id = drive_id or cfg.get("drive_id") or os.environ.get("GDRIVE_CHECKPOINT_ID")
    url = url or cfg.get("url") or os.environ.get("CLOUD_CHECKPOINT_URL")

    # 1. Try Hugging Face Hub
    if hf_repo:
        res = download_from_hf(repo_id=hf_repo, filename="checkpoint_latest.pt", local_dir=checkpoint_dir)
        if res and os.path.exists(dest_path):
            return dest_path

    # 2. Try Google Drive ID
    if drive_id:
        if download_from_drive(file_id=drive_id, dest_path=dest_path):
            return dest_path

    # 3. Try Direct URL
    if url:
        if download_from_url(url=url, dest_path=dest_path):
            return dest_path

    return None if not os.path.exists(dest_path) else dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyTransformer Cloud Checkpoint Sync Tool")
    parser.add_argument("--pull", action="store_true", help="Download latest checkpoint from cloud")
    parser.add_argument("--push", action="store_true", help="Upload local checkpoint to cloud")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. username/TinyTransformer-checkpoints)")
    parser.add_argument("--drive-id", type=str, default=None, help="Google Drive file ID")
    parser.add_argument("--url", type=str, default=None, help="Direct HTTP/HTTPS download URL")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Save to config if provided
    cfg = load_cloud_config()
    if args.hf_repo:
        cfg["hf_repo"] = args.hf_repo
    if args.drive_id:
        cfg["drive_id"] = args.drive_id
    if args.url:
        cfg["url"] = args.url
    if any([args.hf_repo, args.drive_id, args.url]):
        save_cloud_config(cfg)

    ckpt_file = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")

    if args.push:
        hf = args.hf_repo or cfg.get("hf_repo")
        if hf:
            upload_to_hf(ckpt_file, repo_id=hf)
        else:
            print("[!] Please provide --hf-repo to upload checkpoint.")
    elif args.pull:
        sync_cloud_checkpoint(checkpoint_dir=args.checkpoint_dir, hf_repo=args.hf_repo, drive_id=args.drive_id, url=args.url, force_pull=True)
    else:
        print("Usage: python data/cloud_sync.py --pull OR python data/cloud_sync.py --push --hf-repo <repo_id>")
