"""
Amazon ML Challenge 2026 - High-Performance Parallel Image Downloader
====================================================================
Features:
- Multi-threaded ultra-fast downloading (50-100 workers)
- Automatic retry with exponential backoff on timeouts/network glitches
- Resumable: Skips already downloaded and valid images
- Image integrity check (verifies valid image file, prevents 0-byte corrupt files)
- Progress tracking with ETA, speed (images/sec), and success/failure counts
- Logs failed downloads to a CSV for error analysis or targeted retries
"""

import os
import sys
import time
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io

def create_resilient_session(retries=3, backoff_factor=0.3, pool_maxsize=100):
    """Creates a requests Session with automatic retries and connection pooling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "AmazonMLChallengeBot/1.0 (StudentTeam; mailto:team@example.edu) Python-requests/2.31.0",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    return session

def is_valid_image(filepath):
    """Check if the downloaded file exists and is a non-corrupt image."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def download_single_image(args):
    """Worker function to download a single image."""
    idx, url, output_dir, session, timeout, verify_img = args
    
    if not isinstance(url, str) or not url.strip():
        return (idx, url, False, "Empty or invalid URL")
        
    url = url.strip()
    
    # Generate clean filename based on index or URL basename
    url_basename = os.path.basename(url.split("?")[0])
    extension = os.path.splitext(url_basename)[1].lower()
    if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        extension = ".jpg"
    
    filename = f"{idx}_{url_basename}"
    # Remove any illegal characters for Windows filesystem
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        filename = filename.replace(ch, '_')
        
    filepath = os.path.join(output_dir, filename)
    
    # Resume check: if already exists and valid, skip!
    if os.path.exists(filepath):
        if not verify_img or is_valid_image(filepath):
            return (idx, url, True, "Already Exists (Skipped)")
        else:
            try:
                os.remove(filepath)
            except OSError:
                pass

    try:
        response = session.get(url, timeout=timeout)
        if response.status_code == 200:
            content = response.content
            if len(content) < 100:  # Suspiciously small file / error page
                return (idx, url, False, f"File too small ({len(content)} bytes)")
                
            if verify_img:
                # Test in memory before writing to disk
                try:
                    img = Image.open(io.BytesIO(content))
                    img.verify()
                except Exception as e:
                    return (idx, url, False, f"Corrupted image payload: {e}")
            
            with open(filepath, "wb") as f:
                f.write(content)
            return (idx, url, True, "Downloaded")
        else:
            return (idx, url, False, f"HTTP Status {response.status_code}")
    except Exception as e:
        return (idx, url, False, str(e))

def run_parallel_downloader(
    csv_path,
    image_col="image_link",
    id_col=None,
    output_dir="images",
    num_workers=60,
    timeout=10,
    verify_images=True,
    sample_limit=None
):
    """
    Downloads images from a CSV in parallel with maximum throughput.
    """
    print(f"[*] Reading dataset: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[!] Error: File '{csv_path}' does not exist.")
        return

    df = pd.read_csv(csv_path)
    print(f"[*] Total rows in CSV: {len(df):,}")

    if image_col not in df.columns:
        print(f"[!] Error: Column '{image_col}' not found. Available columns: {list(df.columns)}")
        return

    if sample_limit:
        df = df.head(sample_limit)
        print(f"[*] Limiting download to first {sample_limit:,} rows for testing.")

    os.makedirs(output_dir, exist_ok=True)
    session = create_resilient_session(pool_maxsize=num_workers + 10)

    # Prepare download tasks
    tasks = []
    for idx, row in df.iterrows():
        task_id = row[id_col] if id_col and id_col in df.columns else idx
        url = row[image_col]
        tasks.append((task_id, url, output_dir, session, timeout, verify_images))

    print(f"[*] Starting parallel download with {num_workers} worker threads...")
    print(f"[*] Destination directory: {os.path.abspath(output_dir)}")
    start_time = time.time()

    success_count = 0
    skipped_count = 0
    failed_records = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single_image, task): task for task in tasks}
        
        progress_bar = tqdm(as_completed(futures), total=len(tasks), desc="Downloading Images", unit="img")
        
        for future in progress_bar:
            idx, url, success, msg = future.result()
            if success:
                if "Already Exists" in msg:
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_records.append({"id": idx, "url": url, "error": msg})
                
            progress_bar.set_postfix({
                "Downloaded": success_count,
                "Skipped": skipped_count,
                "Failed": len(failed_records)
            })

    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print(f"[*] Download Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} mins)")
    print(f"[*] Successfully Downloaded: {success_count:,}")
    print(f"[*] Skipped (Already existed): {skipped_count:,}")
    print(f"[*] Failed: {len(failed_records):,}")
    print(f"[*] Average Speed: {len(tasks)/elapsed:.1f} images/second")
    print("="*50)

    # Save failed URLs report
    if failed_records:
        failed_df = pd.DataFrame(failed_records)
        failed_csv_path = os.path.join(output_dir, "failed_images.csv")
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"[!] Saved list of failed URLs to: {failed_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge High-Speed Image Downloader")
    parser.add_argument("--csv_path", type=str, default="train.csv", help="Path to CSV containing image links")
    parser.add_argument("--image_col", type=str, default="image_link", help="Column name containing image URLs")
    parser.add_argument("--id_col", type=str, default="index", help="ID column name (or leave default)")
    parser.add_argument("--output_dir", type=str, default="images/train", help="Directory where images should be stored")
    parser.add_argument("--workers", type=int, default=60, help="Number of parallel worker threads (e.g., 50-80)")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds per image request")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Download only first N images to test")
    
    args = parser.parse_args()
    
    run_parallel_downloader(
        csv_path=args.csv_path,
        image_col=args.image_col,
        id_col=args.id_col,
        output_dir=args.output_dir,
        num_workers=args.workers,
        timeout=args.timeout,
        sample_limit=args.limit
    )
