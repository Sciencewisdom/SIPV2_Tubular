import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "http://www.cs.toronto.edu/~vmnih/data"
ROOT_DIR = "/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/data/raw/mass_roads"

splits = ['train', 'valid', 'test']


def fetch_index(split, kind):
    url = f"{BASE_URL}/mass_roads/{split}/{kind}/index.html"
    idx_file = f"/tmp/mass_idx_{split}_{kind}.html"
    try:
        subprocess.run(['wget', '-q', '-O', idx_file, url], check=True, timeout=60)
        with open(idx_file, 'r') as f:
            html = f.read()
        files = []
        for href in re.findall(r'href="([^"]+\.(?:tif|tiff))"', html):
            fname = os.path.basename(href)
            files.append(fname)
        return files
    except Exception as e:
        print(f"Error fetching index for {split}/{kind}: {e}")
        return []


def download_file(args):
    url, out_path = args
    max_retries = 3
    for attempt in range(max_retries):
        if os.path.exists(out_path):
            size = os.path.getsize(out_path)
            if size > 100000:  # Real TIFFs are ~1-2MB, skip if looks good
                return True
            elif size > 0:
                # Partial file, retry
                os.remove(out_path)
        try:
            result = subprocess.run(
                ['wget', '-q', '--timeout=60', '-t', '2', '-O', out_path, url],
                check=True, timeout=180
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
                return True
            else:
                # Download succeeded but file too small -> probably error page
                if os.path.exists(out_path):
                    os.remove(out_path)
                return False
        except Exception as e:
            if os.path.exists(out_path):
                os.remove(out_path)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    return False


def main():
    all_tasks = []
    for split in splits:
        for kind in ['sat', 'map']:
            files = fetch_index(split, kind)
            print(f"{split}/{kind}: {len(files)} files")
            out_dir = os.path.join(ROOT_DIR, split, kind)
            os.makedirs(out_dir, exist_ok=True)
            for fname in files:
                url = f"{BASE_URL}/mass_roads/{split}/{kind}/{fname}"
                out_path = os.path.join(out_dir, fname)
                all_tasks.append((url, out_path))

    print(f"Total files to download: {len(all_tasks)}")
    
    # Pre-clean zero-byte files
    cleaned = 0
    for url, out_path in all_tasks:
        if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
            os.remove(out_path)
            cleaned += 1
    if cleaned:
        print(f"Cleaned {cleaned} zero-byte files")

    success_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, t): t for t in all_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            result = future.result()
            if result:
                success_count += 1
            else:
                fail_count += 1

    print(f"Done. Success: {success_count}, Failed: {fail_count}")


if __name__ == '__main__':
    main()
