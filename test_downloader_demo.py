"""
Quick Test Script to demonstrate the Ultra-Fast Parallel Downloader
Creates a small sample CSV with 10 real sample images and tests downloading.
"""

import pandas as pd
import os
import subprocess
import sys

def create_sample_and_test():
    sample_data = {
        "index": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "product_name": [
            "Sample Product 1", "Sample Product 2", "Sample Product 3",
            "Sample Product 4", "Sample Product 5", "Sample Product 6",
            "Sample Product 7", "Sample Product 8", "Sample Product 9",
            "Sample Product 10"
        ],
        "image_link": [
            f"https://picsum.photos/id/{i*10}/300/300.jpg" for i in range(1, 11)
        ]
    }
    
    test_csv = "sample_test.csv"
    df = pd.DataFrame(sample_data)
    df.to_csv(test_csv, index=False)
    print(f"[+] Created test CSV '{test_csv}' with 10 sample image links.")
    
    cmd = [
        sys.executable, "download_images.py",
        "--csv_path", test_csv,
        "--image_col", "image_link",
        "--id_col", "index",
        "--output_dir", "sample_images",
        "--workers", "10"
    ]
    print(f"[+] Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    create_sample_and_test()
