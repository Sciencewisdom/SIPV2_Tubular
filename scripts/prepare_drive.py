#!/usr/bin/env python3
"""
DRIVE dataset preparation script.

DRIVE (Digital Retinal Images for Vessel Extraction) is a standard benchmark
for retinal vessel segmentation.

Download URL: https://drive.grand-challenge.org/
The dataset is freely available for research purposes but requires registration.

Expected directory structure after download:
    data/raw/DRIVE/
    ├── training/
    │   ├── images/          # 20 .tif files (RGB fundus images)
    │   ├── 1st_manual/      # 20 .gif files (expert 1 annotations)
    │   └── mask/            # 20 .gif files (FOV masks)
    └── test/
        ├── images/          # 20 .tif files
        ├── 1st_manual/      # 20 .gif files (expert 1)
        ├── 2nd_manual/      # 20 .gif files (expert 2)
        └── mask/            # 20 .gif files

This script checks if the dataset is present and validates the structure.
"""
import os
import sys
import glob


def check_drive_dataset(root_dir='data/raw/DRIVE'):
    """Check if DRIVE dataset is present and valid."""
    issues = []

    required = {
        'training/images': '*.tif',
        'training/1st_manual': '*.gif',
        'training/mask': '*.gif',
        'test/images': '*.tif',
        'test/1st_manual': '*.gif',
        'test/mask': '*.gif',
    }

    all_ok = True
    for subdir, pattern in required.items():
        path = os.path.join(root_dir, subdir)
        files = glob.glob(os.path.join(path, pattern))
        if len(files) == 0:
            issues.append(f"  MISSING: {path}/")
            all_ok = False
        else:
            print(f"  OK: {subdir} ({len(files)} files)")

    return all_ok, issues


def main():
    print("="*60)
    print("DRIVE Dataset Preparation")
    print("="*60)

    root_dir = 'data/raw/DRIVE'
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    print(f"\nChecking: {root_dir}")
    print()

    ok, issues = check_drive_dataset(root_dir)

    if ok:
        print("\n" + "="*60)
        print("DRIVE dataset is ready!")
        print("="*60)
        print("\nYou can now run training:")
        print("  bash scripts/run_drive_experiments.sh")
    else:
        print("\n" + "="*60)
        print("DRIVE dataset NOT found!")
        print("="*60)
        print("\nMissing files:")
        for issue in issues:
            print(issue)
        print("\nTo download DRIVE:")
        print("  1. Visit: https://drive.grand-challenge.org/")
        print("  2. Register and download the dataset")
        print("  3. Extract to: data/raw/DRIVE/")
        print("\nExpected structure:")
        print("  data/raw/DRIVE/training/images/*.tif")
        print("  data/raw/DRIVE/training/1st_manual/*.gif")
        print("  data/raw/DRIVE/training/mask/*.gif")
        print("  data/raw/DRIVE/test/images/*.tif")
        print("  data/raw/DRIVE/test/1st_manual/*.gif")
        print("  data/raw/DRIVE/test/mask/*.gif")
        sys.exit(1)


if __name__ == '__main__':
    main()
