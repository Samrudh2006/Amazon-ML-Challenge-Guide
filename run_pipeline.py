"""
Amazon ML Challenge 2026 - Master End-to-End Orchestrator CLI
============================================================
The unified control tower that orchestrates the entire competition pipeline with a single command.

Usage Examples:
1. Run everything end-to-end:
   python run_pipeline.py --stage all --train_csv train.csv --test_csv test.csv

2. Run only Image Downloading:
   python run_pipeline.py --stage download --train_csv train.csv --workers 80

3. Run only OCR Extraction:
   python run_pipeline.py --stage ocr --image_dir images/train

4. Run 5-Fold Multimodal Training & Stacking:
   python run_pipeline.py --stage train --folds 5 --epochs 5

5. Validate Submission:
   python run_pipeline.py --stage validate --sub final_submission.csv --test_csv test.csv
"""

import os
import sys
import time
import argparse
import subprocess

def print_banner(stage_name):
    print("\n" + "#" * 70)
    print(f"  >>> EXECUTING STAGE: {stage_name.upper()} <<<")
    print("#" * 70 + "\n")

def run_cmd(cmd_list):
    print(f"[*] Running command: {' '.join(cmd_list)}")
    start = time.time()
    ret = subprocess.run(cmd_list)
    elapsed = time.time() - start
    if ret.returncode != 0:
        print(f"[!] Stage failed with exit code {ret.returncode} after {elapsed:.2f}s")
        return False
    print(f"[+] Stage completed successfully in {elapsed:.2f}s ({elapsed/60:.2f}m)\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Master Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "download", "ocr", "adversarial", "train", "ensemble", "pseudo", "validate"],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to train CSV")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument("--image_dir", type=str, default="images/train", help="Image storage directory")
    parser.add_argument("--workers", type=int, default=60, help="Downloader worker threads")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--sub", type=str, default="final_submission.csv", help="Submission CSV to validate")
    
    args = parser.parse_args()
    python_bin = sys.executable

    # Intelligent Dataset Fallback
    if not os.path.exists(args.train_csv) and os.path.exists("mock_train.csv"):
        print("[*] 'train.csv' not found. Automatically utilizing 'mock_train.csv' for pipeline execution.")
        args.train_csv = "mock_train.csv"
    if not os.path.exists(args.test_csv) and os.path.exists("mock_test.csv"):
        print("[*] 'test.csv' not found. Automatically utilizing 'mock_test.csv' for pipeline execution.")
        args.test_csv = "mock_test.csv"

    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 - MASTER PIPELINE CONTROLLER")
    print(f"  Stage: {args.stage.upper()} | Train: {args.train_csv} | Test: {args.test_csv}")
    print("=" * 70)

    # 0. AUTO-ADAPT STAGE (100% Zero-Touch Configurator)
    if args.stage in ["all", "adapt"]:
        print_banner("0. Auto-Adaptive Schema & Task Detection")
        if os.path.exists(args.train_csv):
            cmd = [
                python_bin, "auto_adapt.py",
                "--train", args.train_csv,
                "--test", args.test_csv
            ]
            run_cmd(cmd)

    # 1. DOWNLOAD STAGE
    if args.stage in ["all", "download"]:
        print_banner("1. Fast Parallel Image Download")
        cmd = [
            python_bin, "download_images.py",
            "--csv_path", args.train_csv,
            "--output_dir", args.image_dir,
            "--workers", str(args.workers)
        ]
        if not run_cmd(cmd) and args.stage == "all":
            print("[!] Download failed or paused. Continuing to next stage...")

    # 2. OCR EXTRACTION STAGE
    if args.stage in ["all", "ocr"]:
        print_banner("2. Optical Character Recognition (OCR) Extraction")
        cmd = [
            python_bin, "ocr_extractor.py",
            "--image_dir", args.image_dir,
            "--output_csv", "ocr_train.csv"
        ]
        run_cmd(cmd)

    # 3. ADVERSARIAL DRIFT CHECK
    if args.stage in ["all", "adversarial"]:
        print_banner("3. Adversarial Validation & Distribution Drift Check")
        if os.path.exists(args.train_csv) and os.path.exists(args.test_csv):
            cmd = [python_bin, "adversarial_validation.py"]
            run_cmd(cmd)
        else:
            print("[*] Skipping adversarial check (train.csv or test.csv not found locally yet).")

    # 4. MODEL TRAINING (DISPATCHED BY TASK TYPE)
    if args.stage in ["all", "train"]:
        print_banner("4. 5-Fold Model Training (Auto-Dispatched)")
        # Check auto_adapt config if available
        task_type = "UNKNOWN"
        if os.path.exists("competition_config.json"):
            try:
                import json
                with open("competition_config.json", "r") as f:
                    cfg = json.load(f)
                    task_type = cfg.get("task_type", "UNKNOWN")
            except Exception:
                pass

        if "REGRESSION" in task_type:
            print("[*] Task detected as REGRESSION / PRICE PREDICTION. Launching Dual GBDT Champions...")
            # Run LightGBM Baseline
            cmd_lgb = [python_bin, "train_baseline.py", args.train_csv, args.test_csv]
            run_cmd(cmd_lgb)
            # Run CatBoost Champion with Semantic Clusters
            cmd_cb = [python_bin, "catboost_champion.py", args.train_csv, args.test_csv]
            run_cmd(cmd_cb)
        else:
            # Multimodal Neural Network Training
            cmd = [
                python_bin, "train_multimodal.py",
                "--epochs", str(args.epochs)
            ]
            run_cmd(cmd)

    # 5. OOF STACKING & ENSEMBLING
    if args.stage in ["all", "ensemble"]:
        print_banner("5. Kaggle Grandmaster OOF Stacking & Meta-Ensemble")
        if os.path.exists("catboost_oof.npy") and os.path.exists("test_ensemble.py"):
            print("[*] Running Grandmaster Stacker with Nelder-Mead Calibration...")
            cmd = [python_bin, "test_ensemble.py"]
            run_cmd(cmd)
        else:
            cmd = [python_bin, "oof_stacking_ensemble.py"]
            run_cmd(cmd)


    # 6. DISQUALIFICATION SHIELD VALIDATION
    if args.stage in ["all", "validate"]:
        print_banner("6. Final Submission Integrity Verification")
        sub_to_check = args.sub
        if not os.path.exists(sub_to_check):
            for candidate in ["final_champion_submission.csv", "submission.csv"]:
                if os.path.exists(candidate):
                    sub_to_check = candidate
                    break
        cmd = [
            python_bin, "submission_verifier.py",
            sub_to_check, args.test_csv
        ]
        if os.path.exists(args.train_csv):
            cmd.append(args.train_csv)
        run_cmd(cmd)

    print("=" * 70)
    print("  PIPELINE EXECUTION COMPLETE! READY FOR LEADERBOARD.")
    print("=" * 70)

if __name__ == "__main__":
    main()
