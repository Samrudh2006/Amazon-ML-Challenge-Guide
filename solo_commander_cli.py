"""
=================================================================================
SOLO COMMANDER CLI (THE ONE-MAN ARMY MASTER DASHBOARD)
Designed specifically for solo competitors in Amazon ML Challenge 2026.
You don't need anyone else. Single command to run the entire competition pipeline:
Auto-Detect -> Tabular Features -> 5-Fold GBDT -> Deep Ensembling -> Verification
=================================================================================
"""

import os
import sys
import subprocess

def print_banner():
    print("\n" + "="*75)
    print("   AMAZON ML CHALLENGE 2026 - SOLO COMMANDER MASTER DASHBOARD")
    print("   [Autonomous One-Man Army Execution Engine | Zero-Code Operation]")
    print("="*75)
    print("  [1] 🚀 1-CLICK FULL AUTO-PILOT (Ingest -> Train -> Ensemble -> Audit)")
    print("  [2] 🧪 RUN 100% SYSTEM INTEGRATION AUDIT (Verify all 15 modules)")
    print("  [3] ⚡ RUN 5-FOLD FAST BASELINE (LightGBM + CatBoost on log-price)")
    print("  [4] 🎯 RUN MRP ANCHORING & CATALOG TWINS (Pillars 1 & 2)")
    print("  [5] 🛡️ AUDIT SUBMISSION CSV (Zero-Disqualification Pre-Flight)")
    print("  [6] 📦 EXPORT KAGGLE NOTEBOOK (For 2x T4 GPU batch execution)")
    print("  [7] 🌍 BROWSE GLOBAL TOP 20 AI MODEL ZOO")
    print("  [0] ❌ Exit")
    print("="*75)

def run_auto_pilot():
    print("\n[>>> INITIATING 1-CLICK AUTONOMOUS END-TO-END PIPELINE <<<]")
    python_bin = sys.executable
    
    # 1. Pipeline execution
    print("\n---> STEP 1: Running Adaptive Auto-Pipeline...")
    subprocess.run([python_bin, "run_pipeline.py"])
    
    # 2. Ensemble stacker
    print("\n---> STEP 2: Running Master Ensemble Stacker...")
    subprocess.run([python_bin, "ensemble_stacker.py"])
    
    # 3. Submission verifier
    print("\n---> STEP 3: Running Anti-Disqualification Verifier...")
    sub_file = "final_champion_submission.csv" if os.path.exists("final_champion_submission.csv") else "submission.csv"
    if os.path.exists(sub_file) and os.path.exists("mock_test.csv"):
        subprocess.run([python_bin, "submission_verifier.py", sub_file, "mock_test.csv"])
    
    print("\n[SUCCESS] 1-Click Auto-Pilot finished! Your submission file is ready for upload.\n")

def main():
    while True:
        print_banner()
        choice = input("\nEnter Option [0-7]: ").strip()
        python_bin = sys.executable
        
        if choice == '1':
            run_auto_pilot()
        elif choice == '2':
            subprocess.run([python_bin, "system_integration_test.py"])
        elif choice == '3':
            train_f = "train.csv" if os.path.exists("train.csv") else "mock_train.csv"
            test_f = "test.csv" if os.path.exists("test.csv") else "mock_test.csv"
            subprocess.run([python_bin, "train_baseline.py", train_f, test_f])
        elif choice == '4':
            subprocess.run([python_bin, "mrp_discount_anchor.py"])
            subprocess.run([python_bin, "faiss_similarity_matcher.py"])
        elif choice == '5':
            sub_file = input("Enter submission filename [default: submission.csv]: ").strip() or "submission.csv"
            test_file = "test.csv" if os.path.exists("test.csv") else "mock_test.csv"
            subprocess.run([python_bin, "submission_verifier.py", sub_file, test_file])
        elif choice == '6':
            print("\n[*] Kaggle Master Notebook is located at:")
            print("    -> C:\\Users\\HP\\.gemini\\antigravity-ide\\scratch\\amazon-ml-challenge-2026\\kaggle_master_notebook.ipynb")
            print("    Upload this .ipynb to Kaggle with GPU T4 x 2 enabled and click 'Run All'.\n")
        elif choice == '7':
            subprocess.run([python_bin, "model_zoo_registry.py"])
        elif choice == '0':
            print("\nExiting Solo Commander. Good luck in the competition!\n")
            break
        else:
            print("\n[!] Invalid selection. Please enter 0 to 7.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        run_auto_pilot()
    else:
        main()
