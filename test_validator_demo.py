"""
Test runner to demonstrate submission validator on valid vs corrupted submissions.
"""
import pandas as pd
import subprocess
import sys

def test_validator():
    # 1. Create a dummy test.csv
    test_df = pd.DataFrame({
        "index": [101, 102, 103, 104, 105],
        "image_link": ["url1", "url2", "url3", "url4", "url5"]
    })
    test_df.to_csv("dummy_test.csv", index=False)

    # 2. Create a VALID submission.csv
    valid_sub = pd.DataFrame({
        "index": [101, 102, 103, 104, 105],
        "prediction": ["500 gram", "250 millilitre", "1.5 litre", "12 volt", "10 centimetre"]
    })
    valid_sub.to_csv("valid_submission.csv", index=False)

    # 3. Create a BROKEN submission (missing row + NaN)
    broken_sub = pd.DataFrame({
        "index": [101, 102, 103, 104], # missing 105!
        "prediction": ["500 gram", None, "invalid_unit_xyz", "12 volt"]
    })
    broken_sub.to_csv("broken_submission.csv", index=False)

    print("\n--- TEST 1: Scanning Broken Submission ---")
    subprocess.run([
        sys.executable, "submission_validator.py",
        "--submission", "broken_submission.csv",
        "--test_csv", "dummy_test.csv"
    ])

    print("\n--- TEST 2: Scanning Valid Submission ---")
    subprocess.run([
        sys.executable, "submission_validator.py",
        "--submission", "valid_submission.csv",
        "--test_csv", "dummy_test.csv"
    ])

if __name__ == "__main__":
    test_validator()
