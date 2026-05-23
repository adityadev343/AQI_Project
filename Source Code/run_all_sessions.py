#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import ensure_dirs, DATASET_PATH

def main():
    ensure_dirs()
    if not DATASET_PATH.exists():
        print(f"ERROR: Foundational dataset not found at {DATASET_PATH}")
        print("Please run 'python run_pipeline.py' first to generate the dataset.")
        sys.exit(1)
    
    scripts = [
        ("analysis/eda.py", "Running EDA..."),
        ("analysis/official_vs_trb_breakpoints.py", "Running breakpoint analysis..."),
        ("analysis/s1_information_theoretic_audit.py", "Running Session 1: Information audit..."),
        ("analysis/s2_within_bucket_heterogeneity.py", "Running Session 2: Within-bucket heterogeneity...")
    ]
    
    for script, msg in scripts:
        print(msg)
        subprocess.run([sys.executable, script])
    
    print("\nAll analysis completed. Outputs are in 'results/' and 'figures/'.")

if __name__ == "__main__":
    main()