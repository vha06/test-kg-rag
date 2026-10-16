import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from storage.db import DatabaseManager
from evaluation.runner import BenchmarkRunner


def main():
    print("=== STARTING BENCHMARK CLI RUNNER (18 CONFIGURATIONS) ===")
    db = DatabaseManager()
    runner = BenchmarkRunner(db_manager=db)

    # Allow sample limit from CLI argument
    sample_limit = None
    if len(sys.argv) > 1:
        try:
            sample_limit = int(sys.argv[1])
            print(f"Running benchmark on sample of {sample_limit} queries...")
        except ValueError:
            pass

    results = runner.run_matrix(sample_limit=sample_limit)
    print("\n=== BENCHMARK SUMMARY ===")
    print(results["summary_df"].to_string())


if __name__ == "__main__":
    main()
