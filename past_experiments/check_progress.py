import json
import argparse
from pathlib import Path
import sys

def check_cot_progress(run_suffix):
    """
    Checks the CoT agent's initial checkpoint file to count completed items.
    """
    checkpoint_filename = f"step3_cot_{run_suffix}_initial.json"
    checkpoint_path = Path("data") / checkpoint_filename

    total_questions = 25 # Based on data_first_4_videos.csv

    print(f"Checking checkpoint file: {checkpoint_path}")

    if not checkpoint_path.exists():
        print("Checkpoint file not found yet. Step 3 (CoT Agent) might not have started saving progress.")
        return 0

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        if isinstance(predictions, list):
            completed_count = len(predictions)
            print(f"\nFound {completed_count} entries in the checkpoint file.")
            print(f"Progress: {completed_count} / {total_questions} questions processed by CoT agent.")
            return completed_count
        else:
            print("Checkpoint file does not contain a list as expected.")
            return 0

    except json.JSONDecodeError:
        print("Error: Could not decode JSON from the checkpoint file. It might be corrupted or currently being written.")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check CoT Agent progress by reading its checkpoint file.")
    parser.add_argument(
        "--run_suffix",
        type=str,
        required=True,
        help="The run_suffix used when starting the main pipeline script (e.g., first_4_test)."
    )

    args = parser.parse_args()

    check_cot_progress(args.run_suffix)