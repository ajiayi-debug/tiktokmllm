"""
Utility to generate a snapshot of the final formatted output based on existing checkpoints.
Reads initial (and optionally retry) checkpoints, merges them, reads the original CSV,
applies final formatting, and outputs the result for processed QIDs.
"""

import os
import json
import logging
import pandas as pd
import argparse
from collections import defaultdict

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_snapshot")

# --- Helper functions (copied/adapted from utils.py for standalone use) ---

def load_checkpoint(path: str) -> list[dict[str, any]]:
    """Load predictions from checkpoint file."""
    if not os.path.exists(path):
        logger.warning(f"Checkpoint file not found: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        logger.info(f"Loaded {len(predictions)} predictions from {path}")
        return predictions
    except Exception as e:
        logger.error(f"Error loading checkpoint {path}: {e}", exc_info=True)
        return []

def get_combined_question(row):
    """Combine question and prompt if prompt exists."""
    q_text = row.get("question", "")
    prompt_text = row.get("question_prompt", "")
    if prompt_text:
        combined = f"{str(q_text or '').strip()}\n{str(prompt_text or '').strip()}"
        return combined
    else:
        original_q_in_row = str(q_text or '').strip()
        return original_q_in_row

# --- Core Snapshot Logic ---

def generate_snapshot(initial_path: str, retry_path: str | None, csv_path: str) -> list[dict[str, any]]:
    """Generates the snapshot data."""

    # 1. Load and Merge Checkpoints
    logger.info("Loading checkpoints...")
    initial_preds = load_checkpoint(initial_path)
    retry_preds = load_checkpoint(retry_path) if retry_path else []

    # Create map of best predictions (prefer retry results over initial errors)
    merged_preds_map = {str(item["qid"]): item["prediction"] for item in initial_preds}
    updated_count = 0
    for item in retry_preds:
        qid_str = str(item["qid"])
        retry_pred = item["prediction"]
        # Overwrite if initial was error and retry is not, or if qid wasn't in initial
        if qid_str not in merged_preds_map or \
           (merged_preds_map.get(qid_str, {}).get("context", "").startswith("Error:") and \
            not retry_pred.get("context", "").startswith("Error:")):

            if merged_preds_map.get(qid_str, {}).get("context", "").startswith("Error:"):
                 logger.info(f"Overwriting error for QID {qid_str} with retry result.")
                 updated_count +=1
            merged_preds_map[qid_str] = retry_pred

    logger.info(f"Created merged prediction map with {len(merged_preds_map)} entries. Updated {updated_count} from retries.")
    if not merged_preds_map:
        logger.warning("No prediction data found in checkpoints.")
        # Decide whether to return empty list or proceed with only CSV data
        # For a snapshot, returning empty seems appropriate if no checkpoints have data
        return []


    # 2. Load and Format Original CSV Data
    logger.info(f"Loading original data from {csv_path}...")
    if not os.path.exists(csv_path):
        logger.error(f"Original data CSV not found: {csv_path}")
        return []
    try:
        original_df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}", exc_info=True)
        return []

    logger.info("Formatting original data...")
    # Drop 'Unnamed: 0' if exists
    if 'Unnamed: 0' in original_df.columns:
        logger.info("Dropping 'Unnamed: 0' column.")
        original_df = original_df.drop(columns=['Unnamed: 0'])

    # Ensure 'qid' is string
    if 'qid' in original_df.columns:
         original_df['qid'] = original_df['qid'].astype(str)
    else:
         logger.error("'qid' column not found in CSV.")
         return []

    # Apply question concatenation
    required_cols_for_concat = ['qid', 'question']
    if all(col in original_df.columns for col in required_cols_for_concat):
        if 'question_prompt' not in original_df.columns:
             logger.warning("'question_prompt' column not found. Concatenation will only use 'question' field.")
             original_df['question_prompt'] = "" # Add empty column if missing
        original_df['question'] = original_df.apply(get_combined_question, axis=1)
        logger.info("Question concatenation applied.")
    else:
         missing = [col for col in required_cols_for_concat if col not in original_df.columns]
         logger.warning(f"Skipping question concatenation due to missing essential columns: {missing}")


    # 3. Combine and Finalize
    logger.info("Combining formatted data with predictions...")
    snapshot_data_list = []
    processed_qids_in_snapshot = 0
    for _, row in original_df.iterrows():
        original_item = row.to_dict()
        qid_str = original_item['qid']

        # Only include items that have a corresponding prediction in the checkpoints
        if qid_str in merged_preds_map:
            prediction_data = merged_preds_map[qid_str]

            # Combine original data (already formatted) with prediction data
            merged_item = original_item.copy()
            merged_item.update(prediction_data)

            # --- Final Cleanup on Dictionary ---
            # Remove original question_prompt field
            merged_item.pop('question_prompt', None)
            # Explicitly remove 'Unnamed: 0' if somehow still present
            merged_item.pop('Unnamed: 0', None)
            # Replace NaN values with None for JSON compatibility
            for key, value in merged_item.items():
                if pd.isna(value):
                    merged_item[key] = None
            # --- End Cleanup ---

            snapshot_data_list.append(merged_item)
            processed_qids_in_snapshot += 1
        # else:
            # logger.debug(f"Skipping QID {qid_str} - not found in merged checkpoints.")


    logger.info(f"Generated snapshot data for {processed_qids_in_snapshot} processed QIDs.")
    return snapshot_data_list


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a snapshot JSON from agent checkpoints and original data.")
    parser.add_argument(
        "--initial",
        required=True,
        help="Path to the initial checkpoint file (e.g., checkpoints/initial_test_...).",
        metavar="PATH"
    )
    parser.add_argument(
        "--retry",
        required=False,
        default=None,
        help="Path to the retry checkpoint file (e.g., checkpoints/retry_test_...). Optional.",
        metavar="PATH"
    )
    parser.add_argument(
        "--csv",
        default="data/data.csv",
        help="Path to the original data CSV file.",
        metavar="PATH"
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Path to save the output snapshot JSON file. If omitted, prints to console.",
        metavar="PATH"
    )

    args = parser.parse_args()

    logger.info("Starting snapshot generation...")
    snapshot_data = generate_snapshot(
        initial_path=args.initial,
        retry_path=args.retry,
        csv_path=args.csv
    )

    if snapshot_data:
        if args.output:
            logger.info(f"Saving snapshot to {args.output}...")
            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                with open(args.output, "w", encoding="utf-8") as f_out:
                    json.dump(snapshot_data, f_out, indent=2, ensure_ascii=False)
                logger.info("Snapshot saved successfully.")
            except Exception as e:
                logger.error(f"Error saving snapshot to {args.output}: {e}", exc_info=True)
        else:
            logger.info("Printing snapshot to console:")
            print(json.dumps(snapshot_data, indent=2, ensure_ascii=False))
    else:
        logger.warning("No snapshot data generated (check logs for errors or empty checkpoints/CSV).")

    logger.info("Snapshot generation finished.") 