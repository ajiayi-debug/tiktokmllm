"""
Script to specifically retry QIDs marked as errors in a previous checkpoint run.

Reads an existing checkpoint, identifies errors, filters the original CSV data
for those errors, re-runs context retrieval only on the failed items, merges
the new results with the old, and generates a final formatted JSON output.
"""

import os
import json
import logging
import pandas as pd
import argparse
import asyncio
import time
from pathlib import Path

# Configure basic logging (consider making level configurable)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retry_failed_qids")

# --- Import necessary components from the agent --- 
# Note: This assumes the script is run from the project root where 'agents' is accessible
try:
    from agents.context_retriever_agent.utils import (
        load_checkpoint, 
        save_checkpoint, # Needed by process_videos...
        load_error_qids, 
        merge_predictions, 
        merge_with_original_data
    )
    from agents.context_retriever_agent.agent import process_videos_for_context_retrieval
    from agents.context_retriever_agent.config import (
        DEFAULT_VIDEO_DIR, DEFAULT_TEMPERATURE, DEFAULT_WAIT_TIME, 
        DEFAULT_BATCH_SIZE 
    )
except ImportError as e:
    logger.error(f"Failed to import agent components. Make sure you run this script from the project root directory. Error: {e}")
    exit(1)

async def retry_main(
    initial_checkpoint_path: str, 
    original_csv_path: str, 
    retry_results_path: str, 
    merged_checkpoint_path: str,
    final_output_path: str,
    video_dir: str,
    batch_size: int,
    temperature: float,
    video_upload: bool,
    wait_time: int,
    concurrency_limit: int
):
    """Core asynchronous logic for retrying failed QIDs."""
    total_start_time = time.time()
    logger.info(f"Starting retry process for checkpoint: {initial_checkpoint_path}")

    # 1. Identify Failed QIDs
    logger.info("Identifying failed QIDs from initial checkpoint...")
    error_qids = load_error_qids(initial_checkpoint_path)

    if not error_qids:
        logger.info("No QIDs marked with 'Error:' found in the initial checkpoint. Nothing to retry.")
        return
    
    logger.info(f"Found {len(error_qids)} QIDs to retry.")

    # 2. Load Original Data
    logger.info(f"Loading original data from {original_csv_path}...")
    if not os.path.exists(original_csv_path):
        logger.error(f"Original data CSV not found: {original_csv_path}")
        return
    try:
        original_df = pd.read_csv(original_csv_path)
        # Ensure qid column exists and is string
        if 'qid' not in original_df.columns:
             logger.error(f"'qid' column not found in {original_csv_path}")
             return
        original_df['qid'] = original_df['qid'].astype(str)

    except Exception as e:
        logger.error(f"Error reading original CSV {original_csv_path}: {e}", exc_info=True)
        return

    # 3. Filter Original Data for Errors
    logger.info("Filtering original data for failed QIDs...")
    retry_df = original_df[original_df['qid'].isin(error_qids)].copy()
    if retry_df.empty:
        logger.warning("No matching rows found in original CSV for the identified error QIDs. Cannot proceed.")
        return
    logger.info(f"Filtered DataFrame contains {len(retry_df)} rows for retry.")

    # 4. Run Context Retrieval on Filtered Data
    logger.info(f"Starting context retrieval run for {len(retry_df)} failed items...")
    # Ensure the temporary retry results file starts empty
    if os.path.exists(retry_results_path):
        logger.warning(f"Removing existing temporary retry file: {retry_results_path}")
        try:
            os.remove(retry_results_path)
        except OSError as e:
            logger.error(f"Failed to remove existing retry file {retry_results_path}: {e}")
            # Decide if this is fatal or can continue
            # return 
    
    # We pass None for filter_qids here because the DataFrame is *already* filtered
    await process_videos_for_context_retrieval(
        ds=retry_df, 
        checkpoint_path=retry_results_path, # Save results to the new temp retry file
        video_dir=video_dir,
        batch_size=batch_size,
        temperature=temperature,
        filter_qids=None, 
        video_upload=video_upload,
        wait_time=wait_time,
        concurrency_limit=concurrency_limit
    )
    logger.info("Finished context retrieval run for failed items.")

    # 5. Merge Original + Retry Results
    logger.info("Merging original checkpoint with new retry results...")
    if not os.path.exists(retry_results_path):
         logger.warning(f"Retry results file {retry_results_path} was not created. Merging cannot proceed.")
         # Fallback: maybe just run final merge on initial checkpoint?
         # For now, let's stop if retry didn't produce results
         logger.error("Stopping: No retry results generated.")
         return
         
    merge_predictions(
        original_path=initial_checkpoint_path, 
        retry_path=retry_results_path, 
        merged_output_path=merged_checkpoint_path
    )
    logger.info(f"Merged results saved to: {merged_checkpoint_path}")

    # 6. Generate Final Formatted Output
    logger.info("Generating final formatted JSON output...")
    final_output_actual_path = merge_with_original_data(
        intermediate_json_path=merged_checkpoint_path, 
        original_csv_path=original_csv_path, 
        output_json_path=final_output_path
    )
    logger.info(f"Final formatted output saved to: {final_output_actual_path}")

    # --- Total Time Logging --- 
    total_end_time = time.time()
    total_duration_seconds = total_end_time - total_start_time
    total_minutes = int(total_duration_seconds // 60)
    total_seconds = int(total_duration_seconds % 60)
    logger.info(f"Total retry script execution time: {total_minutes} minutes, {total_seconds} seconds.")
    # --- End Total Time Logging ---

# === Main Execution Block ===
if __name__ == "__main__":
    # Assume script is run from project root
    base_dir = Path().resolve() 
    default_initial_checkpoint = base_dir / "checkpoints" / "ContextRetriever_initial_Final.json"
    default_original_csv = base_dir / "data" / "data.csv"
    # Define temporary/output paths for this retry script
    default_retry_results = base_dir / "checkpoints" / "temp_retry_run_results.json"
    default_merged_checkpoint = base_dir / "checkpoints" / "merged_after_retry.json"
    default_final_output = base_dir / "data" / "Step1_Context_RETRY_FIXED.json"
    default_video_dir = base_dir / DEFAULT_VIDEO_DIR # Use config default

    parser = argparse.ArgumentParser(description="Retry failed QIDs from a ContextRetrieverAgent checkpoint run.")
    parser.add_argument(
        "--initial-checkpoint",
        default=str(default_initial_checkpoint),
        help=f"Path to the initial checkpoint file containing errors (default: {default_initial_checkpoint})",
        metavar="PATH"
    )
    parser.add_argument(
        "--original-csv",
        default=str(default_original_csv),
        help=f"Path to the original data CSV file (default: {default_original_csv})",
        metavar="PATH"
    )
    parser.add_argument(
        "--retry-output",
        default=str(default_retry_results),
        help=f"Path to save the temporary results of this retry run (default: {default_retry_results})",
        metavar="PATH"
    )
    parser.add_argument(
        "--merged-checkpoint",
        default=str(default_merged_checkpoint),
        help=f"Path to save the merged checkpoint after incorporating retry results (default: {default_merged_checkpoint})",
        metavar="PATH"
    )
    parser.add_argument(
        "--final-output",
        default=str(default_final_output),
        help=f"Path to save the final formatted JSON output after retries (default: {default_final_output})",
        metavar="PATH"
    )
    # Add processing arguments mirroring agent.py
    parser.add_argument("--video-dir", default=str(default_video_dir), help="Directory containing video files.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing videos.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for generation.")
    parser.add_argument("--video-upload", action='store_true', help="Set flag to upload local video files instead of using URLs.") # Default is False if not present
    parser.add_argument("--wait-time", type=int, default=DEFAULT_WAIT_TIME, help="Wait time parameter for API calls.")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum concurrent API calls.")


    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.retry_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.merged_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.final_output).parent.mkdir(parents=True, exist_ok=True)

    # Run the main async function
    asyncio.run(retry_main(
        initial_checkpoint_path=args.initial_checkpoint,
        original_csv_path=args.original_csv,
        retry_results_path=args.retry_output,
        merged_checkpoint_path=args.merged_checkpoint,
        final_output_path=args.final_output,
        video_dir=args.video_dir,
        batch_size=args.batch_size,
        temperature=args.temperature,
        video_upload=args.video_upload,
        wait_time=args.wait_time,
        concurrency_limit=args.concurrency
    )) 