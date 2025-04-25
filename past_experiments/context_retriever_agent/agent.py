"""Main Context Retriever Agent implementation."""

import os
import asyncio
import logging
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Set, Optional
import json
import time # Import time module

from agents.cot_agent.gemini import GeminiAsync
from agents.cot_agent.rearrange import reorder
from agents.context_retriever_agent.config import (
    DEFAULT_VIDEO_DIR, DEFAULT_TEMPERATURE, DEFAULT_WAIT_TIME, 
    DEFAULT_BATCH_SIZE, get_checkpoint_path
)
from agents.context_retriever_agent.context_builder import get_context_for_questions
from agents.context_retriever_agent.utils import (
    load_checkpoint, save_checkpoint, get_processed_qids, 
    load_error_qids, merge_predictions,
    merge_with_original_data
)

logger = logging.getLogger("context_retriever.agent")


async def process_videos_for_context_retrieval(
    ds: pd.DataFrame,
    checkpoint_path: str,
    video_dir: str = DEFAULT_VIDEO_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    filter_qids: Optional[Set[str]] = None,
    video_upload: bool = False,
    wait_time: int = DEFAULT_WAIT_TIME,
    concurrency_limit: int = 10
):
    """
    Process videos to extract context and validate questions.
    
    Args:
        ds: DataFrame containing video data
        checkpoint_path: Path to save progress
        video_dir: Directory containing local video files
        batch_size: Number of videos to process before saving
        temperature: Temperature for generation
        filter_qids: QIDs to process
        video_upload: Whether to upload local video files
        wait_time: Wait time between API calls
        concurrency_limit: Max concurrent API calls within context builder
    """
    predictions = load_checkpoint(checkpoint_path)
    processed_qids = get_processed_qids(predictions)
    
    # filter and groupby video
    df_filtered = ds.copy()
    if processed_qids:
        df_filtered = df_filtered[~df_filtered['qid'].isin(processed_qids)]
    if filter_qids:
        df_filtered = df_filtered[df_filtered['qid'].isin(filter_qids)]
    
    if df_filtered.empty:
        logger.info("No questions to process")
        return
    
    logger.info(f"Processing {len(df_filtered)} questions")
    
    # group questions by video URL
    video_to_samples = defaultdict(list)
    for _, row in df_filtered.iterrows():
        row_dict = row.to_dict()
        video_to_samples[row_dict['youtube_url']].append(row_dict)
    
    gemini = GeminiAsync()
    
    video_batches = list(video_to_samples.items())
    total_batches = (len(video_batches) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(video_batches)} videos in {total_batches} batches (batch size: {batch_size})")

    for i in range(0, len(video_batches), batch_size):
        batch_start_time = time.time() # Start timer for batch
        current_batch_num = (i // batch_size) + 1
        batch = dict(video_batches[i:i+batch_size])
        batch_video_urls = list(batch.keys()) # Get URLs for logging
        logger.info(f"Processing batch {current_batch_num}/{total_batches} (Videos: {batch_video_urls})")
        
        try:
            batch_predictions = await get_context_for_questions(
                video_questions=batch,
                temperature=temperature,
                wait_time=wait_time,
                video_dir=video_dir,
                use_local_files=video_upload,
                gemini=gemini,
                concurrency_limit=concurrency_limit
            )
            
            predictions.extend(batch_predictions)
            save_checkpoint(predictions, checkpoint_path)

        except Exception as e:
            # Log error for the specific batch
            logger.error(f"Error processing batch {current_batch_num} (Videos: {batch_video_urls}): {e}", exc_info=True)
            # Add error placeholders for all questions in the failed batch
            for video_url, samples in batch.items():
                for s in samples:
                     predictions.append({
                         "qid": s.get("qid", "UNKNOWN"),
                         "prediction": {
                              "context": "Error: Batch processing failed",
                              "is_contextual": False,
                              "explanation": f"Error during batch processing: {str(e)}",
                              "corrected_question": None
                         }
                     })
            # Save checkpoint even after batch error to record the errors
            logger.warning(f"Saving checkpoint after batch {current_batch_num} error to record failed QIDs.")
            save_checkpoint(predictions, checkpoint_path)
            # Optional: decide whether to continue to the next batch or raise/stop
            # continue # Continue to next batch (default behavior)

        finally:
             # Log time taken for the batch regardless of success or failure
             batch_end_time = time.time()
             batch_duration = batch_end_time - batch_start_time
             logger.info(f"Batch {current_batch_num}/{total_batches} (Videos: {batch_video_urls}) finished processing in {batch_duration:.2f} seconds.")


async def ContextRetrieverAgent(
    df: pd.DataFrame,
    checkpoint_path_initial: str,
    checkpoint_path_retry: str,
    final_output_suffix: str,
    number_of_iterations: int = 1,
    temperature: float = DEFAULT_TEMPERATURE,
    video_upload: bool = False,
    wait_time: int = DEFAULT_WAIT_TIME,
    concurrency_limit: int = 10
) -> str:
    """
    Main Context Retriever Agent function.
    Generates context predictions and merges them with original data,
    saving the final result to data/Step1_Context_<final_output_suffix>.json.

    Args:
        df: DataFrame containing video data
        checkpoint_path_initial: Initial checkpoint path suffix
        checkpoint_path_retry: Retry checkpoint path suffix
        final_output_suffix: Suffix for naming intermediate and final files
        number_of_iterations: Not used but kept for API compatibility
        temperature: Temperature for generation
        video_upload: Whether to upload local video files
        wait_time: Wait time between API calls
        concurrency_limit: Max concurrent API calls

    Returns:
        The absolute path to the final merged JSON file (e.g., data/Step1_Context_<final_output_suffix>.json)
    """
    total_start_time = time.time() # Start total timer
    logger.info("Starting Context Retriever Agent")

    # --- Path Setup ---
    initial_checkpoint_path = get_checkpoint_path(checkpoint_path_initial)
    retry_checkpoint_path = get_checkpoint_path(checkpoint_path_retry)
    # Path to the JSON holding predictions before merging with original data
    intermediate_json_path = get_checkpoint_path(final_output_suffix)
    # Define the final output path directly
    final_merged_json_path = os.path.join("data", f"Step1_Context_{final_output_suffix}.json")
    # Ensure the final output directory exists
    os.makedirs(os.path.dirname(final_merged_json_path), exist_ok=True)

    # --- Initial Processing --- 
    logger.info("Starting initial processing run")
    await process_videos_for_context_retrieval(
        ds=df,
        checkpoint_path=initial_checkpoint_path,
        video_dir=DEFAULT_VIDEO_DIR,
        batch_size=DEFAULT_BATCH_SIZE,
        temperature=temperature,
        video_upload=video_upload,
        wait_time=wait_time,
        concurrency_limit=concurrency_limit
    )

    # --- Error Handling and Retries --- 
    error_qids = load_error_qids(initial_checkpoint_path)
    if error_qids:
        logger.info(f"Retrying {len(error_qids)} questions with errors")
        await process_videos_for_context_retrieval(
            ds=df,
            checkpoint_path=retry_checkpoint_path,
            video_dir=DEFAULT_VIDEO_DIR,
            batch_size=DEFAULT_BATCH_SIZE,
            temperature=temperature,
            filter_qids=error_qids,
            video_upload=video_upload,
            wait_time=wait_time,
            concurrency_limit=concurrency_limit
        )

        logger.info("Merging initial and retry results into intermediate JSON")
        merge_predictions(
            original_path=initial_checkpoint_path,
            retry_path=retry_checkpoint_path,
            merged_output_path=intermediate_json_path # Save merged predictions here
        )
    else:
        logger.info("No errors to retry, copying initial checkpoint to intermediate path")
        import shutil
        # Added check to ensure source file exists before copying
        if os.path.exists(initial_checkpoint_path):
             shutil.copy2(initial_checkpoint_path, intermediate_json_path)
        else:
             logger.warning(f"Initial checkpoint {initial_checkpoint_path} not found. Cannot copy to intermediate path. Intermediate file may be missing or empty.")

    # --- Final Merging Step (JSON only) --- 
    logger.info("Merging intermediate predictions with original data")
    # The merge_with_original_data function now handles saving directly to final_merged_json_path
    final_json_output_path = merge_with_original_data(
        intermediate_json_path=intermediate_json_path,
        original_csv_path="data/data.csv",  # Original data path
        output_json_path=final_merged_json_path # Target final file path
    )

    # Removed CSV saving and reordering steps

    logger.info(f"Context Retriever Agent completed successfully.")
    
    # --- Total Time Logging --- 
    total_end_time = time.time()
    total_duration_seconds = total_end_time - total_start_time
    total_minutes = int(total_duration_seconds // 60)
    total_seconds = int(total_duration_seconds % 60)
    logger.info(f"Total execution time: {total_minutes} minutes, {total_seconds} seconds.")
    # --- End Total Time Logging ---
    
    logger.info(f"Final JSON output: {final_json_output_path}")

    return final_json_output_path # Return the path to the single final JSON file


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # EXAMPLE USAGE (Updated)
    print("Running standard ContextRetrieverAgent example...")
    df = pd.read_csv('data/data.csv')
    
    # Define a suffix for this run's files
    run_suffix = "Final"
    
    # Agent now only returns the final JSON path
    final_json_path = asyncio.run(ContextRetrieverAgent(
        df=df,
        checkpoint_path_initial=f"ContextRetriever_initial_{run_suffix}",
        checkpoint_path_retry=f"ContextRetriever_retry_{run_suffix}",
        final_output_suffix=f"ContextRetriever_{run_suffix}", # Suffix used for intermediate checkpoint
        temperature=DEFAULT_TEMPERATURE,
        video_upload=True, # Adjust as needed
        wait_time=DEFAULT_WAIT_TIME, # Keep for underlying client if needed
        concurrency_limit=10 # Set desired concurrency for the run
    ))
    
    # Print the path to the final JSON file
    print(f"\nGenerated final JSON output: {final_json_path}") 
    # Expected path: data/Step1_Context_ContextRetriever_Final.json

    # --- Log Final JSON Content (Sample) ---
    try:
        if final_json_path and os.path.exists(final_json_path):
            with open(final_json_path, 'r') as f:
                final_data = json.load(f)
            # Log a sample (e.g., first item) or summary to avoid flooding logs
            log_output = "Final JSON Content Sample (First Item):\n"
            if isinstance(final_data, list) and len(final_data) > 0:
                log_output += json.dumps(final_data[0], indent=2)
            elif isinstance(final_data, dict):
                 log_output += json.dumps(final_data, indent=2) # If it's not a list for some reason
            else:
                 log_output += str(final_data)[:500] # Log first 500 chars if not list/dict
            
            # Use the logger from the agent module
            logger.info(log_output)
        elif final_json_path:
             logger.warning(f"Final JSON file not found at: {final_json_path}. Cannot log content.")
        else:
             logger.warning("Final JSON path was not returned by agent. Cannot log content.")
            
    except Exception as e:
        logger.error(f"Error reading or logging final JSON content from {final_json_path}: {e}")
    # --- End Log Final JSON Content ---


# TO DO: Test using test_single_video.py to confirm the output is as expected

#     [
#   {
#     "qid": "0",
#     "video_id": "0008-0",
#     "question_type": "Primary Open-ended Question",
#     "capability": "Plot Attribute (Montage)",
#     "question": "What is the difference between the two scenes?",
#     "duration": "15.2",
#     "question_prompt": "Watch this video and answer the following question...",
#     "answer": "The first scene shows a person walking in a park, while the second scene shows a different person at a beach.",
#     "youtube_url": "https://www.youtube.com/watch?v=sj81PWrerDk",
#     "context": "This video shows two distinct scenes. In the first scene, a person is walking through a park with trees and a pathway. In the second scene, there is a different person standing on a beach with waves in the background.",
#     "is_contextual": true,
#     "explanation": "The question asks about the difference between scenes and requires visual context from the video.",
#     "corrected_question": "What is the difference between the park scene and the beach scene shown in this video?"
#   },
#   {
#     // next question entry
#   }
# ]