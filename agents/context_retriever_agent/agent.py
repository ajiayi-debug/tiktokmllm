"""Main Context Retriever Agent implementation."""

import os
import asyncio
import logging
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Set, Optional

from agents.cot_agent.gemini import GeminiAsync
from agents.cot_agent.rearrange import reorder
from agents.context_retriever_agent.config import (
    DEFAULT_VIDEO_DIR, DEFAULT_TEMPERATURE, DEFAULT_WAIT_TIME, 
    DEFAULT_BATCH_SIZE, get_checkpoint_path, get_output_path
)
from agents.context_retriever_agent.context_builder import get_context_for_questions
from agents.context_retriever_agent.utils import (
    load_checkpoint, save_checkpoint, get_processed_qids, 
    load_error_qids, merge_predictions, save_context_to_csv,
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
    wait_time: int = DEFAULT_WAIT_TIME
):
    """
    Process videos to extract context and validate questions.
    
    Args:
        ds: DataFrame containing video data
        checkpoint_path: Path to save progress
        video_dir: Directory containing local video files
        batch_size: Number of videos to process before saving
        temperature: Temperature for generation
        filter_qids: Optional set of QIDs to process
        video_upload: Whether to upload local video files
        wait_time: Wait time between API calls
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
    for i in range(0, len(video_batches), batch_size):
        batch = dict(video_batches[i:i+batch_size])
        
        
        batch_predictions = await get_context_for_questions(
            video_questions=batch,
            temperature=temperature,
            wait_time=wait_time,
            video_dir=video_dir,
            use_local_files=video_upload,
            gemini=gemini
        )
        
        
        predictions.extend(batch_predictions)
        save_checkpoint(predictions, checkpoint_path)


async def ContextRetrieverAgent(
    df: pd.DataFrame,
    checkpoint_path_initial: str,
    checkpoint_path_retry: str,
    final_output: str,
    number_of_iterations: int = 1,  
    temperature: float = DEFAULT_TEMPERATURE,
    video_upload: bool = False,
    wait_time: int = DEFAULT_WAIT_TIME
):
    """
    Main Context Retriever Agent function.
    
    Args:
        df: DataFrame containing video data
        checkpoint_path_initial: Initial checkpoint path suffix
        checkpoint_path_retry: Retry checkpoint path suffix
        final_output: Final output filename suffix
        number_of_iterations: Not used but kept for API compatibility
        temperature: Temperature for generation
        video_upload: Whether to upload local video files
        wait_time: Wait time between API calls
        
    Returns:
        Tuple of (final_csv_path, final_json_path) for the output files
    """
    logger.info("Starting Context Retriever Agent")
    
    initial_path = get_checkpoint_path(checkpoint_path_initial)
    retry_path = get_checkpoint_path(checkpoint_path_retry)
    final_json_path = get_checkpoint_path(final_output)
    final_csv_path = get_output_path(final_output)
    final_reordered_csv_path = get_output_path(f"{final_output}_reordered")
    final_complete_csv_path = get_output_path(f"{final_output}_complete")
    
    logger.info("Starting initial processing run")
    await process_videos_for_context_retrieval(
        ds=df,
        checkpoint_path=initial_path,
        video_dir=DEFAULT_VIDEO_DIR,
        batch_size=DEFAULT_BATCH_SIZE,
        temperature=temperature,
        video_upload=video_upload,
        wait_time=wait_time
    )
    
    # load QIDs with errors
    error_qids = load_error_qids(initial_path)
    
    # then, retry only those with errors
    if error_qids:
        logger.info(f"Retrying {len(error_qids)} questions with errors")
        await process_videos_for_context_retrieval(
            ds=df,
            checkpoint_path=retry_path,
            video_dir=DEFAULT_VIDEO_DIR,
            batch_size=DEFAULT_BATCH_SIZE,
            temperature=temperature,
            filter_qids=error_qids,
            video_upload=video_upload,
            wait_time=wait_time
        )
    
        logger.info("Merging initial and retry results")
        merge_predictions(
            original_path=initial_path,
            retry_path=retry_path,
            merged_output_path=final_json_path
        )
    else:
        logger.info("No errors to retry")
        import shutil
        shutil.copy2(initial_path, final_json_path)
    
    # Save final results to CSV
    logger.info("Saving final results to CSV")
    predictions = load_checkpoint(final_json_path)
    save_context_to_csv(predictions, final_csv_path)
    
    # Reorder to match original data.csv order
    logger.info("Reordering results to match original data")
    reorder(final_csv_path, df, final_reordered_csv_path)
    
    # Merge with original data
    logger.info("Merging with original data")
    final_csv, final_json = merge_with_original_data(
        context_csv=final_reordered_csv_path,
        original_csv="data/data.csv",  # Original data path
        output_csv=final_complete_csv_path
    )
    
    logger.info(f"Context Retriever Agent completed successfully.")
    logger.info(f"Final CSV output: {final_csv}")
    logger.info(f"Final JSON output: {final_json}")
    
    return final_csv, final_json


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # EXAMPLE USAGE
    df = pd.read_csv('data/data.csv')
    csv_path, json_path = asyncio.run(ContextRetrieverAgent(
        df=df,
        checkpoint_path_initial="ContextRetriever_initial",
        checkpoint_path_retry="ContextRetriever_retry",
        final_output="ContextRetriever_Final",
        temperature=DEFAULT_TEMPERATURE,
        video_upload=True,
        wait_time=DEFAULT_WAIT_TIME
    ))
    
    print(f"Generated CSV output: {csv_path}")
    print(f"Generated JSON output: {json_path}") 