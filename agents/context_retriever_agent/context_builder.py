"""Context builder for the Context Retriever Agent."""

import logging
import asyncio
import random # For jitter in backoff
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json

from agents.cot_agent.gemini import GeminiAsync
from agents.context_retriever_agent.prompts import combined_context_prompt
from agents.context_retriever_agent.utils import parse_context_response

logger = logging.getLogger("context_retriever.builder")

# --- Configuration for Retry/Backoff ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 5 # Start backoff at 5 seconds
# --- End Configuration ---

class ContextBuilder:
    """Class to build context for videos and questions."""
    
    def __init__(
        self,
        gemini: Optional[GeminiAsync] = None,
        temperature: float = 0.0,
        wait_time: int = 30, # Keep wait_time param for compatibility
        video_dir: str = "Benchmark-AllVideos-HQ-Encoded-challenge",
        concurrency_limit: int = 10 # Added concurrency limit
    ):
        """
        Initialize the context builder.
        
        Args:
            gemini: Optional GeminiAsync instance (will create one if not provided)
            temperature: Temperature for generation
            wait_time: Wait time parameter (primarily for underlying client, not for sleeps here)
            video_dir: Directory containing local video files
            concurrency_limit: Max number of concurrent API calls
        """
        self.gemini = gemini or GeminiAsync()
        self.temperature = temperature
        self.wait_time = wait_time # Store it, pass it down
        self.video_dir = video_dir
        self.semaphore = asyncio.Semaphore(concurrency_limit) # Initialize semaphore
        logger.info(f"ContextBuilder initialized with concurrency limit: {concurrency_limit}")
    
    async def _process_single_with_retry(self, video_url: str, question: str, video_id: str, use_local_file: bool) -> Dict[str, Any]:
        """Internal helper to process a single question with retries and backoff."""
        retries = 0
        backoff_time = INITIAL_BACKOFF_SECONDS
        last_exception = None

        while retries <= MAX_RETRIES:
            try:
                prompt = combined_context_prompt(question)
                local_video_path = None
                if use_local_file and video_id:
                    import os
                    local_video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                    if not os.path.exists(local_video_path):
                        logger.warning(f"Local video file not found for QID {video_id}: {local_video_path}")
                        local_video_path = None

                response = ""
                if use_local_file and local_video_path:
                    logger.debug(f"QID {video_id}: Using local video file: {local_video_path}")
                    response = await self.gemini.generate_from_uploaded_video_file(
                        local_video_path, prompt,
                        temperature=self.temperature,
                        wait_time=self.wait_time # Pass wait_time down
                    )
                else:
                    logger.debug(f"QID {video_id}: Using video URL: {video_url}")
                    response_list = await self.gemini.generate_from_video(
                        video_url, [prompt],
                        temperature=self.temperature,
                        wait_time=self.wait_time # Pass wait_time down
                    )
                    if response_list: # Ensure list is not empty
                         response = response_list[0]
                    else:
                         raise ValueError(f"API returned empty list for QID {video_id} URL {video_url}")
                
                context_data = parse_context_response(response)
                logger.debug(f"QID {video_id}: Context successfully extracted.")
                return context_data # Success!

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                # Check for rate limit or other potentially transient errors
                # Add more specific error checks if needed (e.g., specific exception types)
                if "429" in error_str or "rate limit" in error_str or \
                   "server error" in error_str or "connection error" in error_str or \
                   "unavailable" in error_str:
                    
                    retries += 1
                    if retries > MAX_RETRIES:
                        logger.error(f"QID {video_id}: Max retries ({MAX_RETRIES}) exceeded. Last error: {e}")
                        break # Exit loop after max retries
                    else:
                        # Exponential backoff with jitter
                        wait = backoff_time + random.uniform(0, 1)
                        logger.warning(f"QID {video_id}: Encountered error: {e}. Retrying in {wait:.2f} seconds (Attempt {retries}/{MAX_RETRIES})...")
                        await asyncio.sleep(wait)
                        backoff_time *= 2 # Increase backoff time for next potential retry
                else:
                    # For non-retryable errors, log and break immediately
                    logger.error(f"QID {video_id}: Encountered non-retryable error: {e}", exc_info=True)
                    break # Exit loop
        
        # If loop finished due to errors
        logger.error(f"QID {video_id}: Failed to process after {retries} retries. Last exception: {last_exception}")
        return {
            "context": "Error: Processing failed after retries",
            "is_contextual": False,
            "explanation": f"Failed after {retries} retries. Last error: {str(last_exception)}",
            "corrected_question": None
        }

    async def _process_single_with_semaphore(self, video_url: str, question_data: Dict[str, Any], use_local_file: bool) -> tuple[str, Dict[str, Any]]:
        """Helper to acquire semaphore and call the processing logic."""
        qid = str(question_data.get("qid", "UNKNOWN"))
        question = question_data.get("question", "")
        video_id = str(question_data.get("video_id", ""))
        
        async with self.semaphore: # Acquire semaphore
            logger.debug(f"QID {qid}: Acquired semaphore, starting processing.")
            context_data = await self._process_single_with_retry(
                video_url=video_url,
                question=question,
                video_id=video_id,
                use_local_file=use_local_file
            )
            logger.debug(f"QID {qid}: Released semaphore after processing.")
            return qid, context_data
        # Semaphore released automatically by 'async with'

    async def batch_process_video_questions(
        self,
        video_url: str,
        questions: List[Dict[str, Any]],
        use_local_file: bool = False
    ) -> List[tuple[str, Dict[str, Any]]]:
        """
        Process multiple questions for the same video concurrently using a semaphore.
        
        Args:
            video_url: URL of the video
            questions: List of question dictionaries with 'qid', 'question', and 'video_id'
            use_local_file: Whether to try using a local video file
            
        Returns:
            List of tuples with (qid, context_data)
        """
        if not questions:
            return []

        tasks = []
        for q_data in questions:
            # Create a task for each question using the semaphore helper
            task = asyncio.create_task(self._process_single_with_semaphore(
                video_url=video_url,
                question_data=q_data,
                use_local_file=use_local_file
            ))
            tasks.append(task)
        
        logger.info(f"Video {video_url}: Gathering {len(tasks)} concurrent tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True) # Gather results, capture exceptions
        logger.info(f"Video {video_url}: Finished gathering {len(tasks)} tasks.")

        # Process results, handling potential exceptions returned by gather
        final_results = []
        original_qids = [str(q.get("qid", "UNKNOWN")) for q in questions] # Ensure we have results for all input qids
        results_map = {}

        for result in results:
            if isinstance(result, Exception):
                # If gather returned an exception for a task
                logger.error(f"Video {video_url}: Task failed with unhandled exception: {result}", exc_info=result)
                # We don't know which QID failed here, so we'll fill missing ones later
                continue 
            elif isinstance(result, tuple) and len(result) == 2:
                 qid, context_data = result
                 results_map[qid] = context_data
            else:
                 logger.error(f"Video {video_url}: Unexpected result type from gather: {type(result)} - {result}")
        
        # Ensure all original QIDs have a result, adding error placeholders if missing
        for qid in original_qids:
            if qid in results_map:
                 final_results.append((qid, results_map[qid]))
            else:
                 logger.error(f"Video {video_url}: Result missing for QID {qid}. Assuming task failed.")
                 final_results.append((qid, {
                     "context": "Error: Task failed during concurrent execution",
                     "is_contextual": False,
                     "explanation": "Result was missing after asyncio.gather, likely due to an unhandled exception.",
                     "corrected_question": None
                 }))

        return final_results

async def get_context_for_questions(
    video_questions: Dict[str, List[Dict[str, Any]]],
    temperature: float = 0.0,
    wait_time: int = 30,
    video_dir: str = "Benchmark-AllVideos-HQ-Encoded-challenge",
    use_local_files: bool = False,
    gemini: Optional[GeminiAsync] = None,
    concurrency_limit: int = 10 # Pass concurrency limit down
) -> List[Dict[str, Any]]:
    """
    Process all video questions to get context.
    
    Args:
        video_questions: Dictionary mapping video URLs to lists of question dicts
        temperature: Temperature for generation
        wait_time: Wait time parameter (passed to underlying client)
        video_dir: Directory containing local video files
        use_local_files: Whether to try using local video files
        gemini: Optional GeminiAsync instance
        concurrency_limit: Max number of concurrent API calls
        
    Returns:
        List of prediction dictionaries in the format {'qid': ..., 'prediction': ...}
    """
    builder = ContextBuilder(
        gemini=gemini,
        temperature=temperature,
        wait_time=wait_time, # Pass wait_time along
        video_dir=video_dir,
        concurrency_limit=concurrency_limit # Use the limit
    )
    
    all_predictions = [] 
    
    for video_url, questions in video_questions.items():
        if not questions: # Skip if a video has no questions associated
             logger.warning(f"Skipping video {video_url} as it has no questions listed.")
             continue
             
        num_q = len(questions)
        logger.info(f"Processing video: {video_url} ({num_q} questions) with concurrency...")
        video_start_time = asyncio.get_event_loop().time()
        
        # Get the results for this video concurrently
        results = await builder.batch_process_video_questions(
            video_url=video_url,
            questions=questions,
            use_local_file=use_local_files
        )
        
        video_end_time = asyncio.get_event_loop().time()
        video_duration = video_end_time - video_start_time
        logger.info(f"Finished processing video {video_url} ({num_q} questions) in {video_duration:.2f} seconds.")
        
        # --- Logging Intermediate Results Per Video --- 
        try:
            # Create a map for easy lookup
            results_map = {str(qid): data for qid, data in results}
            
            log_data_for_video = []
            for original_question_data in questions:
                qid_str = str(original_question_data.get("qid", "UNKNOWN"))
                # Get the prediction, default to an error dict if somehow missing
                prediction_data = results_map.get(qid_str, {
                    "context": "Error: Prediction missing after batch processing",
                    "is_contextual": False,
                    "explanation": "Prediction data not found in results map.",
                    "corrected_question": None
                })
                
                # Combine original data with the prediction for logging
                log_item = original_question_data.copy()
                log_item.update(prediction_data) # Add context fields
                log_data_for_video.append(log_item)
            
            # Log the combined data for this video as formatted JSON
            logger.info(f"Intermediate results for video {video_url}:\n{json.dumps(log_data_for_video, indent=2)}")
            
        except Exception as log_e:
            logger.error(f"Error generating intermediate log for video {video_url}: {log_e}")
        # --- End Logging --- 
        
        # Add results to the final list in the expected format
        for qid, context_data in results:
            all_predictions.append({
                "qid": qid,
                "prediction": context_data
            })
            
    return all_predictions 