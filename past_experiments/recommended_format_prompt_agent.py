"""
Recommended version of the format prompt agent using asyncio for concurrency.
This is an improved version of format_prompt_agent.py with the following key enhancements:

1. Concurrent Processing: Uses asyncio to process multiple questions simultaneously
2. Rate Limit Handling: Implements exponential backoff and retry logic
3. Error Recovery: Falls back to original questions when rewrites fail
4. Progress Tracking: Uses tqdm for visual progress and detailed logging
5. Configurable: Command-line arguments for tuning performance

Key differences from original format_prompt_agent.py:
- Processes questions concurrently instead of sequentially
- Handles API rate limits and errors gracefully
- Provides better feedback and progress tracking
- More robust error handling and recovery
- Configurable via command line arguments
"""

import openai
import os
import json
import asyncio
import logging
import random
from dotenv import load_dotenv
from pathlib import Path
import argparse
import re
from tqdm.asyncio import tqdm 


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
# Initialize async OpenAI client (different from original which used sync client)
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default settings that can be overridden via command line
CONCURRENCY_LIMIT = 20  # How many questions to process simultaneously
MAX_RETRIES = 3        # How many times to retry failed API calls
INITIAL_BACKOFF_SECONDS = 2  # Starting delay for exponential backoff
# --- End Configuration ---

# Set up logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recommended_format_prompt_agent")

async def rewrite_question_with_context_async(
    question: str,
    context_chunks: list[str],
    template: str,
    qid: str 
) -> tuple[str | None, str]:
    """
    Asynchronous function to rewrite a question using context.
    This is the core function that interacts with the OpenAI API.
    
    Key improvements from original:
    - Async API calls (non-blocking)
    - Specific error handling for different API errors
    - Exponential backoff for rate limits
    - Validation of model output
    
    Args:
        question: The original question to rewrite
        context_chunks: List of context pieces from the video
        template: The prompt template to use
        qid: Question ID for logging
    
    Returns:
        Tuple of (rewritten question or None if failed, qid)
    """
    # Format the context and create the prompt
    context = '\n'.join(f"- {chunk}" for chunk in context_chunks)
    prompt = template.format(context=context, question=question)

    retries = 0
    # Use global constants that might be updated by CLI args
    global INITIAL_BACKOFF_SECONDS, MAX_RETRIES 
    backoff_time = INITIAL_BACKOFF_SECONDS 
    last_exception = None

    while retries <= MAX_RETRIES:
        try:
            # Make the API call asynchronously
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            rewritten = response.choices[0].message.content.strip()
            
            # Clean up the response if it starts with "Rewritten Question:"
            if rewritten.lower().startswith("rewritten question:"):
                rewritten = rewritten[len("rewritten question:"):].strip(' "\n')
            
            # Validate that the model actually rewrote the question
            if rewritten and rewritten.lower() != question.lower():
                return rewritten, qid # Success
            else:
                logger.warning(f"QID {qid}: Model did not provide a distinct rewritten question. Original: '{question[:50]}...' Rewritten: '{rewritten[:50]}...'")
                last_exception = ValueError("Model did not provide a distinct rewrite.")
                break # Fail immediately if rewrite is same as original

        except openai.RateLimitError as e:
            # Handle rate limits with exponential backoff
            last_exception = e
            retries += 1
            if retries > MAX_RETRIES:
                logger.error(f"QID {qid}: Max retries ({MAX_RETRIES}) exceeded due to RateLimitError. Last error: {e}")
                break
            else:
                wait = backoff_time + random.uniform(0, 0.5) # Add jitter to prevent thundering herd
                logger.warning(f"QID {qid}: Rate limit exceeded. Retrying in {wait:.2f} seconds (Attempt {retries}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                backoff_time *= 2 # Exponential backoff

        # Handle specific potentially retryable errors
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            last_exception = e
            retries += 1
            if retries > MAX_RETRIES:
                logger.error(f"QID {qid}: Max retries ({MAX_RETRIES}) exceeded due to API error. Last error: {e}")
                break
            else:
                wait = backoff_time + random.uniform(0, 0.5)
                logger.warning(f"QID {qid}: Encountered API error: {e}. Retrying in {wait:.2f} seconds (Attempt {retries}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                backoff_time *= 2
        
        # Handle specific non-retryable errors
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError, openai.UnprocessableEntityError, openai.BadRequestError) as e:
             last_exception = e
             logger.error(f"QID {qid}: Encountered non-retryable API error: {e}. Failing immediately.", exc_info=True)
             break # Do not retry these errors

        # Catch any other unexpected errors
        except Exception as e:
            last_exception = e
            logger.error(f"QID {qid}: Encountered unexpected error during API call: {e}. Failing immediately.", exc_info=True)
            break # Do not retry unknown errors

    logger.error(f"QID {qid}: Failed to get rewritten question after {retries} retries or due to non-retryable error. Last exception: {last_exception}")
    return None, qid # Indicate failure

async def process_item_async(
    item: dict,
    template: str,
    semaphore: asyncio.Semaphore
) -> dict | None:
    """
    Process a single item from the input JSON concurrently.
    This function handles the preparation of data and manages the semaphore for API calls.
    
    Key improvements from original:
    - Handles both list and string context formats
    - Better input validation
    - Uses semaphore for concurrency control
    - Falls back to original question on failure
    
    Args:
        item: Dictionary containing the question data
        template: The prompt template
        semaphore: Semaphore for controlling concurrent API calls
    
    Returns:
        Dictionary with processed data or None if processing failed
    """
    qid = item.get("qid", "UNKNOWN_QID")
    # Handle both possible question field names
    original_question = item.get("question_with_prompt", item.get("question")) 
    context_text = item.get("context")
    video_id = item.get("video_id", "")
    youtube_url = item.get("youtube_url", "")

    # Validate required fields
    if not original_question:
        logger.warning(f"QID {qid}: Skipping item due to missing 'question' or 'question_with_prompt' field.")
        return None
    if context_text is None: # Allow empty string but not None
         logger.warning(f"QID {qid}: Skipping item due to missing 'context' field.")
         return None

    # Handle different context formats
    context_chunks = []
    if isinstance(context_text, list):
         # If context is already a list of strings
         context_chunks = [str(chunk).strip() for chunk in context_text if str(chunk).strip()]
         logger.debug(f"QID {qid}: Using pre-chunked context list ({len(context_chunks)} chunks).")
    elif isinstance(context_text, str):
         # If context is a string, split by periods
         context_chunks = [chunk.strip() for chunk in context_text.split('.') if chunk.strip()]
         logger.debug(f"QID {qid}: Splitting context string into {len(context_chunks)} chunks by period.")
    else:
         logger.error(f"QID {qid}: Skipping item due to unexpected context format: {type(context_text)}")
         return None # Skip items with unexpected context type

    # Skip if no valid chunks found
    if not context_chunks:
        logger.warning(f"QID {qid}: Skipping item because context text resulted in empty chunks.")
        return None

    # Use semaphore to control concurrent API calls
    global CONCURRENCY_LIMIT 
    async with semaphore: 
        logger.debug(f"QID {qid}: Acquired semaphore, calling rewrite function.")
        revised_question_result, _ = await rewrite_question_with_context_async(original_question, context_chunks, template, qid)
        logger.debug(f"QID {qid}: Released semaphore.")

    # If rewrite failed, use the original question
    final_corrected_question = revised_question_result if revised_question_result is not None else original_question

    # Return the processed data
    return {
        "qid": qid,
        "question": original_question, # Always include original for reference
        "youtube_url": youtube_url,
        "context": context_chunks, # Keep the chunked context
        "video_id": video_id,
        "corrected_question": final_corrected_question # Rewritten or original if failed
    }

"""
Recommended version of the format prompt agent using asyncio for concurrency.
This is an improved version of format_prompt_agent.py with the following key enhancements:

1. Concurrent Processing: Uses asyncio to process multiple questions simultaneously
2. Rate Limit Handling: Implements exponential backoff and retry logic
3. Error Recovery: Falls back to original questions when rewrites fail
4. Progress Tracking: Uses tqdm for visual progress and detailed logging
5. Configurable: Command-line arguments for tuning performance

Key differences from original format_prompt_agent.py:
- Processes questions concurrently instead of sequentially
- Handles API rate limits and errors gracefully
- Provides better feedback and progress tracking
- More robust error handling and recovery
- Configurable via command line arguments
"""

import openai
import os
import json
import asyncio
import logging
import random
from dotenv import load_dotenv
from pathlib import Path
import argparse
from tqdm.asyncio import tqdm 
# [unchanged imports up top...]
import aiofiles  # NEW: for async checkpoint writing

CHECKPOINT_EVERY = 50  # NEW: how often to checkpoint

async def save_checkpoint(results: list[dict], path: Path):
    checkpoint_path = path.with_suffix('.checkpoint.json')
    try:
        async with aiofiles.open(checkpoint_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info(f"Checkpoint saved to {checkpoint_path} with {len(results)} items.")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

async def main(input_path_str: str, output_path_str: str, template_path_str: str):
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    template_path = Path(template_path_str)

    if not input_path.exists():
        logger.error(f"Input file not found at: {input_path}")
        return
    if not template_path.exists():
        logger.error(f"Template file not found at: {template_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading template from {template_path}")
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except Exception as e:
        logger.error(f"Failed to read template file: {e}", exc_info=True)
        return

    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    processed_qids = set()
    results = []

    if checkpoint_path.exists():
        logger.info(f"Checkpoint file found at {checkpoint_path}, loading progress...")
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_qids = {item["qid"] for item in results if "qid" in item}
            logger.info(f"Loaded {len(results)} items from checkpoint.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            results = []
            processed_qids = set()

    logger.info(f"Reading input data from {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read input JSON: {e}", exc_info=True)
        return

    if not isinstance(all_data, list):
        logger.error(f"Input JSON is not a list. Found type: {type(all_data)}")
        return

    data = [item for item in all_data if item.get("qid") not in processed_qids]
    if not data:
        logger.info("No remaining items to process. Checkpoint already complete.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    for item in data:
        if isinstance(item, dict):
            tasks.append(asyncio.create_task(process_item_async(item, template, semaphore)))
        else:
            logger.warning(f"Skipping invalid item: {item}")

    logger.info(f"Running {len(tasks)} tasks...")
    failed_qids = []

    for idx, task in enumerate(tqdm.as_completed(tasks, desc="Formatting Prompts", unit="item")):
        result = await task
        if result:
            results.append(result)
            if result["corrected_question"] == result["question"]:
                failed_qids.append(result.get("qid", "UNKNOWN_FAILED_QID"))
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            await save_checkpoint(results, output_path)

    logger.info(f"Processing complete. Success: {len(results) - len(failed_qids)}, Failures: {len(failed_qids)}")
    if failed_qids:
        logger.warning(f"Failed QIDs: {failed_qids}")

    logger.info(f"Saving output to {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Final output saved.")
    except Exception as e:
        logger.error(f"Failed to save output: {e}", exc_info=True)

    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info("Checkpoint file deleted after success.")
        except Exception as e:
            logger.warning(f"Could not delete checkpoint: {e}")


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
# Initialize async OpenAI client (different from original which used sync client)
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default settings that can be overridden via command line
CONCURRENCY_LIMIT = 20  # How many questions to process simultaneously
MAX_RETRIES = 3        # How many times to retry failed API calls
INITIAL_BACKOFF_SECONDS = 2  # Starting delay for exponential backoff
# --- End Configuration ---

# Set up logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recommended_format_prompt_agent")

async def rewrite_question_with_context_async(
    question: str,
    context_chunks: list[str],
    template: str,
    qid: str 
) -> tuple[str | None, str]:
    """
    Asynchronous function to rewrite a question using context.
    This is the core function that interacts with the OpenAI API.
    
    Key improvements from original:
    - Async API calls (non-blocking)
    - Specific error handling for different API errors
    - Exponential backoff for rate limits
    - Validation of model output
    
    Args:
        question: The original question to rewrite
        context_chunks: List of context pieces from the video
        template: The prompt template to use
        qid: Question ID for logging
    
    Returns:
        Tuple of (rewritten question or None if failed, qid)
    """
    # Format the context and create the prompt
    context = '\n'.join(f"- {chunk}" for chunk in context_chunks)
    prompt = template.format(context=context, question=question)

    retries = 0
    # Use global constants that might be updated by CLI args
    global INITIAL_BACKOFF_SECONDS, MAX_RETRIES 
    backoff_time = INITIAL_BACKOFF_SECONDS 
    last_exception = None

    while retries <= MAX_RETRIES:
        try:
            # Make the API call asynchronously
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            rewritten = response.choices[0].message.content.strip()
            
            # Clean up the response if it starts with "Rewritten Question:"
            if rewritten.lower().startswith("rewritten question:"):
                rewritten = rewritten[len("rewritten question:"):].strip(' "\n')
            
            # Validate that the model actually rewrote the question
            if rewritten and rewritten.lower() != question.lower():
                return rewritten, qid # Success
            else:
                logger.warning(f"QID {qid}: Model did not provide a distinct rewritten question. Original: '{question[:50]}...' Rewritten: '{rewritten[:50]}...'")
                last_exception = ValueError("Model did not provide a distinct rewrite.")
                break # Fail immediately if rewrite is same as original

        except openai.RateLimitError as e:
            # Handle rate limits with exponential backoff
            last_exception = e
            retries += 1
            if retries > MAX_RETRIES:
                logger.error(f"QID {qid}: Max retries ({MAX_RETRIES}) exceeded due to RateLimitError. Last error: {e}")
                break
            else:
                wait = backoff_time + random.uniform(0, 0.5) # Add jitter to prevent thundering herd
                logger.warning(f"QID {qid}: Rate limit exceeded. Retrying in {wait:.2f} seconds (Attempt {retries}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                backoff_time *= 2 # Exponential backoff

        # Handle specific potentially retryable errors
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            last_exception = e
            retries += 1
            if retries > MAX_RETRIES:
                logger.error(f"QID {qid}: Max retries ({MAX_RETRIES}) exceeded due to API error. Last error: {e}")
                break
            else:
                wait = backoff_time + random.uniform(0, 0.5)
                logger.warning(f"QID {qid}: Encountered API error: {e}. Retrying in {wait:.2f} seconds (Attempt {retries}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                backoff_time *= 2
        
        # Handle specific non-retryable errors
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError, openai.UnprocessableEntityError, openai.BadRequestError) as e:
             last_exception = e
             logger.error(f"QID {qid}: Encountered non-retryable API error: {e}. Failing immediately.", exc_info=True)
             break # Do not retry these errors

        # Catch any other unexpected errors
        except Exception as e:
            last_exception = e
            logger.error(f"QID {qid}: Encountered unexpected error during API call: {e}. Failing immediately.", exc_info=True)
            break # Do not retry unknown errors

    logger.error(f"QID {qid}: Failed to get rewritten question after {retries} retries or due to non-retryable error. Last exception: {last_exception}")
    return None, qid # Indicate failure

async def process_item_async(
    item: dict,
    template: str,
    semaphore: asyncio.Semaphore
) -> dict | None:
    """
    Process a single item from the input JSON concurrently.
    This function handles the preparation of data and manages the semaphore for API calls.
    
    Key improvements from original:
    - Handles both list and string context formats
    - Better input validation
    - Uses semaphore for concurrency control
    - Falls back to original question on failure
    
    Args:
        item: Dictionary containing the question data
        template: The prompt template
        semaphore: Semaphore for controlling concurrent API calls
    
    Returns:
        Dictionary with processed data or None if processing failed
    """
    qid = item.get("qid", "UNKNOWN_QID")
    # Handle both possible question field names
    original_question = item.get("question_with_prompt", item.get("question")) 
    context_text = item.get("context")
    video_id = item.get("video_id", "")
    youtube_url = item.get("youtube_url", "")

    # Validate required fields
    if not original_question:
        logger.warning(f"QID {qid}: Skipping item due to missing 'question' or 'question_with_prompt' field.")
        return None
    if context_text is None: # Allow empty string but not None
         logger.warning(f"QID {qid}: Skipping item due to missing 'context' field.")
         return None

    # Handle different context formats
    context_chunks = []
    if isinstance(context_text, list):
         # If context is already a list of strings
         context_chunks = [str(chunk).strip() for chunk in context_text if str(chunk).strip()]
         logger.debug(f"QID {qid}: Using pre-chunked context list ({len(context_chunks)} chunks).")
    elif isinstance(context_text, str):
         # If context is a string, split by periods
         context_chunks = [chunk.strip() for chunk in context_text.split('.') if chunk.strip()]
         logger.debug(f"QID {qid}: Splitting context string into {len(context_chunks)} chunks by period.")
    else:
         logger.error(f"QID {qid}: Skipping item due to unexpected context format: {type(context_text)}")
         return None # Skip items with unexpected context type

    # Skip if no valid chunks found
    if not context_chunks:
        logger.warning(f"QID {qid}: Skipping item because context text resulted in empty chunks.")
        return None

    # Use semaphore to control concurrent API calls
    global CONCURRENCY_LIMIT 
    async with semaphore: 
        logger.debug(f"QID {qid}: Acquired semaphore, calling rewrite function.")
        revised_question_result, _ = await rewrite_question_with_context_async(original_question, context_chunks, template, qid)
        logger.debug(f"QID {qid}: Released semaphore.")

    # If rewrite failed, use the original question
    final_corrected_question = revised_question_result if revised_question_result is not None else original_question

    # Return the processed data
    return {
        "qid": qid,
        "question": original_question, # Always include original for reference
        "youtube_url": youtube_url,
        "context": context_chunks, # Keep the chunked context
        "video_id": video_id,
        "corrected_question": final_corrected_question # Rewritten or original if failed
    }

async def main(input_path_str: str, output_path_str: str, template_path_str: str):
    """
    Main asynchronous function to process the data.
    This is the orchestrator that manages the entire process.
    
    Key improvements from original:
    - Concurrent processing of items
    - Progress bar with tqdm
    - Detailed success/failure tracking
    - Better error handling and reporting
    
    Args:
        input_path_str: Path to input JSON file
        output_path_str: Path to save output JSON
        template_path_str: Path to prompt template
    """
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    template_path = Path(template_path_str)

    # Validate input files exist
    if not input_path.exists():
        logger.error(f"Input file not found at: {input_path}")
        return
    if not template_path.exists():
        logger.error(f"Template file not found at: {template_path}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read template once (more efficient than reading for each item)
    logger.info(f"Reading template from {template_path}")
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except Exception as e:
        logger.error(f"Failed to read template file: {e}", exc_info=True)
        return

    # Read and validate input data
    logger.info(f"Reading input data from {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse input JSON file: {e}", exc_info=True)
        return

    if not isinstance(data, list):
        logger.error(f"Input JSON is not a list. Found type: {type(data)}")
        return

    # Set up concurrency control
    global CONCURRENCY_LIMIT
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    logger.info(f"Creating {len(data)} processing tasks with concurrency limit {CONCURRENCY_LIMIT}...")
    valid_items_for_processing = []
    for item in data:
        if isinstance(item, dict):
            valid_items_for_processing.append(item)
            tasks.append(asyncio.create_task(process_item_async(item, template, semaphore)))
        else:
            logger.warning(f"Skipping invalid item in input data (not a dict): {item}")

    # Process items concurrently with progress bar
    results = []
    failed_qids = []
    logger.info(f"Running {len(tasks)} tasks concurrently...")
    for result_task in tqdm.as_completed(tasks, desc="Formatting Prompts", unit="item"):
        result = await result_task
        if result is not None: 
            results.append(result)
            # Track failed rewrites (where corrected_question equals original)
            original_question_for_check = result.get("question", "")
            corrected_question_for_check = result.get("corrected_question", "")
            if corrected_question_for_check == original_question_for_check:
                failed_qids.append(result.get("qid", "UNKNOWN_FAILED_QID"))

    # Log summary of results
    successful_count = len(results) - len(failed_qids)
    logger.info(f"Finished processing. Success: {successful_count}, Failures: {len(failed_qids)}")
    if failed_qids:
        logger.warning(f"Failed to rewrite questions for QIDs: {failed_qids}")

    # Save results
    logger.info(f"Saving {len(results)} results to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Output saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save output JSON file: {e}", exc_info=True)

# === Main Execution Block ===
if __name__ == "__main__":
    # Set up default paths relative to script location
    base_dir = Path(__file__).resolve().parent.parent 
    default_input = base_dir / "data" / "updated_original.json"
    default_output = base_dir / "data" / "recommended_formatted_output.json"
    default_template = base_dir / "templates" / "prompt_template.txt"

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Rewrite questions based on context using OpenAI API concurrently.")
    parser.add_argument(
        "--input",
        default=str(default_input),
        help=f"Path to the input JSON file (default: {default_input})",
        metavar="PATH"
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help=f"Path to save the output JSON file (default: {default_output})",
        metavar="PATH"
    )
    parser.add_argument(
        "--template",
        default=str(default_template),
        help=f"Path to the prompt template file (default: {default_template})",
        metavar="PATH"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY_LIMIT,
        help=f"Maximum number of concurrent API requests (default: {CONCURRENCY_LIMIT})",
        metavar="N"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Maximum number of retries for failed API calls (default: {MAX_RETRIES})",
        metavar="N"
    )
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=INITIAL_BACKOFF_SECONDS,
        help=f"Initial backoff delay in seconds for retries (default: {INITIAL_BACKOFF_SECONDS})",
        metavar="SECONDS"
    )

    # Parse arguments and update global settings
    args = parser.parse_args()
    CONCURRENCY_LIMIT = args.concurrency
    MAX_RETRIES = args.max_retries
    INITIAL_BACKOFF_SECONDS = args.initial_backoff

    # Run the main async function
    asyncio.run(main(args.input, args.output, args.template)) 