#!/usr/bin/env python3
"""
Test script for Context Retriever Agent - processes a single video 
and outputs JSON and CSV with combined question format.
"""

import asyncio
import os
import pandas as pd
import json
import logging
import shutil  # Import shutil for file copying

from agents.context_retriever_agent.agent import ContextRetrieverAgent
from agents.context_retriever_agent.config import get_checkpoint_path # Import helper

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_single_video") # Use specific logger name

async def test_single_video():
    """Process a single specific video and display the results."""
    print("\n==== Context Retriever Agent - Specific Video Test ====\n")
    
    # --- Target Video --- 
    target_url = "https://www.youtube.com/shorts/mLbuK7IU7_Y"
    logger.info(f"Target video URL for testing: {target_url}")
    
    # Load data.csv
    print("Loading data...")
    try:
        df = pd.read_csv('data/data.csv')
    except FileNotFoundError:
        logger.error("Error: data/data.csv not found.")
        print("Error: data/data.csv not found. Please ensure the file exists.")
        return
    except Exception as e:
        logger.error(f"Error loading data.csv: {e}")
        print(f"Error loading data.csv: {e}")
        return
    
    # Filter to the specific video URL
    logger.info(f"Filtering data for URL: {target_url}")
    df_single_video = df[df['youtube_url'] == target_url].copy()
    
    # Check if video data was found
    if df_single_video.empty:
        logger.error(f"No data found in data.csv for the target URL: {target_url}")
        print(f"Error: No data found for URL {target_url} in data/data.csv.")
        return
        
    # Get info *after* filtering
    video_id = df_single_video['video_id'].iloc[0] # Get the actual video ID
    video_url_check = df_single_video['youtube_url'].iloc[0] # Verify URL
    num_questions = len(df_single_video)
    
    print(f"Selected video ID: {video_id}")
    print(f"Video URL: {video_url_check}")
    print(f"Number of questions for this video: {num_questions}")
    
    # Create output suffix using the *actual* video ID from the filtered data
    test_suffix = f"test_{video_id}" 

    # Use shorter checkpoint/suffix names
    initial_ckpt_name = f"initial_{test_suffix}"
    retry_ckpt_name = f"retry_{test_suffix}"
    intermediate_suffix = f"intermediate_{test_suffix}" # Used for intermediate json and final json name

    # Run the agent on just this video's data
    print("\nProcessing video with ContextRetrieverAgent...")
    try:
        final_json_path = await ContextRetrieverAgent(
            df=df_single_video, # Pass the filtered DataFrame
            checkpoint_path_initial=initial_ckpt_name, 
            checkpoint_path_retry=retry_ckpt_name,     
            final_output_suffix=intermediate_suffix, # This suffix determines the final output JSON name
            video_upload=True, # Assuming you want to test upload mode
            wait_time=10
        )
    except Exception as e:
        logger.error(f"Error running ContextRetrieverAgent: {e}", exc_info=True)
        print(f"An error occurred while running the agent: {e}")
        return

    # --- Results Display and Test Output Generation ---
    print("\n==== RESULTS ====\n")
    print(f"Agent processing complete.")
    print(f"Final Merged JSON output expected at: {final_json_path}")

    # --- Create data/test_output.json (filtered from the final agent output) ---
    test_output_json_path = "data/test_output.json"

    print(f"\nAttempting to create filtered test output: {test_output_json_path}")
    print(f"Reading final merged JSON from: {final_json_path}")

    if os.path.exists(final_json_path):
        try:
            # Get the QIDs that were part of this specific test run
            test_qids = df_single_video['qid'].astype(str).tolist() # Ensure string comparison
            logger.info(f"Filtering final JSON for QIDs: {test_qids}")

            with open(final_json_path, 'r', encoding='utf-8') as f_in:
                all_merged_data = json.load(f_in)
            
            # Filter the merged data to include only items processed in this run
            filtered_merged_data = [item for item in all_merged_data if str(item.get('qid')) in test_qids]

            os.makedirs(os.path.dirname(test_output_json_path), exist_ok=True)

            with open(test_output_json_path, 'w', encoding='utf-8') as f_out:
                json.dump(filtered_merged_data, f_out, indent=2) # Write filtered data
            print(f"Filtered test output saved to: {test_output_json_path} ({len(filtered_merged_data)} items)")

            # Display a sample from the test_output.json file
            print("\nTest Output JSON Example (from test_output.json):")
            if filtered_merged_data:
                first_item = filtered_merged_data[0]
                # Select fields relevant to the Context Retriever's output
                sample = {
                    "qid": first_item.get("qid", "N/A"),
                    "video_id": first_item.get("video_id", "N/A"),
                    "original_question": first_item.get("question", "N/A"), # Original question field name might vary post-merge
                    "context": str(first_item.get("context", "N/A"))[:150] + "...", # Show start of context
                    "is_contextual": first_item.get("is_contextual", "N/A"),
                    "explanation": str(first_item.get("explanation", "N/A"))[:150] + "...", # Show start of explanation
                    "corrected_question": first_item.get("corrected_question", None)
                }
                print(json.dumps(sample, indent=2))
            else:
                print("No items found matching the test QIDs in the final JSON.")

        except Exception as e:
            logger.error(f"Error processing final merged JSON ({final_json_path}) or saving test output: {e}", exc_info=True)
            print(f"Error processing agent output or saving test_output.json: {e}")
    else:
        logger.warning(f"Could not find final merged JSON file produced by agent at {final_json_path}.")
        print(f"Warning: Could not find final merged JSON file at {final_json_path}.")
        print(f"         -> {test_output_json_path} was not created.")

    return final_json_path # Return the path generated by the agent

if __name__ == "__main__":
    # Add basic error handling for the async run
    try:
        final_json = asyncio.run(test_single_video())
        if final_json: # Only print success if the function didn't return early due to error
             print("\nTest completed successfully!") 
    except Exception as main_e:
        logger.critical(f"An unexpected error occurred during the test run: {main_e}", exc_info=True)
        print(f"\nAn unexpected critical error occurred: {main_e}") 