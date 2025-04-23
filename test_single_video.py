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

async def test_single_video():
    """Process a single video and display the results."""
    print("\n==== Context Retriever Agent - Single Video Test ====\n")
    
    # Load data.csv
    print("Loading data...")
    df = pd.read_csv('data/data.csv')
    
    # Filter to just one video (using the first unique video ID)
    first_video_id = df['video_id'].unique()[0]
    df_single_video = df[df['video_id'] == first_video_id].copy()
    
    # Print info about the selected video
    video_url = df_single_video['youtube_url'].iloc[0]
    num_questions = len(df_single_video)
    print(f"Selected video: {first_video_id}")
    print(f"Video URL: {video_url}")
    print(f"Number of questions: {num_questions}")
    
    # Create output suffix for this test - SIMPLIFIED
    # Include the video ID to make intermediate/final names unique per video test
    # test_suffix = f"ContextRetriever_test_{first_video_id}" # Original Long version
    test_suffix = f"test_{first_video_id}" # Shorter version

    # Use shorter checkpoint/suffix names
    initial_ckpt_name = f"initial_{test_suffix}"
    retry_ckpt_name = f"retry_{test_suffix}"
    intermediate_suffix = f"intermediate_{test_suffix}" # Used for intermediate json and final json name

    # Run the agent on just this video
    # It now returns only the path to the final merged JSON
    print("\nProcessing video...")
    final_json_path = await ContextRetrieverAgent(
        df=df_single_video,
        # checkpoint_path_initial=f"ContextRetriever_initial_{test_suffix}", # Original Long version
        # checkpoint_path_retry=f"ContextRetriever_retry_{test_suffix}",     # Original Long version
        # final_output_suffix=test_suffix, # Original Long version
        checkpoint_path_initial=initial_ckpt_name, # Shorter name
        checkpoint_path_retry=retry_ckpt_name,     # Shorter name
        final_output_suffix=intermediate_suffix, # Shorter name
        video_upload=True,
        wait_time=10
    )

    # --- Results Display and Test Output Generation ---
    print("\n==== RESULTS ====\n")
    # The agent now directly produces the desired final JSON file
    print(f"Final Merged JSON output saved to: {final_json_path}")
    # Expected path: data/Step1_Context_ContextRetriever_test_VIDEOID.json

    # --- Create data/test_output.json (filtered from the final agent output) ---
    test_output_json_path = "data/test_output.json"

    print(f"\nAttempting to create filtered test output: {test_output_json_path}")
    print(f"Reading final merged JSON from: {final_json_path}")

    if os.path.exists(final_json_path):
        try:
            # Get the QIDs that were part of this specific test run
            test_qids = df_single_video['qid'].astype(str).tolist() # Ensure string comparison
            print(f"Filtering for QIDs: {test_qids}")

            with open(final_json_path, 'r') as f_in:
                all_merged_data = json.load(f_in)
            
            # Filter the merged data to include only items processed in this run
            filtered_merged_data = [item for item in all_merged_data if str(item.get('qid')) in test_qids]

            # Ensure the data directory exists
            os.makedirs(os.path.dirname(test_output_json_path), exist_ok=True)

            with open(test_output_json_path, 'w') as f_out:
                json.dump(filtered_merged_data, f_out, indent=2) # Write filtered data
            print(f"Filtered test output saved to: {test_output_json_path} ({len(filtered_merged_data)} items)")

            # Display a sample from the test_output.json file
            print("\nTest Output JSON Example (from test_output.json):")
            if filtered_merged_data:
                first_item = filtered_merged_data[0]
                sample = {
                    "qid": first_item.get("qid", ""),
                    "video_id": first_item.get("video_id", ""),
                    "original_question": first_item.get("original_question", ""), 
                    "context": first_item.get("context", "")[:100] + "..." if isinstance(first_item.get("context"), str) and len(first_item.get("context")) > 100 else first_item.get("context", ""),
                    "is_contextual": first_item.get("is_contextual", ""),
                    "explanation": first_item.get("explanation", "")
                }
                print(json.dumps(sample, indent=2))

        except Exception as e:
            print(f"Error processing final merged JSON or saving test output: {e}")
            logging.error(f"Error processing {final_json_path} or saving to {test_output_json_path}: {e}")
    else:
        print(f"Warning: Could not find final merged JSON file at {final_json_path}.")
        print(f"         -> {test_output_json_path} was not created.")

    return final_json_path # Return the path generated by the agent

if __name__ == "__main__":
    final_json = asyncio.run(test_single_video())
    print("\nTest completed successfully!") 