"""Utility functions for the Context Retriever Agent."""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Set, Optional
import re # Import regex module (Keep import in case needed elsewhere, or remove if truly unused)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("context_retriever.utils")


def parse_context_response(response: str) -> Dict[str, Any]:
    """
    Parse the context response string (expected JSON) into a dictionary.
    Handles potential JSON decoding errors and missing keys robustly.
    
    Args:
        response: The raw response string from Gemini
        
    Returns:
        A dictionary containing the parsed context data
    """
    # Define the default error structure
    default_error_output = {
        "context": "Error: Failed to parse response",
        "is_contextual": False,
        "explanation": "Could not decode or extract valid fields from the model output.",
        "corrected_question": None
    }
    
    if not response or not isinstance(response, str):
        logger.warning(f"Received empty or non-string response to parse: {response!r}")
        return default_error_output.copy() # Return a copy

    try:
        # Clean the response: remove potential markdown code fences
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.startswith("```"):
             cleaned_response = cleaned_response[len("```"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")]

        # Attempt to parse the JSON
        data = json.loads(cleaned_response)
        
        # Validate required keys and types carefully
        parsed_output = {}
        parsed_output["context"] = str(data.get("context", default_error_output["context"]))
        
        # Handle is_contextual: check type, default to False
        is_contextual_val = data.get("is_contextual")
        if isinstance(is_contextual_val, bool):
            parsed_output["is_contextual"] = is_contextual_val
        else:
            logger.warning(f"Invalid type for 'is_contextual': {type(is_contextual_val)}. Defaulting to False.")
            parsed_output["is_contextual"] = False # Default to False on type mismatch
            
        parsed_output["explanation"] = str(data.get("explanation", default_error_output["explanation"]))
        # Corrected question is optional, default to None
        parsed_output["corrected_question"] = data.get("corrected_question") # Allow None or string

        # Add a check: if essential fields defaulted, log it
        if parsed_output["context"] == default_error_output["context"]:
             logger.warning(f"Could not find 'context' field in parsed JSON: {cleaned_response}")
        if parsed_output["explanation"] == default_error_output["explanation"]:
             logger.warning(f"Could not find 'explanation' field in parsed JSON: {cleaned_response}")


        logger.debug(f"Successfully parsed context response for QID (unknown here): {parsed_output}")
        return parsed_output

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError parsing context response: {e}. Response was: {response!r}")
        return default_error_output.copy()
    except Exception as e:
        # Catch any other unexpected errors during parsing/validation
        logger.error(f"Unexpected error parsing context response: {e}. Response was: {response!r}")
        return default_error_output.copy()


def save_checkpoint(predictions: List[Dict[str, Any]], path: str):
    """
    Save predictions to checkpoint file.
    
    Args:
        predictions: List of prediction dictionaries
        path: Path to save the checkpoint
    """
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(predictions, f, indent=2)
        
        if os.path.exists(temp_path):
            if os.path.exists(path):
                os.replace(temp_path, path)
            else:
                os.rename(temp_path, path)
            logger.info(f"Checkpoint saved: {path} ({len(predictions)} predictions)")
        else:
            logger.error(f"Failed to save checkpoint: {path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    """
    Load predictions from checkpoint file.
    
    Args:
        path: Path to the checkpoint file
        
    Returns:
        List of prediction dictionaries or empty list if file doesn't exist
    """
    if not os.path.exists(path):
        logger.info(f"No checkpoint found at {path}")
        return []
    
    try:
        with open(path, "r") as f:
            predictions = json.load(f)
        logger.info(f"Loaded {len(predictions)} predictions from {path}")
        return predictions
    except Exception as e:
        logger.error(f"Error loading checkpoint {path}: {e}")
        return []


def get_processed_qids(predictions: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract set of QIDs that have been processed.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Set of processed QIDs
    """
    return {p["qid"] for p in predictions}


def load_error_qids(path: str) -> Set[str]:
    """
    Load QIDs that resulted in errors.
    
    Args:
        path: Path to the checkpoint file
        
    Returns:
        Set of QIDs with errors
    """
    predictions = load_checkpoint(path)
    error_qids = {
        p["qid"] for p in predictions 
        if p.get("prediction", {}).get("context") == "Error"
    }
    logger.info(f"Found {len(error_qids)} QIDs with errors")
    return error_qids


def merge_predictions(original_path: str, retry_path: str, merged_output_path: str):
    """
    Merge original and retry predictions.
    
    Args:
        original_path: Path to original predictions
        retry_path: Path to retry predictions
        merged_output_path: Path to save merged predictions
    """
    # Load original predictions
    original_preds = load_checkpoint(original_path)
    original = {item["qid"]: item["prediction"] for item in original_preds}
    
    # Load retry predictions
    retry_preds = load_checkpoint(retry_path)
    retry = {item["qid"]: item["prediction"] for item in retry_preds}
    
    # Overwrite errors in original with retry values
    updated_count = 0
    for qid, retry_pred in retry.items():
        if original.get(qid, {}).get("context") == "Error" and retry_pred.get("context") != "Error":
            logger.info(f"Overwriting error for QID {qid}")
            original[qid] = retry_pred
            updated_count += 1
    
    # Convert back to list of dicts
    merged = [{"qid": qid, "prediction": pred} for qid, pred in original.items()]
    
    # Save merged results
    with open(merged_output_path, "w") as f:
        json.dump(merged, f, indent=2)
    
    logger.info(f"Merged predictions saved to: {merged_output_path}")
    logger.info(f"Total predictions: {len(merged)}, Updated: {updated_count}")


def merge_with_original_data(intermediate_json_path: str, original_csv_path: str, output_json_path: str) -> str:
    """
    Merge intermediate predictions with original data and save to a final JSON file.
    Also handles concatenating 'question' and 'question_prompt' into the 'question' field
    and removing the original 'question_prompt' field.

    Args:
        intermediate_json_path: Path to the intermediate JSON predictions file
                                  (contains list of {'qid': ..., 'prediction': {...}}).
        original_csv_path: Path to the original data CSV file.
        output_json_path: Path to save the final merged JSON file.

    Returns:
        The path to the saved final merged JSON file.
    """
    logger.info(f"Merging intermediate predictions from {intermediate_json_path} with {original_csv_path} into {output_json_path}")
    try:
        # Load intermediate predictions (JSON)
        intermediate_preds = load_checkpoint(intermediate_json_path)
        if not intermediate_preds:
            logger.warning(f"Intermediate predictions file is empty or not found: {intermediate_json_path}. Output will only contain original data.")
            intermediate_preds_map = {}
        else:
             # Convert list to dict keyed by qid for easier lookup
             # Ensure qids are strings for consistent merging
            intermediate_preds_map = {str(item['qid']): item.get('prediction', {}) 
                                      for item in intermediate_preds}

        # Load original data (CSV)
        if not os.path.exists(original_csv_path):
            logger.error(f"Original data CSV not found: {original_csv_path}")
            raise FileNotFoundError(f"Original data CSV not found: {original_csv_path}")
        original_df = pd.read_csv(original_csv_path)
        # Drop the 'Unnamed: 0' column if it exists (often created when saving CSV with index)
        if 'Unnamed: 0' in original_df.columns:
            logger.info("Dropping 'Unnamed: 0' column from original data.")
            original_df = original_df.drop(columns=['Unnamed: 0'])

        # Convert qid to string in the original DataFrame as well
        if 'qid' in original_df.columns:
             original_df['qid'] = original_df['qid'].astype(str)
        else:
             logger.error(f"'qid' column not found in {original_csv_path}")
             raise ValueError(f"'qid' column not found in {original_csv_path}")

        def get_combined_question(row):
            # Always try to combine question and prompt if prompt exists
            q_text = row.get("question", "") # Get original value safely
            prompt_text = row.get("question_prompt", "") # Get prompt value safely
            
            # Use logger for potential debugging if needed later
            # logger.info(f"Processing QID {row.get('qid', 'N/A')}: HasQText={bool(q_text)}, HasPrompt={bool(prompt_text)}")

            if prompt_text: # Check if prompt_text is not None and not empty
                # Concatenate using the format from format_gemini_prompt (strip and join with newline)
                combined = f"{str(q_text or '').strip()}\n{str(prompt_text or '').strip()}"
                # logger.info(f"QID {row.get('qid', 'N/A')}: Returning COMBINED question: {combined[:100]}...")
                return combined
            else:
                # Otherwise, return the original 'question' value from the row (stripped)
                original_q_in_row = str(q_text or '').strip()
                # logger.info(f"QID {row.get('qid', 'N/A')}: Returning ORIGINAL question: {original_q_in_row[:100]}...")
                return original_q_in_row

        # Apply the function to update the 'question' column in the DataFrame
        # Remove the check for specific columns needed only for MC logic if always concatenating
        logger.info("Applying question concatenation logic to DataFrame for all rows.")
        # Ensure 'question' and 'question_prompt' columns exist before applying
        required_cols_for_concat = ['qid', 'question'] # 'question_prompt' is optional
        if all(col in original_df.columns for col in required_cols_for_concat):
            # Make sure 'question_prompt' exists if we need it, handle gracefully if not
            if 'question_prompt' not in original_df.columns:
                 logger.warning("'question_prompt' column not found. Concatenation will only use 'question' field.")
                 # Create an empty 'question_prompt' column to avoid errors in apply
                 original_df['question_prompt'] = "" 
                 
            original_df['question'] = original_df.apply(get_combined_question, axis=1)
            logger.info("Question concatenation applied.")
        else:
             missing = [col for col in required_cols_for_concat if col not in original_df.columns]
             logger.warning(f"Skipping question concatenation due to missing essential columns: {missing}")


        # --- End Concatenation in DataFrame --- 

        # Perform the merge logic (now uses the potentially modified original_df)
        merged_data_list = []
        for _, row in original_df.iterrows():
            original_item = row.to_dict()
            qid_str = original_item['qid'] # Already ensured to be string
            # Get prediction data from the intermediate map, default to empty dict if not found
            prediction_data = intermediate_preds_map.get(qid_str, {}) 
            
            # Combine original data with prediction data
            merged_item = original_item.copy() # Start with original data
            merged_item.update(prediction_data) # Add/overwrite with prediction fields (context, is_contextual, etc.)

            # --- Remove original question_prompt field ---
            merged_item.pop('question_prompt', None) # Remove the key if it exists, do nothing otherwise

            # --- Explicitly remove 'Unnamed: 0' if it survived the DataFrame drop ---
            merged_item.pop('Unnamed: 0', None)
            
            # --- Replace NaN values with None for JSON compatibility ---
            for key, value in merged_item.items():
                # Use pandas.isna() as it handles various NaN types robustly
                if pd.isna(value):
                    merged_item[key] = None
            # --- End NaN Replacement ---

            merged_data_list.append(merged_item)

        # --- Debug Log Before Save ---
        for item in merged_data_list:
            if item.get('qid') == '0008-7':
                 logger.info(f"DEBUG QID 0008-7: Final 'question' value BEFORE saving: {item.get('question', 'NOT FOUND')[:150]}...") # DEBUG
                 break # Found it, no need to check further
        # --- End Debug Log ---

        # Save the merged data to the output JSON file
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        with open(output_json_path, "w") as f:
            json.dump(merged_data_list, f, indent=2)
        
        logger.info(f"Merged data saved to: {output_json_path} ({len(merged_data_list)} rows)")
        return output_json_path

    except Exception as e:
        logger.error(f"Error merging data and saving to JSON {output_json_path}: {e}")
        raise # Re-raise the exception to indicate failure 