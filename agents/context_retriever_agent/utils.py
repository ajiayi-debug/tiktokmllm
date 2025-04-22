"""Utility functions for the Context Retriever Agent."""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Set, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("context_retriever")


def parse_context_response(response: str) -> Dict[str, Any]:
    """
    Parse the response from Gemini to extract structured context data.
    
    Args:
        response: The raw response string from Gemini
        
    Returns:
        A dictionary containing the parsed context data
    """
    # Default values in case parsing fails
    default_context = {
        "context": "Failed to extract context",
        "is_contextual": False,
        "explanation": "Response parsing error",
        "corrected_question": None
    }
    
    try:
        # Try to find and parse JSON in the response
        # Look for JSON-like structure
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx + 1]
            context_data = json.loads(json_str)
            
            # Ensure all required fields are present
            result = {
                "context": context_data.get("context", default_context["context"]),
                "is_contextual": context_data.get("is_contextual", default_context["is_contextual"]),
                "explanation": context_data.get("explanation", default_context["explanation"]),
                "corrected_question": context_data.get("corrected_question", default_context["corrected_question"])
            }
            return result
        else:
            # If no JSON structure found, treat the entire response as context
            logger.warning("No JSON structure found in response, using raw text as context")
            return {
                "context": response,
                "is_contextual": True,  # Assume it's contextual
                "explanation": "Direct context extraction without validation",
                "corrected_question": None
            }
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from response")
        # If JSON parsing fails, use the raw response as context
        return {
            "context": response,
            "is_contextual": True,  # Assume it's contextual
            "explanation": "Failed to parse structured context; using raw response",
            "corrected_question": None
        }
    except Exception as e:
        logger.error(f"Error parsing context response: {e}")
        return default_context


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
    Merge intermediate prediction JSON data with the original dataset CSV 
    and save the final merged data as a JSON file.

    Args:
        intermediate_json_path: Path to the intermediate JSON file 
                                  (contains list of {'qid': ..., 'prediction': {...}}).
        original_csv_path: Path to the original data CSV file.
        output_json_path: Path to save the final merged JSON file.

    Returns:
        The path to the saved final merged JSON file.
    """
    try:
        # Load intermediate predictions (JSON)
        intermediate_preds = load_checkpoint(intermediate_json_path)
        if not intermediate_preds:
            logger.warning(f"Intermediate predictions file is empty or not found: {intermediate_json_path}")
            # Decide how to handle this - perhaps return empty structure or raise error
            # For now, let's try to proceed assuming original data might still be useful alone
            # or create an empty output file.
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

        # Perform the merge logic
        merged_data_list = []
        for _, row in original_df.iterrows():
            original_item = row.to_dict()
            qid_str = original_item['qid'] # Already ensured to be string
            prediction_data = intermediate_preds_map.get(qid_str, {}) # Get prediction by string qid

            # Combine original data with prediction data
            merged_item = original_item.copy() # Start with original data
            merged_item.update(prediction_data) # Add/overwrite with prediction fields
            
            # Optionally recreate combined question fields if needed (example from old logic)
            # merged_item['original_question'] = merged_item.get('question', '')
            # merged_item['question_with_prompt'] = f"{merged_item.get('question_prompt', '')} {merged_item.get('original_question', '')}".strip()
            # merged_item['question'] = merged_item['question_with_prompt']
            
            merged_data_list.append(merged_item)

        # Save the merged data to the output JSON file
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        with open(output_json_path, "w") as f:
            json.dump(merged_data_list, f, indent=2)
        
        logger.info(f"Merged data saved to: {output_json_path} ({len(merged_data_list)} rows)")
        return output_json_path

    except Exception as e:
        logger.error(f"Error merging data and saving to JSON {output_json_path}: {e}")
        # Depending on desired behavior, re-raise, return None, or return the path anyway
        raise # Re-raise the exception to indicate failure 