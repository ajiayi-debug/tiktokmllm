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


def save_context_to_csv(predictions: List[Dict[str, Any]], csv_path: str):
    """
    Convert JSON predictions to CSV with context fields.
    
    Args:
        predictions: List of prediction dictionaries
        csv_path: Path to save the CSV file
    """
    rows = []
    for item in predictions:
        qid = str(item.get("qid", "")).strip()
        prediction = item.get("prediction", {})
        
        row = {
            "qid": qid,
            "context": prediction.get("context", ""),
            "is_contextual": prediction.get("is_contextual", False),
            "explanation": prediction.get("explanation", ""),
            "corrected_question": prediction.get("corrected_question", "")
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(rows)} context predictions to {csv_path}")


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


def merge_with_original_data(context_csv: str, original_csv: str, output_csv: str):
    """
    Merge context data with original dataset.
    
    Args:
        context_csv: Path to context predictions CSV
        original_csv: Path to original data CSV
        output_csv: Path to save merged CSV
    """
    try:
        # Load datasets
        context_df = pd.read_csv(context_csv)
        original_df = pd.read_csv(original_csv)
        
        # Set index for easy merging
        context_df = context_df.set_index("qid")
        original_df = original_df.set_index("qid")
        
        # Merge datasets
        merged_df = original_df.join(context_df, how="left")
        
        # Reset index to bring qid back as column
        merged_df = merged_df.reset_index()
        
        # Save to CSV
        merged_df.to_csv(output_csv, index=False)
        logger.info(f"Merged data saved to: {output_csv} ({len(merged_df)} rows)")
        
        # Also save to JSON for easier consumption by format_prompt_agent
        output_json = output_csv.replace(".csv", ".json")
        merged_df.to_json(output_json, orient="records")
        logger.info(f"Merged data also saved to JSON: {output_json}")
        
        return output_csv, output_json
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        return output_csv, None 