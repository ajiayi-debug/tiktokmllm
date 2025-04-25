"""Configuration settings for the Context Retriever Agent."""

import os

# Default paths
DEFAULT_VIDEO_DIR = "Benchmark-AllVideos-HQ-Encoded-challenge"
DEFAULT_DATA_DIR = "data"
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# API settings
DEFAULT_TEMPERATURE = 0.0
DEFAULT_WAIT_TIME = 30
DEFAULT_ITERATIONS = 1
DEFAULT_BATCH_SIZE = 1

# File path templates
def get_checkpoint_path(checkpoint_name):
    """Get the full path for a checkpoint file inside the checkpoint directory."""
    # Ensure the directory exists
    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True) 
    # Return path inside the checkpoint directory
    return os.path.join(DEFAULT_CHECKPOINT_DIR, f"{checkpoint_name}.json")

# Removed get_output_path as it's no longer used by the agent
# def get_output_path(output_name, extension="csv"):
#     """Get the full path for an output file."""
#     return os.path.join(DEFAULT_DATA_DIR, f"{output_name}.{extension}") 