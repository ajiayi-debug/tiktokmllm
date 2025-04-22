"""Configuration settings for the Context Retriever Agent."""

import os

# Default paths
DEFAULT_VIDEO_DIR = "Benchmark-AllVideos-HQ-Encoded-challenge"
DEFAULT_DATA_DIR = "data"

# API settings
DEFAULT_TEMPERATURE = 0.0
DEFAULT_WAIT_TIME = 30
DEFAULT_ITERATIONS = 1
DEFAULT_BATCH_SIZE = 1

# File path templates
def get_checkpoint_path(checkpoint_name):
    """Get the full path for a checkpoint file."""
    return os.path.join(DEFAULT_DATA_DIR, f"{checkpoint_name}.json")

def get_output_path(output_name, extension="csv"):
    """Get the full path for an output file."""
    return os.path.join(DEFAULT_DATA_DIR, f"{output_name}.{extension}") 