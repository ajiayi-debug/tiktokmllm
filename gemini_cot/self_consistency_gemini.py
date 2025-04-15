from datasets import load_dataset
import pandas as pd
from gemini_qa import *

def CotAgent(df):
    # Initial run
    process_all_video_questions_list_gemini(
        ds=df,
        iterations=1,
        checkpoint_path="data/GeminiPredictionsSelfConsistency.json",
        video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
        batch_size=5,
        temperature=0.8
    )

    # Load QIDs with errors
    error_qids = load_error_qids("data/GeminiPredictionsSelfConsistency.json")

    # Retry only those
    process_all_video_questions_list_gemini(
        ds=df,
        iterations=1,
        checkpoint_path="data/GeminiPredictions_Retry.json",
        video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
        batch_size=5,
        temperature=0.8,
        filter_qids=error_qids
    )

    # Merge and overwrite errors
    merge_predictions(
        original_path="data/GeminiPredictions.json",
        retry_path="data/GeminiPredictions_Retry.json",
        merged_output_path="data/GeminiPredictionsjson"
    )

    # Save final merged predictions to CSV
    save_predictions_to_csv(
        json_path="data/GeminiPredictions.json",
        csv_path="data/GeminiPredictions.csv"
    )


if __name__ == "__main__":
    df=load_dataset("lmms-lab/AISG_Challenge")
    CotAgent(df)