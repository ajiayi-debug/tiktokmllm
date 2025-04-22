import json
import csv
import os
from collections import defaultdict
from tqdm import tqdm

#to determine which questions and videos are NOT answered yet.
def checkpointing_qns(df,checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            predictions = json.load(f)
        processed_qids={p['qid'] for p in predictions}
        print(f"Loaded {len(predictions)} predictions from checkpoint")
    else:
        predictions = []
        processed_qids = set()
    # Filter and group by unprocessed qids
    df_filtered = df[~df['qid'].isin(processed_qids)]
    return df_filtered


def save_predictions_to_csv(
    json_path,
    csv_path
):
    """
    Converts JSON predictions to CSV with columns: qid, pred.

    Args:
        json_path (str): Path to the JSON file containing predictions.
        csv_path (str): Output path for the CSV file.
    """
    # Load the original predictions
    with open(json_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Prepare rows
    rows = []
    for item in predictions:
        qid = str(item.get("qid", "")).strip()
        pred = str(item.get("prediction", "")).strip()
        rows.append([qid, pred])

    # Save to CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "pred"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {csv_path}")


def format_gemini_prompt(question, prompt):
    return f"""\
{str(question or '').strip()}
{str(prompt or '').strip()}"""