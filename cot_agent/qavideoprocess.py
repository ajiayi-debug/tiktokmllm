import os
import json
import torch
import traceback
from tqdm import tqdm
from collections import defaultdict
import csv


def process_all_video_questions_list(
    ds,
    video_fn,
    iterations=1,
    checkpoint_path="internPredictions.json",
    video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
    batch_size=5,
):
    """
    Processes grouped questions per video using a vision-language model.

    Args:
        ds: a dataset (or a dict containing a list in ds['test']) with fields 'qid', 'video_id', 'question' and optionally 'question_prompt'
        video_fn: function(video_path, list_of_questions) -> list_of_predictions
        iterations: Number of times model repeats output for each question
        checkpoint_path: path to save intermediate and final predictions
        video_dir: folder containing videos named as <video_id>.mp4
        batch_size: number of predictions before intermediate save
    """
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            predictions = json.load(f)
        processed_qids = {p["qid"] for p in predictions}
        print(f"Loaded {len(predictions)} predictions from checkpoint.")
    else:
        predictions = []
        processed_qids = set()

    # Group by video
    video_to_samples = defaultdict(list)
    for sample in ds["test"]:
        if sample["qid"] not in processed_qids:
            video_to_samples[sample["video_id"]].append(sample)

    # Inference loop
    with torch.no_grad():
        for video_id, samples in tqdm(video_to_samples.items(), desc="Processing grouped videos"):
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            questions = [s["question"] + s.get("question_prompt", "") for s in samples]

            try:
                outputs = video_fn(video_path, questions, num_repeats=iterations)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                print(f"Error on video {video_id}: {e}")
                traceback.print_exc()
                for s in samples:
                    predictions.append({
                        "qid": s["qid"],
                        "prediction": "Error"
                    })
                _save_checkpoint(predictions, checkpoint_path)
                continue

            for sample, response in zip(samples, outputs):
                predictions.append({
                    "qid": sample["qid"],
                    "prediction": response
                })
                print(f"{sample['qid']}: {response}")

            if len(predictions) % batch_size == 0:
                _save_checkpoint(predictions, checkpoint_path)

    _save_checkpoint(predictions, checkpoint_path)


def _save_checkpoint(predictions, path):
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2)


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

