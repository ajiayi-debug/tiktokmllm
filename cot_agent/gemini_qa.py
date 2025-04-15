import os
import json
import traceback
from tqdm import tqdm
from collections import defaultdict
import csv
from gemini import Gemini
import time

import time

def gemini_video_fn(video_uri, questions, num_repeats=1, wait_time=65):
    gemini = Gemini()
    predictions = []

    for question in questions:
        for attempt in range(2):  # try once, then retry after error
            try:
                print(f"Asking: {question}")
                output = gemini.generate_from_video(video_uri, [question], num_repeats=num_repeats)[0]
                print(output)
                predictions.append(output)
                break  # success!
            except Exception as e:
                print(f"⚠️ Error on question: {question}\n➡️ {e}")
                if attempt == 0:
                    print(f"Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)
                else:
                    print("Failed after retry.")
                    predictions.append("Error")
        # Enforce delay after each question regardless of success or fail
        time.sleep(wait_time)

    return predictions

def process_all_video_questions_list_gemini(
    ds,
    iterations=1,
    checkpoint_path="geminipredictions.json",
    video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
    batch_size=5,
    filter_qids=None
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
        if sample["qid"] not in processed_qids and (filter_qids is None or sample["qid"] in filter_qids):
            video_to_samples[sample["youtube_url"]].append(sample)

    # Inference loop
    for video_id, samples in tqdm(video_to_samples.items(), desc="Processing grouped videos"):
        # video_path = os.path.join(video_dir, f"{video_id}.mp4")
        video_path=video_id
        print(video_id)
        # questions = [s["question"] + (s.get("question_prompt") or "") for s in samples]

        questions = []
        for s in samples:
            qid = s.get("qid", "UNKNOWN")
            try:
                question = s.get("question") or ""
                prompt = s.get("question_prompt") or ""
                questions.append(question + prompt)
            except Exception as e:
                print(f"Failed to build question for QID {qid}: {e}")
                predictions.append({"qid": qid, "prediction": "Error"})
        try:
            outputs = gemini_video_fn(video_path, questions, num_repeats=iterations)
            print(outputs)
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

def load_error_qids(json_path):
    with open(json_path, "r") as f:
        predictions = json.load(f)
    return {item["qid"] for item in predictions if item.get("prediction") == "Error"}


def merge_predictions(original_path, retry_path, merged_output_path):
    # Load original predictions
    with open(original_path, "r") as f:
        original = {item["qid"]: item["prediction"] for item in json.load(f)}

    # Load retry predictions
    with open(retry_path, "r") as f:
        retry = {item["qid"]: item["prediction"] for item in json.load(f)}

    # Overwrite errors in original with retry values (if retry didn't also fail)
    for qid, retry_pred in retry.items():
        if original.get(qid) == "Error" and retry_pred != "Error":
            print(f"✅ Overwriting error for QID {qid}")
            original[qid] = retry_pred

    # Convert back to list of dicts
    merged = [{"qid": qid, "prediction": pred} for qid, pred in original.items()]

    # Save merged results
    with open(merged_output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged predictions saved to: {merged_output_path}")
    print(f"Total predictions: {len(merged)}")

