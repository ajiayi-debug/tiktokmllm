import os
import json
import traceback
from tqdm import tqdm
from collections import defaultdict
import csv
from agents.cot_agent.gemini import GeminiAsync
import time
import pandas as pd
import asyncio
from typing import Iterable, List, Sequence
import openai


class RateLimitError(Exception):
    """Raised when we hit a 429 and want to abort the entire run."""
    pass

async def gemini_video_fn_async(
    *,
    video_uri: str,
    questions: Sequence[list],
    num_repeats: int = 1,
    wait_time: float = 30,
    temperature: float = 0.0,
    local_video_path: str | None = None,
    iterate_prompt: bool = True,
    video_upload: bool = False,
    iteration_in_prompt=8,
    concurrency: int = 5,
) -> list[str]:
    """Async drop‑in replacement for the old gemini_video_fn."""

    sem = asyncio.Semaphore(concurrency)

    g=GeminiAsync()

    async def _single(q: str) -> str:
        async with sem:
            try:
                if video_upload:
                    return await g.generate_from_uploaded_video_file(
                        local_video_path or "", q,
                        temperature=temperature,
                        iterate_prompt=iterate_prompt,
                        wait_time=wait_time,
                        iteration_in_prompt=iteration_in_prompt
                    )
                # try direct first
                try:
                    return (
                        await g.generate_from_video(
                            video_uri,
                            [q],
                            temperature=temperature,
                            iterate_prompt=iterate_prompt,
                            wait_time=wait_time,
                            iteration_in_prompt=iteration_in_prompt
                        )
                    )[0]
                except Exception as e:
                    if "403" in str(e) and local_video_path:
                        return await g.generate_from_uploaded_video_file(
                            local_video_path,
                            q,
                            temperature=temperature,
                            iterate_prompt=iterate_prompt,
                            wait_time=wait_time,
                            iteration_in_prompt=iteration_in_prompt
                        )
                    # if it’s a 429, bail out
                    if "429" in str(e):
                        print(f"[RateLimit] Question `{q}` hit 429; skipping.")
                        return "Error:RateLimit"
                    # otherwise re‑raise so outer catches it
                    raise
            except RateLimitError:
                # re‑raise so the outer gather catches it
                raise
            except Exception as final_err:
                print("Fail", final_err)
                return "Error"

    all_preds: list[str] = []
    for _ in range(num_repeats):
        try:
            # if any _single raises RateLimitError, gather will raise it here
            batch = await asyncio.gather(*[_single(q) for q in questions])
        except RateLimitError:
            print("[RateLimit] Received 429; aborting remaining iterations.")
            break

        all_preds.extend(batch)
        if wait_time:
            await asyncio.sleep(wait_time)

    return all_preds

def format_gemini_prompt(question, prompt):
    return f"""\
{str(question or '').strip()}
{str(prompt or '').strip()}"""


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
            print(f"Overwriting error for QID {qid}")
            original[qid] = retry_pred

    # Convert back to list of dicts
    merged = [{"qid": qid, "prediction": pred} for qid, pred in original.items()]

    # Save merged results
    with open(merged_output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged predictions saved to: {merged_output_path}")
    print(f"Total predictions: {len(merged)}")



async def process_all_video_questions_list_gemini_df(
    ds,
    iterations=1,
    checkpoint_path="geminipredictions.json",
    video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
    batch_size=5,
    temperature=0,
    filter_qids=None,
    iterate_prompt=True,
    video_upload=False,
    wait_time=30,
    iteration_in_prompt=8
):
    """
    Processes grouped questions per video using a vision-language model.

    Args:
        df: A pandas DataFrame with columns 'qid', 'youtube_url', 'question' and optionally 'question_prompt'
        iterations: Number of times model repeats output for each question
        checkpoint_path: Path to save intermediate and final predictions
        video_dir: Folder containing videos named as <video_id>.mp4 (unused if using URLs)
        batch_size: Number of predictions before intermediate save
        temperature: Temperature setting for generation
        filter_qids: Optional set of QIDs to process
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

    # Filter and group by video
    df_filtered = ds[~ds['qid'].isin(processed_qids)]
    if filter_qids is not None:
        df_filtered = df_filtered[df_filtered['qid'].isin(filter_qids)]

    video_to_samples = defaultdict(list)
    for _, row in df_filtered.iterrows():
        video_to_samples[row['youtube_url']].append(row)

    # Inference loop
    for video_url, samples in tqdm(video_to_samples.items(), desc="Processing grouped videos"):
        print(video_url)

        questions = []
        for s in samples:
            qid = s.get("qid", "UNKNOWN")
            try:
                formatted_prompt=s['question']
                
                questions.append([formatted_prompt])
                print(questions)
            except Exception as e:
                print(f"Failed to build question for QID {qid}: {e}")
                predictions.append({"qid": qid, "prediction": "Error"})

        # Prepare fallback to local .mp4
        video_id = samples[0].get("video_id") or samples[0].get("qid").split("_")[0]  # safe fallback
        local_video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(local_video_path):
            local_video_path = None

        try:
            outputs = await gemini_video_fn_async(
                video_uri=video_url,
                questions=questions,
                num_repeats=iterations,
                wait_time=wait_time,
                temperature=temperature,
                local_video_path=local_video_path,
                iterate_prompt=iterate_prompt,
                video_upload=video_upload,
                iteration_in_prompt=iteration_in_prompt
            )
            print(outputs)
        except Exception as e:
            print(f"Error on video {video_url}: {e}")
            traceback.print_exc()
            for s in samples:
                predictions.append({"qid": s["qid"], "prediction": "Error"})
            _save_checkpoint(predictions, checkpoint_path)
            continue

        for s, response in zip(samples, outputs):
            predictions.append({"qid": s["qid"], "prediction": response})
            print(f"{s['qid']}: {response}")

        if len(predictions) % batch_size == 0:
            _save_checkpoint(predictions, checkpoint_path)

    _save_checkpoint(predictions, checkpoint_path)

