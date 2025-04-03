#Script to download finetune dataset, dont need to run if u already have the data.
#Make sure to download full dataset of videos first
import pandas as pd
from datasets import load_dataset
import os

def sample_and_save(parquet_url, output_path, frac=0.2, seed=42):
    df = pd.read_parquet(parquet_url)
    sampled_df = df.sample(frac=frac, random_state=seed)
    sampled_df.to_csv(output_path, index=False)
    print(f"Saved {len(sampled_df)} rows to {output_path}")

#cot
ds = load_dataset("Fahad-S/videobench_cotv01")
ds = ds["test"]

ds.to_csv("data/cot.csv")

# Generic
sample_and_save("hf://datasets/lmms-lab/VideoChatGPT/Generic/test-00000-of-00001.parquet", "data/generic.csv")

# Consistency
sample_and_save("hf://datasets/lmms-lab/VideoChatGPT/Consistency/test-00000-of-00001.parquet", "data/consistency.csv")

# Temporal
sample_and_save("hf://datasets/lmms-lab/VideoChatGPT/Temporal/test-00000-of-00001.parquet", "data/temporal.csv")

# Step 1: Load sampled CSVs (only concerned about unlabelled large dataset.)
csv_paths = [
    "data/generic.csv",
    "data/consistency.csv",
    "data/temporal.csv",
]

# Step 2: Collect all video_ids used across datasets
all_video_ids = set()
for path in csv_paths:
    df = pd.read_csv(path)
    if "video_name" in df.columns:
        all_video_ids.update(df["video_name"].astype(str).tolist())
    else:
        print(f"Warning: No 'video_name' column found in {path}")

print(f"Total unique video_ids used: {len(all_video_ids)}")

# Step 3: Delete videos not in this list
video_folder = "Videos/Test_Videos"  # <--- Change this to your actual folder path
video_extensions = (".mp4", ".mov", ".avi", ".mkv")  # Adjust as needed

deleted_count = 0

for fname in os.listdir(video_folder):
    if fname.endswith(video_extensions):
        video_id = os.path.splitext(fname)[0]  # Remove .mp4 etc.
        if video_id not in all_video_ids:
            file_path = os.path.join(video_folder, fname)
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted unused video: {fname}")

print(f"Cleanup complete. Deleted {deleted_count} videos.")