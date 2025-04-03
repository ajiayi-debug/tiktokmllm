import pandas as pd
from datasets import load_dataset

ds = load_dataset("Fahad-S/videobench_cotv01")

df = pd.read_parquet("hf://datasets/lmms-lab/VideoChatGPT/Generic/test-00000-of-00001.parquet")
df.to_csv('data/generic.csv')

df = pd.read_parquet("hf://datasets/lmms-lab/VideoChatGPT/Consistency/test-00000-of-00001.parquet")
df.to_csv('data/consistency.csv')

df = pd.read_parquet("hf://datasets/lmms-lab/VideoChatGPT/Temporal/test-00000-of-00001.parquet")
df.to_csv('data/temporal.csv')

