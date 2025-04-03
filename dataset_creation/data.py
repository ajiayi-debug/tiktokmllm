import pandas as pd

df = pd.read_parquet("hf://datasets/lmms-lab/VideoChatGPT/Generic/test-00000-of-00001.parquet")
print(df.head())