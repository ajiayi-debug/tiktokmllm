import pandas as pd


df=pd.read_csv('data/GeminiPredictions.csv')
df_data=pd.read_csv('data/data.csv')

# Reorder df based on df_data's qid order
df = df.set_index("qid")
df_data = df_data.set_index("qid")

# Reindex and reset index (to bring qid back as a column)
df_reordered = df.reindex(df_data.index).reset_index()

# Save without index
df_reordered.to_csv("data/GeminiPrediction_reordered.csv", index=False)

