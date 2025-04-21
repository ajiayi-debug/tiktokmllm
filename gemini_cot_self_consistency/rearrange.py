import pandas as pd

# df=pd.read_csv('data/GeminiPredictionsFinal.csv')
# df_data=pd.read_csv('data/data.csv')

# # Reorder df based on df_data's qid order
# df = df.set_index("qid")
# df_data = df_data.set_index("qid")

# # Reindex and reset index (to bring qid back as a column)
# df_reordered = df.reindex(df_data.index).reset_index()

# # Save without index
# df_reordered.to_csv("data/GeminiPredictionFinal_reordered.csv", index=False)

def reorder(csv, data, final):
    df=pd.read_csv(csv)
    df_data=data

    # Reorder df based on df_data's qid order
    df = df.set_index("qid")
    df_data = df_data.set_index("qid")

    # Reindex and reset index (to bring qid back as a column)
    df_reordered = df.reindex(df_data.index).reset_index()

    # Save without index
    df_reordered.to_csv(f"data/{final}", index=False)

    print(f"Reordered output, saved to data/{final}")
