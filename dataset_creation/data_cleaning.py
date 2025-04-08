import pandas as pd

# Load the CSV
df = pd.read_csv("data/consistency.csv")

# Combine question_1 and question_2 with logic
def combine_questions(row):
    q1 = row["question_1"]
    q2 = row["question_2"]

    if pd.isna(q1):
        return q2
    elif pd.isna(q2):
        return q1
    else:
        return f"{q1.strip()} {q2.strip()}"

# Apply the combination
df["question"] = df.apply(combine_questions, axis=1)

# Drop the original columns
df.drop(columns=["question_1", "question_2"], inplace=True)

# Save the new CSV
df.to_csv("data/consistency.csv", index=False)

