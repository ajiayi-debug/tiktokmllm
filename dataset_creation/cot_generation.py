from gpt import *
from prompt import *
from gpt_functions import *
import asyncio
import pandas as pd
from tqdm import tqdm

# question="What is the difference in actions between the last person and the first two persons?"

# prompt=get_prompt(question, qbreakdown)

# asyncio.run(openai_gpt4o_mini_async(prompt, max_tokens=9999, output_json=False ))


# Async task runner
async def process_row(row):
    prompt = generate_qbreakdown(row)
    try:
        result = await openai_gpt4o_mini_async(prompt, max_tokens=9999, output_json=False)
        return result
    except Exception as e:
        print(f"Error on row {row['video_name']}: {e}")
        return None

# Batching with tqdm
async def process_in_batches(df, batch_size=10):
    results = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
        batch = df.iloc[i:i+batch_size]
        tasks = [process_row(row) for _, row in batch.iterrows()]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    return results

async def run_multiple(csv_paths, batch_size=5):
    for path in tqdm(csv_paths, desc="Processing CSV files"):
        await breakdown(path, batch_size=batch_size)


# Main breakdown function
async def breakdown(csv, batch_size=5):
    name=csv+".csv"
    df = pd.read_csv(name)
    outputs = await process_in_batches(df, batch_size=batch_size)
    df["question_breakdown"] = outputs
    output_path = f"{csv}_with_question_breakdown.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

csv = ["data/generic","data/temporal","data/consistency"]
# Run
if __name__ == "__main__":
    
    asyncio.run(run_multiple(csv))