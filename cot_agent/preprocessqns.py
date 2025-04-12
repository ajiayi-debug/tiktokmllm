from gpt_call import *
from datasets import load_dataset
import asyncio
import pandas as pd
import random
from tqdm.asyncio import tqdm_asyncio

def load_Dataset(name, traintestsplit):
    ds = load_dataset(name)
    df=ds[traintestsplit].to_pandas()
    df.to_csv("data.csv")
    return df

def convert_to_excel(json, name):
    df=pd.read_json(json)
    df.to_csv(name)
    return df

def stepbystep_prompt(qns):
    return f"""Break down the following questions into steps such that a visual Large Language Model can follow the steps to solve the question with the video. 
Take note that you are not able to watch the video so just break down the question into steps. OUTPUT THE STEPS ONLY!
question: {qns}
steps:"""

def breakdown_prompt(qns):
    return f"""Breakdown the questions into questions that a Visual Large Language Model needs to answer to give a Large Language Model (text only) details to solve the question. Output in the form of a python list of strings where each string is a question.
**For example:** 
question: What is different in the action between the first two people and the last person?
questions to ask Visual Large Language Model to solve the question: ['What are the actions of the first person?', 'What are the actions of the second person?', 'What are the actions of the third person?']

**Your answer:**
question: {qns}
questions to ask Visual Large Language Model to solve the question:"""


async def universal_qns_processor(
    df,
    question,
    question_prompt=None,
    prompt_fn=None,
    result_column="result",
    model="gpt-4o-mini",
    max_retries=3,
    batch_size=10,
    max_parallel=5  # control concurrency
):
    semaphore = asyncio.Semaphore(max_parallel)
    all_results = []

    # Split into batches
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    async def process_batch(batch_df):
        async with semaphore:
            batch_outputs = []

            for _, row in batch_df.iterrows():
                if question_prompt:
                    qns = row[question] + " " + row[question_prompt]
                else:
                    qns = row[question]

                prompt = prompt_fn(qns)

                success = False
                for attempt in range(max_retries):
                    try:
                        if model == "gpt-4o-mini":
                            result = await openai_gpt4o_mini_async(prompt)
                        else:
                            result = await openai_o3_mini_async(prompt)

                        batch_outputs.append(result)
                        success = True
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            batch_outputs.append(qns)  # fallback after final failure
                if not success:
                    continue  # skip to next row

            return batch_outputs

    # Kick off all batch tasks with tqdm progress
    all_batch_outputs = await tqdm_asyncio.gather(
        *(process_batch(batch) for batch in batches),
        desc=f"Generating {result_column}",
    )

    # Flatten and assign result column
    flat_results = [item for sublist in all_batch_outputs for item in sublist]
    df[result_column] = flat_results
    return df
