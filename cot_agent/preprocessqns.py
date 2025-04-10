from gpt_call import *
from datasets import load_dataset
import asyncio
import pandas as pd
import random

def load_Dataset(name, traintestsplit):
    ds = load_dataset(name)
    df=ds[traintestsplit].to_pandas()
    df.to_csv("data.csv")
    return df


async def qns_stepbystep(df, question, question_prompt=None, model="gpt-4o-mini", max_retries=3, batch_size=10):
    all_steps = []

    for batch_start in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_start:batch_start + batch_size]
        batch_steps = []

        for _, row in batch_df.iterrows():
            # Construct the question
            if question_prompt:
                qns = row[question] + " " + row[question_prompt]
            else:
                qns = row[question]

            # Prompt construction
            prompt = f"""Break down the following questions into steps such that a visual Large Language Model can follow the steps to solve the question with the video. 
            Take note that you are not able to watch the video so just break down the question into steps.
            question: {qns}
            steps:"""

            # Retry with exponential backoff
            for attempt in range(max_retries):
                try:
                    if model == "gpt-4o-mini":
                        steps = await openai_gpt4o_mini_async(prompt)
                    else:
                        steps = await openai_o3_mini_async(prompt)

                    batch_steps.append(steps)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt + random.random()
                        print(f"Retrying ({attempt+1}/{max_retries}) after error: {e} — waiting {wait:.2f}s")
                        await asyncio.sleep(wait)
                    else:
                        print(f"Failed after {max_retries} attempts: {e}")
                        batch_steps.append("ERROR")

        all_steps.extend(batch_steps)

    df['steps'] = all_steps
    return df


async def qns_breakdown(df, question, question_prompt=None, model="gpt-4o-mini", max_retries=3, batch_size=10):
    all_break = []

    for batch_start in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_start:batch_start + batch_size]
        batch_break = []

        for _, row in batch_df.iterrows():
            # Construct the question
            if question_prompt:
                qns = row[question] + " " + row[question_prompt]
            else:
                qns = row[question]

            # Prompt construction
            prompt = f"""Breakdown the questions into questions that a Visual Large Language Model needs to answer to give a Large Language Model (text only) details to solve the question. Output in the form of a python list of strings where each string is a question.
            **For example:** 
            question: What is different in the action between the first two people and the last person?
            questions to ask Visual Large Language Model to solve the question: ['What are the actions of the first person?', 'What are the actions of the second person?', 'What are the actions of the third person?']
            
            **Your answer:**
            question: {qns}
            questions to ask Visual Large Language Model to solve the question:"""

            # Retry with exponential backoff
            for attempt in range(max_retries):
                try:
                    if model == "gpt-4o-mini":
                        breakdown = await openai_gpt4o_mini_async(prompt)
                    else:
                        breakdown = await openai_o3_mini_async(prompt)

                    batch_break.append(breakdown)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt + random.random()
                        print(f"Retrying ({attempt+1}/{max_retries}) after error: {e} — waiting {wait:.2f}s")
                        await asyncio.sleep(wait)
                    else:
                        print(f"Failed after {max_retries} attempts: {e}")
                        batch_break.append("ERROR")

        all_break.extend(batch_break)

    df['breakdown'] = all_break
    return df
