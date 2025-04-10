from gpt_call import *
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import ast
import ast
import asyncio
import logging
import random
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from collections import Counter

import os
import pandas as pd
import hashlib

def generate_checkpoint_name(input_csv, model, batch_size, description):
    base_name = os.path.basename(input_csv).replace(".csv", "")
    desc_flag = "desc" if description else "nodesc"
    identifier = f"{base_name}_{model}_{batch_size}_{desc_flag}"
    hash_id = hashlib.md5(identifier.encode()).hexdigest()[:8]
    return f"checkpoint_{hash_id}.csv"



def best_answer_prompt(video_description, question, list_of_answer, description=True, synthesize=False):
    if synthesize:
        if description:
            prompt = f"""You are an expert assistant tasked with synthesizing the best possible answer to the following question related to a video.

Question: {question}
Video Description: {video_description}

You are provided with {len(list_of_answer)} chain-of-thought answers that reason through the question:

{list_of_answer}

Your task:
- Synthesize the best final answer using the most relevant and logical elements from all the answers.
- Your answer must follow the rubrics below:
    1) Answer the question accurately based on the video description.
    2) Follow a logical reasoning structure.
    3) Be coherent, complete, and concise.

If important explanations are found in the chain-of-thought reasoning steps but are missing from the final answer, integrate them appropriately.

Your synthesized final answer:"""
        else:
            prompt = f"""You are an expert assistant tasked with synthesizing the best possible answer to the following question.

Question: {question}

You are provided with {len(list_of_answer)} chain-of-thought answers that reason through the question:

{list_of_answer}

Your task:
- Synthesize the best final answer using the most relevant and logical elements from all the answers.
- Your answer must follow the rubrics below:
    1) Answer the question accurately.
    2) Follow a logical reasoning structure.
    3) Be coherent, complete, and concise.

If important explanations are found in the chain-of-thought reasoning steps but are missing from the final answer, integrate them appropriately.

Your synthesized final answer:"""
    else:
        if description:
            prompt = f"""You are in charge of choosing the best answer among a list of {len(list_of_answer)} chain of thought answers to the following question: 
{question}
The question is related to a video whereby its description is as follows: {video_description}
The list of chain of thought answers are as follows: {list_of_answer}
Choose the best chain of thought answer that follows as closely as possible to the following rubrics:
Rubrics:
1) Answers the question according to the video description
2) Has the most logical chain of thought process
3) If none of the answers make sense, choose the answer that has the majority in terms of final answer

Output the final answer of the chosen chain of thought answer. (Meaning don't include the step-by-step thought process of the answers)
If the final answer is missing any explanation that the question requires but the explanation can be found in the chain of thought process, include a summary of the chain of thought process in the final answer.
Your chosen final answer:"""
        else:
            prompt = f"""You are in charge of choosing the best answer among a list of {len(list_of_answer)} chain of thought answers to the following question: 
{question}
The list of chain of thought answers are as follows: {list_of_answer}
Choose the best chain of thought answer that follows as closely as possible to the following rubrics:
Rubrics:
1) Makes sense in answering the question as much as possible
2) Has the most logical chain of thought process
3) If none of the answers make sense, choose the answer that has the majority in terms of final answer

Output the final answer of the chosen chain of thought answer. (Meaning don't include the step-by-step thought process of the answers)
If the final answer is missing any explanation that the question requires but the explanation can be found in the chain of thought process, include a summary of the chain of thought process in the final answer.
Your chosen final answer:"""

    return prompt



# Safely parse prediction column into list
def parse_pred(pred):
    if isinstance(pred, list):
        return pred
    try:
        return ast.literal_eval(pred)
    except Exception as e:
        logging.warning(f"Could not parse prediction list: {e}")
        return []

# Retry delay extractor
def extract_retry_after(exception):
    if hasattr(exception, 'response') and exception.response is not None:
        retry_after = exception.response.headers.get('Retry-After')
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                logging.warning(f"Retry-After header not integer: {retry_after}")
    return None

# Single batch handler: fallback immediately to first answer on error
async def process_batch_gpt(model, batch, description, synthesize):
    results = []
    local_errors = []

    for row in batch:
        qid = row['qid']
        video_description = row['video_description']
        question = row['question'] + row['question_prompt']
        list_of_answer = parse_pred(row['pred'])

        if not list_of_answer:
            results.append({"qid": qid, "pred": "Error: Empty answer list."})
            local_errors.append({
                "qid": qid,
                "error": "Empty prediction list",
                "fallback_used": True
            })
            continue

        try:
            prompt = best_answer_prompt(video_description, question, list_of_answer, description, synthesize)
            if model == "gpt-4o-mini":
                chosen_answer = await openai_gpt4o_mini_async(prompt)
            else:
                chosen_answer = await openai_o3_mini_async(prompt)
            results.append({"qid": qid, "pred": chosen_answer})
        except Exception as e:
            fallback = list_of_answer[0]
            results.append({"qid": qid, "pred": fallback})
            local_errors.append({
                "qid": qid,
                "error": str(e),
                "fallback_used": True
            })

    return pd.DataFrame(results), local_errors

# Master batch orchestrator
async def process_all_batches_gpt(df, model="gpt-4o-mini", batch_size=10, description=True, synthesize=False, max_parallel=4):
    error_log = []
    results = []

    semaphore = asyncio.Semaphore(max_parallel)

    async def safe_process(batch):
        async with semaphore:
            return await process_batch_gpt(model, batch, description, synthesize)

    # Prepare batches
    batches = [df.iloc[i:i + batch_size].to_dict('records') for i in range(0, len(df), batch_size)]
    tasks = [safe_process(batch) for batch in batches]

    # Execute with progress tracking
    batch_outputs = await tqdm_asyncio.gather(*tasks, desc="Processing Batches")

    for batch_df, batch_errors in batch_outputs:
        results.append(batch_df)
        error_log.extend(batch_errors)

    final_df = pd.concat(results, ignore_index=True)
    return final_df, pd.DataFrame(error_log)


# # Process a single row
# async def process_row(row, description):
#     qid = row['qid']
#     video_description = row['video_description']
#     question = row['question'] + row['question_prompt']
#     list_of_answer = row['pred']
#     prompt = best_answer_prompt(video_description, question, list_of_answer, description)
#     try:
#         chosen_answer = await openai_gpt4o_mini_async(prompt)
#     except Exception as e:
#         list_of_answer = ast.literal_eval(row['pred'])
#         chosen_answer = list_of_answer[0]
#     return {"qid": qid, "pred": chosen_answer}

# # Process one batch of rows
# async def process_batch(batch_rows, description):
#     tasks = [process_row(row, description) for row in batch_rows]
#     return await asyncio.gather(*tasks)

# # Process all batches with tqdm
# async def process_all_batches(df, batch_size=10, description=True):
#     results = []
#     for i in tqdm_asyncio(range(0, len(df), batch_size), desc="Processing Batches"):
#         batch = [row for _, row in df.iloc[i:i+batch_size].iterrows()]
#         batch_result = await process_batch(batch, description)
#         results.extend(batch_result)
#     return pd.DataFrame(results)

# # Run the entire process
# def obtain_final_result(input_csv, output_csv, batch_size=10, description=True, model="gpt-4o-mini"):
#     df = pd.read_csv(input_csv)
#     result_df = asyncio.run(process_all_batches_gpt(df,model,batch_size=batch_size, description=description))
#     result_df.to_csv(output_csv, index=False)
#     print(f"Output saved to {output_csv}")


def obtain_final_result(input_csv, output_csv, batch_size=10, description=True, model="gpt-4o-mini", synthesize=False):
    df = pd.read_csv(input_csv)
    checkpoint_path = generate_checkpoint_name(input_csv, model, batch_size, description)

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        done_df = pd.read_csv(checkpoint_path)
        completed_ids = set(done_df['qid'])
        df = df[~df['qid'].isin(completed_ids)]  # only process unfinished
    else:
        done_df = pd.DataFrame()

    # If all done, skip
    if df.empty:
        print("All entries already processed.")
        done_df.to_csv(output_csv, index=False)
        return

    # Run remaining processing
    result_df, error_df = asyncio.run(
        process_all_batches_gpt(df, model=model, batch_size=batch_size, description=description, synthesize=synthesize)
    )

    # Append results and save checkpoint
    final_df = pd.concat([done_df, result_df], ignore_index=True)
    final_df.to_csv(checkpoint_path, index=False)
    final_df.to_csv(output_csv, index=False)

    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Final output saved to {output_csv}")
