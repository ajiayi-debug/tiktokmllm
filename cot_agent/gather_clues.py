import pandas as pd
import asyncio
from gpt_call import *
import logging
import ast 
from tqdm.asyncio import tqdm_asyncio
import hashlib

def get_respective_answers(data,answers,merged):
    data=pd.read_csv(data)
    answers=pd.read_csv(answers)
    answers.rename(columns={'id': 'qid', 'pred': 'answers'}, inplace=True)
    merged_df = data.merge(answers, on='qid', how='left')
    merged_df.to_csv(merged)
    return merged_df


def answer_to_question(question,answer):
    prompt=f"""You are given the following list of questions (in python format):{question}
            You are also given the following list of answers that are not structured properly in terms of a list of answers but just a list of a single line of string that represents the answers to all questions: {answer}
            Output the correct format of answers to the question in the form of a list of string like ['answer1','answer2','answer3',...]. If the answer does not make sense, try to make sense of it and make an answer to the question.
            The answer:"""
    return prompt

def finalise_from_answer(QUESTION,question,answer):

    prompt=f"""You are given the main question: {QUESTION}
            You are also given the following list of questions that are formed to try to gather clues to solve the main question (in python format):{question}
            You are also given the following list of answers to the questions: {answer}
            Using the list of questions and the respective answers, answer the main question.
            Your answer:
            """
    return prompt

def generate_checkpoint_name(input_csv, func, batch_size):
    base_name = os.path.basename(input_csv).replace(".csv", "")
    identifier = f"{base_name}_{func}_{batch_size}"
    hash_id = hashlib.md5(identifier.encode()).hexdigest()[:8]
    return f"checkpoint_{hash_id}.csv"





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
async def process_batch_gpt(batch, func):
    results = []
    local_errors = []

    for row in batch:
        qid = row['qid']
        answer=row['answer']
        question = row['breakdown']
        QUESTION = row['question']+row['question_prompt']

        try:
            if func==answer_to_question:
                prompt = answer_to_question(question,answer)
                chosen_answer = await openai_gpt4o_mini_async(prompt)
            else:
                prompt = finalise_from_answer(QUESTION,question,answer)
                chosen_answer = await openai_o3_mini_async(prompt)
            results.append({"qid": qid, "answer": chosen_answer})
        except Exception as e:
            fallback = answer
            results.append({"qid": qid, "pred": fallback})
            local_errors.append({
                "qid": qid,
                "error": str(e),
                "fallback_used": True
            })

    return pd.DataFrame(results), local_errors

# Master batch orchestrator
async def process_all_batches_gpt(df, func, batch_size=10,max_parallel=4):
    error_log = []
    results = []

    semaphore = asyncio.Semaphore(max_parallel)

    async def safe_process(batch):
        async with semaphore:
            return await process_batch_gpt(batch, func)

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

def obtain_final_result(input_csv, output_csv, func, batch_size=10):
    df = pd.read_csv(input_csv)
    checkpoint_path = generate_checkpoint_name(input_csv, func, batch_size)

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
        process_all_batches_gpt(df, func=func, batch_size=batch_size)
    )

    # Append results
    updated_results = pd.concat([done_df, result_df], ignore_index=True)

    if func==answer_to_question:
        # Merge updated answers back into the original input dataframe
        original_df = pd.read_csv(input_csv)
        original_df.drop(columns=['answer'], inplace=True)  # Drop old answer column
        original_df = original_df.merge(updated_results[['qid', 'answer']], on='qid', how='left')

        # Save final outputs
        original_df.to_csv(checkpoint_path, index=False)
        original_df.to_csv(output_csv, index=False)
    else:
        # Append results and save checkpoint
        final_df = pd.concat([done_df, result_df], ignore_index=True)
        final_df.to_csv(checkpoint_path, index=False)
        final_df.to_csv(output_csv, index=False)

    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Final output saved to {output_csv}")


def remove_col(df,column_name,df_name):
    df.drop(columns=[column_name], inplace=True)
    df.to_csv(df_name,index=False)
    return df