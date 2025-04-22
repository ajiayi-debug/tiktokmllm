from datasets import load_dataset
import pandas as pd
from agents.gemini_qa import *
from agents.rearrange import reorder
import asyncio

async def CotAgent(df,checkpoint_path_initial,checkpoint_path_retry, number_of_iterations=1, temperature=0, iterate_prompt="", video_upload=False, wait_time=30, iteration_in_prompt=8):
    # Initial run
    await process_all_video_questions_list_gemini_df(
        ds=df,
        iterations=number_of_iterations,
        checkpoint_path=f"data/{checkpoint_path_initial}.json",
        video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
        batch_size=1,
        temperature=temperature,
        iterate_prompt=iterate_prompt,
        video_upload=video_upload,
        wait_time=wait_time,
        iteration_in_prompt=iteration_in_prompt
    )

    # Load QIDs with errors
    error_qids = load_error_qids(f"data/{checkpoint_path_initial}.json")

    # Retry only those
    await process_all_video_questions_list_gemini_df(
        ds=df,
        iterations=number_of_iterations,
        checkpoint_path=f"data/{checkpoint_path_retry}.json",
        video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
        batch_size=1,
        temperature=temperature,
        filter_qids=error_qids,
        iterate_prompt=iterate_prompt,
        video_upload=video_upload,
        wait_time=wait_time,
        iteration_in_prompt=iteration_in_prompt
    )

    # Merge and overwrite errors
    merge_predictions(
        original_path=f"data/{checkpoint_path_initial}.json",
        retry_path=f"data/{checkpoint_path_retry}.json",
        merged_output_path=f"data/{checkpoint_path_initial}.json"
    )

    # delete the retry file – it's no longer needed
    retry_path = f"data/{checkpoint_path_retry}.json"
    if os.path.isfile(retry_path):
        os.remove(retry_path)
        print("🗑️  removed", retry_path)

    # Save final merged predictions to CSV
    save_predictions_to_csv(
        json_path=f"data/{checkpoint_path_initial}.json",
        csv_path=f"data/{checkpoint_path_initial}.csv"
    )

    reorder(f"data/{checkpoint_path_initial}.csv",df,f"{checkpoint_path_initial}_rearranged.csv")




if __name__ == "__main__":
    #df=load_dataset("lmms-lab/AISG_Challenge")
    iteration=8
    iterate_prompt=f"""Generate your top {iteration} highest confidence scoring answers. Dont rank the answers."""
    df=pd.read_csv('data/data.csv')
    asyncio.run(CotAgent(df, "Gemini_top8", "Gemini_top8_retry", number_of_iterations=1, iterate_prompt=iterate_prompt, video_upload=True, wait_time=10, iteration_in_prompt=iteration))