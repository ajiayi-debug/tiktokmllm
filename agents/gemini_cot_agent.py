from datasets import load_dataset
import pandas as pd
from gemini_qa import *
from rearrange import reorder
import asyncio

async def CotAgent(df,checkpoint_path_initial,checkpoint_path_retry,final_output, number_of_iterations=1, temperature=0, iterate_prompt="", video_upload=False, wait_time=30):
    # Initial run
    await process_all_video_questions_list_gemini_df(
        ds=df,
        iterations=number_of_iterations,
        checkpoint_path=f"data/{checkpoint_path_initial}.json",
        video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
        batch_size=1,
        temperature=temperature,
        video_upload=video_upload,
        wait_time=wait_time
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
        video_upload=video_upload,
        wait_time=wait_time
    )

    # Merge and overwrite errors
    merge_predictions(
        original_path=f"data/{checkpoint_path_initial}.json",
        retry_path=f"data/{checkpoint_path_retry}.json",
        merged_output_path=f"data/{final_output}.json"
    )

    # Save final merged predictions to CSV
    save_predictions_to_csv(
        json_path=f"data/{final_output}.json",
        csv_path=f"data/{final_output}.csv"
    )

    reorder(f"data/{final_output}.csv",df,f"data/{final_output}_rearranged.csv")




if __name__ == "__main__":
    #df=load_dataset("lmms-lab/AISG_Challenge")
    df=pd.read_csv('data/data.csv')
    asyncio.run(CotAgent(df, "Gemini_guided", "Gemini_guided_retry", "Gemini_guided_Final", number_of_iterations=1,video_upload=False, wait_time=10))