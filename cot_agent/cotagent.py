from determine_best_answer import obtain_final_result
from preprocessqns import *
import pandas as pd


def main():
    #obtain_final_result("descriptioninternpredictions.csv", "o3nodescriptionresults.csv", description=False, model="o3-mini")
    #obtain_final_result("descriptioninternpredictions.csv", "o3results.csv", description=True, model="o3-mini")
    remove_video_description("descriptioninternpredictions.csv","answer.csv")
    add_videodescription("answer.csv","video.json", "data_with_video_description.csv")
    obtain_final_result("data_with_video_description.csv","descriptionresult.csv")
    # ds=load_Dataset("lmms-lab/AISG_Challenge","test")
    # ds.to_csv("data.csv")
    # df = pd.read_csv("data.csv")

    # # Ensure unique IDs if needed
    # if 'qid' not in df.columns:
    #     df['qid'] = df.index.astype(str)

    # # Run both processors sequentially in one event loop
    # async def run_all():
    #     df_steps = await universal_qns_processor(
    #         df.copy(),  # make a copy to avoid overwriting
    #         question='question',
    #         question_prompt='question_prompt',
    #         prompt_fn=stepbystep_prompt,
    #         result_column='steps',
    #         model='gpt-4o-mini',
    #         max_retries=3,
    #         batch_size=10
    #     )
    #     df_steps.to_csv("datasteps.csv", index=False)

    #     df_breakdown = await universal_qns_processor(
    #         df.copy(),
    #         question='question',
    #         question_prompt='question_prompt',
    #         prompt_fn=breakdown_prompt,
    #         result_column='breakdown',
    #         model='gpt-4o-mini',
    #         max_retries=3,
    #         batch_size=10
    #     )
    #     df_breakdown.to_csv("qnsdata.csv", index=False)

    # asyncio.run(run_all())


if __name__ == "__main__":
    main()
