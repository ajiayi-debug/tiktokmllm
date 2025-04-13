from determine_best_answer import obtain_final_result
from preprocessqns import *
import pandas as pd
from qavideoprocess import *
from internmodel import InternVLVideoQA

videoqa=InternVLVideoQA()
#df=load_Dataset("lmms-lab/AISG_Challenge", "test")

def CotAgent(df):
    #run qa for cot model 8 times (testing do 3 times only)
    #need to add a column called context / column that has the integrated question with ontext then I call that column directly.
    process_all_video_questions_list(
        ds=df,
        video_fn=videoqa.ask,
        iterations=8,
        checkpoint_path="data/InternPredictions8.json",
        video_dir="Benchmark-AllVideos-HQ-Encoded-challenge",
        batch_size=5,
    )
    #save predictions to csv
    save_predictions_to_csv(
        json_path="data/InternPredictions8.json",
        csv_path="data/InternPredictions8.csv"
    )
    #choose best answer
    obtain_final_result("data/InternPredictions8.csv", "data/Results8.csv", description=False, model="gpt-4o-mini")


    
   



if __name__ == "__main__":
    CotAgent()
