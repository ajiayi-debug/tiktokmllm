from determine_best_answer import obtain_final_result
from preprocessqns import *

#obtain_final_result("descriptioninternpredictions.csv", "o3nodescriptionresults.csv", description=False, model="o3-mini")
#obtain_final_result("descriptioninternpredictions.csv", "o3results.csv", description=True, model="o3-mini")

def main():
    #load dataset
    df=load_dataset("lmms-lab/AISG_Challenge")
    df.to_csv("data.csv")


