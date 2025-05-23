from agents.cot_agent.gemini_cot_agent import CotAgent
import asyncio
import pandas as pd
from pathlib import Path
from datasets import load_dataset

async def main():
    ds = load_dataset("lmms-lab/AISG_Challenge")
    ds['test'].to_csv("Data.csv", index=False)
    iteration=32  
     
    
    df=pd.read_csv('data/Data.csv') 
    await CotAgent(                    
        df, f"Gemini_{iteration}", f"Gemini_{iteration}_retry",
        number_of_iterations=1,
        iterate_prompt=True,
        video_upload=True,
        wait_time=10,
        iteration_in_prompt=iteration
    )

if __name__ == "__main__":
    asyncio.run(main())


