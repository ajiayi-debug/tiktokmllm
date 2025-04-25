#from agents.format_prompt_agent import run_format_prompt_agent
from agents.cot_agent.gemini_cot_agent import CotAgent
import asyncio
import pandas as pd
from pathlib import Path
from datasets import load_dataset

async def main():
    #run_format_prompt_agent()
    ds = load_dataset("lmms-lab/AISG_Challenge")
    ds['test'].to_csv("Data.csv", index=False)
    iteration=32  
    prompt_path = Path(__file__).parent / "templates"/"iterate_prompt.txt" 
    template = prompt_path.read_text(encoding="utf-8").strip()
    iterate_prompt = template.format(iteration=iteration)      
    #df=pd.read_json('data/testjson.json')
    #df = pd.read_json("data/recommended_formatted_output.json")
    df=pd.read_csv('data/Data.csv')
    print(df)
    await CotAgent(                    
        df, f"Gemini_{iteration}_context", f"Gemini_{iteration}_retry_context",
        number_of_iterations=1,
        iterate_prompt=iterate_prompt,
        video_upload=True,
        wait_time=10,
        iteration_in_prompt=iteration
    )

if __name__ == "__main__":
    asyncio.run(main())