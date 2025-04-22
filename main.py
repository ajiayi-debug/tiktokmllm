#from agents.format_prompt_agent import run_format_prompt_agent
from agents.cot_agent.gemini_cot_agent import CotAgent
import asyncio
import pandas as pd

async def main():
    #run_format_prompt_agent()
    iteration=32         
    iterate_prompt=f"""Generate your top {iteration} highest confidence scoring answers. Dont rank the answers. For multiple choice answers, provide a brief explanation on why that choice is chosen (ignore the second instruction of the question)."""
    df = pd.read_csv('data/data.csv') 
    await CotAgent(                    
        df, "Gemini_top8", "Gemini_top8_retry",
        number_of_iterations=1,
        iterate_prompt=iterate_prompt,
        video_upload=True,
        wait_time=10,
        iteration_in_prompt=iteration
    )

if __name__ == "__main__":
    asyncio.run(main())