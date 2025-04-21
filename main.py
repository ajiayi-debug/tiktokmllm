from agents.format_prompt_agent import run_format_prompt_agent
from agents.gemini_cot_agent import CotAgent
import asyncio
import pandas as pd

async def main():
    run_format_prompt_agent()         
    iterate_prompt="""Generate your top 8 highest confidence scoring answers. Dont rank the answers."""
    df = pd.read_csv('data/data.csv') 
    await CotAgent(                    
        df, "Gemini_top8", "Gemini_top8_retry", "Gemini_top8_Final",
        number_of_iterations=1,
        iterate_prompt=iterate_prompt,
        video_upload=True,
        wait_time=10
    )

if __name__ == "__main__":
    asyncio.run(main())