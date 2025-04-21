from agents.format_prompt_agent import run_format_prompt_agent
from agents.gemini_cot_agent import CotAgent
import asyncio
import pandas as pd

if __name__ == "__main__":
    run_format_prompt_agent()
    df=pd.read_csv('data/data.csv')
    iterate_prompt="""Generate your top 8 highest confidence scoring answers. Dont rank the answers."""
    asyncio.run(CotAgent(df, "Gemini_top8", "Gemini_top8_retry", "Gemini_top8_Final", number_of_iterations=1, iterate_prompt=iterate_prompt, video_upload=True, wait_time=10))