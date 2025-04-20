from agents.format_prompt_agent import run_format_prompt_agent
from agents.gemini_cot_agent import CotAgent
import asyncio
import pandas as pd

if __name__ == "__main__":
    run_format_prompt_agent()
    df=pd.read_csv('data/data.csv')
    asyncio.run(CotAgent(df, "Gemini_guided", "Gemini_guided_retry", "Gemini_guided_Final", number_of_iterations=1,video_upload=False, wait_time=10))