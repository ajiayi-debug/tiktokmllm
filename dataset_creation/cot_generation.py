from gpt import *
from prompt import *
from gpt_functions import *
import asyncio
from data import *

question="What is the difference in actions between the last person and the first two persons?"

prompt=get_prompt(question, qbreakdown)

asyncio.run(openai_gpt4o_mini_async(prompt, max_tokens=9999, output_json=False ))