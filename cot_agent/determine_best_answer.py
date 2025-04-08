from gpt_call import *
import pandas as pd

def best_answer_prompt(video_description, question, list_of_answer):
    prompt=f"""You are in charge of choosing the best answer among a list of {len(list_of_answer)} chain of thought answers to the following question: 
    {question}
    The question is related to a video whereby its description is as follows: {video_description}
    The list of chain of thought answers are as follows: {list_of_answer}
    Choose the best chain of thought answer that follows as closely as possible to the following rubrics:
    Rubrics:
    1) Answers the question according to the video
    2) Has the most logical chain of thought process
    3) If none of the answers make sense, choose the answer that has the majority in terms of final answer
    
    Output the final answer of the chosen chain of thought answer. (Meaning dont include the step by step thought process of the answers)
    Your chosen final aswer:"""
    return prompt

def output(df):
    return 