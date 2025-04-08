from prompt import *

def get_prompt(question, prompt_template):
    return prompt_template.format(question=question)

def generate_qbreakdown(row):
    return get_prompt(row["question"], qbreakdown)