from pathlib import Path
import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_question_with_context(question: str, context_chunks: list[str]) -> str:
    template_path = Path(__file__).resolve().parent.parent / "templates" / "prompt_template.txt"
    
    with open(template_path, 'r') as f:
        template = f.read()

    context = '\n'.join(f"- {chunk}" for chunk in context_chunks)
    prompt = template.format(context=context, question=question)

    response = client.chat.completions.create(
        #model="gpt-3.5-turbo"
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}]
        #temperature=0.3
    )
    return response.choices[0].message.content.strip()

# TEST
def run_format_prompt_agent():
    question = "Why did they KICK the ball?"
    context = [
        "The person is in front of a goalpost.",
        "They are on a field."
    ]

    revised_question = rewrite_question_with_context(question, context)

    print("=== Question ===\n", question)
    print("=== Question integrated with context ===\n", revised_question)
    print("=== Context ===\n", context)