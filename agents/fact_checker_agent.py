# fact_checker_agent.py

import openai
import json

def fact_check_answer(question, context, final_answer, api_key):
    """
    Verifies if the final answer is supported by the context using OpenAI GPT.
    Returns a structured JSON response with support status and explanation.
    """
    openai.api_key = api_key
    context_text = "\n".join(f"- {line}" for line in context)

    system_prompt = "You are a fact-checking assistant that checks consistency between answers and context."

    user_prompt = f"""
Context:
{context_text}

Question:
{question}

Final Answer:
"{final_answer}"

Instructions:
Check if the answer is consistent with the context.

Respond ONLY in valid JSON using one of the formats below.

✔️ Supported:
{{ 
  "is_supported": true, 
  "final_verified_answer": "<same as final_answer>", 
  "explanation": "..." 
}}

❌ Contradiction:
{{ 
  "is_supported": false, 
  "revised_answer": "<a corrected version>", 
  "explanation": "..." 
}}
""".strip()

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    reply = response['choices'][0]['message']['content']
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        return {"error": "Could not parse response", "raw_response": reply}

# Example test run
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()  

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    question = "Why did the person pour the wine down the sink?"
    context = [
        "The person uncorks a wine bottle.",
        "They pour the wine into a glass and drink it."
    ]
    final_answer = "They poured it down the sink because it smelled off."

    result = fact_check_answer(question, context, final_answer, OPENAI_API_KEY)
    print(json.dumps(result, indent=2))
