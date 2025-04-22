import openai
import json

def fact_check(question: str, context: list[str], qid: str, final_answer: str, api_key: str) -> dict:
    """
    Checks if the final_answer is consistent with the provided context.

    Returns:
        {
            "is_supported": True/False,
            "explanation": "...",
            "qid": "..."
        }
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

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        reply = response['choices'][0]['message']['content']
        result = json.loads(reply)

        return {
            "is_supported": result.get("is_supported", False),
            "explanation": result.get("explanation", "No explanation provided."),
            "qid": qid
        }
    except Exception as e:
        return {
            "is_supported": False,
            "explanation": f"Error during fact check: {e}",
            "qid": qid
        }


# Example test run
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    result = fact_check(
        question="Why did the person pour the wine down the sink?",
        context=[
            "The person uncorks a wine bottle.",
            "They pour the wine into a glass and drink it."
        ],
        qid="Q001",
        final_answer="They poured it down the sink because it smelled off.",
        api_key=api_key
    )

    print(json.dumps(result, indent=2))