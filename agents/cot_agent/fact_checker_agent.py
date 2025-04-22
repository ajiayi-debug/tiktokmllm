
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fact_check(
    question: str,
    context: list[str],
    final_answer: str
) -> tuple[bool, str]:
    """
    Checks if the final_answer is consistent with the provided context.
    Returns:
      (is_supported: bool, explanation: str)
    """
    context_text = "\n".join(f"- {line}" for line in context)
    system_prompt = (
        "You are a fact-checking assistant that checks consistency between answers and context."
    )
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
        resp = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt}
            ]
        )
        reply = resp.choices[0].message.content.strip()
        try:
            result = json.loads(reply)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from model. Raw reply:\n{reply}") from e
        return (
            result.get("is_supported", False),
            result.get("explanation", "No explanation provided.")
        )
    except Exception as e:
        return (False, f"Error during fact check: {e}")


# # Example test run
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     import os

#     load_dotenv()
#     api_key = os.getenv("OPENAI_API_KEY")

#     result = fact_check(
#         question="Why did the person pour the wine down the sink?",
#         context=[
#             "The person uncorks a wine bottle.",
#             "They pour the wine into a glass and drink it."
#         ],
#         qid="Q001",
#         final_answer="They poured it down the sink because it smelled off.",
#         api_key=api_key
#     )

#     print(json.dumps(result, indent=2))



