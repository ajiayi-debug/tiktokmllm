import os
import json
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def fact_check(
    question: str,
    final_answer: str
) -> tuple[bool, str]:
    """
    Checks if the final_answer is a sound and reasonable response to the question.
    Returns:
      (is_supported: bool, explanation: str)
    """
    system_prompt = (
        "You are a fact-checking assistant. Your job is to check if a given final answer is a reasonable and logically sound response to the provided question."
    )

    user_prompt = f"""
Question:
{question}

Final Answer:
"{final_answer}"

Instructions:
Assess if the final answer makes sense as a response to the question. 
It should be relevant, logically coherent, and plausible.

Respond ONLY in valid JSON using one of the formats below.

✔️ Supported:
{{ 
  "is_supported": true, 
  "explanation": "..." 
}}

❌ Unsupported:
{{ 
  "is_supported": false, 
  "explanation": "..." 
}}
""".strip()

    try:
        resp = openai.ChatCompletion.create(
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

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    test_cases = [
        ("What is the difference in the action between the first two people and the last person? Please state your answer with a brief explanation.", "The first two people tap to open a bottle while the last person flicks his finger to open a bottle"),
        ("What is the difference in the action between the first two people and the last person? Please state your answer with a brief explanation.", "Person 1 and 2 were holding a knife while person 3 is holding nothing"),
        ("What is the capital of France?", "Paris"),
        ("What is the capital of France?", "France is known for wine and cheese."),
        
    ]

    for i, (question, final_answer) in enumerate(test_cases, 1):
        is_supported, explanation = fact_check(question, final_answer)
        status = "✅ Reasonable" if is_supported else "❌ Not Reasonable"
        print(f"\nTest {i}: {status}")
        print(f"Q: {question}")
        print(f"A: {final_answer}")
        print(f"→ Explanation: {explanation}")



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



