import openai
import json
import asyncio
import json
import pandas as pd
from typing import List

load_dotenv()  # loads from .env

openai.api_key = os.getenv("OPENAI_API_KEY")

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


async def run_fact_checks(predictions_path: str, df: pd.DataFrame) -> List[dict]:
    """
    predictions_path: str — path to JSONL or JSON array file with [{"qid": ..., "answers": [...]}]
    df: DataFrame — must contain columns ["qid", "question", "context"] (context as list[str])
    Returns a list of fact-check results (dicts)
    """

    # Load predictions
    with open(predictions_path) as f:
        preds = [json.loads(line) for line in f] if predictions_path.endswith(".jsonl") else json.load(f)

    # Convert to DataFrame and keep only qid + top prediction
    pred_df = pd.DataFrame([
        {"qid": p["qid"], "final_answer": p["answers"][0] if p["answers"] else ""}
        for p in preds
    ])

    # Merge with original df (must contain qid, question, context)
    merged_df = df.merge(pred_df, on="qid")

    # Run fact_check per row, async via thread pool
    async def check_row(row):
        return await asyncio.to_thread(
            fact_check,
            question=row["question"],
            context=row["context"],
            qid=row["qid"],
            final_answer=row["final_answer"],
            api_key=os.getenv("OPENAI_API_KEY")
        )

    tasks = [check_row(row) for _, row in merged_df.iterrows()]
    results = await asyncio.gather(*tasks)
    return results




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


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    df = pd.DataFrame([{
        "qid": "Q001",
        "question": "What is the difference between the action of the last person in the video and the actions of the first two people? Please state your answer with a brief explanation.",
        "context": 
            "The video shows three different people interacting with capped glass bottles. The first person, a man with a beard, taps the top of a beer bottle with a small object. The second person, a woman in an Adidas sweatshirt, taps the top of a Corona beer bottle with her finger. In both these instances, the bottle cap remains securely on the bottle. The third person, a man in a black shirt, first taps the top of a Coca-Cola bottle with a chopstick, then waves his hand over it, then taps the side of the bottle near the bottom with his knuckles, and finally waves his hand over the top again. After the final hand wave, the bottle cap pops off, and the soda fizzes."
    }])

    load_dotenv()

    prediction={"qid":"Q001", "prediction":"The first two people only tapped the bottle cap repeatedly with a small object, which failed to open the bottle. The last person, after initially tapping, successfully opened the bottle by using a different technique: a forceful hand strike near the cap."}
    results = asyncio.run(run_fact_checks(prediction, df))
    print(json.dumps(results, indent=2))
