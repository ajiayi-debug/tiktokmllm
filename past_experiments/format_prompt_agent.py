import openai
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_question_with_context(question: str, context_chunks: list[str], template_path: str = None) -> str:
    if template_path is None:
        template_path = Path(__file__).resolve().parent.parent / "templates" / "prompt_template.txt"

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    context = '\n'.join(f"- {chunk}" for chunk in context_chunks)
    prompt = template.format(context=context, question=question)

    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def run_format_prompt_agent(qid: str, question: str, context_text: str, video_id: str, youtube_url: str):
    context_chunks = [chunk.strip() for chunk in context_text.split('.') if chunk.strip()]
    revised_question_raw = rewrite_question_with_context(question, context_chunks)
    revised_question = revised_question_raw.replace("Rewritten Question:", "").strip(' "\n')
    return {
        "qid": qid,
        "question": question,
        "youtube_url": youtube_url,
        "context": context_chunks,
        "video_id": video_id,
        "corrected_question": revised_question
    }

# === BATCH MODE ===
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent  # tiktokmllm/
    input_path = base_dir / "data" / "test_output.json"
    output_path = base_dir / "data" / "formatted_output.json"

    if not input_path.exists():
        raise FileNotFoundError(f" Input file not found at: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        qid = item.get("qid", "")
        question = item.get("question_with_prompt", "")
        context_text = item.get("context", "")
        video_id = item.get("video_id", "")
        youtube_url = item.get("youtube_url", "")

        formatted = run_format_prompt_agent(qid, question, context_text, video_id, youtube_url)
        results.append(formatted)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f" Finished formatting {len(results)} entries. Output saved to {output_path}")
