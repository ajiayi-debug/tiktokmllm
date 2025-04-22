import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import asyncio
from typing import List
from tqdm import tqdm
from agents.cot_agent.fact_checker_agent import fact_check

load_dotenv()

model="gemini-2.5-pro-preview-03-25"

#model="gemini-2.5-flash-preview-04-17"


def choose_best_answer_prompt(question,output,iteration_in_prompt):
    prompt=f"""Based on the {iteration_in_prompt} answers and the video, determine the best answer to the question: {question}
    The best answer can be a combination of answers from the top {iteration_in_prompt} answers. Just answer the question and dont explain why you chose that answer among the top {iteration_in_prompt} answer and dont reference the answers.
    The {iteration_in_prompt} answers:
    {output}"""
    return prompt

class GeminiAsync:
    """Async wrapper around Google Gen‑AI Video QA pipeline (two‑step logic)."""

    def __init__(self, api_key: str | None = None, model: str = model) -> None:
        # sync client is still handy for file uploads
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.aio = self.client.aio  # async surface
        self.model = model

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    async def _stream_text(self, contents: list[types.Content], temperature: float = 0.0) -> str:
        cfg = types.GenerateContentConfig(
            temperature=temperature, response_mime_type="text/plain"
        )

        # --- non‑streaming, simplest & compatible ---
        resp = await self.aio.models.generate_content(
            model=self.model, contents=contents, config=cfg
        )
        if resp.text:
            return resp.text.strip()
        else:
            return None

    async def _wait_until_file_active(self, file_obj, timeout: int = 30, poll: int = 2):
        """Polls until the uploaded file reaches ACTIVE state."""
        start = time.time()
        while time.time() - start < timeout:
            fo = await self.aio.files.get(name=file_obj.name)
            if getattr(fo, "state", "") == "ACTIVE":
                return fo
            await asyncio.sleep(poll)
        raise TimeoutError(f"{file_obj.name} not ACTIVE within {timeout}s")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------



    async def generate_from_video(
        self,
        video_uri: str,
        questions: list[str],
        *,
        temperature: float = 0.0,
        wait_time: int = 30,
        iterate_prompt: str = "",
        iteration_in_prompt: int = 8,
        step4_prompt_template: str = ""
    ) -> list[str]:
        """
        Runs QA in three phases: 
        1) generate n candidate answers,
        2) select the best answer,
        3) iteratively fact-check and refine until supported.
        Returns the final answers only.
        """
        final_answers: list[str] = []

        for q in tqdm(questions, desc="Answering questions", unit="q", leave=False):
            # ---- Step 1: generate multiple answers ----
            print(q[2])
            if iterate_prompt:
                # build prompt with iterate_prompt to get candidates
                contents_multi = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=video_uri, mime_type="video/*"),
                            types.Part.from_text(text=f"{q[2]} {iterate_prompt}"),
                        ],
                    )
                ]
                try:
                    multi = await self._stream_text(contents_multi, temperature)
                except Exception as e:
                    print(f"Gemini API error in step 1: {e}")
                    final_answers.append("Error")
                    continue
                print(f"Step 1 answers for '{q[2]}':\n{multi}\n")

                # ---- Step 2: pick best answer from candidates ----
                best_prompt = choose_best_answer_prompt(q[0], multi, iteration_in_prompt)
                contents_best = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=video_uri, mime_type="video/*"),
                            types.Part.from_text(text=best_prompt),
                        ],
                    )
                ]
                try:
                    answer = await self._stream_text(contents_best, temperature)
                    print(f"step 2 answer: {answer}")
                except Exception as e:
                    print(f"Gemini API error in step 2: {e}")
                    final_answers.append("Error")
                    continue
            else:
                # if no iterate_prompt, single-shot QA
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=video_uri, mime_type="video/*"),
                            types.Part.from_text(text=q),
                        ],
                    )
                ]
                try:
                    answer = await self._stream_text(contents, temperature)
                except Exception as e:
                    print(f"Gemini API error in single-shot: {e}")
                    final_answers.append("Error")
                    continue

            # ---- Iterative Step 3 & 4: fact-check & refine ----
            while True:
                print(f"Context: {q[1]}")
                is_supported, explanation = await asyncio.to_thread(
                    fact_check,
                    question=q[0],
                    context=[q[1]],
                    final_answer=answer
                )
                print(is_supported)
                print(explanation)
                if is_supported or not step4_prompt_template:
                    break

                # refine prompt using explanation
                prompt = step4_prompt_template.format(
                    question=q,
                    answer=answer,
                    explanation=explanation
                )
                print(prompt)
                contents_refine = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=video_uri, mime_type="video/*"),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ]
                try:
                    answer = await self._stream_text(contents_refine, temperature)
                    print(f"step4 answer:{answer}")
                except Exception as e:
                    print(f"Gemini API error in step 4 refine: {e}")
                    break
                # throttle before next fact-check
                await asyncio.sleep(wait_time)

            final_answers.append(answer)
            # throttle between Qs
            await asyncio.sleep(wait_time)

        return final_answers

    async def generate_from_uploaded_video_file(
        self,
        file_path: str,
        question: str,
        *,
        temperature: float = 0.0,
        iterate_prompt: str = "",
        iteration_in_prompt=8,
        wait_time: int = 30,
    ) -> str:
        """Uploads a local video then runs the two‑step QA flow."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        # Blocking upload – push to executor so event‑loop stays free
        upload_obj = await asyncio.to_thread(self.client.files.upload, file=file_path)
        upload_obj = await self._wait_until_file_active(upload_obj)
        uri = upload_obj.uri

        answers = await self.generate_from_video(
            uri,
            [question],
            temperature=temperature,
            wait_time=wait_time,
            iterate_prompt=iterate_prompt,
            iteration_in_prompt=iteration_in_prompt
        )
        return answers[0]

    async def generate(
        self, input_text: str, *, temperature: float = 0.7, stream: bool = False
    ) -> str:
        """Plain text generation (non‑video)."""
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
        ]
        if stream:
            return await self._stream_text(contents, temperature)
        cfg = types.GenerateContentConfig(
            temperature=temperature, response_mime_type="text/plain"
        )
        resp = await self.aio.models.generate_content(
            model=self.model, contents=contents, config=cfg
        )
        return resp.text.strip()


# -------------------------------------------------------------------------
# Quick CLI test – run:  python gemini_async.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo():
        g = GeminiAsync()
        answers = await g.generate_from_video(
            video_uri="https://youtu.be/sj81PWrerDk",
            questions=[["Did the last person in the video open the bottle with a knife while the first two people failed in their attempts? Please state your answer with a brief explanation.","The video displays three separate clips. In the first clip, a man with a beard taps the cap of a glass bottle (appears to be Bundaberg Ginger Beer) with a small object, possibly a lighter or another cap. The clip ends before the outcome is shown. In the second clip, a woman taps the cap of a Corona beer bottle with a thin, stick-like object. This clip also ends before the outcome is revealed. In the third clip, a man attempts to open a Coca-Cola bottle. He first taps the cap with a chopstick, then tries flicking it with a folded piece of paper, taps it again with the chopstick, and finally makes a sweeping hand gesture towards the bottle, after which the cap appears to fly off. This final action seems like a magic trick or video edit rather than using a physical tool like a knife.","What methods did the individuals in the video use to try and open their bottles, and did the final person appear to successfully open the bottle using an unconventional technique or trick?"]],
            iterate_prompt="Generate your top 8 highest confidence scoring answers. Dont rank the answers.",
            iteration_in_prompt=8,
            step4_prompt_template=(
                "The previous answer was '{answer}', but it was contradicted because {explanation}. Please answer the question '{question}' correctly."
            ),
            temperature=0,
            wait_time=10
        )
        print("Final Answer:", answers)
    asyncio.run(_demo())