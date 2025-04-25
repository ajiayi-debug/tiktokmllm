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
#from fact_checker_agent import fact_check

load_dotenv()

model="gemini-2.5-pro-preview-03-25"

#model="gemini-2.5-flash-preview-04-17"

def refiner(question, answer, explanation):
    prompt=f"""The previous answer was '{answer}', but it contradicts the question '{question}' because {explanation}. Please answer the question '{question}' correctly by refining your answer based on the explanation and the video.
            Return the refined answer ONLY and make sure it answers the question: {question}.
            """
    return prompt
    

def choose_best_answer_prompt(question,output,iteration_in_prompt):
    prompt=f"""Based on the {iteration_in_prompt} answers and the video, select/craft an answer to the question using the {iteration_in_prompt} answers.
    If the question '{question}' is a multiple choice question:
        -If you only see one selected answer, check if you agree with the answer after watching the video before giving your answer in reference to the video.
        -If the {iteration_in_prompt} answers selects E as all of its option, IGNORE THE {iteration_in_prompt} ANSWERS and watch the video yourself to determine the answer.
        -If you conclude the answer as 'E: None of the above', rewatch the video and re-evaluate your options as E is NOT A answer. If you still think it's 'E', just choose the NEXT BEST PLAUSIBLE ANSWER.
        -ELSE: If the {iteration_in_prompt} answers is NOT E, select from there your final answer after confirming by watching the video.
        -IF multiple multiple choice questions CAN be the answer (e.g A and B can be the answer), you select E as NONE of the multiple choice answer IS the answer (make sure E stands for None of the above).
        -ALWAYS MAKE SURE YOUR FINAL ANSWER TO MULTIPLE CHOICE QUESTION IS A SINGULAR LETTER FROM THE OPTIONS.
    else:
        -Final answer can be a combination of answers from the {iteration_in_prompt} answers OR you select the most logical answers OR you use the answers to create the final answer.
    Just answer the question and dont explain why you gave that answer among the top {iteration_in_prompt} answer and dont reference the answers.
    The {iteration_in_prompt} answers:
    {output}
    The question to answer: 
    {question}
    Your answer:"""
    return prompt


def use_context(context,iteration,question):
    prompt=f"""Here is some textual context of the video: {context}

    Watch the video and then, generate your top {iteration} highest confidence scoring answers to the question. 
    Dont rank the answers. For multiple choice answers, provide a brief explanation on why that choice is chosen for each answer. 

    Some special cases:
    Try to answer the question in terms of the context of the video. DONT simply answer the question AS WHAT YOU SEE. REMEMBER YOUR CONTEXT (both from watching the video and from the text if it makes sense) 
    For example:
    A girl is hanging on a basketball rim with the help of a boy and she is shown hanging on the rim at the end of the video.
    Sample question: What is the girl doing at the end of the video
    Sample Answer: The girl is PRETENDING to dunk.

    Sometimes the actions are also a causation due to external actions OUTSIDE THE VIDEO (e.g a cut on a bottle is made, allowing one to just smack the bottle open easily using their finger). TRY not to assume everything is video editing or magic and think about what other kind of plausible external physical causation can cause what happen in the video for incidents that are unexplainable in the video.
    
    Consider the exact activity or context or situation that is going on, and ensure that EVERY WORD in the user’s question is mapped to this exact activity or context or situation. For example, in golf a successful hit is “golf club makes contact with ball and hits the target” and NOT JUST “golf club makes contact with ball” do not assume the meaning of the word in the user’s question but LINK IT BACK TO THE SITUATION”
    E.g: The video shows a golf ball flying past a lamp
    Sample question: Did the golfer successfully hit the lamp
    Sample answer: As the ball flew past the lamp, the golfer did NOT successfully hit the lamp.
    
    For simple questions, if the video shows details that go beyong the question, try to incorporate the details in your answer, then conclude your answer after thinking about the details. For example:
    The video shows a real panda and a human dressed as a panda.
    Sample question: How many pandas are in the video?
    Sample answer: There is one real panda and one human dressed like a panda, so therefore, only one panda.
    
    Question:{question}

    Your {iteration} answers: """
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

    async def delete_file(self, file_name_or_uri: str) -> None:
        """
        Delete a file you previously uploaded via Gemini API.

        Examples
        --------
        await g.delete_file("files/abc123def")        # from list() call
        await g.delete_file(uploaded_file.uri)        # after upload
        """
        # extract canonical file name if a full URI is passed
        file_name = file_name_or_uri.split("/")[-1]
        file_path = f"files/{file_name}"

        try:
            await self.aio.files.delete(name=file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"File delete failed for {file_path}: {e}")
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
            step1=use_context(q[1],iteration_in_prompt,q[0])
            if iterate_prompt:
                # build prompt with iterate_prompt to get candidates
                contents_multi = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=video_uri, mime_type="video/*"),
                            # types.Part.from_text(text=f"{q[2]} {iterate_prompt}"),
                            types.Part.from_text(text=step1),
                        ],
                    )
                ]
                try:
                    multi = await self._stream_text(contents_multi, temperature)
                except Exception as e:
                    print(f"Gemini API error in step 1: {e}")
                    final_answers.append("Error")
                    raise
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
                    raise
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
                    raise


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
        await self.delete_file(uri)
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
            temperature=0,
            wait_time=10
        )
        print("Final Answer:", answers)
    asyncio.run(_demo())