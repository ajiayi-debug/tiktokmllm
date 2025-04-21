import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import asyncio
from typing import List
from tqdm import tqdm

load_dotenv()

model="gemini-2.5-pro-preview-03-25"

#model="gemini-2.5-flash-preview-04-17"


def choose_best_answer_prompt(question,output, number_of_iterations):
    prompt=f"""Based on the {number_of_iterations} answers and the video, determine the best answer to the question: {question}
    The best answer is either decided based on the majority answers (most common answer) or the best chain of thought thinking. 
    The best answer can also be a combination of answers from the {number_of_iterations} answers. Just answer the question and dont explain why you chose that answer among the {number_of_iterations} answer.
    The {number_of_iterations} answers:
    {output}"""
    return prompt

class GeminiAsync:
    """Async wrapper around Google Gen‑AI Video QA pipeline (two‑step logic)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_preview: str = "gemini-2.5-pro-preview-03-25",
        model_flash:  str = "gemini-2.5-flash-preview-04-17",
    ) -> None:
        self.client        = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.aio           = self.client.aio
        self.model_preview = model_preview
        self.model_flash   = model_flash

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    async def _stream_text(
        self,
        contents: list[types.Content],
        temperature: float,
        model_name: str,
    ) -> str:
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="text/plain",
        )
        resp = await self.aio.models.generate_content(
            model=model_name,        # <-- use override here
            contents=contents,
            config=cfg,
        )
        return resp.text.strip()
    
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
        questions: List[str],
        *,
        temperature: float = 0.0,
        wait_time: int = 30,
        number_of_iterations=1
    ) -> List[str]:
        """Two‑step generation directly from a YouTube/URI video."""
        results: list[str] = []
        for q in tqdm(questions, desc="Answering questions", unit="q", leave=False):
            answers=[]
            for n in tqdm(range(number_of_iterations), desc="Iterations", leave=False):
                full_q = q 

                # Step 1 – get n candidate answers
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=video_uri, mime_type="video/*"),
                            types.Part.from_text(text=full_q),
                        ],
                    )
                ]
                try:
                    multi = await self._stream_text(contents, temperature, self.model_preview)
                    answers.append(multi)
                except Exception as e:
                    print(f"Gemini API error during multi‑answer gen: {e}")
                    answers.append("Error")
                    continue

            print("\nQ:", full_q)          # shows the *exact* prompt sent
            print("Step‑1 answers:\n", answers, "\n")
            await asyncio.sleep(wait_time)
            # Step 2 – optional: have Gemini pick the best
            best_prompt = choose_best_answer_prompt(q, answers, number_of_iterations)
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
                best = await self._stream_text(contents_best, temperature, self.model_preview)
                results.append(best)
            except Exception as e:
                print(f"Gemini API error during best‑answer selection: {e}")
                results.append("Error")
        
            await asyncio.sleep(wait_time)
        return results

    async def generate_from_uploaded_video_file(
        self,
        file_path: str,
        question: str,
        *,
        temperature: float = 0.0,
        wait_time: int = 30,
         number_of_iterations=1
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
            number_of_iterations=number_of_iterations
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
        ans = await g.generate_from_video(
            "https://youtu.be/sj81PWrerDk",
            [
                "What is the difference between the action of the last person in the video and the actions of the first two people?"
            ],
            temperature=0.5,
            number_of_iterations=5
        )
        print("\nAnswer:", ans)

    asyncio.run(_demo())
