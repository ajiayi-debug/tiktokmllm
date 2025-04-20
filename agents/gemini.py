import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import asyncio
from typing import List
from tqdm import tqdm
import re

load_dotenv()

model="gemini-2.5-pro-preview-03-25"

#model="gemini-2.5-flash-preview-04-17"


def generate_question_then_solve(question):
    prompt=f"""Generate questions to gather clues to solve the following question:
    {question}
    Then, solve each question you generated with reference to the video, then use the question answers to solve the main question. Output your final answer to the main question with the tag: 
    [Final answer: (answer to main question)]"""
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
        return resp.text.strip()

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
    ) -> List[str]:
        results: list[str] = []

        for q in tqdm(questions, desc="Answering questions", unit="q", leave=False):
            full_q = generate_question_then_solve(q)

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
                multi = await self._stream_text(contents, temperature)
            except Exception as e:
                print("Gemini API error during multi‑answer gen:", e)
                results.append("Error")
                continue

            print("\nQ:", full_q)
            print("Step‑1 answers:\n", multi, "\n")

            # pull out [Final answer: ... ]
            pattern = r"\[\s*Final\s+Answer:\s*(.*?)\s*\]"   # allow spaces, any case
            match   = re.search(pattern, multi, flags=re.DOTALL | re.IGNORECASE)

            if match:
                answer = match.group(1).strip()
                results.append(answer)          # ← append extracted answer
                print("Extracted answer:", answer)
            else:
                results.append(multi)           # fallback: keep whole text

            await asyncio.sleep(wait_time)
        return results[0]                          # ← return the correct list

    async def generate_from_uploaded_video_file(
        self,
        file_path: str,
        question: str,
        *,
        temperature: float = 0.0,
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
            wait_time=wait_time
        )
        await self.delete_file(uri)
        return answers

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
            temperature=0,
        )
        print("\nAnswer:", ans)

    asyncio.run(_demo())
