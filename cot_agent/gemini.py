import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class Gemini:
    def __init__(self, api_key=None, model="gemini-2.5-pro-preview-03-25"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def generate_from_video(self, video_uri, questions, num_repeats=1):
        predictions = []
        for _ in range(num_repeats):
            for question in questions:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=video_uri,
                                mime_type="video/*",
                            ),
                            types.Part.from_text(text=question),
                        ],
                    ),
                ]

                generate_content_config = types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="text/plain",
                )

                try:
                    output = ""
                    for chunk in self.client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        output += chunk.text
                    predictions.append(output.strip())
                except Exception as e:
                    print(f"Gemini API error: {e}")
                    raise e  # ← critical: re-raise so the caller (you) can handle it

        # If multiple repeats, we could return majority vote, but now just return one set
        return predictions

    def generate_from_youtube(self, video_url, input_text):
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=video_url,
                        mime_type="video/*",
                    ),
                    types.Part.from_text(text=input_text),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")


if __name__ == "__main__":
    gemini = Gemini()
    gemini.generate_from_youtube(
        video_url="https://youtu.be/sj81PWrerDk",
        input_text="What is the difference between the action of the last person in the video and the actions of the first two people?"
    )

