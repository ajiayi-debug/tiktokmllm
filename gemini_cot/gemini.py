import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()


class Gemini:
    def __init__(self, api_key=None, model="gemini-2.5-pro-preview-03-25"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def generate_from_video(self, video_uri, questions, temperature=0, num_repeats=1, wait_time=30):
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
                    temperature=temperature,
                    response_mime_type="text/plain",
                )

                try:
                    output = ""
                    for chunk in self.client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if chunk.text is not None:
                            output += chunk.text
                        else:
                            output += ""
                    predictions.append(output.strip())
                except Exception as e:
                    print(f"Gemini API error: {e}")
                    raise e
                time.sleep(wait_time)
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

    def wait_until_file_active(self, uploaded_file, timeout=30, poll_interval=2):
        file_id = uploaded_file.name  # e.g., "files/rwlgemkra6vn"
        start_time = time.time()
        while time.time() - start_time < timeout:
            file = self.client.files.get(name=file_id)
            state = getattr(file, "state", "UNKNOWN")
            if state == "ACTIVE":
                print(f"File {file_id} is ACTIVE.")
                return file
            print(f"Waiting for file {file_id} to become ACTIVE (current: {state})...")
            time.sleep(poll_interval)
        raise TimeoutError(f"File {file_id} did not become ACTIVE within {timeout} seconds.")



    def generate_from_uploaded_video_file(self, file_path, question, temperature=0):
        print(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        try:
            uploaded_file = self.client.files.upload(file=file_path)
            uploaded_file = self.wait_until_file_active(uploaded_file)
            print(f"Uploaded file: {uploaded_file.uri}")

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                        types.Part.from_text(text=question),
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="text/plain",
            )

            output = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    output += chunk.text
                else:
                    output += ""
            return output.strip()

        except Exception as e:
            print(f"Error generating from video: {e}")
            return "Error"


    def generate(self, input_text, temperature=0.7):
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=input_text),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="text/plain",
        )

        try:
            output = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:
                    output += chunk.text
                else:
                    output += ""
            return output.strip()
        except Exception as e:
            print(f"Gemini text generation error: {e}")
            return "Error"




if __name__ == "__main__":
    gemini = Gemini()
    gemini.generate_from_youtube(
        video_url="https://youtu.be/sj81PWrerDk",
        input_text="What is the difference between the action of the last person in the video and the actions of the first two people?"
    )

