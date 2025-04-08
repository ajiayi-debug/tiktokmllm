import os
import cv2
import base64
import asyncio
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY_US2"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_US2"),
    api_version="2024-11-20"
)

async def gpt4o_vision_async(pil_image, prompt="Describe this frame in JSON"):
    """Send image and prompt to GPT-4o vision model (Azure OpenAI)."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    b64_img = base64.b64encode(buffered.getvalue()).decode()

    messages = [
        {
            "role": "system",
            "content": "You are an Assistant to output JSON with key 'response'."
        },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
                { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{b64_img}" } }
            ]
        }
    ]

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=999,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content.strip()

def extract_frames(video_path, frame_interval=1):
    """Extract frames at 1-second intervals."""
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_gap = int(fps * frame_interval)
    frames = []
    count = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_gap == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            frames.append(pil_img)
        count += 1

    vidcap.release()
    return frames

async def main(video_path):
    frames = extract_frames(video_path)
    for i, frame in enumerate(frames):
        print(f"\n--- Frame {i+1} ---")
        try:
            result = await gpt4o_vision_async(frame, prompt="What is happening in this frame? Respond in JSON with key 'response'.")
            print(result)
        except Exception as e:
            print(f"Error on frame {i+1}: {e}")

if __name__ == "__main__":
    video_file = "videos/2435100235.mp4"  # Change this to your video path
    asyncio.run(main(video_file))
