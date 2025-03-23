from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
import av
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import cv2

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

videoprocessor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
videomodel = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

#process video

"""Decoding video"""
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder and overlay frame numbers.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames with frame numbers (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            img = frame.to_ndarray(format="rgb24")

            # Draw frame number using OpenCV
            img = cv2.putText(img,
                              f"Frame {i}",
                              (10, 30),               # Position
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1,                      # Font scale
                              (255, 0, 0),            # Color (Blue in RGB)
                              2,                      # Thickness
                              cv2.LINE_AA)

            frames.append(img)

    return np.stack(frames)


"""Sample video frames"""
def video_frame(video_path, frames):
  container = av.open(video_path)

  # sample uniformly n chosen frames from the video (we can sample more for longer videos)
  total_frames = container.streams.video[0].frames
  indices = np.arange(0, total_frames, total_frames / frames).astype(int)
  clip = read_video_pyav(container, indices)
  return clip

"""To view video"""
def animate_video(decoded_video):
  fig = plt.figure()
  im = plt.imshow(decoded_video[0,:,:,:])
  plt.close() # this is required to not display the generated image

  def init():
      im.set_data(decoded_video[0,:,:,:])

  def animate(i):
      im.set_data(decoded_video[i,:,:,:])
      return im

  anim = animation.FuncAnimation(fig, animate, init_func=init, frames=decoded_video.shape[0],
                                interval=100)
  return HTML(anim.to_html5_video())


# insert video and question functions for LLaVA-NeXT-Video-DPO (7B)
"""To insert prompt format into video generator"""

def prompt_video(prompt):
  conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
                ],
        },
  ]
  Prompt=videoprocessor.apply_chat_template(conversation, add_generation_prompt=True)

  return Prompt

def process_inputs(Prompt, clip):
  inputs = videoprocessor([Prompt], videos=[clip], padding=True, return_tensors="pt").to(videomodel.device)
  return inputs


def output(inputs):
  generate_kwargs = {"max_new_tokens": 3000, "do_sample": True, "top_p": 0.9}

  output = videomodel.generate(**inputs, **generate_kwargs)
  generated_text = videoprocessor.batch_decode(output, skip_special_tokens=True)
  return generated_text

