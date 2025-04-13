import os
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLVideoQA:
    def __init__(self, model_path='OpenGVLab/InternVL2_5-8B-MPO', image_size=448):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        self.image_size = image_size
        self.transform = self._build_transform()

    def _build_transform(self):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect = ratio[0] / ratio[1]
            diff = abs(aspect_ratio - target_aspect)
            if diff < best_ratio_diff or (diff == best_ratio_diff and area > 0.5 * self.image_size ** 2 * ratio[0] * ratio[1]):
                best_ratio_diff = diff
                best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, use_thumbnail=False):
        width, height = image.size
        aspect_ratio = width / height
        target_ratios = sorted({
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        }, key=lambda x: x[0] * x[1])
        target = self._find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height)

        target_w, target_h = self.image_size * target[0], self.image_size * target[1]
        blocks = target[0] * target[1]
        resized = image.resize((target_w, target_h))
        processed_images = [
            resized.crop((
                (i % (target_w // self.image_size)) * self.image_size,
                (i // (target_w // self.image_size)) * self.image_size,
                ((i % (target_w // self.image_size)) + 1) * self.image_size,
                ((i // (target_w // self.image_size)) + 1) * self.image_size
            ))
            for i in range(blocks)
        ]
        if use_thumbnail and blocks != 1:
            processed_images.append(image.resize((self.image_size, self.image_size)))
        return processed_images

    def _get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        start = max(first_idx, round((bound[0] if bound else -1e5) * fps))
        end = min(round((bound[1] if bound else 1e5) * fps), max_frame)
        seg_size = (end - start) / num_segments
        return np.array([
            int(start + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

    def _get_num_frames(self, duration):
        local_num_frames = 4
        segments = max(1, duration // local_num_frames)
        frames = local_num_frames * segments
        return min(512, max(128, frames))

    def _load_video(self, video_path, bound=None, num_segments=32, max_num=1, get_frame_by_duration=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        duration = max_frame / fps

        if get_frame_by_duration:
            num_segments = min(self._get_num_frames(duration), 32)

        indices = self._get_index(bound, fps, max_frame, num_segments=num_segments)
        pixel_batches, patch_counts = [], []

        for idx in indices:
            frame = Image.fromarray(vr[idx].asnumpy()).convert('RGB')
            tiles = self._dynamic_preprocess(frame, use_thumbnail=True, max_num=max_num)
            pixels = torch.stack([self.transform(tile) for tile in tiles])
            pixel_batches.append(pixels)
            patch_counts.append(pixels.shape[0])

        pixel_values = torch.cat(pixel_batches).to(dtype=next(self.model.parameters()).dtype, device=self.model.device)
        return pixel_values, patch_counts

    def ask(self, video_path, questions, your_prompt=None, num_repeats=3):
        pixel_values, patch_counts = self._load_video(video_path)
        frame_prompt = "".join([f"Frame{i+1}: <image>\n" for i in range(len(patch_counts))])
        all_responses = []

        for q in tqdm(questions, desc=f"QnA on {video_path}"):
            if your_prompt:
                prompt = frame_prompt + your_prompt + q
            else:
                prompt = frame_prompt + \
                        "Your task is to answer the question below. Give step by step reasoning before you answer, " \
                        "and when you're ready to answer, please use the format 'Final answer: ..'" + q
            responses = []
            for _ in tqdm(range(num_repeats), leave=False, desc=" Repeats"):
                res, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    self.generation_config,
                    num_patches_list=patch_counts,
                    history=None,
                    return_history=True
                )
                responses.append(res)
            all_responses.append(responses)

        return all_responses
    

# Sample usage:
# from internmodel import InternVLVideoQA

# videoqa = InternVLVideoQA()
# video_path = "Benchmark-AllVideos-HQ-Encoded-challenge/example_video.mp4"
# questions = [
#     "What is the person doing in the video?",
#     "Describe the background and setting."
# ]

# results = videoqa.ask(video_path, questions)
# for i, res in enumerate(results):
#     print(f"Q{i+1}: {questions[i]}")
#     for r in res:
#         print("-", r)