import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVideoVL:
    def __init__(
        self,
        model_name="FanqingM/MM-Eureka-Qwen-7B",
        device="cuda",
        attn_impl="flash_attention_2",
        torch_dtype=torch.bfloat16,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    ):
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def ask(self, video_path: str, question: str, prompt='', fps: float = 2.0, max_pixels: int = 360 * 420, max_new_tokens: int = 500):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": max_pixels,
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": prompt+question,
                    },
                ],
            }
        ]

        # Prepare text prompt
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process visual inputs
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # Tokenize everything
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.device)

        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return response[0] if response else None
    


#sample usage:
# from qwenmodel import QwenVideoVL

# qwen_vl = QwenVideoVL()
# video_path = "Benchmark-AllVideos-HQ-Encoded-challenge/example.mp4"
# question = "What is happening in this video?"

# answer = qwen_vl.ask(video_path,question)
# print("Answer:", answer)