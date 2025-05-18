from agents.cot_agent.gemini import GeminiAsync
import asyncio


class VideoProcessingService:
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key

    async def get_gemini_alone_response(self, video_url: str, user_question: str) -> tuple[str, str | None]:
        try:
            if not self.gemini_api_key:
                return ("Error (Gemini Alone): GEMINI_API_KEY not provided to service.", None)
            gemini_client = GeminiAsync(api_key=self.gemini_api_key)
            formatted_question_list = [[user_question, "", user_question]]
            
            response_list = await gemini_client.generate_from_video(
                video_uri=video_url,
                questions=formatted_question_list,
                iterate_prompt=False,
                temperature=0.0
            )
            if response_list and len(response_list) > 0:
                return response_list[0]
            else:
                return ("Error: No response from Gemini.", None)
        except Exception as e:
            return (f"Error (Gemini Alone): {str(e)}", None)

    async def get_agentic_workflow_response(self, video_url: str, user_question: str, num_candidates: int) -> tuple[str, str | None]:
        try:
            if not self.gemini_api_key:
                return ("Error (Agentic Workflow): GEMINI_API_KEY not provided to service.", None)
            gemini_client = GeminiAsync(api_key=self.gemini_api_key)
            formatted_question_list = [[user_question, "", user_question]]

            response_list = await gemini_client.generate_from_video(
                video_uri=video_url,
                questions=formatted_question_list,
                iterate_prompt=True,
                iteration_in_prompt=num_candidates,
                temperature=0.0 
            )
            if response_list and len(response_list) > 0:
                return response_list[0]
            else:
                return ("Error: No response from Agentic Workflow.", None)
        except Exception as e:
            return (f"Error (Agentic Workflow): {str(e)}", None) 