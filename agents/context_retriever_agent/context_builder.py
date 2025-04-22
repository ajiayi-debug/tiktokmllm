"""Context builder for the Context Retriever Agent."""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from collections import defaultdict

from agents.cot_agent.gemini import GeminiAsync
from agents.context_retriever_agent.prompts import combined_context_prompt
from agents.context_retriever_agent.utils import parse_context_response

logger = logging.getLogger("context_retriever.builder")


class ContextBuilder:
    """Class to build context for videos and questions."""
    
    def __init__(
        self, 
        gemini: Optional[GeminiAsync] = None,
        temperature: float = 0.0,
        wait_time: int = 30,
        video_dir: str = "Benchmark-AllVideos-HQ-Encoded-challenge"
    ):
        """
        Initialize the context builder.
        
        Args:
            gemini: Optional GeminiAsync instance (will create one if not provided)
            temperature: Temperature for generation
            wait_time: Wait time between API calls
            video_dir: Directory containing local video files
        """
        self.gemini = gemini or GeminiAsync()
        self.temperature = temperature
        self.wait_time = wait_time
        self.video_dir = video_dir
    
    async def process_video_question(
        self,
        video_url: str,
        question: str,
        video_id: str = "",
        use_local_file: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single video question to extract context and validate.
        
        Args:
            video_url: URL of the video
            question: Question to process
            video_id: ID of the video (for local file lookup)
            use_local_file: Whether to try using a local video file
            
        Returns:
            Dict containing context data
        """
        try:
            prompt = combined_context_prompt(question)
            
            # Try local file if requested
            local_video_path = None
            if use_local_file and video_id:
                import os
                local_video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                if not os.path.exists(local_video_path):
                    logger.warning(f"Local video file not found: {local_video_path}")
                    local_video_path = None
            
            # Process with Gemini
            if use_local_file and local_video_path:
                logger.info(f"Using local video file: {local_video_path}")
                response = await self.gemini.generate_from_uploaded_video_file(
                    local_video_path,
                    prompt,
                    temperature=self.temperature,
                    wait_time=self.wait_time
                )
            else:
                logger.info(f"Using video URL: {video_url}")
                response = (await self.gemini.generate_from_video(
                    video_url,
                    [prompt],
                    temperature=self.temperature,
                    wait_time=self.wait_time
                ))[0]
            
            # Parse the response
            context_data = parse_context_response(response)
            logger.info(f"Context extracted for question about video {video_id or video_url}")
            
            return context_data
            
        except Exception as e:
            logger.error(f"Error processing video {video_id or video_url}: {e}")
            return {
                "context": "Error",
                "is_contextual": False,
                "explanation": f"Error during processing: {str(e)}",
                "corrected_question": None
            }
    
    async def batch_process_video_questions(
        self,
        video_url: str,
        questions: List[Dict[str, Any]],
        use_local_file: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions for the same video in batch.
        
        Args:
            video_url: URL of the video
            questions: List of question dictionaries with 'qid', 'question', and 'video_id'
            use_local_file: Whether to try using a local video file
            
        Returns:
            List of tuples with (qid, context_data)
        """
        results = []
        
        for q in questions:
            qid = q.get("qid", "UNKNOWN")
            question = q.get("question", "")
            video_id = q.get("video_id", "")
            
            try:
                context_data = await self.process_video_question(
                    video_url=video_url,
                    question=question,
                    video_id=video_id,
                    use_local_file=use_local_file
                )
                
                results.append((qid, context_data))
                
            except Exception as e:
                logger.error(f"Error in batch processing for qid {qid}: {e}")
                results.append((
                    qid, 
                    {
                        "context": "Error",
                        "is_contextual": False,
                        "explanation": f"Error during batch processing: {str(e)}",
                        "corrected_question": None
                    }
                ))
        
        return results


async def get_context_for_questions(
    video_questions: Dict[str, List[Dict[str, Any]]],
    temperature: float = 0.0,
    wait_time: int = 30,
    video_dir: str = "Benchmark-AllVideos-HQ-Encoded-challenge",
    use_local_files: bool = False,
    gemini: Optional[GeminiAsync] = None
) -> List[Dict[str, Any]]:
    """
    Process all video questions to get context.
    
    Args:
        video_questions: Dictionary mapping video URLs to lists of question dicts
        temperature: Temperature for generation
        wait_time: Wait time between API calls
        video_dir: Directory containing local video files
        use_local_files: Whether to try using local video files
        gemini: Optional GeminiAsync instance
        
    Returns:
        List of prediction dictionaries
    """
    builder = ContextBuilder(
        gemini=gemini,
        temperature=temperature,
        wait_time=wait_time,
        video_dir=video_dir
    )
    
    predictions = []
    
    for video_url, questions in video_questions.items():
        logger.info(f"Processing video: {video_url} ({len(questions)} questions)")
        
        results = await builder.batch_process_video_questions(
            video_url=video_url,
            questions=questions,
            use_local_file=use_local_files
        )
        
        for qid, context_data in results:
            predictions.append({
                "qid": qid,
                "prediction": context_data
            })
    
    return predictions 