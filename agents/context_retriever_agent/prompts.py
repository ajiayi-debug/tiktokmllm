"""Dynamic prompt templates for the Context Retriever Agent."""

def context_extraction_prompt(question):
    """
    Generate a prompt for extracting contextual information from a video.
    
    Args:
        question: The original question about the video
        
    Returns:
        A formatted prompt string
    """
    return f"""Extract detailed contextual information from this video that would be useful for answering: "{question}"
    
Focus on identifying facts, entities, events, and relationships visible in the video.
Include timestamps for important events when possible.
Do not answer the question directly - only provide objective context that would help someone answer it.
Format your response as a structured summary of the video content relevant to the question."""


def question_validation_prompt(question, context):
    """
    Generate a prompt for validating if a question is factually aligned with video content.
    
    Args:
        question: The question to validate
        context: The previously extracted context
        
    Returns:
        A formatted prompt string
    """
    return f"""Based on the extracted context from the video and the video itself, determine if the following question is factually aligned with the video content:

Question: "{question}"

Extracted Context:
{context}

Analyze whether the question:
1. Contains accurate assumptions about what's in the video
2. References elements, people, or events that actually appear in the video
3. Can be reasonably answered based on the video content

Provide a structured JSON response with the following fields:
- "is_factually_aligned": boolean (true if the question is factually aligned, false otherwise)
- "explanation": brief explanation of your reasoning
- "corrected_question": a reformulation of the question that aligns with the video content (only if needed)"""


def combined_context_prompt(question):
    """
    Combined prompt for both context extraction and question validation in one call.
    
    Args:
        question: The question to analyze
        
    Returns:
        A formatted prompt string
    """
    return f"""Task: Analyze this video in relation to the question: "{question}"

1. First, extract detailed contextual information from the video relevant to this question.
2. Then, determine if the question is factually aligned with the video content.
3. If the question contains false assumptions or references things not in the video, provide a corrected version.

Return a JSON object with the following structure:
{{
  "context": "Detailed factual description of the video content relevant to the question",
  "is_contextual": true/false,
  "explanation": "Brief explanation of why the question is or isn't aligned with the video",
  "corrected_question": "A reformulated question that aligns with the actual video content (if needed)"
}}""" 