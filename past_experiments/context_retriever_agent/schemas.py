"""Pydantic schemas for the Context Retriever Agent."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ContextOutput(BaseModel):
    """Schema for context retrieval output."""
    context: str = Field(description="Extracted context information from the video")
    is_contextual: bool = Field(description="Whether the question is aligned with video content")
    explanation: str = Field(description="Explanation of the alignment analysis")
    corrected_question: Optional[str] = Field(None, description="Corrected version of the question if needed")


class ContextPrediction(BaseModel):
    """Schema for a single prediction result."""
    qid: str = Field(description="Question ID")
    prediction: Dict[str, Any] = Field(description="The context prediction data")


class ContextBatchResult(BaseModel):
    """Schema for a batch of prediction results."""
    predictions: List[ContextPrediction] = Field(description="List of predictions")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors")
    
    def add_prediction(self, qid: str, prediction: Dict[str, Any]):
        """Add a prediction to the batch result."""
        self.predictions.append(ContextPrediction(qid=qid, prediction=prediction))
    
    def add_error(self, qid: str, error_message: str):
        """Add an error to the batch result."""
        self.errors.append({
            "qid": qid,
            "error": error_message
        })
        self.add_prediction(qid, {
            "context": "Error",
            "is_contextual": False,
            "explanation": f"Error during processing: {error_message}",
            "corrected_question": None
        }) 