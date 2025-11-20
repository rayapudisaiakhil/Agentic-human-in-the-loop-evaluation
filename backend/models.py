"""
Pydantic models for request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionStatus(str, Enum):
    """Status of a prediction in the pipeline."""
    PENDING = "PENDING"
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    REVIEWED = "REVIEWED"


class SentimentLabel(str, Enum):
    """Sentiment labels for classification."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    text: str = Field(..., min_length=1, max_length=1000)
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    id: int
    text: str
    model_label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: PredictionStatus
    requires_review: bool
    timestamp: str


class PendingItem(BaseModel):
    """Model for pending review items."""
    id: int
    text: str
    model_label: SentimentLabel
    confidence: float
    timestamp: str


class PendingResponse(BaseModel):
    """Response model for pending endpoint."""
    count: int
    items: List[PendingItem]


class FeedbackRequest(BaseModel):
    """Request model for feedback submission."""
    id: int = Field(..., gt=0)
    human_label: SentimentLabel
    reviewer_notes: Optional[str] = Field(None, max_length=500)

    @field_validator("reviewer_notes")
    @classmethod
    def clean_notes(cls, v: Optional[str]) -> Optional[str]:
        """Clean up reviewer notes."""
        if v:
            return v.strip()
        return v


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    message: str
    sample_id: int
    agreement: bool  # Whether human agreed with model


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    total_predictions: int
    status_breakdown: Dict[str, int]
    agreement_rate: float
    total_reviewed: int
    confidence_stats: Dict[str, Dict[str, float]]
    label_distribution: Dict[str, int]
    recent_reviews: List[Dict[str, Any]]
    system_health: Dict[str, Any]


class BatchInferenceRequest(BaseModel):
    """Request model for batch inference."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None, description="Batch identifier")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate all texts in batch."""
        cleaned = []
        for text in v:
            if not text.strip():
                raise ValueError("Batch contains empty text")
            cleaned.append(text.strip())
        return cleaned


class BatchInferenceResponse(BaseModel):
    """Response model for batch inference."""
    batch_id: Optional[str]
    total_processed: int
    results: List[InferenceResponse]
    summary: Dict[str, int]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    database: bool
    model_loaded: bool
    uptime_seconds: float
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RetrainingRequest(BaseModel):
    """Request model for retraining endpoint (Phase 2 placeholder)."""
    use_feedback: bool = True
    epochs: int = Field(default=3, ge=1, le=10)
    learning_rate: float = Field(default=2e-5, gt=0, lt=1)


class RetrainingResponse(BaseModel):
    """Response model for retraining (Phase 2 placeholder)."""
    success: bool
    message: str
    samples_used: int
    new_accuracy: Optional[float] = None