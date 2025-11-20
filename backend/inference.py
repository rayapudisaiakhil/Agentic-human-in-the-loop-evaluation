"""
Model inference module using Hugging Face transformers.
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from backend.models import SentimentLabel, PredictionStatus

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CONFIDENCE_THRESHOLD = 0.8
MAX_LENGTH = 512


class ModelInference:
    """Handles model loading and inference operations."""

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the inference module.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of model components."""
        if not self._initialized:
            logger.info(f"Loading model: {self.model_name}")
            try:
                # Use pipeline for simpler interface
                # Determine device: CUDA > MPS > CPU
                if torch.cuda.is_available():
                    device = 0
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = -1

                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=device,
                    max_length=MAX_LENGTH,
                    truncation=True,
                )

                # Also load tokenizer and model for more control if needed
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )

                self._initialized = True
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, text: str) -> Tuple[str, float, str]:
        """
        Make a prediction on a single text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (label, confidence, status)
        """
        self._lazy_init()

        try:
            # Get prediction from pipeline
            results = self.pipeline(text, top_k=2)  # Get top 2 for confidence analysis

            # Extract top prediction
            top_result = results[0]
            label = self._map_label(top_result["label"])
            confidence = top_result["score"]

            # Determine status based on confidence
            status = self._determine_status(confidence)

            logger.debug(
                f"Prediction - Text: '{text[:50]}...', "
                f"Label: {label}, Confidence: {confidence:.3f}, Status: {status}"
            )

            return label, confidence, status

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return low confidence negative as fallback
            return SentimentLabel.NEGATIVE, 0.5, PredictionStatus.PENDING

    def batch_predict(self, texts: List[str]) -> List[Tuple[str, float, str]]:
        """
        Make predictions on multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of (label, confidence, status) tuples
        """
        self._lazy_init()

        results = []
        for text in texts:
            results.append(self.predict(text))

        return results

    @staticmethod
    def _map_label(raw_label: str) -> str:
        """
        Map model output labels to our schema.

        Args:
            raw_label: Raw label from model

        Returns:
            Mapped label (POSITIVE or NEGATIVE)
        """
        # DistilBERT SST-2 outputs POSITIVE/NEGATIVE directly
        # But we normalize to ensure consistency
        label_map = {
            "POSITIVE": SentimentLabel.POSITIVE,
            "NEGATIVE": SentimentLabel.NEGATIVE,
            "LABEL_0": SentimentLabel.NEGATIVE,  # Some models use numeric labels
            "LABEL_1": SentimentLabel.POSITIVE,
        }

        return label_map.get(raw_label.upper(), SentimentLabel.NEGATIVE)

    @staticmethod
    def _determine_status(confidence: float) -> str:
        """
        Determine prediction status based on confidence.

        Args:
            confidence: Model confidence score (0-1)

        Returns:
            Status (AUTO_ACCEPTED or PENDING)
        """
        if confidence > CONFIDENCE_THRESHOLD:
            return PredictionStatus.AUTO_ACCEPTED
        else:
            return PredictionStatus.PENDING

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        self._lazy_init()

        return {
            "model_name": self.model_name,
            "loaded": self._initialized,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "max_length": MAX_LENGTH,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    def compute_embeddings(self, text: str) -> Optional[torch.Tensor]:
        """
        Compute embeddings for a text (useful for similarity analysis).

        Args:
            text: Input text

        Returns:
            Embedding tensor or None if error
        """
        self._lazy_init()

        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the [CLS] token embedding from last hidden layer
                embeddings = outputs.hidden_states[-1][:, 0, :]

            return embeddings

        except Exception as e:
            logger.error(f"Embedding computation error: {e}")
            return None

    def analyze_confidence_distribution(
        self, texts: List[str]
    ) -> Dict[str, List[float]]:
        """
        Analyze confidence distribution across multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Dictionary with confidence statistics
        """
        self._lazy_init()

        confidences = []
        for text in texts:
            _, conf, _ = self.predict(text)
            confidences.append(conf)

        return {
            "all_confidences": confidences,
            "high_confidence": [c for c in confidences if c > CONFIDENCE_THRESHOLD],
            "low_confidence": [c for c in confidences if c <= CONFIDENCE_THRESHOLD],
            "mean": sum(confidences) / len(confidences) if confidences else 0,
            "min": min(confidences) if confidences else 0,
            "max": max(confidences) if confidences else 0,
        }


# Global singleton instance
_model_instance: Optional[ModelInference] = None


def get_model() -> ModelInference:
    """
    Get or create the global model instance.

    Returns:
        ModelInference instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelInference()
    return _model_instance


def warmup_model():
    """Warm up the model by loading it and making a test prediction."""
    model = get_model()
    model._lazy_init()
    # Make a dummy prediction to ensure everything works
    model.predict("Test warmup prediction")
    logger.info("Model warmed up successfully")