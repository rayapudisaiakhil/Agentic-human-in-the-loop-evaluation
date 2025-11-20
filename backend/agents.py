"""
Multi-agent validation module for Phase 2 extension.

This module implements a multi-agent voting system where multiple
models or heuristic agents can vote on predictions, with disagreements
being escalated to human review.
"""

import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import Counter

from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of validation agents."""
    MODEL = "model"
    HEURISTIC = "heuristic"
    VARIATION = "variation"


@dataclass
class AgentPrediction:
    """Container for agent predictions."""
    agent_id: str
    agent_type: AgentType
    label: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict] = None


class ValidationAgent(ABC):
    """Abstract base class for validation agents."""

    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type

    @abstractmethod
    def predict(self, text: str) -> AgentPrediction:
        """
        Make a prediction on input text.

        Args:
            text: Input text to classify

        Returns:
            AgentPrediction with label and confidence
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        """Get agent information and configuration."""
        pass


class PrimaryModelAgent(ValidationAgent):
    """
    Primary agent using the main DistilBERT model.
    """

    def __init__(self):
        super().__init__("primary_model", AgentType.MODEL)
        self.pipeline = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of the model."""
        if not self._initialized:
            # Determine device
            if torch.cuda.is_available():
                device = 0
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1

            self.pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
            self._initialized = True

    def predict(self, text: str) -> AgentPrediction:
        """Make prediction using primary model."""
        self._lazy_init()

        result = self.pipeline(text)[0]

        return AgentPrediction(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            label=result["label"],
            confidence=result["score"],
            reasoning="Primary DistilBERT model prediction"
        )

    def get_info(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "description": "Primary sentiment analysis model"
        }


class VariationAgent(ValidationAgent):
    """
    Agent with slight variations to simulate model diversity.
    """

    def __init__(self, variation_id: int, noise_factor: float = 0.05):
        super().__init__(f"variation_{variation_id}", AgentType.VARIATION)
        self.variation_id = variation_id
        self.noise_factor = noise_factor
        self.pipeline = None
        self._initialized = False
        # Set different seed for each variation
        random.seed(variation_id * 42)

    def _lazy_init(self):
        """Lazy initialization of the model."""
        if not self._initialized:
            # Determine device
            if torch.cuda.is_available():
                device = 0
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1

            self.pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
            self._initialized = True

    def predict(self, text: str) -> AgentPrediction:
        """Make prediction with added noise/variation."""
        self._lazy_init()

        result = self.pipeline(text)[0]

        # Add noise to confidence
        original_confidence = result["score"]
        noise = random.uniform(-self.noise_factor, self.noise_factor)
        confidence = max(0.0, min(1.0, original_confidence + noise))

        # Occasionally flip label if confidence is low
        label = result["label"]
        if confidence < 0.6 and random.random() < 0.2:
            label = "NEGATIVE" if label == "POSITIVE" else "POSITIVE"
            confidence = 1.0 - confidence

        return AgentPrediction(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            label=label,
            confidence=confidence,
            reasoning=f"Variation agent {self.variation_id} with {self.noise_factor} noise",
            metadata={"original_confidence": original_confidence, "noise": noise}
        )

    def get_info(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "variation_id": self.variation_id,
            "noise_factor": self.noise_factor,
            "description": f"Model variation with controlled noise"
        }


class HeuristicAgent(ValidationAgent):
    """
    Rule-based heuristic agent for validation.
    """

    def __init__(self):
        super().__init__("heuristic_v1", AgentType.HEURISTIC)
        # Define positive and negative keywords
        self.positive_keywords = [
            "excellent", "amazing", "wonderful", "fantastic", "great",
            "love", "perfect", "best", "awesome", "incredible", "outstanding",
            "brilliant", "superb", "exceptional", "phenomenal"
        ]
        self.negative_keywords = [
            "terrible", "horrible", "awful", "worst", "hate", "disgusting",
            "pathetic", "poor", "bad", "disappointing", "failure", "disaster",
            "waste", "useless", "garbage"
        ]

    def predict(self, text: str) -> AgentPrediction:
        """Apply heuristic rules for sentiment prediction."""
        text_lower = text.lower()

        # Count positive and negative keywords
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        # Determine label and confidence based on counts
        if positive_count > negative_count:
            label = "POSITIVE"
            confidence = min(0.95, 0.6 + positive_count * 0.1)
            reasoning = f"Found {positive_count} positive keywords"
        elif negative_count > positive_count:
            label = "NEGATIVE"
            confidence = min(0.95, 0.6 + negative_count * 0.1)
            reasoning = f"Found {negative_count} negative keywords"
        else:
            # Neutral or unclear
            label = "POSITIVE" if len(text) < 50 else "NEGATIVE"
            confidence = 0.5
            reasoning = "No clear sentiment keywords found"

        # Adjust confidence based on text characteristics
        if "!" in text:
            confidence = min(1.0, confidence + 0.05)
        if "?" in text:
            confidence = max(0.0, confidence - 0.05)

        return AgentPrediction(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "positive_keywords": positive_count,
                "negative_keywords": negative_count
            }
        )

    def get_info(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "description": "Rule-based heuristic sentiment analyzer",
            "num_positive_keywords": len(self.positive_keywords),
            "num_negative_keywords": len(self.negative_keywords)
        }


class MultiAgentValidator:
    """
    Coordinator for multi-agent validation system.
    """

    def __init__(self, agents: List[ValidationAgent], agreement_threshold: float = 0.66):
        self.agents = agents
        self.agreement_threshold = agreement_threshold
        self.validation_count = 0
        self.disagreement_count = 0

    def validate(self, text: str) -> Tuple[str, float, str, List[AgentPrediction], float]:
        """
        Run multi-agent validation on text.

        Args:
            text: Input text to validate

        Returns:
            Tuple of (consensus_label, consensus_confidence, status, all_predictions, agreement_ratio)
        """
        # Collect predictions from all agents
        predictions = []
        for agent in self.agents:
            try:
                prediction = agent.predict(text)
                predictions.append(prediction)
                logger.debug(f"Agent {agent.agent_id}: {prediction.label} ({prediction.confidence:.2f})")
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} failed: {e}")

        if not predictions:
            # Fallback if all agents fail
            return "NEGATIVE", 0.5, "PENDING", [], 0.0

        # Calculate consensus
        labels = [p.label for p in predictions]
        label_counts = Counter(labels)
        consensus_label = label_counts.most_common(1)[0][0]

        # Calculate agreement ratio
        agreement_ratio = label_counts[consensus_label] / len(predictions)

        # Calculate average confidence for consensus label
        consensus_confidences = [p.confidence for p in predictions if p.label == consensus_label]
        consensus_confidence = sum(consensus_confidences) / len(consensus_confidences)

        # Determine status based on agreement
        if agreement_ratio >= self.agreement_threshold and consensus_confidence > 0.8:
            status = "AUTO_ACCEPTED"
        elif agreement_ratio < self.agreement_threshold:
            status = "NEEDS_HUMAN"  # Disagreement among agents
            self.disagreement_count += 1
        else:
            status = "PENDING"  # Low confidence but agreement

        self.validation_count += 1

        logger.info(
            f"Multi-agent validation - Label: {consensus_label}, "
            f"Confidence: {consensus_confidence:.2f}, "
            f"Agreement: {agreement_ratio:.2f}, Status: {status}"
        )

        return consensus_label, consensus_confidence, status, predictions, agreement_ratio

    def calculate_disagreement_score(self, predictions: List[AgentPrediction]) -> float:
        """
        Calculate disagreement score among agents.

        Args:
            predictions: List of agent predictions

        Returns:
            Disagreement score (0 = full agreement, 1 = max disagreement)
        """
        if len(predictions) <= 1:
            return 0.0

        # Calculate label disagreement
        labels = [p.label for p in predictions]
        label_counts = Counter(labels)

        # If all agree on label, disagreement is 0
        if len(label_counts) == 1:
            return 0.0

        # Calculate entropy-based disagreement
        total = len(labels)
        entropy = 0.0
        for count in label_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * (p if p == 0 else p * (1 if p == 1 else p.bit_length()))

        # Normalize to 0-1 range
        max_entropy = -(0.5 * -1 + 0.5 * -1)  # Maximum entropy for binary classification
        disagreement = min(1.0, entropy / max_entropy if max_entropy > 0 else 0)

        return disagreement

    def get_agent_stats(self) -> Dict[str, Dict]:
        """
        Get performance statistics for the validation system.

        Returns:
            Dictionary of validation metrics
        """
        disagreement_rate = (
            self.disagreement_count / self.validation_count
            if self.validation_count > 0 else 0.0
        )

        return {
            "total_validations": self.validation_count,
            "disagreements": self.disagreement_count,
            "disagreement_rate": disagreement_rate,
            "num_agents": len(self.agents),
            "agreement_threshold": self.agreement_threshold,
            "agents": [agent.get_info() for agent in self.agents]
        }


# Factory functions for creating agents

def create_default_agents() -> List[ValidationAgent]:
    """
    Create default set of validation agents.

    Returns:
        List of configured agents
    """
    agents = [
        PrimaryModelAgent(),
        VariationAgent(variation_id=1, noise_factor=0.05),
        VariationAgent(variation_id=2, noise_factor=0.08),
        HeuristicAgent()
    ]

    logger.info(f"Created {len(agents)} default agents")
    return agents


def setup_multi_agent_validation(agreement_threshold: float = 0.66) -> Optional[MultiAgentValidator]:
    """
    Set up multi-agent validation system.

    Args:
        agreement_threshold: Minimum agreement ratio for consensus

    Returns:
        Configured MultiAgentValidator or None if setup fails
    """
    try:
        agents = create_default_agents()
        if not agents:
            logger.warning("No agents available for multi-agent validation")
            return None

        validator = MultiAgentValidator(agents, agreement_threshold)
        logger.info(f"Multi-agent validation system initialized with {len(agents)} agents")
        return validator

    except Exception as e:
        logger.error(f"Failed to set up multi-agent validation: {e}")
        return None


# Export key components
__all__ = [
    "ValidationAgent",
    "PrimaryModelAgent",
    "VariationAgent",
    "HeuristicAgent",
    "MultiAgentValidator",
    "AgentPrediction",
    "AgentType",
    "create_default_agents",
    "setup_multi_agent_validation",
]