"""
Utility functions and configuration for HITL pipeline.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Configuration constants
CONFIDENCE_THRESHOLD = 0.8
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DB_PATH = "hitl_demo.db"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "feedback_events.log"
GENERAL_LOG_FILE = LOG_DIR / "app.log"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA + Colors.BOLD,
    }

    def format(self, record):
        """Format log record with color."""
        log_color = self.COLORS.get(record.levelno, Colors.WHITE)
        record.levelname = f"{log_color}{record.levelname}{Colors.RESET}"
        record.msg = f"{log_color}{record.msg}{Colors.RESET}"
        return super().format(record)


def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        name: Logger name (None for root logger)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for general logs
    file_handler = logging.FileHandler(GENERAL_LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def setup_feedback_logger() -> logging.Logger:
    """
    Set up dedicated logger for feedback events.

    Returns:
        Configured feedback logger
    """
    feedback_logger = logging.getLogger("feedback")
    feedback_logger.setLevel(logging.INFO)
    feedback_logger.handlers = []

    # File handler for feedback events
    feedback_handler = logging.FileHandler(LOG_FILE)
    feedback_handler.setLevel(logging.INFO)
    feedback_formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    feedback_handler.setFormatter(feedback_formatter)
    feedback_logger.addHandler(feedback_handler)

    return feedback_logger


def log_feedback_event(
    sample_id: int,
    text: str,
    model_label: str,
    human_label: str,
    confidence: float,
    agreement: bool
):
    """
    Log a feedback event to dedicated log file.

    Args:
        sample_id: ID of the sample
        text: Sample text (truncated)
        model_label: Model's prediction
        human_label: Human's label
        confidence: Model confidence
        agreement: Whether human agreed with model
    """
    feedback_logger = logging.getLogger("feedback")
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "sample_id": sample_id,
        "text_preview": text[:100] if len(text) > 100 else text,
        "model_label": model_label,
        "human_label": human_label,
        "confidence": round(confidence, 3),
        "agreement": agreement,
    }
    feedback_logger.info(f"FEEDBACK_EVENT: {event}")


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp in ISO format.

    Args:
        dt: Datetime object (None for current time)

    Returns:
        ISO formatted timestamp string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def calculate_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate additional metrics from basic stats.

    Args:
        stats: Basic statistics dictionary

    Returns:
        Enhanced metrics dictionary
    """
    total = stats.get("total_predictions", 0)
    if total == 0:
        return {
            "accuracy": 0.0,
            "automation_rate": 0.0,
            "review_rate": 0.0,
            "pending_rate": 0.0,
        }

    status_counts = stats.get("status_counts", {})
    auto_accepted = status_counts.get("auto_accepted", 0)
    reviewed = status_counts.get("reviewed", 0)
    pending = status_counts.get("pending", 0)

    return {
        "accuracy": stats.get("agreement_rate", 0.0),
        "automation_rate": round(auto_accepted / total, 3) if total > 0 else 0.0,
        "review_rate": round(reviewed / total, 3) if total > 0 else 0.0,
        "pending_rate": round(pending / total, 3) if total > 0 else 0.0,
        "total_processed": auto_accepted + reviewed,
        "efficiency_score": round((auto_accepted + reviewed * stats.get("agreement_rate", 0)) / total, 3) if total > 0 else 0.0,
    }


def validate_environment() -> Dict[str, bool]:
    """
    Validate that all required components are available.

    Returns:
        Dictionary of component availability
    """
    checks = {}

    # Check Python version
    checks["python_version"] = sys.version_info >= (3, 11)

    # Check if model can be loaded
    try:
        import transformers
        checks["transformers"] = True
    except ImportError:
        checks["transformers"] = False

    # Check if torch is available
    try:
        import torch
        checks["torch"] = True
        checks["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        checks["torch"] = False
        checks["cuda_available"] = False

    # Check if database path is writable
    try:
        Path(DB_PATH).parent.mkdir(exist_ok=True)
        checks["database_writable"] = True
    except Exception:
        checks["database_writable"] = False

    # Check if log directory is writable
    try:
        LOG_DIR.mkdir(exist_ok=True)
        checks["logs_writable"] = True
    except Exception:
        checks["logs_writable"] = False

    return checks


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.

    Returns:
        System information dictionary
    """
    import platform

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = "Not installed"

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "Not installed"

    return info


# Version management utilities
def bump_version(version: str) -> str:
    """
    Increment version number.

    Args:
        version: Current version string (e.g., "v1.0")

    Returns:
        Incremented version string (e.g., "v1.1")
    """
    if not version.startswith("v"):
        version = f"v{version}"

    parts = version[1:].split(".")
    if len(parts) == 1:
        # Simple version like "v1" -> "v2"
        major = int(parts[0])
        return f"v{major + 1}"
    elif len(parts) == 2:
        # Version like "v1.0" -> "v1.1"
        major, minor = int(parts[0]), int(parts[1])
        return f"v{major}.{minor + 1}"
    else:
        # Just increment the last part
        parts[-1] = str(int(parts[-1]) + 1)
        return f"v{'.'.join(parts)}"


def load_metadata() -> Dict[str, Any]:
    """
    Load metadata from metadata.json.

    Returns:
        Metadata dictionary
    """
    metadata_path = Path("metadata.json")
    if not metadata_path.exists():
        # Create default metadata
        default_metadata = {
            "model_version": "v1.0",
            "dataset_version": "v1.0",
            "accuracy": 0.85,
            "last_retrain": datetime.utcnow().isoformat() + "Z",
            "total_predictions": 0,
            "total_retrains": 0,
            "version_history": [],
            "agent_config": {
                "multi_agent_enabled": False,
                "num_agents": 3,
                "agreement_threshold": 0.66,
                "agent_types": ["primary", "variation1", "variation2"]
            }
        }
        save_metadata(default_metadata)
        return default_metadata

    with open(metadata_path, "r") as f:
        return json.load(f)


def save_metadata(metadata: Dict[str, Any]):
    """
    Save metadata to metadata.json.

    Args:
        metadata: Metadata dictionary to save
    """
    metadata_path = Path("metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def simulate_accuracy_improvement(current_accuracy: float) -> float:
    """
    Simulate accuracy improvement after retraining.

    Args:
        current_accuracy: Current model accuracy

    Returns:
        Improved accuracy (1-3% increase, capped at 0.95)
    """
    import random

    # Random improvement between 1-3%
    improvement = random.uniform(0.01, 0.03)
    new_accuracy = current_accuracy + improvement

    # Cap at 95% to remain realistic
    return min(new_accuracy, 0.95)


def create_checkpoint_file(model_version: str, accuracy: float):
    """
    Create a dummy checkpoint file for the given model version.

    Args:
        model_version: Model version string
        accuracy: Model accuracy to include in metadata
    """
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / f"model_{model_version}.pt"

    # Create dummy checkpoint data
    checkpoint_data = {
        "version": model_version,
        "accuracy": accuracy,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": MODEL_NAME,
        "framework": "pytorch",
        "notes": f"Simulated checkpoint for {model_version}"
    }

    # Save as JSON (simulating a model checkpoint)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Created checkpoint: {checkpoint_path}")


def create_dataset_snapshot(dataset_version: str, num_samples: int):
    """
    Create a dummy dataset snapshot for versioning.

    Args:
        dataset_version: Dataset version string
        num_samples: Number of samples in the dataset
    """
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)

    snapshot_path = dataset_dir / f"dataset_{dataset_version}.json"

    # Create dummy dataset metadata
    dataset_data = {
        "version": dataset_version,
        "num_samples": num_samples,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "splits": {
            "train": int(num_samples * 0.8),
            "validation": int(num_samples * 0.1),
            "test": int(num_samples * 0.1)
        },
        "notes": f"Dataset snapshot for {dataset_version}"
    }

    with open(snapshot_path, "w") as f:
        json.dump(dataset_data, f, indent=2)

    logger.info(f"Created dataset snapshot: {snapshot_path}")


# Initialize loggers on module import
logger = setup_logging(__name__)
feedback_logger = setup_feedback_logger()

if __name__ == "__main__":
    # Test utilities
    logger.info("Testing logging setup")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test feedback logging
    log_feedback_event(
        sample_id=1,
        text="This is a test sample",
        model_label="POSITIVE",
        human_label="NEGATIVE",
        confidence=0.75,
        agreement=False
    )

    # Validate environment
    checks = validate_environment()
    print("\nEnvironment Validation:")
    for component, status in checks.items():
        status_str = "✓" if status else "✗"
        print(f"  {component}: {status_str}")

    # System info
    print("\nSystem Information:")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")