"""
FastAPI backend for HITL validation pipeline.
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import database, utils
from backend.inference import get_model, warmup_model
from backend.agents import setup_multi_agent_validation, MultiAgentValidator
from backend.models import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    PendingResponse,
    PredictionStatus,
    SentimentLabel,
    StatsResponse,
)

# Setup logging
logger = utils.setup_logging(__name__)

# Track application start time
START_TIME = time.time()

# Global multi-agent validator instance (lazy-loaded)
_multi_agent_validator: Optional[MultiAgentValidator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown.
    """
    # Startup
    logger.info("Starting HITL Validation Pipeline Backend")

    # Initialize database
    database.init_database()
    logger.info("Database initialized")

    # Warm up model - commented out for now
    # Model will be lazy-loaded on first request
    # try:
    #     warmup_model()
    #     logger.info("Model warmed up successfully")
    # except Exception as e:
    #     logger.warning(f"Model warmup failed (will lazy-load): {e}")
    logger.info("Model will be lazy-loaded on first request")

    # Log system info
    system_info = utils.get_system_info()
    logger.info(f"System info: {system_info}")

    yield

    # Shutdown
    logger.info("Shutting down HITL Validation Pipeline Backend")


# Create FastAPI app
app = FastAPI(
    title="HITL Validation Pipeline",
    description="Human-in-the-Loop validation system for model predictions",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HITL Validation Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        System health status
    """
    try:
        # Check database connection
        stats = database.get_stats()
        db_ok = stats is not None
    except Exception:
        db_ok = False

    # Check model
    try:
        model = get_model()
        model_ok = model._initialized
    except Exception:
        model_ok = False

    uptime = time.time() - START_TIME

    return HealthResponse(
        status="healthy" if db_ok and model_ok else "degraded",
        database=db_ok,
        model_loaded=model_ok,
        uptime_seconds=round(uptime, 2),
    )


@app.post("/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run model inference on input text.

    Args:
        request: Inference request with text

    Returns:
        Prediction with confidence and status
    """
    try:
        logger.info(f"Running inference on text: '{utils.truncate_text(request.text)}'")

        # Check if multi-agent mode is enabled
        metadata = utils.load_metadata()
        use_multi_agent = metadata.get("agent_config", {}).get("multi_agent_enabled", False)

        if use_multi_agent:
            # Use multi-agent validation
            global _multi_agent_validator
            if _multi_agent_validator is None:
                _multi_agent_validator = setup_multi_agent_validation(
                    metadata.get("agent_config", {}).get("agreement_threshold", 0.66)
                )
                if _multi_agent_validator is None:
                    # Fallback to single model if multi-agent setup fails
                    logger.warning("Multi-agent setup failed, falling back to single model")
                    use_multi_agent = False

        if use_multi_agent and _multi_agent_validator:
            # Multi-agent validation
            label, confidence, status, predictions, agreement_ratio = _multi_agent_validator.validate(request.text)

            # Log agent predictions for debugging
            logger.info(
                f"Multi-agent validation - Consensus: {label} ({confidence:.3f}), "
                f"Agreement: {agreement_ratio:.2f}, Status: {status}"
            )

            # Store additional metadata if needed
            metadata_dict = {
                "multi_agent": True,
                "agreement_ratio": agreement_ratio,
                "num_agents": len(predictions),
                "agent_predictions": [
                    {"agent": p.agent_id, "label": p.label, "confidence": p.confidence}
                    for p in predictions
                ]
            }

            # Log individual agent predictions
            for pred in predictions:
                logger.info(f"  Agent {pred.agent_id}: {pred.label} (confidence: {pred.confidence:.3f})")
        else:
            # Single model inference (original behavior)
            model = get_model()
            label, confidence, status = model.predict(request.text)
            metadata_dict = {"multi_agent": False}

        # Store in database
        sample_id = database.insert_prediction(
            text=request.text,
            model_label=label,
            confidence=confidence,
            status=status,
        )

        # Prepare response
        response = InferenceResponse(
            id=sample_id,
            text=request.text,
            model_label=label,
            confidence=round(confidence, 3),
            status=status,
            requires_review=(status == PredictionStatus.PENDING or status == "NEEDS_HUMAN"),
            timestamp=utils.format_timestamp(),
        )

        logger.info(
            f"Inference complete - ID: {sample_id}, Label: {label}, "
            f"Confidence: {confidence:.3f}, Status: {status}, "
            f"Multi-agent: {use_multi_agent}"
        )

        return response

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/batch_infer", response_model=BatchInferenceResponse)
async def run_batch_inference(request: BatchInferenceRequest):
    """
    Run inference on multiple texts.

    Args:
        request: Batch inference request

    Returns:
        Batch results with summary
    """
    try:
        logger.info(f"Running batch inference on {len(request.texts)} texts")

        model = get_model()
        results = []
        summary = {
            "total": len(request.texts),
            "auto_accepted": 0,
            "pending": 0,
        }

        for text in request.texts:
            # Run inference
            label, confidence, pred_status = model.predict(text)

            # Store in database
            sample_id = database.insert_prediction(
                text=text,
                model_label=label,
                confidence=confidence,
                status=pred_status,
            )

            # Add to results
            results.append(
                InferenceResponse(
                    id=sample_id,
                    text=text,
                    model_label=label,
                    confidence=round(confidence, 3),
                    status=pred_status,
                    requires_review=(pred_status == PredictionStatus.PENDING),
                    timestamp=utils.format_timestamp(),
                )
            )

            # Update summary
            if pred_status == PredictionStatus.AUTO_ACCEPTED:
                summary["auto_accepted"] += 1
            else:
                summary["pending"] += 1

        return BatchInferenceResponse(
            batch_id=request.batch_id,
            total_processed=len(results),
            results=results,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )


@app.get("/pending", response_model=PendingResponse)
async def get_pending_samples(limit: Optional[int] = None):
    """
    Get samples pending human review.

    Args:
        limit: Optional limit on number of samples

    Returns:
        Pending samples for review
    """
    try:
        samples = database.get_pending_samples(limit=limit)

        # Convert to response models
        items = [
            {
                "id": s["id"],
                "text": s["text"],
                "model_label": s["model_label"],
                "confidence": s["confidence"],
                "timestamp": s["timestamp"],
            }
            for s in samples
        ]

        response = PendingResponse(count=len(items), items=items)

        logger.info(f"Retrieved {len(items)} pending samples")
        return response

    except Exception as e:
        logger.error(f"Error retrieving pending samples: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pending samples: {str(e)}",
        )


@app.post("/submit_feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit human feedback for a sample.

    Args:
        request: Feedback with sample ID and human label

    Returns:
        Feedback submission result
    """
    try:
        # Get original sample
        sample = database.get_sample_by_id(request.id)
        if not sample:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample {request.id} not found",
            )

        # Check if already reviewed
        if sample["status"] == PredictionStatus.REVIEWED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sample {request.id} has already been reviewed",
            )

        # Submit feedback
        success = database.submit_feedback(request.id, request.human_label)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit feedback",
            )

        # Check agreement
        agreement = sample["model_label"] == request.human_label

        # Log feedback event
        utils.log_feedback_event(
            sample_id=request.id,
            text=sample["text"],
            model_label=sample["model_label"],
            human_label=request.human_label,
            confidence=sample["confidence"],
            agreement=agreement,
        )

        logger.info(
            f"Feedback submitted - ID: {request.id}, "
            f"Human: {request.human_label}, Agreement: {agreement}"
        )

        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            sample_id=request.id,
            agreement=agreement,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}",
        )


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Get system statistics and metrics.

    Returns:
        System statistics
    """
    try:
        # Get basic stats from database
        stats = database.get_stats()

        # Calculate additional metrics
        metrics = utils.calculate_metrics(stats)

        # Get system health
        health = await health_check()

        response = StatsResponse(
            total_predictions=stats["total_predictions"],
            status_breakdown=stats["status_counts"],
            agreement_rate=stats["agreement_rate"],
            total_reviewed=stats["total_reviewed"],
            confidence_stats=stats["confidence_stats"],
            label_distribution=stats["label_distribution"],
            recent_reviews=stats["recent_reviews"],
            system_health={
                "status": health.status,
                "database": health.database,
                "model_loaded": health.model_loaded,
                "uptime_seconds": health.uptime_seconds,
                **metrics,
            },
        )

        logger.debug(f"Statistics retrieved: {stats['total_predictions']} total predictions")
        return response

    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )


@app.get("/export_feedback", response_model=List[Dict[str, Any]])
async def export_feedback():
    """
    Export all reviewed samples with feedback.

    Returns:
        List of reviewed samples
    """
    try:
        feedback_data = database.export_feedback_data()
        logger.info(f"Exported {len(feedback_data)} feedback samples")
        return feedback_data

    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export feedback: {str(e)}",
        )


@app.delete("/clear", response_model=Dict[str, str])
async def clear_data():
    """
    Clear all data from database (for testing).

    Returns:
        Confirmation message
    """
    try:
        database.clear_database()
        logger.warning("Database cleared")
        return {"message": "Database cleared successfully"}

    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear database: {str(e)}",
        )


@app.post("/retrain")
async def retrain_model():
    """
    Simulate model retraining with feedback data.

    Returns:
        Retraining results with new version info
    """
    try:
        # Load current metadata
        metadata = utils.load_metadata()

        # Get feedback data for "retraining"
        feedback_data = database.export_feedback_data()
        num_samples = len(feedback_data)

        # Simulate retraining only if we have feedback
        if num_samples == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No feedback data available for retraining"
            )

        # Bump versions
        old_model_version = metadata["model_version"]
        old_dataset_version = metadata["dataset_version"]
        new_model_version = utils.bump_version(old_model_version)
        new_dataset_version = utils.bump_version(old_dataset_version)

        # Simulate accuracy improvement
        old_accuracy = metadata["accuracy"]
        new_accuracy = utils.simulate_accuracy_improvement(old_accuracy)

        # Create checkpoint and dataset snapshot
        utils.create_checkpoint_file(new_model_version, new_accuracy)
        utils.create_dataset_snapshot(new_dataset_version, num_samples)

        # Update metadata
        retrain_timestamp = utils.format_timestamp() + "Z"

        # Add to version history
        version_entry = {
            "model_version": new_model_version,
            "dataset_version": new_dataset_version,
            "accuracy": round(new_accuracy, 4),
            "timestamp": retrain_timestamp,
            "samples_used": num_samples,
            "improvement": round(new_accuracy - old_accuracy, 4),
            "notes": f"Retrained with {num_samples} feedback samples"
        }

        metadata["model_version"] = new_model_version
        metadata["dataset_version"] = new_dataset_version
        metadata["accuracy"] = round(new_accuracy, 4)
        metadata["last_retrain"] = retrain_timestamp
        metadata["total_retrains"] = metadata.get("total_retrains", 0) + 1
        metadata["version_history"].append(version_entry)

        # Save updated metadata
        utils.save_metadata(metadata)

        logger.info(
            f"Model retrained: {old_model_version} → {new_model_version}, "
            f"Accuracy: {old_accuracy:.2%} → {new_accuracy:.2%}"
        )

        return {
            "success": True,
            "message": "Model retrained successfully",
            "old_version": {
                "model": old_model_version,
                "dataset": old_dataset_version,
                "accuracy": round(old_accuracy, 4)
            },
            "new_version": {
                "model": new_model_version,
                "dataset": new_dataset_version,
                "accuracy": round(new_accuracy, 4)
            },
            "improvement": round(new_accuracy - old_accuracy, 4),
            "samples_used": num_samples,
            "timestamp": retrain_timestamp
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrain model: {str(e)}"
        )


@app.get("/metadata")
async def get_metadata():
    """
    Get current model metadata and version info.

    Returns:
        Current metadata including versions and accuracy
    """
    try:
        metadata = utils.load_metadata()
        return metadata
    except Exception as e:
        logger.error(f"Metadata retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metadata: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url)},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=utils.BACKEND_PORT,
        reload=True,
        log_level="info",
    )