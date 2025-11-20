"""
Database module for HITL validation pipeline.
Manages SQLite database operations for storing predictions and feedback.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

# Database configuration
DB_PATH = Path("hitl_demo.db")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database with required schema."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create main predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                model_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                human_label TEXT,
                timestamp TEXT NOT NULL,
                feedback_timestamp TEXT,
                dataset_version TEXT DEFAULT 'v1.0',
                model_version TEXT DEFAULT 'distilbert-sst2-v1'
            )
        """)

        # Create index on status for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON predictions(status)
        """)

        # Create index on timestamp for time-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON predictions(timestamp)
        """)


def insert_prediction(
    text: str,
    model_label: str,
    confidence: float,
    status: str
) -> int:
    """
    Insert a new prediction into the database.

    Args:
        text: Input text that was classified
        model_label: Model's predicted label
        confidence: Model's confidence score (0-1)
        status: Prediction status (PENDING, AUTO_ACCEPTED)

    Returns:
        ID of inserted record
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()

        cursor.execute("""
            INSERT INTO predictions (text, model_label, confidence, status, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (text, model_label, confidence, status, timestamp))

        return cursor.lastrowid


def get_pending_samples(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all pending samples for human review.

    Args:
        limit: Optional limit on number of samples to return

    Returns:
        List of pending samples with all fields
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = """
            SELECT id, text, model_label, confidence, timestamp
            FROM predictions
            WHERE status = 'PENDING'
            ORDER BY timestamp ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]


def submit_feedback(sample_id: int, human_label: str) -> bool:
    """
    Update a sample with human feedback.

    Args:
        sample_id: ID of the sample to update
        human_label: Human-provided label

    Returns:
        True if update successful, False otherwise
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        feedback_timestamp = datetime.utcnow().isoformat()

        cursor.execute("""
            UPDATE predictions
            SET status = 'REVIEWED',
                human_label = ?,
                feedback_timestamp = ?
            WHERE id = ? AND status = 'PENDING'
        """, (human_label, feedback_timestamp, sample_id))

        return cursor.rowcount > 0


def get_stats() -> Dict[str, Any]:
    """
    Calculate and return system statistics.

    Returns:
        Dictionary containing various metrics
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get status counts
        cursor.execute("""
            SELECT
                status,
                COUNT(*) as count
            FROM predictions
            GROUP BY status
        """)
        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM predictions")
        total = cursor.fetchone()['total']

        # Calculate agreement rate for reviewed items
        cursor.execute("""
            SELECT
                COUNT(CASE WHEN model_label = human_label THEN 1 END) as agreements,
                COUNT(*) as total_reviewed
            FROM predictions
            WHERE status = 'REVIEWED' AND human_label IS NOT NULL
        """)
        agreement_data = cursor.fetchone()
        agreement_rate = (
            agreement_data['agreements'] / agreement_data['total_reviewed']
            if agreement_data['total_reviewed'] > 0 else 0.0
        )

        # Get average confidence by status
        cursor.execute("""
            SELECT
                status,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence
            FROM predictions
            GROUP BY status
        """)
        confidence_stats = {
            row['status']: {
                'avg': round(row['avg_confidence'], 3) if row['avg_confidence'] else 0,
                'min': round(row['min_confidence'], 3) if row['min_confidence'] else 0,
                'max': round(row['max_confidence'], 3) if row['max_confidence'] else 0,
            }
            for row in cursor.fetchall()
        }

        # Get label distribution
        cursor.execute("""
            SELECT
                model_label,
                COUNT(*) as count
            FROM predictions
            GROUP BY model_label
        """)
        label_distribution = {row['model_label']: row['count'] for row in cursor.fetchall()}

        # Get recent activity (last 10 feedbacks)
        cursor.execute("""
            SELECT
                id,
                text,
                model_label,
                human_label,
                confidence,
                feedback_timestamp
            FROM predictions
            WHERE status = 'REVIEWED'
            ORDER BY feedback_timestamp DESC
            LIMIT 10
        """)
        recent_reviews = [dict(row) for row in cursor.fetchall()]

        return {
            'total_predictions': total,
            'status_counts': {
                'auto_accepted': status_counts.get('AUTO_ACCEPTED', 0),
                'pending': status_counts.get('PENDING', 0),
                'reviewed': status_counts.get('REVIEWED', 0),
            },
            'agreement_rate': round(agreement_rate, 3),
            'total_reviewed': agreement_data['total_reviewed'],
            'confidence_stats': confidence_stats,
            'label_distribution': label_distribution,
            'recent_reviews': recent_reviews,
        }


def get_sample_by_id(sample_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific sample by ID.

    Args:
        sample_id: ID of the sample to retrieve

    Returns:
        Sample data if found, None otherwise
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions WHERE id = ?
        """, (sample_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def bulk_insert_samples(samples: List[Dict[str, str]]) -> int:
    """
    Bulk insert samples for inference.

    Args:
        samples: List of dictionaries with 'text' key

    Returns:
        Number of samples inserted
    """
    count = 0
    for sample in samples:
        if 'text' in sample:
            # These will be processed through the inference endpoint
            # For now, just count them
            count += 1
    return count


def clear_database():
    """Clear all data from the database (useful for testing)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")


def export_feedback_data() -> List[Dict[str, Any]]:
    """
    Export all reviewed samples with feedback.
    Useful for retraining or analysis.

    Returns:
        List of reviewed samples with human labels
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                text,
                model_label,
                human_label,
                confidence,
                feedback_timestamp
            FROM predictions
            WHERE status = 'REVIEWED' AND human_label IS NOT NULL
            ORDER BY feedback_timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


# Initialize database on module import
if __name__ == "__main__":
    init_database()
    print(f"Database initialized at {DB_PATH}")
else:
    init_database()