"""
Streamlit frontend for HITL validation pipeline.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.exceptions import ConnectionError, RequestException

# Configuration
BACKEND_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds

# Page config
st.set_page_config(
    page_title="HITL Validation Pipeline",
    page_icon="↻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .pending-card {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-bottom: 20px;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


class APIClient:
    """Client for backend API communication."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def health_check(self) -> Optional[Dict]:
        """Check backend health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except RequestException:
            pass
        return None

    def get_pending(self) -> List[Dict]:
        """Get pending samples for review."""
        try:
            response = requests.get(f"{self.base_url}/pending", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("items", [])
        except RequestException as e:
            st.error(f"Failed to fetch pending items: {e}")
        return []

    def submit_feedback(self, sample_id: int, human_label: str) -> Optional[Dict]:
        """Submit human feedback."""
        try:
            response = requests.post(
                f"{self.base_url}/submit_feedback",
                json={"id": sample_id, "human_label": human_label},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except RequestException as e:
            st.error(f"Failed to submit feedback: {e}")
        return None

    def get_stats(self) -> Optional[Dict]:
        """Get system statistics."""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except RequestException as e:
            st.error(f"Failed to fetch statistics: {e}")
        return None

    def run_inference(self, text: str) -> Optional[Dict]:
        """Run inference on new text."""
        try:
            response = requests.post(
                f"{self.base_url}/infer",
                json={"text": text},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except RequestException as e:
            st.error(f"Inference failed: {e}")
        return None

    def load_dataset(self) -> List[Dict]:
        """Load and process the sample dataset."""
        try:
            response = requests.post(
                f"{self.base_url}/batch_infer",
                json={
                    "texts": [s["text"] for s in st.session_state.dataset_samples],
                    "batch_id": "initial_load"
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except RequestException as e:
            st.error(f"Failed to process dataset: {e}")
        return None

    def get_metadata(self) -> Optional[Dict]:
        """Get model metadata and version info."""
        try:
            response = requests.get(f"{self.base_url}/metadata", timeout=5)
            if response.status_code == 200:
                return response.json()
        except RequestException as e:
            st.error(f"Failed to fetch metadata: {e}")
        return None

    def retrain_model(self) -> Optional[Dict]:
        """Trigger model retraining."""
        try:
            response = requests.post(f"{self.base_url}/retrain", timeout=30)
            if response.status_code == 200:
                return response.json()
        except RequestException as e:
            st.error(f"Retraining failed: {e}")
        return None


def init_session_state():
    """Initialize session state variables."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(BACKEND_URL)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Review Queue"

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = []

    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False

    if "dataset_samples" not in st.session_state:
        try:
            with open("dataset/samples.json", "r") as f:
                st.session_state.dataset_samples = json.load(f)
        except FileNotFoundError:
            st.session_state.dataset_samples = []


def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.title("HITL Pipeline")
        st.markdown("---")

        # Backend status
        health = st.session_state.api_client.health_check()
        if health:
            if health["status"] == "healthy":
                st.success("Backend Connected")
            else:
                st.warning("Backend Degraded")
        else:
            st.error("Backend Disconnected")

        st.markdown("---")

        # Navigation
        pages = ["Review Queue", "Dashboard", "Test Inference", "Load Dataset"]
        selected_page = st.radio("Navigation", pages)
        st.session_state.current_page = selected_page

        st.markdown("---")

        # Version Info
        metadata = st.session_state.api_client.get_metadata()
        if metadata:
            st.markdown("### Model Version")
            st.info(f"Model: {metadata['model_version']}")
            st.info(f"Dataset: {metadata['dataset_version']}")
            st.info(f"Accuracy: {metadata['accuracy']*100:.1f}%")

        st.markdown("---")

        # Quick stats
        stats = st.session_state.api_client.get_stats()
        if stats:
            st.markdown("### Quick Stats")
            st.metric("Total Predictions", stats["total_predictions"])
            st.metric("Pending Review", stats["status_breakdown"]["pending"])
            st.metric("Agreement Rate", f"{stats['agreement_rate']*100:.1f}%")


def render_review_queue():
    """Render the review queue page."""
    st.header("Review Queue")
    st.markdown("Review and correct low-confidence model predictions")

    # Get pending samples
    pending = st.session_state.api_client.get_pending()

    if not pending:
        st.info("No samples pending review")
        return

    st.markdown(f"**{len(pending)} samples pending review**")
    st.markdown("---")

    # Review interface
    for idx, sample in enumerate(pending[:5]):  # Show max 5 at a time
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### Sample #{sample['id']}")
                st.markdown(f"**Text:** {sample['text']}")
                st.markdown(f"**Model Prediction:** `{sample['model_label']}`")
                st.markdown(f"**Confidence:** {sample['confidence']*100:.1f}%")

                # Confidence bar
                confidence_color = "green" if sample['confidence'] > 0.8 else "orange"
                st.progress(sample['confidence'])

            with col2:
                st.markdown("### Your Review")
                human_label = st.radio(
                    "Select correct label:",
                    ["POSITIVE", "NEGATIVE"],
                    key=f"label_{sample['id']}",
                    index=0 if sample['model_label'] == "POSITIVE" else 1
                )

                if st.button(f"Submit", key=f"submit_{sample['id']}"):
                    result = st.session_state.api_client.submit_feedback(
                        sample['id'], human_label
                    )
                    if result:
                        if result['agreement']:
                            st.success(f"Submitted (Agreed with model)")
                        else:
                            st.warning(f"Submitted (Corrected model prediction)")
                        st.session_state.feedback_submitted.append(sample['id'])
                        time.sleep(1)
                        st.rerun()

            st.markdown("---")

    # Auto-refresh option
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh Queue"):
            st.rerun()


def render_dashboard():
    """Render the dashboard page."""
    st.header("Dashboard")
    st.markdown("System metrics and performance statistics")

    # Get statistics
    stats = st.session_state.api_client.get_stats()
    if not stats:
        st.error("Failed to load statistics")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Predictions",
            stats["total_predictions"],
            delta=None
        )
    with col2:
        st.metric(
            "Auto-Accepted",
            stats["status_breakdown"]["auto_accepted"],
            delta=f"{stats['system_health'].get('automation_rate', 0)*100:.1f}%"
        )
    with col3:
        st.metric(
            "Human Reviewed",
            stats["status_breakdown"]["reviewed"],
            delta=f"Agreement: {stats['agreement_rate']*100:.1f}%"
        )
    with col4:
        st.metric(
            "Pending Review",
            stats["status_breakdown"]["pending"],
            delta=None
        )

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Status distribution pie chart
        if stats["total_predictions"] > 0:
            fig_pie = px.pie(
                values=list(stats["status_breakdown"].values()),
                names=list(stats["status_breakdown"].keys()),
                title="Status Distribution",
                color_discrete_map={
                    "auto_accepted": "#4CAF50",
                    "pending": "#FFC107",
                    "reviewed": "#2196F3"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Confidence statistics
        if stats.get("confidence_stats"):
            conf_data = []
            for status, conf_info in stats["confidence_stats"].items():
                conf_data.append({
                    "Status": status,
                    "Average": conf_info["avg"],
                    "Min": conf_info["min"],
                    "Max": conf_info["max"]
                })
            if conf_data:
                df_conf = pd.DataFrame(conf_data)
                fig_bar = px.bar(
                    df_conf,
                    x="Status",
                    y="Average",
                    title="Average Confidence by Status",
                    error_y=[d["Max"] - d["Average"] for d in conf_data],
                    error_y_minus=[d["Average"] - d["Min"] for d in conf_data]
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # Recent reviews table
    st.markdown("### Recent Reviews")
    if stats.get("recent_reviews"):
        df_reviews = pd.DataFrame(stats["recent_reviews"])
        if not df_reviews.empty:
            # Select and reorder columns
            columns_to_show = ["id", "text", "model_label", "human_label", "confidence"]
            df_display = df_reviews[columns_to_show]
            df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x*100:.1f}%")
            df_display["text"] = df_display["text"].str[:50] + "..."
            st.dataframe(df_display, use_container_width=True)

    # Model Management Section
    st.markdown("---")
    st.markdown("### Model Management")

    metadata = st.session_state.api_client.get_metadata()
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Model", metadata["model_version"])
        with col2:
            st.metric("Dataset Version", metadata["dataset_version"])
        with col3:
            st.metric("Model Accuracy", f"{metadata['accuracy']*100:.1f}%")
        with col4:
            # Retrain button
            if st.button("Retrain Model", type="primary", key="retrain_btn"):
                if stats["total_reviewed"] > 0:
                    with st.spinner("Retraining model..."):
                        retrain_result = st.session_state.api_client.retrain_model()
                        if retrain_result and retrain_result.get("success"):
                            st.success(
                                f"Model retrained successfully! "
                                f"New version: {retrain_result['new_version']['model']} "
                                f"(+{retrain_result['improvement']*100:.1f}% accuracy)"
                            )
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Retraining failed. Check if you have feedback data.")
                else:
                    st.warning("No feedback data available for retraining")

        # Version History
        if metadata.get("version_history") and len(metadata["version_history"]) > 0:
            st.markdown("#### Version History")

            # Create DataFrame for version history
            history_data = []
            for entry in metadata["version_history"]:
                history_data.append({
                    "Model Version": entry["model_version"],
                    "Dataset Version": entry["dataset_version"],
                    "Accuracy": f"{entry['accuracy']*100:.1f}%",
                    "Samples Used": entry.get("samples_used", 0),
                    "Improvement": f"+{entry.get('improvement', 0)*100:.1f}%",
                    "Timestamp": entry["timestamp"][:16]  # Trim to minute precision
                })

            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)

            # Accuracy trend chart
            if len(metadata["version_history"]) > 1:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=[h["model_version"] for h in metadata["version_history"]],
                    y=[h["accuracy"]*100 for h in metadata["version_history"]],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#4CAF50', width=2),
                    marker=dict(size=8)
                ))
                fig_trend.update_layout(
                    title="Model Accuracy Over Versions",
                    xaxis_title="Model Version",
                    yaxis_title="Accuracy (%)",
                    yaxis_range=[80, 100],
                    showlegend=False
                )
                st.plotly_chart(fig_trend, use_container_width=True)

    # System health
    st.markdown("---")
    st.markdown("### System Health")
    col1, col2, col3 = st.columns(3)
    with col1:
        uptime_hours = stats["system_health"]["uptime_seconds"] / 3600
        st.metric("Uptime", f"{uptime_hours:.2f} hours")
    with col2:
        st.metric(
            "Model Status",
            "Loaded" if stats["system_health"]["model_loaded"] else "Not Loaded"
        )
    with col3:
        st.metric(
            "Database Status",
            "Connected" if stats["system_health"]["database"] else "Disconnected"
        )


def render_test_inference():
    """Render the test inference page."""
    st.header("Test Inference")
    st.markdown("Test the model with custom text input")

    # Input form
    with st.form("inference_form"):
        text_input = st.text_area(
            "Enter text for sentiment analysis:",
            placeholder="Type or paste your text here...",
            height=100
        )
        submitted = st.form_submit_button("Run Inference")

    if submitted and text_input:
        with st.spinner("Running inference..."):
            result = st.session_state.api_client.run_inference(text_input)

        if result:
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", result["model_label"])
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            with col3:
                st.metric("Status", result["status"])

            # Show confidence bar
            st.progress(result['confidence'])

            if result["requires_review"]:
                st.warning("This prediction requires human review due to low confidence")
            else:
                st.success("High confidence prediction - automatically accepted")

            # Show full response
            with st.expander("View full response"):
                st.json(result)


def render_load_dataset():
    """Render the dataset loading page."""
    st.header("Load Dataset")
    st.markdown("Load and process the sample dataset")

    # Show dataset info
    if st.session_state.dataset_samples:
        st.info(f"Found {len(st.session_state.dataset_samples)} samples in dataset")

        # Preview samples
        with st.expander("Preview dataset samples"):
            for i, sample in enumerate(st.session_state.dataset_samples[:5]):
                st.markdown(f"**Sample {i+1}:** {sample['text']}")

    else:
        st.error("No dataset found at dataset/samples.json")
        return

    # Load button
    if st.button("Process Dataset", type="primary"):
        with st.spinner(f"Processing {len(st.session_state.dataset_samples)} samples..."):
            result = st.session_state.api_client.load_dataset()

        if result:
            st.success(f"Successfully processed {result['total_processed']} samples")

            # Show summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Auto-Accepted", result["summary"]["auto_accepted"])
            with col2:
                st.metric("Sent for Review", result["summary"]["pending"])

            st.session_state.dataset_loaded = True

            # Auto-refresh stats
            time.sleep(2)
            st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Route to selected page
    page_map = {
        "Review Queue": render_review_queue,
        "Dashboard": render_dashboard,
        "Test Inference": render_test_inference,
        "Load Dataset": render_load_dataset,
    }

    page_function = page_map.get(st.session_state.current_page)
    if page_function:
        page_function()


if __name__ == "__main__":
    main()