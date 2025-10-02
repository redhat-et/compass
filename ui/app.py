"""Streamlit UI for AI Pre-Deployment Assistant.

This module provides the main Streamlit interface for the assistant, featuring:
1. Chat interface for conversational requirement gathering
2. Recommendation display with all specification details
3. Editable specification component for user review/modification
4. Integration with FastAPI backend
"""

import streamlit as st
import requests
from typing import Optional, Dict, Any
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="AI Pre-Deployment Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .warning-badge {
        background-color: #ffc107;
        color: black;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None
if "editing_mode" not in st.session_state:
    st.session_state.editing_mode = False


def main():
    """Main application entry point."""

    # Header
    st.markdown('<div class="main-header">🤖 AI Pre-Deployment Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">From concept to production-ready LLM deployment</div>', unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Main content area - two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("💬 Conversation")
        render_chat_interface()

    with col2:
        if st.session_state.recommendation:
            st.subheader("📊 Recommendation")
            render_recommendation()
        else:
            st.info("👈 Start a conversation to get deployment recommendations")


def render_sidebar():
    """Render sidebar with app information and quick actions."""
    with st.sidebar:
        st.markdown("### 🤖 AI Pre-Deployment Assistant")

        st.markdown("---")
        st.markdown("### 🎯 Quick Start")
        st.markdown("""
        1. Describe your LLM use case
        2. Review the recommendation
        3. Edit specifications if needed
        4. Deploy to Kubernetes
        """)

        st.markdown("---")
        st.markdown("### 📚 Example Prompts")

        example_prompts = [
            "Customer service chatbot for 5000 users, low latency critical",
            "Code generation assistant for 500 developers, quality over speed",
            "Document summarization pipeline, high throughput, cost efficient"
        ]

        for i, prompt in enumerate(example_prompts, 1):
            if st.button(f"Example {i}", key=f"example_{i}", use_container_width=True):
                st.session_state.current_prompt = prompt

        st.markdown("---")

        # Reset conversation
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.recommendation = None
            st.session_state.editing_mode = False
            st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This assistant helps you:
        - Define LLM deployment requirements
        - Get GPU recommendations
        - Review SLO targets
        - Deploy to Kubernetes
        """)


def render_chat_interface():
    """Render the chat interface for conversational requirement gathering."""

    # Display chat messages
    chat_container = st.container(height=400)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Describe your LLM deployment requirements...")

    # Check if we have a prompt from example button
    if "current_prompt" in st.session_state:
        prompt = st.session_state.current_prompt
        del st.session_state.current_prompt

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Get recommendation from API
        with st.spinner("Analyzing requirements and generating recommendation..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/recommend",
                    json={"message": prompt},
                    timeout=30
                )

                if response.status_code == 200:
                    recommendation = response.json()
                    st.session_state.recommendation = recommendation

                    # Add assistant response
                    assistant_message = format_recommendation_summary(recommendation)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

                    st.rerun()
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def format_recommendation_summary(rec: Dict[str, Any]) -> str:
    """Format recommendation as a chat message."""

    meets_slo = rec.get("meets_slo", False)
    slo_status = "✅ Meets SLO" if meets_slo else "⚠️ Does not meet SLO"

    summary = f"""
I've analyzed your requirements and generated a recommendation:

**{rec['model_name']}** on **{rec['gpu_config']['gpu_count']}x {rec['gpu_config']['gpu_type']}**

**Performance:**
- TTFT p90: {rec['predicted_ttft_p90_ms']}ms
- TPOT p90: {rec['predicted_tpot_p90_ms']}ms
- E2E p95: {rec['predicted_e2e_p95_ms']}ms
- Throughput: {rec['predicted_throughput_qps']:.1f} QPS

**Cost:** ${rec['cost_per_month_usd']:,.2f}/month

**Status:** {slo_status}

{rec['reasoning']}

👉 Review the full details in the right panel, or ask me to adjust the configuration!
"""
    return summary.strip()


def render_recommendation():
    """Render the recommendation display and specification editor."""

    rec = st.session_state.recommendation

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "⚙️ Specifications", "📊 Performance", "💰 Cost"])

    with tab1:
        render_overview_tab(rec)

    with tab2:
        render_specifications_tab(rec)

    with tab3:
        render_performance_tab(rec)

    with tab4:
        render_cost_tab(rec)


def render_overview_tab(rec: Dict[str, Any]):
    """Render overview tab with key information."""

    # SLO Status Badge
    meets_slo = rec.get("meets_slo", False)
    if meets_slo:
        st.markdown('<span class="success-badge">✅ MEETS SLO</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warning-badge">⚠️ DOES NOT MEET SLO</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Model and GPU
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 Model")
        st.markdown(f"**{rec['model_name']}**")
        st.caption(f"ID: `{rec['model_id']}`")

    with col2:
        st.markdown("### 🖥️ GPU Configuration")
        gpu_config = rec['gpu_config']
        st.markdown(f"**{gpu_config['gpu_count']}x {gpu_config['gpu_type']}**")
        st.caption(f"Tensor Parallel: {gpu_config['tensor_parallel']}, Replicas: {gpu_config['replicas']}")

    st.markdown("---")

    # Key Metrics
    st.markdown("### 📈 Key Metrics")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric("TTFT p90", f"{rec['predicted_ttft_p90_ms']}ms")

    with metrics_col2:
        st.metric("TPOT p90", f"{rec['predicted_tpot_p90_ms']}ms")

    with metrics_col3:
        st.metric("E2E p95", f"{rec['predicted_e2e_p95_ms']}ms")

    with metrics_col4:
        st.metric("Throughput", f"{rec['predicted_throughput_qps']:.1f} QPS")

    st.markdown("---")

    # Reasoning
    st.markdown("### 💡 Reasoning")
    st.info(rec['reasoning'])


def render_specifications_tab(rec: Dict[str, Any]):
    """Render specifications tab with editable fields."""

    st.markdown("### 🔧 Deployment Specifications")
    st.caption("Review and modify the specifications before deployment")

    # Intent
    st.markdown("#### Use Case & Requirements")
    intent = rec['intent']

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Use Case", value=intent['use_case'], disabled=True)
        st.number_input("Users", value=intent['user_count'], disabled=True)

    with col2:
        st.text_input("Latency Requirement", value=intent['latency_requirement'], disabled=True)
        st.text_input("Budget Constraint", value=intent['budget_constraint'], disabled=True)

    # Traffic Profile
    st.markdown("#### Traffic Profile")
    traffic = rec['traffic_profile']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("Expected QPS", value=traffic['expected_qps'], format="%.2f", disabled=not st.session_state.editing_mode)

    with col2:
        st.number_input("Avg Prompt Tokens", value=traffic['prompt_tokens_mean'], disabled=not st.session_state.editing_mode)

    with col3:
        st.number_input("Avg Generation Tokens", value=traffic['generation_tokens_mean'], disabled=not st.session_state.editing_mode)

    # SLO Targets
    st.markdown("#### SLO Targets")
    slo = rec['slo_targets']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("TTFT p90 (ms)", value=slo['ttft_p90_target_ms'], disabled=not st.session_state.editing_mode)

    with col2:
        st.number_input("TPOT p90 (ms)", value=slo['tpot_p90_target_ms'], disabled=not st.session_state.editing_mode)

    with col3:
        st.number_input("E2E p95 (ms)", value=slo['e2e_p95_target_ms'], disabled=not st.session_state.editing_mode)

    # Edit mode toggle
    st.markdown("---")
    if not st.session_state.editing_mode:
        if st.button("✏️ Enable Editing", use_container_width=True):
            st.session_state.editing_mode = True
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                st.success("Changes saved! (Note: Re-recommendation not yet implemented)")
                st.session_state.editing_mode = False
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.editing_mode = False
                st.rerun()


def render_performance_tab(rec: Dict[str, Any]):
    """Render performance tab with detailed metrics."""

    st.markdown("### 📊 Predicted Performance")

    slo = rec['slo_targets']

    # TTFT
    st.markdown("#### Time to First Token (TTFT)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted p90", f"{rec['predicted_ttft_p90_ms']}ms")
    with col2:
        delta_ms = rec['predicted_ttft_p90_ms'] - slo['ttft_p90_target_ms']
        st.metric("Target p90", f"{slo['ttft_p90_target_ms']}ms",
                 delta=f"{delta_ms}ms", delta_color="inverse")

    # TPOT
    st.markdown("#### Time Per Output Token (TPOT)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted p90", f"{rec['predicted_tpot_p90_ms']}ms")
    with col2:
        delta_ms = rec['predicted_tpot_p90_ms'] - slo['tpot_p90_target_ms']
        st.metric("Target p90", f"{slo['tpot_p90_target_ms']}ms",
                 delta=f"{delta_ms}ms", delta_color="inverse")

    # E2E Latency
    st.markdown("#### End-to-End Latency")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted p95", f"{rec['predicted_e2e_p95_ms']}ms")
    with col2:
        delta_ms = rec['predicted_e2e_p95_ms'] - slo['e2e_p95_target_ms']
        st.metric("Target p95", f"{slo['e2e_p95_target_ms']}ms",
                 delta=f"{delta_ms}ms", delta_color="inverse")

    # Throughput
    st.markdown("#### Throughput")
    st.metric("Requests/sec", f"{rec['predicted_throughput_qps']:.1f} QPS")


def render_cost_tab(rec: Dict[str, Any]):
    """Render cost tab with pricing details."""

    st.markdown("### 💰 Cost Breakdown")

    # Main cost metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Hourly Cost")
        st.markdown(f"## ${rec['cost_per_hour_usd']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Monthly Cost")
        st.markdown(f"## ${rec['cost_per_month_usd']:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # GPU details
    gpu_config = rec['gpu_config']
    st.markdown("#### GPU Configuration")
    st.markdown(f"""
    - **GPU Type:** {gpu_config['gpu_type']}
    - **Total GPUs:** {gpu_config['gpu_count']}
    - **Tensor Parallel:** {gpu_config['tensor_parallel']}
    - **Replicas:** {gpu_config['replicas']}
    """)

    st.markdown("---")

    # Cost assumptions
    st.info("""
    **💡 Cost Assumptions:**
    - Pricing based on typical cloud GPU rates
    - 730 hours/month (24/7 operation)
    - Does not include networking, storage, or egress costs
    - Actual costs may vary by cloud provider
    """)

    # Next steps
    st.markdown("---")
    st.markdown("### 🚀 Next Steps")

    if st.button("📄 Generate Deployment YAML", use_container_width=True):
        st.info("YAML generation will be implemented in Sprint 4!")

    if st.button("🚢 Deploy to Kubernetes", use_container_width=True):
        st.info("Kubernetes deployment will be implemented in Sprint 6!")


if __name__ == "__main__":
    main()
