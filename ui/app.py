"""Streamlit UI for Compass.

This module provides the main Streamlit interface for Compass, featuring:
1. Chat interface for conversational requirement gathering
2. Recommendation display with all specification details
3. Editable specification component for user review/modification
4. Integration with FastAPI backend
"""

import time
from typing import Any

import requests
import streamlit as st

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Compass",
    page_icon="docs/compass-logo.ico",
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
if "deployment_id" not in st.session_state:
    st.session_state.deployment_id = None
if "deployment_files" not in st.session_state:
    st.session_state.deployment_files = None
if "cluster_accessible" not in st.session_state:
    st.session_state.cluster_accessible = None
if "deployed_to_cluster" not in st.session_state:
    st.session_state.deployed_to_cluster = False


def main():
    """Main application entry point."""

    # Header
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("docs/compass-logo.svg", width=50)
    with col2:
        st.markdown('<div class="main-header">Compass</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">From concept to production-ready LLM deployment</div>', unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Top-level tabs
    main_tabs = st.tabs(["🤖 Assistant", "📊 Recommendation Details", "📦 Deployment Management"])

    with main_tabs[0]:
        render_assistant_tab()

    with main_tabs[1]:
        render_recommendation_details_tab()

    with main_tabs[2]:
        render_deployment_management_tab()


def render_assistant_tab():
    """Render the AI assistant tab with chat interface."""
    st.subheader("💬 Conversation")
    render_chat_interface()

    # Action buttons below chat
    if st.session_state.recommendation:
        st.markdown("---")
        st.markdown("### 🚀 Actions")

        # Check cluster status on first render
        if st.session_state.cluster_accessible is None:
            check_cluster_status()

        # Enable button if cluster is accessible and not already deployed
        button_disabled = not st.session_state.cluster_accessible or st.session_state.deployed_to_cluster
        button_label = "✅ Deployed" if st.session_state.deployed_to_cluster else "🚢 Deploy to Kubernetes"
        button_help = "Already deployed to cluster" if st.session_state.deployed_to_cluster else (
            "Deploy to Kubernetes cluster (YAML auto-generated)" if st.session_state.cluster_accessible else
            "Kubernetes cluster not accessible"
        )

        if st.button(button_label, use_container_width=True, type="primary", disabled=button_disabled, help=button_help):
            deploy_to_cluster(get_selected_option())

        if st.session_state.recommendation.get("yaml_generated", False):
            st.caption("💡 YAML files auto-generated. View in the **Recommendation Details** tab.")


def get_selected_option():
    """Get the currently selected deployment option (recommended or alternative)."""
    rec = st.session_state.recommendation
    selected_idx = st.session_state.get('selected_option_idx', 0)

    if selected_idx == 0:
        # Return the recommended option (current recommendation)
        return rec
    else:
        # Return the selected alternative
        alternatives = rec.get('alternative_options', [])
        if selected_idx - 1 < len(alternatives):
            alt = alternatives[selected_idx - 1]
            # Create a recommendation dict from the alternative
            selected_rec = rec.copy()
            selected_rec['model_name'] = alt['model_name']
            selected_rec['model_id'] = alt['model_id']
            selected_rec['gpu_config'] = alt['gpu_config']
            selected_rec['predicted_ttft_p90_ms'] = alt['predicted_ttft_p90_ms']
            selected_rec['predicted_tpot_p90_ms'] = alt['predicted_tpot_p90_ms']
            selected_rec['predicted_e2e_p95_ms'] = alt['predicted_e2e_p95_ms']
            selected_rec['predicted_throughput_qps'] = alt['predicted_throughput_qps']
            selected_rec['cost_per_hour_usd'] = alt['cost_per_hour_usd']
            selected_rec['cost_per_month_usd'] = alt['cost_per_month_usd']
            selected_rec['reasoning'] = alt['reasoning']
            return selected_rec
        else:
            # Fallback to recommended if invalid index
            return rec


def render_recommendation_details_tab():
    """Render the recommendation details tab."""
    if st.session_state.recommendation:
        render_recommendation()

        # Add deploy button at bottom
        st.markdown("---")
        st.markdown("### 🚀 Deploy")

        # Check cluster status
        if st.session_state.cluster_accessible is None:
            check_cluster_status()

        button_disabled = not st.session_state.cluster_accessible or st.session_state.deployed_to_cluster
        button_label = "✅ Deployed" if st.session_state.deployed_to_cluster else "🚢 Deploy to Kubernetes"
        button_help = "Already deployed to cluster" if st.session_state.deployed_to_cluster else (
            "Deploy to Kubernetes cluster (YAML auto-generated)" if st.session_state.cluster_accessible else
            "Kubernetes cluster not accessible"
        )

        # Show which option will be deployed
        selected_idx = st.session_state.get('selected_option_idx', 0)
        if selected_idx == 0:
            st.caption("📌 **Recommended** option will be deployed")
        else:
            st.caption(f"📌 **Option {selected_idx+1}** will be deployed")

        if st.button(button_label, key="deploy_from_details", use_container_width=True, type="primary", disabled=button_disabled, help=button_help):
            deploy_to_cluster(get_selected_option())
    else:
        st.info("👈 Start a conversation in the **Assistant** tab to get deployment recommendations")


def render_deployment_management_tab():
    """Render the deployment management tab with list of services and details."""
    st.markdown("### 📦 Cluster Deployments")

    # Load all deployments from cluster
    all_deployments = load_all_deployments()

    if all_deployments is None:
        st.warning("⚠️ Could not connect to cluster to list deployments")
        st.info("""
        **Troubleshooting:**
        - Ensure Kubernetes cluster is running (e.g., KIND cluster)
        - Check that kubectl can access the cluster: `kubectl cluster-info`
        - Verify backend API is running on http://localhost:8000
        """)
        return

    if len(all_deployments) == 0:
        st.info("""
        📦 **No deployments found in cluster**

        To create a deployment:
        1. Go to the **Assistant** tab
        2. Describe your use case in the conversation
        3. Review the recommendation
        4. Click "Deploy to Kubernetes"
        """)
        return

    # Show deployments table
    st.markdown(f"**Found {len(all_deployments)} deployment(s)**")

    # Create table data
    table_data = []
    for dep in all_deployments:
        dep_id = dep["deployment_id"]
        status = dep.get("status", {})
        pods = dep.get("pods", [])

        # Get service info (cluster IP, ports)
        ready = status.get("ready", False)
        ready_icon = "✅" if ready else "⏳"

        # Extract basic info
        cluster_ip = "N/A"
        port = "N/A"

        table_data.append({
            "Status": ready_icon,
            "Name": dep_id,
            "Pods": len(pods),
            "Ready": "Yes" if ready else "No"
        })

    # Display as table with clickable rows
    import pandas as pd
    df = pd.DataFrame(table_data)

    # Use dataframe to display (not editable)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔍 Select Deployment to Manage")

    # Deployment selector
    deployment_options = {d["deployment_id"]: d for d in all_deployments}
    deployment_ids = list(deployment_options.keys())

    # Initialize selected deployment if not set
    if "selected_deployment" not in st.session_state or st.session_state.selected_deployment not in deployment_ids:
        st.session_state.selected_deployment = deployment_ids[0] if deployment_ids else None

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Choose a deployment:",
            deployment_ids,
            index=deployment_ids.index(st.session_state.selected_deployment) if st.session_state.selected_deployment in deployment_ids else 0,
            key="deployment_selector_mgmt"
        )
        st.session_state.selected_deployment = selected

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_mgmt"):
            st.rerun()

    if not st.session_state.selected_deployment:
        return

    deployment_info = deployment_options[st.session_state.selected_deployment]

    st.markdown("---")

    # Show detailed management for selected deployment
    render_deployment_management(deployment_info, context="mgmt")
    st.markdown("---")
    render_k8s_status_for_deployment(deployment_info, context="mgmt")
    st.markdown("---")
    render_inference_testing_for_deployment(deployment_info, context="mgmt")


def render_sidebar():
    """Render sidebar with app information and quick actions."""
    with st.sidebar:
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("docs/compass-logo.svg", width=30)
        with col2:
            st.markdown("### Compass")

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
        st.markdown("### 📦 Deployments")

        # Load deployments from cluster
        deployments = load_all_deployments()

        if deployments is None:
            st.caption("⚠️ Cluster not accessible")
        elif len(deployments) == 0:
            st.caption("No deployments found")
        else:
            st.caption(f"{len(deployments)} deployment(s) in cluster")

            for dep in deployments:
                dep_id = dep["deployment_id"]
                status = dep.get("status", {})
                ready = status.get("ready", False)
                status_icon = "✅" if ready else "⏳"

                # Show full name (it will truncate based on sidebar width)
                # Tooltip shows full name on hover
                if st.button(
                    f"{status_icon} {dep_id}",
                    key=f"sidebar_dep_{dep_id}",
                    use_container_width=True,
                    help=dep_id  # Tooltip shows full deployment ID
                ):
                    st.session_state.selected_deployment = dep_id
                    st.session_state.show_monitoring = True
                    st.rerun()

        if st.button("🔄 Refresh Deployments", use_container_width=True):
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

                    # Reset deployment state for new recommendation
                    st.session_state.deployed_to_cluster = False
                    st.session_state.deployment_id = None

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


def format_recommendation_summary(rec: dict[str, Any]) -> str:
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

👉 Review the full details on the **Recommendation Details** tab above, ask me to adjust the configuration, or deploy to Kubernetes using the button below!
"""
    return summary.strip()


def render_recommendation():
    """Render the recommendation display and specification editor."""

    rec = st.session_state.recommendation

    # Tabs for different views
    tabs = st.tabs(["📋 Overview", "⚙️ Specifications", "📊 Performance", "💰 Cost", "📄 YAML Preview"])

    with tabs[0]:
        render_overview_tab(rec)

    with tabs[1]:
        render_specifications_tab(rec)

    with tabs[2]:
        render_performance_tab(rec)

    with tabs[3]:
        render_cost_tab(rec)

    with tabs[4]:
        render_yaml_preview_tab(rec)


def render_overview_tab(rec: dict[str, Any]):
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
        st.markdown("### Model")
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

    # Alternative Options
    st.markdown("---")
    st.markdown("### 🔄 Deployment Options Comparison")

    alternatives = rec.get('alternative_options')
    if alternatives and len(alternatives) > 0:
        st.caption("Click Select button to choose which option to deploy")

        # Build table-like layout with buttons

        # Initialize selected_option_idx in session state if not present
        # This just tracks which option is selected, doesn't modify the recommendation
        if 'selected_option_idx' not in st.session_state:
            st.session_state.selected_option_idx = 0  # 0 = recommended

        # Header row
        header_cols = st.columns([0.8, 1.5, 2.5, 1.2, 0.8, 1, 1, 1, 1, 1])
        header_cols[0].markdown("**Select**")
        header_cols[1].markdown("**Option**")
        header_cols[2].markdown("**Model**")
        header_cols[3].markdown("**GPU Config**")
        header_cols[4].markdown("**Replicas**")
        header_cols[5].markdown("**TTFT p90**")
        header_cols[6].markdown("**TPOT p90**")
        header_cols[7].markdown("**E2E p95**")
        header_cols[8].markdown("**Max QPS**")
        header_cols[9].markdown("**Cost/Month**")

        # Recommended option row
        cols = st.columns([0.8, 1.5, 2.5, 1.2, 0.8, 1, 1, 1, 1, 1])
        is_selected = st.session_state.selected_option_idx == 0
        button_label = "✅" if is_selected else "⚫"

        with cols[0]:
            if st.button(button_label, key="select_rec", use_container_width=True, disabled=is_selected):
                st.session_state.selected_option_idx = 0
                st.rerun()

        cols[1].markdown("**Recommended**")
        cols[2].markdown(rec['model_name'])
        cols[3].markdown(f"{rec['gpu_config']['gpu_count']}x {rec['gpu_config']['gpu_type']}")
        cols[4].markdown(f"{rec['gpu_config']['replicas']}")
        cols[5].markdown(f"{rec['predicted_ttft_p90_ms']}")
        cols[6].markdown(f"{rec['predicted_tpot_p90_ms']}")
        cols[7].markdown(f"{rec['predicted_e2e_p95_ms']}")
        cols[8].markdown(f"{rec['predicted_throughput_qps']:.0f}")
        cols[9].markdown(f"${rec['cost_per_month_usd']:,.0f}")

        # Alternative rows
        for i, alt in enumerate(alternatives, 1):
            cols = st.columns([0.8, 1.5, 2.5, 1.2, 0.8, 1, 1, 1, 1, 1])
            is_selected = st.session_state.selected_option_idx == i
            button_label = "✅" if is_selected else "⚫"

            with cols[0]:
                if st.button(button_label, key=f"select_alt_{i}", use_container_width=True, disabled=is_selected):
                    st.session_state.selected_option_idx = i
                    st.rerun()

            cols[1].markdown(f"**Option {i+1}**")
            cols[2].markdown(alt['model_name'])
            cols[3].markdown(f"{alt['gpu_config']['gpu_count']}x {alt['gpu_config']['gpu_type']}")
            cols[4].markdown(f"{alt['gpu_config']['replicas']}")
            cols[5].markdown(f"{alt['predicted_ttft_p90_ms']}")
            cols[6].markdown(f"{alt['predicted_tpot_p90_ms']}")
            cols[7].markdown(f"{alt['predicted_e2e_p95_ms']}")
            cols[8].markdown(f"{alt['predicted_throughput_qps']:.0f}")
            cols[9].markdown(f"${alt['cost_per_month_usd']:,.0f}")
    else:
        st.info("💡 No alternative options available. This is the only configuration that meets your SLO requirements.")


def render_specifications_tab(rec: dict[str, Any]):
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


def render_performance_tab(rec: dict[str, Any]):
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
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Capacity", f"{rec['predicted_throughput_qps']:.1f} QPS")
    with col2:
        expected_qps = rec['traffic_profile']['expected_qps']
        delta_qps = rec['predicted_throughput_qps'] - expected_qps
        st.metric("Expected Load", f"{expected_qps:.1f} QPS",
                 delta=f"+{delta_qps:.1f} headroom" if delta_qps > 0 else f"{delta_qps:.1f} over capacity",
                 delta_color="normal" if delta_qps > 0 else "inverse")


def render_cost_tab(rec: dict[str, Any]):
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


def check_cluster_status():
    """Check if Kubernetes cluster is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/cluster-status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            st.session_state.cluster_accessible = status.get("accessible", False)
            return status
        return {"accessible": False}
    except:
        st.session_state.cluster_accessible = False
        return {"accessible": False}


def generate_deployment_yaml(rec: dict[str, Any]):
    """Generate deployment YAML files via API."""
    try:
        with st.spinner("Generating deployment YAML files..."):
            response = requests.post(
                f"{API_BASE_URL}/api/deploy",
                json={"recommendation": rec, "namespace": "default"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                st.session_state.deployment_id = result["deployment_id"]
                st.session_state.deployment_files = result["files"]
                # Reset deployment flag when generating new YAML files
                st.session_state.deployed_to_cluster = False

                st.success("✅ Deployment files generated successfully!")
                st.info(f"**Deployment ID:** `{result['deployment_id']}`")

                # Show file paths
                st.markdown("**Generated Files:**")
                for config_type, file_path in result["files"].items():
                    st.code(file_path, language="text")

                # Check cluster status
                cluster_status = check_cluster_status()
                if cluster_status.get("accessible"):
                    st.markdown("---")
                    st.success("✅ Kubernetes cluster is accessible!")
                    st.markdown("**Next:** Click **Deploy to Kubernetes** to deploy to the cluster!")
                else:
                    st.markdown("---")
                    st.warning("⚠️ Kubernetes cluster not accessible. YAML files generated but not deployed.")
                    st.markdown("**Next:** Go to the **Monitoring** tab to see simulated observability metrics!")

            else:
                st.error(f"Failed to generate YAML: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"❌ Error generating deployment: {str(e)}")


def deploy_to_cluster(rec: dict[str, Any]):
    """Deploy model to Kubernetes cluster."""
    try:
        with st.spinner("Deploying to Kubernetes cluster..."):
            response = requests.post(
                f"{API_BASE_URL}/api/deploy-to-cluster",
                json={"recommendation": rec, "namespace": "default"},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                st.session_state.deployment_id = result["deployment_id"]
                st.session_state.deployment_files = result["files"]
                st.session_state.deployed_to_cluster = True

                st.success("✅ Successfully deployed to Kubernetes cluster!")
                st.info(f"**Deployment ID:** `{result['deployment_id']}`")
                st.info(f"**Namespace:** `{result['namespace']}`")

                # Show deployment details
                st.markdown("**Deployed Resources:**")
                deployment_result = result.get("deployment_result", {})
                for applied_file in deployment_result.get("applied_files", []):
                    st.markdown(f"- ✅ {applied_file['file']}")

                st.markdown("---")
                st.markdown("**Next:** Go to the **Monitoring** tab to see actual deployment status!")

            elif response.status_code == 503:
                st.error("❌ Kubernetes cluster not accessible. Ensure KIND cluster is running.")
                st.code("kind get clusters", language="bash")
            else:
                st.error(f"Failed to deploy: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"❌ Error deploying to cluster: {str(e)}")


def render_deployments_page():
    """Render standalone deployments page (when no recommendation exists)."""

    # Load all deployments from cluster
    all_deployments = load_all_deployments()

    if all_deployments is None:
        st.warning("⚠️ Could not connect to cluster to list deployments")
        return

    if len(all_deployments) == 0:
        st.info("""
        📦 **No deployments found in cluster**

        To create a deployment:
        1. Start a conversation describing your use case
        2. Review the recommendation
        3. Click "Deploy to Kubernetes" in the Cost tab
        """)
        return

    # Show deployment selector
    st.markdown(f"**Found {len(all_deployments)} deployment(s) in cluster**")

    deployment_options = {d["deployment_id"]: d for d in all_deployments}
    deployment_ids = list(deployment_options.keys())

    # Initialize selected deployment if not set
    if "selected_deployment" not in st.session_state or st.session_state.selected_deployment not in deployment_ids:
        st.session_state.selected_deployment = deployment_ids[0] if deployment_ids else None

    # Deployment selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Select deployment to monitor:",
            deployment_ids,
            index=deployment_ids.index(st.session_state.selected_deployment) if st.session_state.selected_deployment in deployment_ids else 0,
            key="deployment_selector_standalone"
        )
        st.session_state.selected_deployment = selected

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_standalone"):
            st.rerun()

    if not st.session_state.selected_deployment:
        return

    deployment_info = deployment_options[st.session_state.selected_deployment]

    # Show deployment management options
    render_deployment_management(deployment_info, context="standalone")
    st.markdown("---")

    # Show K8s status and inference testing
    render_k8s_status_for_deployment(deployment_info, context="standalone")
    st.markdown("---")
    render_inference_testing_for_deployment(deployment_info, context="standalone")


def load_all_deployments():
    """Load all InferenceServices from the cluster."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/deployments", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("deployments", [])
        elif response.status_code == 503:
            return None  # Cluster not accessible
        else:
            st.error(f"Failed to load deployments: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Error loading deployments: {str(e)}")
        return None


def render_deployment_management(deployment_info: dict[str, Any], context: str = "default"):
    """Render deployment management controls (delete, etc.).

    Args:
        deployment_info: Deployment information dictionary
        context: Unique context identifier to avoid key collisions (e.g., 'mgmt', 'monitoring', 'assistant')
    """
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})

    st.markdown("#### 🎛️ Deployment Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        ready_status = "✅ Ready" if status.get("ready") else "⏳ Pending"
        st.metric("Status", ready_status)

    with col2:
        pods = deployment_info.get("pods", [])
        st.metric("Pods", len(pods))

    with col3:
        if st.button("🗑️ Delete Deployment", use_container_width=True, type="secondary", key=f"delete_btn_{context}_{deployment_id}"):
            if st.session_state.get(f"confirm_delete_{deployment_id}"):
                # Actually delete
                with st.spinner("Deleting deployment..."):
                    try:
                        response = requests.delete(
                            f"{API_BASE_URL}/api/deployments/{deployment_id}",
                            timeout=30
                        )
                        if response.status_code == 200:
                            st.success(f"✅ Deleted {deployment_id}")
                            # Clear session state for this deployment
                            if st.session_state.deployment_id == deployment_id:
                                st.session_state.deployment_id = None
                                st.session_state.deployed_to_cluster = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete: {response.text}")
                    except Exception as e:
                        st.error(f"Error deleting deployment: {str(e)}")
                # Reset confirmation
                st.session_state[f"confirm_delete_{deployment_id}"] = False
            else:
                # Ask for confirmation
                st.session_state[f"confirm_delete_{deployment_id}"] = True
                st.warning(f"⚠️ Click again to confirm deletion of {deployment_id}")


def render_k8s_status_for_deployment(deployment_info: dict[str, Any], context: str = "default"):
    """Render Kubernetes status for a specific deployment.

    Args:
        deployment_info: Deployment information dictionary
        context: Unique context identifier to avoid key collisions
    """
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})
    pods = deployment_info.get("pods", [])

    st.markdown("#### ☸️ Kubernetes Status")

    # InferenceService status
    if status.get("exists"):
        st.success(f"✅ InferenceService: **{deployment_id}**")
        st.markdown(f"**Ready:** {'✅ Yes' if status.get('ready') else '⏳ Not yet'}")

        if status.get("url"):
            st.markdown(f"**URL:** `{status['url']}`")

        # Show conditions
        with st.expander("📋 Resource Conditions"):
            for condition in status.get("conditions", []):
                status_icon = "✅" if condition.get("status") == "True" else "⏳"
                st.markdown(f"{status_icon} **{condition.get('type')}**: {condition.get('message', 'N/A')}")
    else:
        st.warning(f"⚠️ InferenceService not found: {status.get('error', 'Unknown error')}")

    # Pod status
    if pods:
        st.markdown(f"**Pods:** {len(pods)} pod(s)")
        with st.expander("🔍 Pod Details"):
            for pod in pods:
                st.markdown(f"**{pod.get('name')}**")
                st.markdown(f"- Phase: {pod.get('phase')}")
                node_name = pod.get('node_name') or 'Not assigned'
                st.markdown(f"- Node: {node_name}")
    else:
        st.info("ℹ️ No pods found yet (may still be creating)")


def render_inference_testing_for_deployment(deployment_info: dict[str, Any], context: str = "default"):
    """Render inference testing for a specific deployment.

    Args:
        deployment_info: Deployment information dictionary
        context: Unique context identifier to avoid key collisions
    """
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})

    st.markdown("#### 🧪 Inference Testing")

    if not status.get("ready"):
        st.info("⏳ Deployment not ready yet. Inference testing will be available once the service is ready.")
        return

    # Test prompt input
    col1, col2 = st.columns([2, 1])
    with col1:
        test_prompt = st.text_area(
            "Test Prompt",
            value="Write a Python function that calculates the fibonacci sequence.",
            height=100,
            key=f"test_prompt_{context}_{deployment_id}"
        )

    with col2:
        max_tokens = st.number_input("Max Tokens", value=150, min_value=10, max_value=500, key=f"max_tokens_{context}_{deployment_id}")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key=f"temperature_{context}_{deployment_id}")

    # Test button
    if st.button("🚀 Send Test Request", use_container_width=True, key=f"test_button_{context}_{deployment_id}"):
        with st.spinner("Sending inference request..."):
            try:
                import json
                import subprocess
                import time

                # Get service name (KServe appends "-predictor" to deployment_id)
                service_name = f"{deployment_id}-predictor"

                # Use kubectl port-forward in background, then send request
                st.info(f"📡 Connecting to service: `{service_name}`")

                # Start port-forward in background
                # KServe services expose port 80 (which maps to container port 8080)
                port_forward_proc = subprocess.Popen(
                    ["kubectl", "port-forward", f"svc/{service_name}", "8080:80"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Give it a moment to establish connection
                time.sleep(3)

                # Check if port-forward is still running
                if port_forward_proc.poll() is not None:
                    # Process exited
                    pf_stdout, pf_stderr = port_forward_proc.communicate()
                    st.error("❌ Port-forward failed to start")
                    st.code(f"stdout: {pf_stdout.decode()}\nstderr: {pf_stderr.decode()}", language="text")
                    return

                try:
                    # Send inference request
                    start_time = time.time()

                    curl_cmd = [
                        "curl", "-s", "-X", "POST",
                        "http://localhost:8080/v1/completions",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps({
                            "prompt": test_prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        })
                    ]

                    # Show the command being executed for debugging
                    with st.expander("🔍 Debug Info"):
                        st.code(" ".join(curl_cmd), language="bash")

                    result = subprocess.run(
                        curl_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    elapsed_time = time.time() - start_time

                    if result.returncode == 0 and result.stdout:
                        try:
                            response_data = json.loads(result.stdout)

                            # Display response
                            st.success(f"✅ Response received in {elapsed_time:.2f}s")

                            # Show the generated text
                            st.markdown("**Generated Response:**")
                            response_text = response_data.get("choices", [{}])[0].get("text", "")
                            st.code(response_text, language=None)

                            # Show usage stats
                            usage = response_data.get("usage", {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prompt Tokens", usage.get("prompt_tokens", 0))
                            with col2:
                                st.metric("Completion Tokens", usage.get("completion_tokens", 0))
                            with col3:
                                st.metric("Total Tokens", usage.get("total_tokens", 0))

                            # Show timing
                            st.metric("Total Latency", f"{elapsed_time:.2f}s")

                            # Show raw response in expander
                            with st.expander("📋 Raw API Response"):
                                st.json(response_data)

                        except json.JSONDecodeError as e:
                            st.error(f"❌ Failed to parse JSON response: {e}")
                            st.markdown("**Raw stdout:**")
                            st.code(result.stdout, language="text")

                    else:
                        st.error(f"❌ Request failed (return code: {result.returncode})")

                        if result.stdout:
                            st.markdown("**stdout:**")
                            st.code(result.stdout, language="text")

                        if result.stderr:
                            st.markdown("**stderr:**")
                            st.code(result.stderr, language="text")

                        if not result.stdout and not result.stderr:
                            st.warning("No output captured from curl command. Port-forward may have failed.")

                finally:
                    # Clean up port-forward process
                    port_forward_proc.terminate()
                    port_forward_proc.wait(timeout=2)

            except subprocess.TimeoutExpired:
                st.error("❌ Request timed out (30s). The model may still be starting up.")
                try:
                    port_forward_proc.terminate()
                except:
                    pass
            except Exception as e:
                st.error(f"❌ Error testing inference: {str(e)}")
                import traceback
                with st.expander("🔍 Full Error Traceback"):
                    st.code(traceback.format_exc(), language="text")

    # Add helpful notes
    with st.expander("ℹ️ How Inference Testing Works"):
        st.markdown("""
        **Process:**
        1. Uses `kubectl port-forward` to connect to the InferenceService
        2. Sends a POST request to `/v1/completions` (OpenAI-compatible API)
        3. Displays the response and metrics

        **Note:** This temporarily forwards port 8080 on your local machine to the service.
        """)


def render_yaml_preview_tab(rec: dict[str, Any]):
    """Render YAML preview tab showing generated deployment files."""

    st.markdown("### 📄 Deployment YAML Files")

    # Check if YAML was auto-generated
    if not rec.get("yaml_generated", False):
        st.warning("⚠️ YAML files were not generated automatically. This may indicate an error during recommendation creation.")
        if st.button("🔄 Regenerate YAML", use_container_width=True):
            generate_deployment_yaml(rec)
            st.rerun()
        return

    st.success("✅ YAML files generated automatically")

    # Fetch YAML files from backend
    try:
        deployment_id = rec.get("deployment_id")
        if not deployment_id:
            deployment_id = f"{rec['intent']['use_case']}-{rec['model_id'].split('/')[-1]}"

        response = requests.get(
            f"{API_BASE_URL}/api/deployments/{deployment_id}/yaml",
            timeout=10
        )

        if response.status_code == 200:
            yaml_data = response.json()

            # Show each YAML file in an expander
            st.markdown("#### Generated Files")

            for filename, content in yaml_data.get("files", {}).items():
                with st.expander(f"📄 {filename}", expanded=False):
                    st.code(content, language="yaml")

                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/yaml"
                    )
        else:
            st.error(f"Failed to fetch YAML files: {response.text}")

    except Exception as e:
        st.error(f"Error fetching YAML preview: {str(e)}")
        st.info("💡 The YAML files are stored on the backend and ready for deployment.")


def render_monitoring_tab(rec: dict[str, Any]):
    """Render monitoring dashboard."""

    st.markdown("### 📡 Deployment Monitoring")

    # Load all deployments from cluster
    all_deployments = load_all_deployments()

    if all_deployments is None:
        st.warning("⚠️ Could not connect to cluster to list deployments")
        return

    if len(all_deployments) == 0:
        st.info("""
        👈 **No deployments found!**

        Click **Deploy to Kubernetes** in the Actions section to deploy to the cluster.

        This dashboard will show all InferenceServices deployed to the cluster, allowing you to:
        - Monitor deployment status
        - Test inference endpoints
        - Delete deployments
        """)
        return

    # Show deployment selector
    st.markdown(f"**Found {len(all_deployments)} deployment(s) in cluster**")

    deployment_options = {d["deployment_id"]: d for d in all_deployments}
    deployment_ids = list(deployment_options.keys())

    # Initialize selected deployment if not set
    if "selected_deployment" not in st.session_state or st.session_state.selected_deployment not in deployment_ids:
        st.session_state.selected_deployment = deployment_ids[0] if deployment_ids else None

    # Deployment selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Select deployment to monitor:",
            deployment_ids,
            index=deployment_ids.index(st.session_state.selected_deployment) if st.session_state.selected_deployment in deployment_ids else 0,
            key="deployment_selector"
        )
        st.session_state.selected_deployment = selected

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_monitoring"):
            st.rerun()

    if not st.session_state.selected_deployment:
        return

    deployment_info = deployment_options[st.session_state.selected_deployment]

    # Show deployment management options
    render_deployment_management(deployment_info, context="monitoring")
    st.markdown("---")

    # Show K8s status and inference testing
    render_k8s_status_for_deployment(deployment_info, context="monitoring")
    st.markdown("---")
    render_inference_testing_for_deployment(deployment_info, context="monitoring")
    st.markdown("---")

    # Fetch mock monitoring data (if available)
    if st.session_state.deployment_id == st.session_state.selected_deployment:
        try:
            with st.spinner("Loading deployment metrics..."):
                response = requests.get(
                    f"{API_BASE_URL}/api/deployments/{st.session_state.selected_deployment}/status",
                    timeout=10
                )

                if response.status_code == 200:
                    status = response.json()
                    render_monitoring_dashboard(status, rec)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API.")
        except Exception as e:
            st.error(f"❌ Error fetching monitoring data: {str(e)}")


def render_k8s_status():
    """Render actual Kubernetes deployment status."""
    st.markdown("#### 🎛️ Kubernetes Cluster Status")

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/deployments/{st.session_state.deployment_id}/k8s-status",
            timeout=10
        )

        if response.status_code == 200:
            k8s_status = response.json()
            isvc = k8s_status.get("inferenceservice", {})
            pods = k8s_status.get("pods", [])

            # InferenceService status
            if isvc.get("exists"):
                st.success(f"✅ InferenceService: **{st.session_state.deployment_id}**")
                st.markdown(f"**Ready:** {'✅ Yes' if isvc.get('ready') else '⏳ Not yet'}")

                if isvc.get("url"):
                    st.markdown(f"**URL:** `{isvc['url']}`")

                # Show conditions
                with st.expander("📋 Resource Conditions"):
                    for condition in isvc.get("conditions", []):
                        status_icon = "✅" if condition.get("status") == "True" else "⏳"
                        st.markdown(f"{status_icon} **{condition.get('type')}**: {condition.get('message', 'N/A')}")
            else:
                st.warning(f"⚠️ InferenceService not found: {isvc.get('error', 'Unknown error')}")

            # Pod status
            if pods:
                st.markdown(f"**Pods:** {len(pods)} pod(s)")
                with st.expander("🔍 Pod Details"):
                    for pod in pods:
                        st.markdown(f"**{pod.get('name')}**")
                        st.markdown(f"- Phase: {pod.get('phase')}")
                        node_name = pod.get('node_name') or 'Not assigned'
                        st.markdown(f"- Node: {node_name}")
            else:
                st.info("ℹ️ No pods found yet (may still be creating)")

        elif response.status_code == 503:
            st.error("❌ Kubernetes cluster not accessible")
        else:
            st.error(f"Failed to fetch K8s status: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API.")
    except Exception as e:
        st.error(f"❌ Error fetching K8s status: {str(e)}")


def render_inference_testing():
    """Render inference testing UI component."""
    st.markdown("####  🧪 Inference Testing")

    st.markdown("""
    Test the deployed model by sending inference requests. This validates that the vLLM simulator
    is running and responding correctly.
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        # Test prompt input
        test_prompt = st.text_area(
            "Test Prompt",
            value="Write a Python function to calculate fibonacci numbers",
            height=100,
            help="Enter a prompt to test the model"
        )

    with col2:
        max_tokens = st.number_input("Max Tokens", value=150, min_value=10, max_value=500)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    # Test button
    if st.button("🚀 Send Test Request", use_container_width=True):
        with st.spinner("Sending inference request..."):
            try:
                import json
                import subprocess
                import time

                # Get service name (KServe appends "-predictor" to deployment_id)
                service_name = f"{st.session_state.deployment_id}-predictor"

                # Use kubectl port-forward in background, then send request
                st.info(f"📡 Connecting to service: `{service_name}`")

                # Start port-forward in background
                # KServe services expose port 80 (which maps to container port 8080)
                port_forward_proc = subprocess.Popen(
                    ["kubectl", "port-forward", f"svc/{service_name}", "8080:80"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Give it a moment to establish connection
                time.sleep(3)

                # Check if port-forward is still running
                if port_forward_proc.poll() is not None:
                    # Process exited
                    pf_stdout, pf_stderr = port_forward_proc.communicate()
                    st.error("❌ Port-forward failed to start")
                    st.code(f"stdout: {pf_stdout.decode()}\nstderr: {pf_stderr.decode()}", language="text")
                    return

                try:
                    # Send inference request
                    start_time = time.time()

                    curl_cmd = [
                        "curl", "-s", "-X", "POST",
                        "http://localhost:8080/v1/completions",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps({
                            "prompt": test_prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        })
                    ]

                    # Show the command being executed for debugging
                    with st.expander("🔍 Debug Info"):
                        st.code(" ".join(curl_cmd), language="bash")

                    result = subprocess.run(
                        curl_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    elapsed_time = time.time() - start_time

                    if result.returncode == 0 and result.stdout:
                        try:
                            response_data = json.loads(result.stdout)

                            # Display response
                            st.success(f"✅ Response received in {elapsed_time:.2f}s")

                            # Show the generated text
                            st.markdown("**Generated Response:**")
                            response_text = response_data.get("choices", [{}])[0].get("text", "")
                            st.code(response_text, language=None)

                            # Show usage stats
                            usage = response_data.get("usage", {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prompt Tokens", usage.get("prompt_tokens", 0))
                            with col2:
                                st.metric("Completion Tokens", usage.get("completion_tokens", 0))
                            with col3:
                                st.metric("Total Tokens", usage.get("total_tokens", 0))

                            # Show timing
                            st.metric("Total Latency", f"{elapsed_time:.2f}s")

                            # Show raw response in expander
                            with st.expander("📋 Raw API Response"):
                                st.json(response_data)

                        except json.JSONDecodeError as e:
                            st.error(f"❌ Failed to parse JSON response: {e}")
                            st.markdown("**Raw stdout:**")
                            st.code(result.stdout, language="text")

                    else:
                        st.error(f"❌ Request failed (return code: {result.returncode})")

                        if result.stdout:
                            st.markdown("**stdout:**")
                            st.code(result.stdout, language="text")

                        if result.stderr:
                            st.markdown("**stderr:**")
                            st.code(result.stderr, language="text")

                        if not result.stdout and not result.stderr:
                            st.warning("No output captured from curl command. Port-forward may have failed.")

                finally:
                    # Clean up port-forward process
                    port_forward_proc.terminate()
                    port_forward_proc.wait(timeout=2)

            except subprocess.TimeoutExpired:
                st.error("❌ Request timed out (30s). The model may still be starting up.")
                try:
                    port_forward_proc.terminate()
                except:
                    pass
            except Exception as e:
                st.error(f"❌ Error testing inference: {str(e)}")
                import traceback
                with st.expander("🔍 Full Error Traceback"):
                    st.code(traceback.format_exc(), language="text")

    # Add helpful notes
    with st.expander("ℹ️ How Inference Testing Works"):
        st.markdown("""
        **Process:**
        1. Creates temporary port-forward to the Kubernetes service
        2. Sends HTTP POST request to `/v1/completions` endpoint
        3. Displays the response and metrics
        4. Closes the port-forward connection

        **Simulator Mode:**
        - Returns canned responses based on prompt patterns
        - Simulates realistic latency (TTFT/TPOT from benchmarks)
        - No actual model inference occurs

        **Production Mode (future):**
        - Would use real vLLM inference
        - Actual token generation
        - Real GPU utilization
        """)


def render_monitoring_dashboard(status: dict[str, Any], rec: dict[str, Any]):
    """Render the actual monitoring dashboard with metrics."""

    deployment_id = status["deployment_id"]

    st.markdown(f"**Deployment ID:** `{deployment_id}`")
    st.markdown(f"**Status:** 🟢 {status['status'].upper()} (Simulated)")

    st.markdown("---")

    # SLO Compliance
    st.markdown("### ✅ SLO Compliance (Last 7 Days) - Simulated Data")

    slo = status["slo_compliance"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ttft_delta = slo["ttft_p90_ms"] - slo["ttft_target_ms"]
        st.metric(
            "TTFT p90",
            f"{slo['ttft_p90_ms']}ms",
            delta=f"{ttft_delta}ms vs target",
            delta_color="inverse"
        )
        st.caption(f"Target: {slo['ttft_target_ms']}ms {'✓' if slo['ttft_compliant'] else '⚠️'}")

    with col2:
        tpot_delta = slo["tpot_p90_ms"] - slo["tpot_target_ms"]
        st.metric(
            "TPOT p90",
            f"{slo['tpot_p90_ms']}ms",
            delta=f"{tpot_delta}ms vs target",
            delta_color="inverse"
        )
        st.caption(f"Target: {slo['tpot_target_ms']}ms {'✓' if slo['tpot_compliant'] else '⚠️'}")

    with col3:
        e2e_delta = slo["e2e_p95_ms"] - slo["e2e_target_ms"]
        st.metric(
            "E2E p95",
            f"{slo['e2e_p95_ms']}ms",
            delta=f"{e2e_delta}ms vs target",
            delta_color="inverse"
        )
        st.caption(f"Target: {slo['e2e_target_ms']}ms {'✓' if slo['e2e_compliant'] else '⚠️'}")

    with col4:
        qps_delta = slo["throughput_qps"] - slo["throughput_target_qps"]
        st.metric(
            "Throughput",
            f"{slo['throughput_qps']} QPS",
            delta=f"{qps_delta:+.0f} vs target"
        )
        st.caption(f"Target: {slo['throughput_target_qps']} QPS {'✓' if slo['throughput_compliant'] else '⚠️'}")

    # Uptime
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        uptime_delta = slo["uptime_pct"] - slo["uptime_target_pct"]
        st.metric(
            "Uptime",
            f"{slo['uptime_pct']:.2f}%",
            delta=f"{uptime_delta:+.2f}% vs target"
        )
        st.caption(f"Target: {slo['uptime_target_pct']}% {'✓' if slo['uptime_compliant'] else '⚠️'}")

    # Resource Utilization
    st.markdown("---")
    st.markdown("### 🖥️ Resource Utilization")

    util = status["resource_utilization"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("GPU Utilization", f"{util['gpu_utilization_pct']}%")
        st.caption("Target: >80% for cost efficiency")
        if util['gpu_utilization_pct'] < 80:
            st.warning("⚠️ Below efficiency target")

    with col2:
        st.metric(
            "GPU Memory",
            f"{util['gpu_memory_used_gb']:.1f} GB",
            delta=f"of {util['gpu_memory_total_gb']} GB"
        )

    with col3:
        st.metric("Avg Batch Size", util['avg_batch_size'])
        st.caption(f"Queue depth: {util['queue_depth']}")

    # Cost Analysis
    st.markdown("---")
    st.markdown("### 💰 Cost Analysis")

    cost = status["cost_analysis"]

    col1, col2 = st.columns(2)

    with col1:
        cost_diff_month = cost["actual_cost_per_month_usd"] - cost["predicted_cost_per_month_usd"]
        st.metric(
            "Monthly Cost",
            f"${cost['actual_cost_per_month_usd']:.0f}",
            delta=f"${cost_diff_month:+.0f} vs predicted"
        )
        st.caption(f"Predicted: ${cost['predicted_cost_per_month_usd']:.0f}")

    with col2:
        st.metric(
            "Cost per 1k Tokens",
            f"${cost['cost_per_1k_tokens_usd']:.3f}"
        )
        st.caption(f"Predicted: ${cost['predicted_cost_per_1k_tokens_usd']:.3f}")

    # Traffic Patterns
    st.markdown("---")
    st.markdown("### 📊 Traffic Patterns")

    traffic = status["traffic_patterns"]

    col1, col2, col3 = st.columns(3)

    with col1:
        prompt_diff = traffic["avg_prompt_tokens"] - traffic["predicted_prompt_tokens"]
        st.metric(
            "Avg Prompt Tokens",
            traffic["avg_prompt_tokens"],
            delta=f"{prompt_diff:+d} vs predicted"
        )
        st.caption(f"Predicted: {traffic['predicted_prompt_tokens']}")

    with col2:
        gen_diff = traffic["avg_generation_tokens"] - traffic["predicted_generation_tokens"]
        st.metric(
            "Avg Generation Tokens",
            traffic["avg_generation_tokens"],
            delta=f"{gen_diff:+d} vs predicted"
        )
        st.caption(f"Predicted: {traffic['predicted_generation_tokens']}")

    with col3:
        qps_diff = traffic["peak_qps"] - traffic["predicted_peak_qps"]
        st.metric(
            "Peak QPS",
            traffic["peak_qps"],
            delta=f"{qps_diff:+d} vs predicted"
        )
        st.caption(f"Predicted: {traffic['predicted_peak_qps']}")

    # Request volume
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Requests (Last Hour)", f"{traffic['requests_last_hour']:,}")
    with col2:
        st.metric("Requests (Last 24h)", f"{traffic['requests_last_24h']:,}")

    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Optimization Recommendations")

    for recommendation in status.get("recommendations", []):
        st.info(recommendation)

    st.markdown("---")
    st.caption("**Note:** This is simulated monitoring data for POC purposes. In production, this would connect to Prometheus/Grafana.")


if __name__ == "__main__":
    main()
