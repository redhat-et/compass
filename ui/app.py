"""Streamlit UI for Compass.

This module provides the main Streamlit interface for Compass, featuring:
1. Chat interface for conversational requirement gathering
2. Recommendation display with all specification details
3. Editable specification component for user review/modification
4. Integration with FastAPI backend
"""

import contextlib
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
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

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
if "ranked_recommendations" not in st.session_state:
    st.session_state.ranked_recommendations = None
if "ranked_request_message" not in st.session_state:
    st.session_state.ranked_request_message = None


def main():
    """Main application entry point."""

    # Header
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("docs/compass-logo.svg", width=50)
    with col2:
        st.markdown('<div class="main-header">Compass</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">From concept to production-ready LLM deployment</div>',
            unsafe_allow_html=True,
        )

    # Sidebar
    render_sidebar()

    # Top-level tabs
    main_tabs = st.tabs(
        ["💬 Chat", "📊 Recommendation Details", "🏆 Ranked Options", "📦 Deployment Management"]
    )

    with main_tabs[0]:
        render_assistant_tab()

    with main_tabs[1]:
        render_recommendation_details_tab()

    with main_tabs[2]:
        render_ranked_options_tab()

    with main_tabs[3]:
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
        button_disabled = (
            not st.session_state.cluster_accessible or st.session_state.deployed_to_cluster
        )
        button_label = (
            "✅ Deployed" if st.session_state.deployed_to_cluster else "🚢 Deploy to Kubernetes"
        )
        button_help = (
            "Already deployed to cluster"
            if st.session_state.deployed_to_cluster
            else (
                "Deploy to Kubernetes cluster (YAML auto-generated)"
                if st.session_state.cluster_accessible
                else "Kubernetes cluster not accessible"
            )
        )

        if st.button(
            button_label,
            use_container_width=True,
            type="primary",
            disabled=button_disabled,
            help=button_help,
        ):
            deploy_to_cluster(get_selected_option())

        if st.session_state.recommendation.get("yaml_generated", False):
            st.caption("💡 YAML files auto-generated. View in the **Recommendation Details** tab.")


def get_selected_option():
    """Get the currently selected deployment option (recommended or alternative)."""
    rec = st.session_state.recommendation
    selected_idx = st.session_state.get("selected_option_idx", 0)

    if selected_idx == 0:
        # Return the recommended option (current recommendation)
        return rec
    else:
        # Return the selected alternative
        alternatives = rec.get("alternative_options", [])
        if selected_idx - 1 < len(alternatives):
            alt = alternatives[selected_idx - 1]
            # Create a recommendation dict from the alternative
            selected_rec = rec.copy()
            selected_rec["model_name"] = alt["model_name"]
            selected_rec["model_id"] = alt["model_id"]
            selected_rec["gpu_config"] = alt["gpu_config"]
            selected_rec["predicted_ttft_p95_ms"] = alt["predicted_ttft_p95_ms"]
            selected_rec["predicted_itl_p95_ms"] = alt["predicted_itl_p95_ms"]
            selected_rec["predicted_e2e_p95_ms"] = alt["predicted_e2e_p95_ms"]
            selected_rec["predicted_throughput_qps"] = alt["predicted_throughput_qps"]
            selected_rec["cost_per_hour_usd"] = alt["cost_per_hour_usd"]
            selected_rec["cost_per_month_usd"] = alt["cost_per_month_usd"]
            selected_rec["reasoning"] = alt["reasoning"]
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

        button_disabled = (
            not st.session_state.cluster_accessible or st.session_state.deployed_to_cluster
        )
        button_label = (
            "✅ Deployed" if st.session_state.deployed_to_cluster else "🚢 Deploy to Kubernetes"
        )
        button_help = (
            "Already deployed to cluster"
            if st.session_state.deployed_to_cluster
            else (
                "Deploy to Kubernetes cluster (YAML auto-generated)"
                if st.session_state.cluster_accessible
                else "Kubernetes cluster not accessible"
            )
        )

        # Show which option will be deployed
        selected_idx = st.session_state.get("selected_option_idx", 0)
        if selected_idx == 0:
            st.caption("📌 **Recommended** option will be deployed")
        else:
            st.caption(f"📌 **Option {selected_idx+1}** will be deployed")

        if st.button(
            button_label,
            key="deploy_from_details",
            use_container_width=True,
            type="primary",
            disabled=button_disabled,
            help=button_help,
        ):
            deploy_to_cluster(get_selected_option())
    else:
        st.info(
            "👈 Start a conversation in the **Assistant** tab to get deployment recommendations"
        )


def fetch_ranked_recommendations(
    message: str, min_accuracy: int | None = None, max_cost: float | None = None
):
    """Fetch ranked recommendations from the API."""
    try:
        request_data = {
            "message": message,
            "include_near_miss": True,
        }
        if min_accuracy and min_accuracy > 0:
            request_data["min_accuracy"] = min_accuracy
        if max_cost and max_cost > 0:
            request_data["max_cost"] = max_cost

        response = requests.post(
            f"{API_BASE_URL}/api/ranked-recommend", json=request_data, timeout=60
        )
        response.raise_for_status()
        st.session_state.ranked_recommendations = response.json()
        st.session_state.ranked_request_message = message
        return True
    except requests.RequestException as e:
        st.error(f"Failed to fetch ranked recommendations: {e}")
        return False


def render_ranked_options_tab():
    """Render the ranked options tab with multi-criteria views."""
    st.markdown("### 🏆 Ranked Deployment Options")
    st.caption("Explore configurations ranked by different criteria")

    # Check if we have a recommendation to base ranking on
    if not st.session_state.recommendation:
        st.info(
            "👈 Start a conversation in the **Chat** tab to get ranked deployment recommendations"
        )
        return

    # Get the original message from the conversation
    user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
    if not user_messages:
        st.warning("No conversation history found")
        return

    last_user_message = user_messages[-1]["content"]

    # Filters section
    with st.expander("🔍 Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_accuracy = st.slider(
                "Minimum Accuracy Score",
                min_value=0,
                max_value=100,
                value=0,
                help="Filter out configurations below this accuracy threshold (0 = no filter)",
            )
        with col2:
            max_cost = st.number_input(
                "Maximum Monthly Cost ($)",
                min_value=0,
                value=0,
                help="Filter out configurations above this cost (0 = no filter)",
            )

        if st.button("Apply Filters & Refresh"):
            with st.spinner("Fetching ranked recommendations..."):
                fetch_ranked_recommendations(
                    last_user_message,
                    min_accuracy if min_accuracy > 0 else None,
                    max_cost if max_cost > 0 else None,
                )
            st.rerun()

    # Fetch rankings if not cached or message changed
    if (
        st.session_state.ranked_recommendations is None
        or st.session_state.ranked_request_message != last_user_message
    ):
        with st.spinner("Fetching ranked recommendations..."):
            if not fetch_ranked_recommendations(last_user_message):
                return

    ranked = st.session_state.ranked_recommendations
    if not ranked:
        st.warning("No ranked recommendations available")
        return

    # Statistics
    st.markdown(
        f"**{ranked.get('total_configs_evaluated', 0)}** configurations evaluated, "
        f"**{ranked.get('configs_after_filters', 0)}** after filters"
    )

    st.markdown("---")

    # Ranking view selector
    ranking_views = {
        "⚖️ Balanced": "balanced",
        "🎯 Best Accuracy": "best_accuracy",
        "💰 Lowest Cost": "lowest_cost",
        "⚡ Lowest Latency": "lowest_latency",
        "🔧 Simplest": "simplest",
    }

    selected_view = st.selectbox(
        "Sort by:",
        options=list(ranking_views.keys()),
        index=0,
        help="Choose how to rank the configurations",
    )

    view_key = ranking_views[selected_view]
    configs = ranked.get(view_key, [])

    if not configs:
        st.info(f"No configurations available for {selected_view} ranking")
        return

    # Display table header
    st.markdown(f"### Top {len(configs)} - {selected_view}")

    header_cols = st.columns([0.5, 1.8, 1.2, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 1])
    header_cols[0].markdown("**#**")
    header_cols[1].markdown("**Model**")
    header_cols[2].markdown("**GPU Config**")
    header_cols[3].markdown("**Replicas**")
    header_cols[4].markdown("**Accuracy**")
    header_cols[5].markdown("**Price**")
    header_cols[6].markdown("**Latency**")
    header_cols[7].markdown("**Simple**")
    header_cols[8].markdown("**Balanced**")
    header_cols[9].markdown("**Cost/Month**")

    # Display each configuration
    for i, config in enumerate(configs, 1):
        scores = config.get("scores", {})
        slo_status = scores.get("slo_status", "unknown")
        status_icon = "✅" if slo_status == "compliant" else "⚠️"

        gpu_config = config.get("gpu_config", {})
        tensor_parallel = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        gpu_str = f"{tensor_parallel}x {gpu_config.get('gpu_type', 'Unknown')}"

        cols = st.columns([0.5, 1.8, 1.2, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 1])
        cols[0].markdown(f"**{i}.**")
        cols[1].markdown(f"{status_icon} {config.get('model_name', 'Unknown')}")
        cols[2].markdown(gpu_str)
        cols[3].markdown(f"{replicas}")
        cols[4].markdown(f"{scores.get('accuracy_score', '-')}")
        cols[5].markdown(f"{scores.get('price_score', '-')}")
        cols[6].markdown(f"{scores.get('latency_score', '-')}")
        cols[7].markdown(f"{scores.get('complexity_score', '-')}")
        cols[8].markdown(f"{scores.get('balanced_score', '-'):.1f}")
        cols[9].markdown(f"${config.get('cost_per_month_usd', 0):,.0f}")

    # Legend
    st.markdown("---")
    st.caption("✅ Meets SLO targets | ⚠️ Near-miss (within 20% of SLO)")
    st.caption("Scores are 0-100 (higher is better)")


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

        table_data.append(
            {
                "Status": ready_icon,
                "Name": dep_id,
                "Pods": len(pods),
                "Ready": "Yes" if ready else "No",
            }
        )

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
    if (
        "selected_deployment" not in st.session_state
        or st.session_state.selected_deployment not in deployment_ids
    ):
        st.session_state.selected_deployment = deployment_ids[0] if deployment_ids else None

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Choose a deployment:",
            deployment_ids,
            index=deployment_ids.index(st.session_state.selected_deployment)
            if st.session_state.selected_deployment in deployment_ids
            else 0,
            key="deployment_selector_mgmt",
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
    st.markdown("---")

    # Show simulated observability metrics if available
    render_simulated_observability(deployment_info, context="mgmt")


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
            "Document summarization pipeline, high throughput, cost efficient",
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
                    help=dep_id,  # Tooltip shows full deployment ID
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
        with chat_container, st.chat_message("user"):
            st.markdown(prompt)

        # Get recommendation from API
        with st.spinner("Analyzing requirements and generating recommendation..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/recommend", json={"message": prompt}, timeout=30
                )

                if response.status_code == 200:
                    recommendation = response.json()
                    st.session_state.recommendation = recommendation

                    # Reset deployment state for new recommendation
                    st.session_state.deployed_to_cluster = False
                    st.session_state.deployment_id = None

                    # Add assistant response
                    assistant_message = format_recommendation_summary(recommendation)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_message}
                    )

                    st.rerun()
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to backend API. Make sure the FastAPI server is running on http://localhost:8000"
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def format_recommendation_summary(rec: dict[str, Any]) -> str:
    """Format recommendation as a chat message."""

    # Check if this is a spec-only response (no viable config found)
    has_config = rec.get("model_name") is not None and rec.get("gpu_config") is not None

    if not has_config:
        # No viable configuration found
        summary = f"""
I've analyzed your requirements:

{rec['reasoning']}

👉 Review the **Specifications** tab to see what I understood from your request, then adjust your SLO targets or requirements to find viable configurations.
"""
        return summary.strip()

    # Has viable configuration
    meets_slo = rec.get("meets_slo", False)
    slo_status = "✅ Meets SLO" if meets_slo else "⚠️ Does not meet SLO"

    summary = f"""
I've analyzed your requirements and recommend the following solution:

**{rec['model_name']}** on **{rec['gpu_config']['gpu_count']}x {rec['gpu_config']['gpu_type']}**

**Performance:**
- TTFT p95: {rec['predicted_ttft_p95_ms']}ms
- ITL p95: {rec['predicted_itl_p95_ms']}ms
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
    tabs = st.tabs(
        ["📋 Overview", "⚙️ Specifications", "📊 Performance", "💰 Cost", "📄 YAML Preview"]
    )

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

    # Check if this is a spec-only response (no viable config found)
    has_config = rec.get("model_name") is not None and rec.get("gpu_config") is not None

    if not has_config:
        # No viable configuration found - show friendly message
        st.error("❌ No viable deployment configurations found meeting SLO targets")
        st.markdown("---")
        st.markdown("### 💡 What you can do:")
        st.markdown("""
        1. **Review the Specifications tab** to see what Compass understood from your request
        2. **Relax your SLO targets** in the Specifications tab and click "Save & Re-Evaluate"
        3. **Adjust traffic parameters** (expected QPS, token lengths) if they seem too high
        4. **Try a different use case** that may have less stringent latency requirements
        """)
        st.markdown("---")
        st.markdown("### 📝 Details")
        st.markdown(rec.get("reasoning", "No additional details available"))
        return

    # SLO Status Badge
    meets_slo = rec.get("meets_slo", False)
    if meets_slo:
        st.markdown('<span class="success-badge">✅ MEETS SLO</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="warning-badge">⚠️ DOES NOT MEET SLO</span>', unsafe_allow_html=True
        )

    st.markdown("---")

    # Model and GPU
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Model")
        st.markdown(f"**{rec['model_name']}**")
        st.caption(f"ID: `{rec['model_id']}`")

    with col2:
        st.markdown("### 🖥️ GPU Configuration")
        gpu_config = rec["gpu_config"]
        st.markdown(f"**{gpu_config['tensor_parallel']}x {gpu_config['gpu_type']}**")
        st.caption(
            f"Tensor Parallel: {gpu_config['tensor_parallel']}, Replicas: {gpu_config['replicas']}"
        )

    st.markdown("---")

    # Key Metrics
    st.markdown("### 📈 Key Metrics")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric("TTFT p95", f"{rec['predicted_ttft_p95_ms']}ms")

    with metrics_col2:
        st.metric("ITL p95", f"{rec['predicted_itl_p95_ms']}ms")

    with metrics_col3:
        st.metric("E2E p95", f"{rec['predicted_e2e_p95_ms']}ms")

    with metrics_col4:
        st.metric("Throughput", f"{rec['predicted_throughput_qps']:.1f} QPS")

    st.markdown("---")

    # Reasoning
    st.markdown("### 💡 Reasoning")
    st.info(rec["reasoning"])

    # Alternative Options
    st.markdown("---")
    st.markdown("### 🔄 Deployment Options Comparison")

    alternatives = rec.get("alternative_options")
    if alternatives and len(alternatives) > 0:
        st.caption("Click Select button to choose which option to deploy")

        # Build table-like layout with buttons

        # Initialize selected_option_idx in session state if not present
        # This just tracks which option is selected, doesn't modify the recommendation
        if "selected_option_idx" not in st.session_state:
            st.session_state.selected_option_idx = 0  # 0 = recommended

        # Header row
        header_cols = st.columns([0.8, 1.5, 2.5, 1.2, 0.8, 1, 1, 1, 1, 1])
        header_cols[0].markdown("**Select**")
        header_cols[1].markdown("**Option**")
        header_cols[2].markdown("**Model**")
        header_cols[3].markdown("**GPU Config**")
        header_cols[4].markdown("**Replicas**")
        header_cols[5].markdown("**TTFT p95**")
        header_cols[6].markdown("**ITL p95**")
        header_cols[7].markdown("**E2E p95**")
        header_cols[8].markdown("**Max QPS**")
        header_cols[9].markdown("**Cost/Month**")

        # Recommended option row
        cols = st.columns([0.8, 1.5, 2.5, 1.2, 0.8, 1, 1, 1, 1, 1])
        is_selected = st.session_state.selected_option_idx == 0
        button_label = "✅" if is_selected else "⚫"

        with cols[0]:
            if st.button(
                button_label, key="select_rec", use_container_width=True, disabled=is_selected
            ):
                st.session_state.selected_option_idx = 0
                st.rerun()

        cols[1].markdown("**Recommended**")
        cols[2].markdown(rec["model_name"])
        cols[3].markdown(f"{rec['gpu_config']['tensor_parallel']}x {rec['gpu_config']['gpu_type']}")
        cols[4].markdown(f"{rec['gpu_config']['replicas']}")
        cols[5].markdown(f"{rec['predicted_ttft_p95_ms']}")
        cols[6].markdown(f"{rec['predicted_itl_p95_ms']}")
        cols[7].markdown(f"{rec['predicted_e2e_p95_ms']}")
        cols[8].markdown(f"{rec['predicted_throughput_qps']:.0f}")
        cols[9].markdown(f"${rec['cost_per_month_usd']:,.0f}")

        # Alternative rows
        for i, alt in enumerate(alternatives, 1):
            cols = st.columns([0.8, 1.5, 2.5, 1.2, 0.8, 1, 1, 1, 1, 1])
            is_selected = st.session_state.selected_option_idx == i
            button_label = "✅" if is_selected else "⚫"

            with cols[0]:
                if st.button(
                    button_label,
                    key=f"select_alt_{i}",
                    use_container_width=True,
                    disabled=is_selected,
                ):
                    st.session_state.selected_option_idx = i
                    st.rerun()

            cols[1].markdown(f"**Option {i+1}**")
            cols[2].markdown(alt["model_name"])
            cols[3].markdown(f"{alt['gpu_config']['tensor_parallel']}x {alt['gpu_config']['gpu_type']}")
            cols[4].markdown(f"{alt['gpu_config']['replicas']}")
            cols[5].markdown(f"{alt['predicted_ttft_p95_ms']}")
            cols[6].markdown(f"{alt['predicted_itl_p95_ms']}")
            cols[7].markdown(f"{alt['predicted_e2e_p95_ms']}")
            cols[8].markdown(f"{alt['predicted_throughput_qps']:.0f}")
            cols[9].markdown(f"${alt['cost_per_month_usd']:,.0f}")
    else:
        st.info(
            "💡 No alternative options available. This is the only configuration that meets your SLO requirements."
        )


def render_specifications_tab(rec: dict[str, Any]):
    """Render specifications tab with editable fields."""

    st.markdown("### 🔧 Deployment Specifications")
    st.caption("Review and modify the specifications to explore different configurations")

    # Initialize session state for tracking edit modes per section
    if "editing_requirements" not in st.session_state:
        st.session_state.editing_requirements = False
    if "editing_traffic" not in st.session_state:
        st.session_state.editing_traffic = False
    if "editing_slo" not in st.session_state:
        st.session_state.editing_slo = False
    if "original_requirements" not in st.session_state:
        st.session_state.original_requirements = None
    if "show_regenerate_warning" not in st.session_state:
        st.session_state.show_regenerate_warning = False
    if "edit_session_key" not in st.session_state:
        st.session_state.edit_session_key = 0

    # Intent Section
    intent = rec["intent"]

    col_header, col_button = st.columns([6, 1])
    with col_header:
        st.markdown("#### Use Case & Requirements")
    with col_button:
        if not st.session_state.editing_requirements:
            if st.button("✏️", key="edit_requirements_btn", help="Edit requirements"):
                st.session_state.editing_requirements = True
                # Store original values
                st.session_state.original_requirements = {
                    "use_case": intent["use_case"],
                    "user_count": intent["user_count"],
                    "latency_requirement": intent["latency_requirement"],
                    "throughput_priority": intent.get("throughput_priority", "medium"),
                    "budget_constraint": intent["budget_constraint"],
                }
                st.rerun()

    # Define enum options - 9 use cases from traffic_and_slos.md
    use_case_options = [
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_generation",
        "summarization_short",
        "document_analysis_rag",
        "summarization_long",
        "research_legal_analysis",
    ]
    latency_options = ["very_high", "high", "medium", "low"]
    throughput_options = ["very_high", "high", "medium", "low"]
    budget_options = ["strict", "moderate", "flexible", "none"]

    # Use session key to force widget recreation on cancel
    session_key = st.session_state.edit_session_key

    col1, col2 = st.columns(2)
    with col1:
        use_case = st.selectbox(
            "Use Case",
            options=use_case_options,
            index=use_case_options.index(intent["use_case"]),
            disabled=not st.session_state.editing_requirements,
            key=f"edit_use_case_{session_key}",
        )
        user_count = st.number_input(
            "Users",
            value=intent["user_count"],
            min_value=1,
            step=100,
            disabled=not st.session_state.editing_requirements,
            key=f"edit_user_count_{session_key}",
        )
        throughput_priority = st.selectbox(
            "Throughput Priority",
            options=throughput_options,
            index=throughput_options.index(intent.get("throughput_priority", "medium")),
            disabled=not st.session_state.editing_requirements,
            key=f"edit_throughput_priority_{session_key}",
        )

    with col2:
        latency_requirement = st.selectbox(
            "Latency Requirement",
            options=latency_options,
            index=latency_options.index(intent["latency_requirement"]),
            disabled=not st.session_state.editing_requirements,
            key=f"edit_latency_requirement_{session_key}",
        )
        budget_constraint = st.selectbox(
            "Budget Constraint",
            options=budget_options,
            index=budget_options.index(intent["budget_constraint"]),
            disabled=not st.session_state.editing_requirements,
            key=f"edit_budget_constraint_{session_key}",
        )

    # Save/Cancel buttons for requirements section
    if st.session_state.editing_requirements:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💾 Regenerate Profile & SLOs",
                key="save_requirements",
                use_container_width=True,
                type="primary",
            ):
                # Check if requirements actually changed
                requirements_changed = (
                    use_case != st.session_state.original_requirements["use_case"]
                    or int(user_count) != st.session_state.original_requirements["user_count"]
                    or latency_requirement
                    != st.session_state.original_requirements["latency_requirement"]
                    or throughput_priority
                    != st.session_state.original_requirements["throughput_priority"]
                    or budget_constraint
                    != st.session_state.original_requirements["budget_constraint"]
                )

                if requirements_changed and not st.session_state.show_regenerate_warning:
                    # Show warning first
                    st.session_state.show_regenerate_warning = True
                    st.rerun()
                else:
                    # Proceed with regeneration
                    st.session_state.show_regenerate_warning = False
                    edited_intent = {
                        "use_case": use_case,
                        "user_count": int(user_count),
                        "latency_requirement": latency_requirement,
                        "throughput_priority": throughput_priority,
                        "budget_constraint": budget_constraint,
                        "domain_specialization": intent.get("domain_specialization", ["general"]),
                        "additional_context": intent.get("additional_context"),
                    }
                    regenerate_and_recommend({"intent": edited_intent})

        with col2:
            if st.button("❌ Cancel", key="cancel_requirements", use_container_width=True):
                # Restore original values
                if st.session_state.original_requirements:
                    st.session_state.recommendation["intent"].update(
                        st.session_state.original_requirements
                    )
                st.session_state.editing_requirements = False
                st.session_state.show_regenerate_warning = False
                st.session_state.original_requirements = None
                # Increment session key to force widget recreation with original values
                st.session_state.edit_session_key += 1
                st.rerun()

        # Show warning if triggered
        if st.session_state.show_regenerate_warning:
            st.warning(
                """
                **Regenerate Profile & SLOs**

                You modified Use Case & Requirements fields. This will regenerate
                Traffic Profile and SLO Targets based on the new requirements.

                Any manual edits to Traffic Profile or SLO Targets will be overwritten.

                Click "Regenerate Profile & SLOs" again to confirm.
                """,
                icon="⚠️",
            )

    # Traffic Profile Section
    traffic = rec["traffic_profile"]

    col_header, col_button = st.columns([6, 1])
    with col_header:
        st.markdown("#### Traffic Profile")
    with col_button:
        if not st.session_state.editing_traffic:
            if st.button("✏️", key="edit_traffic_btn", help="Edit traffic profile"):
                st.session_state.editing_traffic = True
                # Store original values
                st.session_state.original_traffic = {
                    "expected_qps": traffic["expected_qps"],
                    "prompt_tokens": traffic["prompt_tokens"],
                    "output_tokens": traffic["output_tokens"],
                }
                st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        expected_qps = st.number_input(
            "Expected QPS",
            value=float(traffic["expected_qps"]),
            min_value=0.1,
            step=1.0,
            format="%.2f",
            disabled=not st.session_state.editing_traffic,
            key=f"edit_expected_qps_{session_key}",
        )

    with col2:
        prompt_tokens = st.number_input(
            "Prompt Tokens",
            value=traffic["prompt_tokens"],
            min_value=1,
            step=10,
            disabled=not st.session_state.editing_traffic,
            key=f"edit_prompt_tokens_{session_key}",
        )

    with col3:
        output_tokens = st.number_input(
            "Output Tokens",
            value=traffic["output_tokens"],
            min_value=1,
            step=10,
            disabled=not st.session_state.editing_traffic,
            key=f"edit_output_tokens_{session_key}",
        )

    # Save/Cancel buttons for traffic section
    if st.session_state.editing_traffic:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💾 Save & Re-Evaluate",
                key="save_traffic",
                use_container_width=True,
                type="primary",
            ):
                # Collect all current specs with edited traffic
                edited_specs = {
                    "intent": {
                        "use_case": intent["use_case"],
                        "user_count": intent["user_count"],
                        "latency_requirement": intent["latency_requirement"],
                        "throughput_priority": intent.get("throughput_priority", "medium"),
                        "budget_constraint": intent["budget_constraint"],
                        "domain_specialization": intent.get("domain_specialization", ["general"]),
                        "additional_context": intent.get("additional_context"),
                    },
                    "traffic_profile": {
                        "prompt_tokens": int(prompt_tokens),
                        "output_tokens": int(output_tokens),
                        "expected_qps": float(expected_qps),
                    },
                    "slo_targets": rec["slo_targets"],
                }
                re_recommend_with_specs(edited_specs)

        with col2:
            if st.button("❌ Cancel", key="cancel_traffic", use_container_width=True):
                # Restore original values
                if "original_traffic" in st.session_state and st.session_state.original_traffic:
                    st.session_state.recommendation["traffic_profile"].update(
                        st.session_state.original_traffic
                    )
                    st.session_state.original_traffic = None
                st.session_state.editing_traffic = False
                # Increment session key to force widget recreation with original values
                st.session_state.edit_session_key += 1
                st.rerun()

    # SLO Targets Section
    slo = rec["slo_targets"]

    col_header, col_button = st.columns([6, 1])
    with col_header:
        st.markdown("#### SLO Targets")
    with col_button:
        if not st.session_state.editing_slo:
            if st.button("✏️", key="edit_slo_btn", help="Edit SLO targets"):
                st.session_state.editing_slo = True
                # Store original values
                st.session_state.original_slo = {
                    "ttft_p95_target_ms": slo["ttft_p95_target_ms"],
                    "itl_p95_target_ms": slo["itl_p95_target_ms"],
                    "e2e_p95_target_ms": slo["e2e_p95_target_ms"],
                }
                st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        ttft_target = st.number_input(
            "TTFT p95 (ms)",
            value=slo["ttft_p95_target_ms"],
            min_value=1,
            step=10,
            disabled=not st.session_state.editing_slo,
            key=f"edit_ttft_target_{session_key}",
        )

    with col2:
        itl_target = st.number_input(
            "ITL p95 (ms)",
            value=slo["itl_p95_target_ms"],
            min_value=1,
            step=5,
            disabled=not st.session_state.editing_slo,
            key=f"edit_itl_target_{session_key}",
        )

    with col3:
        e2e_target = st.number_input(
            "E2E p95 (ms)",
            value=slo["e2e_p95_target_ms"],
            min_value=1,
            step=50,
            disabled=not st.session_state.editing_slo,
            key=f"edit_e2e_target_{session_key}",
        )

    # Save/Cancel buttons for SLO section
    if st.session_state.editing_slo:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💾 Save & Re-Evaluate", key="save_slo", use_container_width=True, type="primary"
            ):
                # Collect all current specs with edited SLOs
                edited_specs = {
                    "intent": {
                        "use_case": intent["use_case"],
                        "user_count": intent["user_count"],
                        "latency_requirement": intent["latency_requirement"],
                        "throughput_priority": intent.get("throughput_priority", "medium"),
                        "budget_constraint": intent["budget_constraint"],
                        "domain_specialization": intent.get("domain_specialization", ["general"]),
                        "additional_context": intent.get("additional_context"),
                    },
                    "traffic_profile": rec["traffic_profile"],
                    "slo_targets": {
                        "ttft_p95_target_ms": int(ttft_target),
                        "itl_p95_target_ms": int(itl_target),
                        "e2e_p95_target_ms": int(e2e_target),
                    },
                }
                re_recommend_with_specs(edited_specs)

        with col2:
            if st.button("❌ Cancel", key="cancel_slo", use_container_width=True):
                # Restore original values
                if "original_slo" in st.session_state and st.session_state.original_slo:
                    st.session_state.recommendation["slo_targets"].update(
                        st.session_state.original_slo
                    )
                    st.session_state.original_slo = None
                st.session_state.editing_slo = False
                # Increment session key to force widget recreation with original values
                st.session_state.edit_session_key += 1
                st.rerun()


def render_performance_tab(rec: dict[str, Any]):
    """Render performance tab with detailed metrics."""

    # Check if this is a spec-only response (no viable config found)
    has_config = rec.get("model_name") is not None and rec.get("gpu_config") is not None

    if not has_config:
        st.error("❌ No viable deployment configurations found meeting SLO targets")
        st.markdown("---")
        st.markdown("### 📝 SLO Targets")
        st.markdown(
            "These are the performance targets Compass is searching for, based on your requirements:"
        )
        slo = rec["slo_targets"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TTFT p95 Target", f"{slo['ttft_p95_target_ms']}ms")
        with col2:
            st.metric("ITL p95 Target", f"{slo['itl_p95_target_ms']}ms")
        with col3:
            st.metric("E2E p95 Target", f"{slo['e2e_p95_target_ms']}ms")

        st.markdown("---")
        st.info(
            "💡 Try relaxing these targets in the **Specifications** tab and clicking **Save & Re-Evaluate**"
        )
        return

    st.markdown("### 📊 Predicted Performance")

    slo = rec["slo_targets"]

    # TTFT
    st.markdown("#### Time to First Token (TTFT)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted p95", f"{rec['predicted_ttft_p95_ms']}ms")
    with col2:
        delta_ms = rec["predicted_ttft_p95_ms"] - slo["ttft_p95_target_ms"]
        st.metric(
            "Target p95",
            f"{slo['ttft_p95_target_ms']}ms",
            delta=f"{delta_ms}ms",
            delta_color="inverse",
        )

    # ITL
    st.markdown("#### Inter-Token Latency (ITL)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted p95", f"{rec['predicted_itl_p95_ms']}ms")
    with col2:
        delta_ms = rec["predicted_itl_p95_ms"] - slo["itl_p95_target_ms"]
        st.metric(
            "Target p95",
            f"{slo['itl_p95_target_ms']}ms",
            delta=f"{delta_ms}ms",
            delta_color="inverse",
        )

    # E2E Latency
    st.markdown("#### End-to-End Latency")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted p95", f"{rec['predicted_e2e_p95_ms']}ms")
    with col2:
        delta_ms = rec["predicted_e2e_p95_ms"] - slo["e2e_p95_target_ms"]
        st.metric(
            "Target p95",
            f"{slo['e2e_p95_target_ms']}ms",
            delta=f"{delta_ms}ms",
            delta_color="inverse",
        )

    # Throughput
    st.markdown("#### Throughput")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Capacity", f"{rec['predicted_throughput_qps']:.1f} QPS")
    with col2:
        expected_qps = rec["traffic_profile"]["expected_qps"]
        delta_qps = rec["predicted_throughput_qps"] - expected_qps
        st.metric(
            "Expected Load",
            f"{expected_qps:.1f} QPS",
            delta=f"+{delta_qps:.1f} headroom"
            if delta_qps > 0
            else f"{delta_qps:.1f} over capacity",
            delta_color="normal" if delta_qps > 0 else "inverse",
        )


def render_cost_tab(rec: dict[str, Any]):
    """Render cost tab with pricing details."""

    # Check if this is a spec-only response (no viable config found)
    has_config = rec.get("model_name") is not None and rec.get("gpu_config") is not None

    if not has_config:
        st.error("❌ No viable deployment configurations found meeting SLO targets")
        st.markdown("---")
        st.info(
            """
            **Cannot estimate cost without a viable GPU configuration.**

            Cost estimates depend on:
            - GPU type (L4, A100, H100, etc.)
            - Number of GPUs needed
            - Tensor parallelism configuration
            - Number of replicas

            💡 Adjust your requirements in the **Specifications** tab to find configurations that meet your SLO targets.
            """
        )
        return

    st.markdown("### 💰 Cost Breakdown")

    # Main cost metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Hourly Cost")
        st.markdown(f"## ${rec['cost_per_hour_usd']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Monthly Cost")
        st.markdown(f"## ${rec['cost_per_month_usd']:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # GPU details
    gpu_config = rec["gpu_config"]
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
    except Exception:
        st.session_state.cluster_accessible = False
        return {"accessible": False}


def generate_deployment_yaml(rec: dict[str, Any]):
    """Generate deployment YAML files via API."""
    try:
        with st.spinner("Generating deployment YAML files..."):
            response = requests.post(
                f"{API_BASE_URL}/api/deploy",
                json={"recommendation": rec, "namespace": "default"},
                timeout=30,
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
                for _config_type, file_path in result["files"].items():
                    st.code(file_path, language="text")

                # Check cluster status
                cluster_status = check_cluster_status()
                if cluster_status.get("accessible"):
                    st.markdown("---")
                    st.success("✅ Kubernetes cluster is accessible!")
                    st.markdown(
                        "**Next:** Click **Deploy to Kubernetes** to deploy to the cluster!"
                    )
                else:
                    st.markdown("---")
                    st.warning(
                        "⚠️ Kubernetes cluster not accessible. YAML files generated but not deployed."
                    )
                    st.markdown(
                        "**Next:** Go to the **Monitoring** tab to see simulated observability metrics!"
                    )

            else:
                st.error(f"Failed to generate YAML: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"❌ Error generating deployment: {str(e)}")


def regenerate_and_recommend(edited_specs: dict[str, Any]):
    """Regenerate traffic profile and SLO targets from requirements, then re-recommend."""
    try:
        with st.spinner("Regenerating traffic profile and SLO targets..."):
            response = requests.post(
                f"{API_BASE_URL}/api/regenerate-and-recommend",
                json={"intent": edited_specs["intent"]},
                timeout=30,
            )

            if response.status_code == 200:
                new_recommendation = response.json()
                st.session_state.recommendation = new_recommendation

                # Reset all edit mode flags
                st.session_state.editing_requirements = False
                st.session_state.editing_traffic = False
                st.session_state.editing_slo = False
                st.session_state.original_requirements = None
                st.session_state.show_regenerate_warning = False

                # Reset deployment state for new recommendation
                st.session_state.deployed_to_cluster = False
                st.session_state.deployment_id = None

                # Check if we got a viable config or just a spec
                has_config = (
                    new_recommendation.get("model_name") is not None
                    and new_recommendation.get("gpu_config") is not None
                )

                if has_config:
                    st.success(
                        "✅ Traffic Profile and SLO Targets regenerated! New recommendation generated."
                    )
                    st.info(
                        f"**New Recommendation:** {new_recommendation['model_name']} on "
                        f"{new_recommendation['gpu_config']['gpu_count']}x {new_recommendation['gpu_config']['gpu_type']}"
                    )
                else:
                    st.warning(
                        "⚠️ Specification regenerated, but no viable configurations found meeting SLO targets."
                    )
                    st.info("Review the **Specifications** tab and adjust your requirements.")

                st.rerun()
            else:
                st.error(f"Failed to regenerate: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"❌ Error during regeneration: {str(e)}")


def re_recommend_with_specs(edited_specs: dict[str, Any]):
    """Re-generate recommendation with edited specifications (no regeneration)."""
    try:
        with st.spinner("Re-evaluating model and GPU choices..."):
            response = requests.post(
                f"{API_BASE_URL}/api/re-recommend",
                json={"specifications": edited_specs},
                timeout=30,
            )

            if response.status_code == 200:
                new_recommendation = response.json()
                st.session_state.recommendation = new_recommendation

                # Reset all edit mode flags
                st.session_state.editing_requirements = False
                st.session_state.editing_traffic = False
                st.session_state.editing_slo = False
                st.session_state.original_requirements = None
                st.session_state.show_regenerate_warning = False

                # Reset deployment state for new recommendation
                st.session_state.deployed_to_cluster = False
                st.session_state.deployment_id = None

                # Check if we got a viable config or just a spec
                has_config = (
                    new_recommendation.get("model_name") is not None
                    and new_recommendation.get("gpu_config") is not None
                )

                if has_config:
                    st.success("✅ Re-evaluation complete! New recommendation generated.")
                    st.info(
                        f"**New Recommendation:** {new_recommendation['model_name']} on "
                        f"{new_recommendation['gpu_config']['gpu_count']}x {new_recommendation['gpu_config']['gpu_type']}"
                    )
                else:
                    st.warning(
                        "⚠️ Re-evaluation complete, but no viable configurations found meeting SLO targets."
                    )
                    st.info("Review the **Specifications** tab and adjust your SLO targets.")

                st.rerun()
            else:
                st.error(f"Failed to re-evaluate: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"❌ Error during re-evaluation: {str(e)}")


def deploy_to_cluster(rec: dict[str, Any]):
    """Deploy model to Kubernetes cluster."""
    try:
        with st.spinner("Deploying to Kubernetes cluster..."):
            response = requests.post(
                f"{API_BASE_URL}/api/deploy-to-cluster",
                json={"recommendation": rec, "namespace": "default"},
                timeout=60,
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
                st.markdown(
                    "**Next:** Go to the **Monitoring** tab to see actual deployment status!"
                )

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
    if (
        "selected_deployment" not in st.session_state
        or st.session_state.selected_deployment not in deployment_ids
    ):
        st.session_state.selected_deployment = deployment_ids[0] if deployment_ids else None

    # Deployment selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Select deployment to monitor:",
            deployment_ids,
            index=deployment_ids.index(st.session_state.selected_deployment)
            if st.session_state.selected_deployment in deployment_ids
            else 0,
            key="deployment_selector_standalone",
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
        if st.button(
            "🗑️ Delete Deployment",
            use_container_width=True,
            type="secondary",
            key=f"delete_btn_{context}_{deployment_id}",
        ):
            if st.session_state.get(f"confirm_delete_{deployment_id}"):
                # Actually delete
                with st.spinner("Deleting deployment..."):
                    try:
                        response = requests.delete(
                            f"{API_BASE_URL}/api/deployments/{deployment_id}", timeout=30
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
                st.markdown(
                    f"{status_icon} **{condition.get('type')}**: {condition.get('message', 'N/A')}"
                )
    else:
        st.warning(f"⚠️ InferenceService not found: {status.get('error', 'Unknown error')}")

    # Pod status
    if pods:
        st.markdown(f"**Pods:** {len(pods)} pod(s)")
        with st.expander("🔍 Pod Details"):
            for pod in pods:
                st.markdown(f"**{pod.get('name')}**")
                st.markdown(f"- Phase: {pod.get('phase')}")
                node_name = pod.get("node_name") or "Not assigned"
                st.markdown(f"- Node: {node_name}")
    else:
        st.info("ℹ️ No pods found yet (may still be creating)")


def render_inference_testing_for_deployment(
    deployment_info: dict[str, Any], context: str = "default"
):
    """Render inference testing for a specific deployment.

    Args:
        deployment_info: Deployment information dictionary
        context: Unique context identifier to avoid key collisions
    """
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})

    st.markdown("#### 🧪 Inference Testing")

    if not status.get("ready"):
        st.info(
            "⏳ Deployment not ready yet. Inference testing will be available once the service is ready."
        )
        return

    # Test prompt input
    col1, col2 = st.columns([2, 1])
    with col1:
        test_prompt = st.text_area(
            "Test Prompt",
            value="Write a Python function that calculates the fibonacci sequence.",
            height=100,
            key=f"test_prompt_{context}_{deployment_id}",
        )

    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            value=150,
            min_value=10,
            max_value=500,
            key=f"max_tokens_{context}_{deployment_id}",
        )
        temperature = st.slider(
            "Temperature", 0.0, 2.0, 0.7, 0.1, key=f"temperature_{context}_{deployment_id}"
        )

    # Test button
    if st.button(
        "🚀 Send Test Request",
        use_container_width=True,
        key=f"test_button_{context}_{deployment_id}",
    ):
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
                    stderr=subprocess.PIPE,
                )

                # Give it a moment to establish connection
                time.sleep(3)

                # Check if port-forward is still running
                if port_forward_proc.poll() is not None:
                    # Process exited
                    pf_stdout, pf_stderr = port_forward_proc.communicate()
                    st.error("❌ Port-forward failed to start")
                    st.code(
                        f"stdout: {pf_stdout.decode()}\nstderr: {pf_stderr.decode()}",
                        language="text",
                    )
                    return

                try:
                    # Send inference request
                    start_time = time.time()

                    curl_cmd = [
                        "curl",
                        "-s",
                        "-X",
                        "POST",
                        "http://localhost:8080/v1/completions",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(
                            {
                                "prompt": test_prompt,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            }
                        ),
                    ]

                    # Show the command being executed for debugging
                    with st.expander("🔍 Debug Info"):
                        st.code(" ".join(curl_cmd), language="bash")

                    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)

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
                            st.warning(
                                "No output captured from curl command. Port-forward may have failed."
                            )

                finally:
                    # Clean up port-forward process
                    port_forward_proc.terminate()
                    port_forward_proc.wait(timeout=2)

            except subprocess.TimeoutExpired:
                st.error("❌ Request timed out (30s). The model may still be starting up.")
                with contextlib.suppress(Exception):
                    port_forward_proc.terminate()
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

    # Check if this is a spec-only response (no viable config found)
    has_config = rec.get("model_name") is not None and rec.get("gpu_config") is not None

    if not has_config:
        st.error("❌ No viable deployment configurations found meeting SLO targets")
        st.markdown("---")
        st.info(
            """
            **Cannot generate deployment YAML without a viable configuration.**

            YAML deployment files require:
            - Model selection (which LLM to deploy)
            - GPU configuration (type, count, tensor parallelism)
            - Resource requests and limits
            - Autoscaling parameters

            💡 Adjust your requirements in the **Specifications** tab to find configurations that can be deployed.
            """
        )
        return

    # Check if YAML was auto-generated
    if not rec.get("yaml_generated", False):
        st.warning(
            "⚠️ YAML files were not generated automatically. This may indicate an error during recommendation creation."
        )
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

        response = requests.get(f"{API_BASE_URL}/api/deployments/{deployment_id}/yaml", timeout=10)

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
                        mime="text/yaml",
                    )
        else:
            st.error(f"Failed to fetch YAML files: {response.text}")

    except Exception as e:
        st.error(f"Error fetching YAML preview: {str(e)}")
        st.info("💡 The YAML files are stored on the backend and ready for deployment.")


def render_k8s_status():
    """Render actual Kubernetes deployment status."""
    st.markdown("#### 🎛️ Kubernetes Cluster Status")

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/deployments/{st.session_state.deployment_id}/k8s-status",
            timeout=10,
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
                        st.markdown(
                            f"{status_icon} **{condition.get('type')}**: {condition.get('message', 'N/A')}"
                        )
            else:
                st.warning(f"⚠️ InferenceService not found: {isvc.get('error', 'Unknown error')}")

            # Pod status
            if pods:
                st.markdown(f"**Pods:** {len(pods)} pod(s)")
                with st.expander("🔍 Pod Details"):
                    for pod in pods:
                        st.markdown(f"**{pod.get('name')}**")
                        st.markdown(f"- Phase: {pod.get('phase')}")
                        node_name = pod.get("node_name") or "Not assigned"
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
            help="Enter a prompt to test the model",
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
                    stderr=subprocess.PIPE,
                )

                # Give it a moment to establish connection
                time.sleep(3)

                # Check if port-forward is still running
                if port_forward_proc.poll() is not None:
                    # Process exited
                    pf_stdout, pf_stderr = port_forward_proc.communicate()
                    st.error("❌ Port-forward failed to start")
                    st.code(
                        f"stdout: {pf_stdout.decode()}\nstderr: {pf_stderr.decode()}",
                        language="text",
                    )
                    return

                try:
                    # Send inference request
                    start_time = time.time()

                    curl_cmd = [
                        "curl",
                        "-s",
                        "-X",
                        "POST",
                        "http://localhost:8080/v1/completions",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(
                            {
                                "prompt": test_prompt,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            }
                        ),
                    ]

                    # Show the command being executed for debugging
                    with st.expander("🔍 Debug Info"):
                        st.code(" ".join(curl_cmd), language="bash")

                    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)

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
                            st.warning(
                                "No output captured from curl command. Port-forward may have failed."
                            )

                finally:
                    # Clean up port-forward process
                    port_forward_proc.terminate()
                    port_forward_proc.wait(timeout=2)

            except subprocess.TimeoutExpired:
                st.error("❌ Request timed out (30s). The model may still be starting up.")
                with contextlib.suppress(Exception):
                    port_forward_proc.terminate()
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


def render_simulated_observability(deployment_info: dict[str, Any], context: str = "default"):
    """Render simulated observability metrics from sample_outcomes.json.

    Args:
        deployment_info: Deployment information dictionary
        context: Unique context identifier to avoid key collisions
    """
    status = deployment_info.get("status", {})

    # Only show if deployment is ready
    if not status.get("ready"):
        st.info("⏳ Observability metrics will be available once deployment is ready")
        return

    st.markdown("#### 📊 Simulated Observability Metrics")
    st.caption(
        "These metrics are simulated from sample deployment outcomes. In production, these would come from Prometheus/Grafana."
    )

    # Load sample outcomes data
    try:
        import json
        import random

        with open("data/sample_outcomes.json") as f:
            data = json.load(f)
            outcomes = data.get("deployment_outcomes", [])

        if not outcomes:
            st.info("No sample outcome data available")
            return

        # Pick a random outcome for demonstration
        # In a real system, this would match the actual deployment
        outcome = random.choice(outcomes)

        # Display metrics in tabs
        metrics_tabs = st.tabs(["🎯 SLO Compliance", "🖥️ Resources", "💰 Cost", "📈 Traffic"])

        with metrics_tabs[0]:
            render_slo_compliance_metrics(outcome)

        with metrics_tabs[1]:
            render_resource_metrics(outcome)

        with metrics_tabs[2]:
            render_cost_metrics(outcome)

        with metrics_tabs[3]:
            render_traffic_metrics(outcome)

    except FileNotFoundError:
        st.warning("⚠️ Sample outcomes data file not found (data/sample_outcomes.json)")
    except Exception as e:
        st.error(f"Error loading simulated metrics: {str(e)}")


def render_slo_compliance_metrics(outcome: dict[str, Any]):
    """Render SLO compliance metrics from outcome data."""
    st.markdown("##### SLO Performance vs Targets")

    predicted = outcome["predicted_slos"]
    actual = outcome["actual_slos"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "TTFT p90",
            f"{actual['ttft_p90_ms']}ms",
            delta=f"{actual['ttft_p90_ms'] - predicted['ttft_p90_ms']}ms vs predicted",
            delta_color="inverse",
        )
        compliant = actual["ttft_p90_ms"] <= predicted["ttft_p90_ms"] * 1.1
        st.caption(f"Predicted: {predicted['ttft_p90_ms']}ms {'✅' if compliant else '⚠️'}")

    with col2:
        st.metric(
            "TPOT p90",
            f"{actual['tpot_p90_ms']}ms",
            delta=f"{actual['tpot_p90_ms'] - predicted['tpot_p90_ms']}ms vs predicted",
            delta_color="inverse",
        )
        compliant = actual["tpot_p90_ms"] <= predicted["tpot_p90_ms"] * 1.1
        st.caption(f"Predicted: {predicted['tpot_p90_ms']}ms {'✅' if compliant else '⚠️'}")

    with col3:
        st.metric(
            "E2E p90",
            f"{actual['e2e_p90_ms']}ms",
            delta=f"{actual['e2e_p90_ms'] - predicted['e2e_p90_ms']}ms vs predicted",
            delta_color="inverse",
        )
        compliant = actual["e2e_p90_ms"] <= predicted["e2e_p90_ms"] * 1.1
        st.caption(f"Predicted: {predicted['e2e_p90_ms']}ms {'✅' if compliant else '⚠️'}")

    # Throughput
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Actual QPS",
            f"{outcome['actual_qps']} QPS",
            delta=f"{outcome['actual_qps'] - outcome['predicted_qps']} vs predicted",
        )
        st.caption(f"Predicted: {outcome['predicted_qps']} QPS")

    with col2:
        success_icon = "✅" if outcome["deployment_success"] else "❌"
        st.metric(
            "Deployment Status",
            f"{success_icon} {'Success' if outcome['deployment_success'] else 'Failed'}",
        )

    # Notes
    if outcome.get("notes"):
        st.markdown("---")
        st.info(f"**Notes:** {outcome['notes']}")


def render_resource_metrics(outcome: dict[str, Any]):
    """Render resource utilization metrics."""
    st.markdown("##### GPU Configuration & Utilization")

    gpu_config = outcome["gpu_config"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("GPU Type", gpu_config["gpu_type"])
        st.metric("GPU Count", gpu_config["gpu_count"])

    with col2:
        st.metric("Tensor Parallel", gpu_config["tensor_parallel"])
        st.metric("Replicas", gpu_config["replicas"])

    with col3:
        # Simulated utilization (would come from Prometheus in production)
        import random

        gpu_util = random.randint(75, 95)
        st.metric("GPU Utilization", f"{gpu_util}%")
        st.caption("Simulated metric")


def render_cost_metrics(outcome: dict[str, Any]):
    """Render cost analysis metrics."""
    st.markdown("##### Cost Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Monthly Cost", f"${outcome['cost_per_month_usd']:,.0f}")

    with col2:
        # Calculate cost per 1k tokens (simulated)
        # Assume ~1M tokens/day for estimation
        tokens_per_month = 30_000_000
        cost_per_1k = (outcome["cost_per_month_usd"] / tokens_per_month) * 1000
        st.metric("Cost per 1k Tokens", f"${cost_per_1k:.3f}")
        st.caption("Estimated from usage")

    st.markdown("---")
    st.markdown("**GPU Configuration:**")
    st.markdown(f"- {outcome['gpu_config']['tensor_parallel']}x {outcome['gpu_config']['gpu_type']}")
    st.markdown(
        f"- Tensor Parallel: {outcome['gpu_config']['tensor_parallel']}, Replicas: {outcome['gpu_config']['replicas']}"
    )


def render_traffic_metrics(outcome: dict[str, Any]):
    """Render traffic pattern metrics."""
    st.markdown("##### Traffic Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Use Case", outcome["use_case"].replace("_", " ").title())
        st.metric("User Count", f"{outcome['user_count']:,}")

    with col2:
        st.metric("Model", outcome["model_id"].split("/")[-1])
        st.metric("Timestamp", outcome["timestamp"].split("T")[0])

    # Simulated request volume
    st.markdown("---")
    st.markdown("**Request Volume (Simulated):**")

    import random

    requests_per_hour = outcome["actual_qps"] * 3600
    requests_per_day = requests_per_hour * 24

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Requests/Hour", f"{requests_per_hour:,.0f}")
    with col2:
        st.metric("Requests/Day", f"{requests_per_day:,.0f}")
    with col3:
        # Add some variance for "last 7 days"
        variance = random.uniform(0.9, 1.1)
        st.metric("Requests (7d)", f"{int(requests_per_day * 7 * variance):,.0f}")


if __name__ == "__main__":
    main()
