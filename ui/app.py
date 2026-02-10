"""
NeuralNav - E2E LLM Deployment Recommendation System

A beautiful, presentation-ready Streamlit application demonstrating:
1. Business Context Extraction (Qwen 2.5 7B @ 95.1% accuracy)
2. MCDM Scoring (204 Open-Source Models)
3. Full Explainability with visual score breakdowns
4. SLO & Workload Impact Analysis
5. Hardware-aware recommendations

Usage:
    streamlit run ui/poc_app.py
"""

import io
import json
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

import streamlit as st
import streamlit.components.v1 as components

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL NAME NORMALIZATION - Consistent Provider/Model-Name Format
# =============================================================================

# Provider mapping for model name normalization
PROVIDER_MAPPING = {
    "gpt-oss": "OpenAI",
    "gptoss": "OpenAI",
    "kimi": "Moonshot",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "llama": "Meta",
    "mistral": "Mistral",
    "gemma": "Google",
    "minimax": "MiniMax",
    "phi": "Microsoft",
    "falcon": "TII",
    "yi": "01.AI",
    "internlm": "Shanghai AI Lab",
    "baichuan": "Baichuan",
    "chatglm": "Zhipu",
    "glm": "Zhipu",
    "starcoder": "BigCode",
    "codellama": "Meta",
    "aya": "Cohere",
}


def normalize_model_name(raw_name: str) -> str:
    """
    Normalize model name to consistent Provider/Model-Name format.
    
    Examples:
        "GPT-OSS 120B" → "OpenAI/GPT-OSS-120B"
        "gpt-oss-120b" → "OpenAI/GPT-OSS-120B"
        "Moonshot/Kimi-K2-Thinking" → "Moonshot/Kimi-K2-Thinking"
        "kimi-k2-thinking" → "Moonshot/Kimi-K2-Thinking"
        "DeepSeek/DeepSeek-V3.1-Reasoning" → "DeepSeek/DeepSeek-V3.1-Reasoning"
    """
    if not raw_name:
        return "Unknown"
    
    name = raw_name.strip()
    
    # Already has provider prefix (e.g., "Moonshot/Kimi-K2")
    if "/" in name:
        parts = name.split("/", 1)
        provider = parts[0].strip()
        model = parts[1].strip()
        # Normalize model part: title case with hyphens
        model_normalized = "-".join(word.title() if not word.isupper() else word 
                                     for word in model.replace("_", "-").split("-"))
        return f"{provider}/{model_normalized}"
    
    # No provider - need to detect and add it
    name_lower = name.lower().replace(" ", "-").replace("_", "-")
    
    # Find matching provider
    detected_provider = None
    for keyword, provider in PROVIDER_MAPPING.items():
        if keyword in name_lower:
            detected_provider = provider
            break
    
    if not detected_provider:
        detected_provider = "Unknown"
    
    # Normalize model name: title case with hyphens
    # Keep version numbers and special terms intact
    model_parts = name.replace(" ", "-").replace("_", "-").split("-")
    normalized_parts = []
    for part in model_parts:
        if not part:
            continue
        # Keep uppercase terms (like "OSS", "W4A16", "B200")
        if part.isupper() or any(c.isdigit() for c in part):
            normalized_parts.append(part.upper() if part.isalpha() else part)
        else:
            normalized_parts.append(part.title())
    
    model_normalized = "-".join(normalized_parts)
    
    return f"{detected_provider}/{model_normalized}"


def format_display_name(raw_name: str) -> str:
    """
    Format model name for display (uppercase, spaces instead of hyphens).
    
    Examples:
        "OpenAI/GPT-OSS-120B" → "OPENAI / GPT OSS 120B"
        "Moonshot/Kimi-K2-Thinking" → "MOONSHOT / KIMI K2 THINKING"
    """
    normalized = normalize_model_name(raw_name)
    if "/" in normalized:
        provider, model = normalized.split("/", 1)
        model_display = model.replace("-", " ")
        return f"{provider.upper()} / {model_display.upper()}"
    return normalized.upper().replace("-", " ")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NeuralNav",
    page_icon="docs/neuralnav-logo-32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# MINIMAL CSS OVERRIDES
# =============================================================================
st.markdown("""
<style>
    /* Hide Streamlit deploy button */
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "models_df" not in st.session_state:
    st.session_state.models_df = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
# Workflow approval state
if "extraction_approved" not in st.session_state:
    st.session_state.extraction_approved = None  # None = pending, True = approved, False = editing
if "slo_approved" not in st.session_state:
    st.session_state.slo_approved = None
if "edited_extraction" not in st.session_state:
    st.session_state.edited_extraction = None
# Tab switching flags
if "switch_to_tab2" not in st.session_state:
    st.session_state.switch_to_tab2 = False
if "switch_to_tab3" not in st.session_state:
    st.session_state.switch_to_tab3 = False
# Editable SLO values
if "edit_slo" not in st.session_state:
    st.session_state.edit_slo = False
if "custom_ttft" not in st.session_state:
    st.session_state.custom_ttft = None
if "custom_itl" not in st.session_state:
    st.session_state.custom_itl = None
if "custom_e2e" not in st.session_state:
    st.session_state.custom_e2e = None
if "custom_qps" not in st.session_state:
    st.session_state.custom_qps = None
# SLO percentile selection (Mean, P90, P95, P99)
if "slo_percentile" not in st.session_state:
    st.session_state.slo_percentile = "p95"  # Default to P95

# Ranking weights are initialized in the PRIORITIES section of the Tech Spec tab
# using values from priority_weights.json. Do NOT hardcode defaults here.
# See fetch_priority_weights() and the PRIORITIES section around line 5317.

if "include_near_miss" not in st.session_state:
    st.session_state.include_near_miss = False

# Category expansion state (for inline expand/collapse of additional options)
if "expanded_categories" not in st.session_state:
    st.session_state.expanded_categories = set()

# Winner dialog state - must be explicitly initialized to False
if "show_winner_dialog" not in st.session_state:
    st.session_state.show_winner_dialog = False
if "balanced_winner" not in st.session_state:
    st.session_state.balanced_winner = None
if "winner_priority" not in st.session_state:
    st.session_state.winner_priority = "balanced"
if "winner_extraction" not in st.session_state:
    st.session_state.winner_extraction = {}

# Use case tracking
if "detected_use_case" not in st.session_state:
    st.session_state.detected_use_case = "chatbot_conversational"

# Category exploration dialog state
if "show_category_dialog" not in st.session_state:
    st.session_state.show_category_dialog = False
if "explore_category" not in st.session_state:
    st.session_state.explore_category = "balanced"
if "show_full_table_dialog" not in st.session_state:
    st.session_state.show_full_table_dialog = False
if "show_options_list_expanded" not in st.session_state:
    st.session_state.show_options_list_expanded = False
if "top5_balanced" not in st.session_state:
    st.session_state.top5_balanced = []
if "top5_accuracy" not in st.session_state:
    st.session_state.top5_accuracy = []
if "top5_latency" not in st.session_state:
    st.session_state.top5_latency = []
if "top5_cost" not in st.session_state:
    st.session_state.top5_cost = []
if "top5_simplest" not in st.session_state:
    st.session_state.top5_simplest = []
# Deployment tab state
if "deployment_selected_config" not in st.session_state:
    st.session_state.deployment_selected_config = None
if "deployment_yaml_files" not in st.session_state:
    st.session_state.deployment_yaml_files = {}
if "deployment_id" not in st.session_state:
    st.session_state.deployment_id = None
if "deployment_yaml_generated" not in st.session_state:
    st.session_state.deployment_yaml_generated = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_use_case_name(use_case: str) -> str:
    """Format use case name with proper capitalization for acronyms."""
    if not use_case:
        return "Unknown"
    # Replace underscores and title case
    formatted = use_case.replace('_', ' ').title()
    # Fix common acronyms
    acronyms = {
        'Rag': 'RAG',
        'Llm': 'LLM', 
        'Ai': 'AI',
        'Api': 'API',
        'Gpu': 'GPU',
        'Cpu': 'CPU',
        'Slo': 'SLO',
        'Qps': 'QPS',
        'Rps': 'RPS',
    }
    for wrong, right in acronyms.items():
        formatted = formatted.replace(wrong, right)
    return formatted

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_206_models() -> pd.DataFrame:
    """Load all 206 models from backend API."""
    try:
        # Fetch benchmark data from backend API
        response = requests.get(
            f"{API_BASE_URL}/api/v1/benchmarks",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Convert JSON to DataFrame
        if data.get("success") and data.get("benchmarks"):
            df = pd.DataFrame(data["benchmarks"])
            # Ensure Model Name column exists and is clean
            if 'Model Name' in df.columns:
                df = df.dropna(subset=['Model Name'])
                df = df[df['Model Name'].str.strip() != '']
            return df
        else:
            logger.warning("No benchmark data returned from API")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load benchmarks from API: {e}")
        return pd.DataFrame()

# =============================================================================
# RANKED RECOMMENDATIONS (Backend API Integration)
# =============================================================================

@st.cache_data(ttl=300)
def fetch_slo_defaults(use_case: str) -> dict | None:
    """Fetch default SLO values for a use case from the backend API.

    Returns dict with ttft_ms, itl_ms, e2e_ms each containing min, max, default.
    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/slo-defaults/{use_case}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("slo_defaults")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch SLO defaults for {use_case}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_expected_rps(use_case: str, user_count: int) -> dict | None:
    """Fetch expected RPS for a use case from the backend API.

    Uses research-backed workload patterns to calculate:
    - expected_rps: average requests per second
    - peak_rps: peak capacity needed
    - workload_params: active_fraction, requests_per_min, etc.

    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/expected-rps/{use_case}",
            params={"user_count": user_count},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch expected RPS for {use_case}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_workload_profile(use_case: str) -> dict | None:
    """Fetch workload profile for a use case from the backend API.

    Returns dict with workload_profile containing:
    - prompt_tokens: average input token count
    - output_tokens: average output token count
    - peak_multiplier: peak traffic multiplier
    - distribution: workload distribution type

    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/workload-profile/{use_case}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("workload_profile")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch workload profile for {use_case}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_priority_weights() -> dict | None:
    """Fetch priority weights configuration from the backend API.

    Returns dict with priority_weights mapping and defaults.
    Cached for 1 hour (config rarely changes).
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/priority-weights",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("priority_weights")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch priority weights: {e}")
        return None


def fetch_ranked_recommendations(
    use_case: str,
    user_count: int,
    prompt_tokens: int,
    output_tokens: int,
    expected_qps: float,
    ttft_target_ms: int,
    itl_target_ms: int,
    e2e_target_ms: int,
    weights: dict = None,
    include_near_miss: bool = False,
    percentile: str = "p95",
    preferred_gpu_types: list[str] = None,
) -> dict | None:
    """Fetch ranked recommendations from the backend API.

    Args:
        use_case: Use case identifier (e.g., "chatbot_conversational")
        user_count: Number of concurrent users
        prompt_tokens: Input prompt token count
        output_tokens: Output generation token count
        expected_qps: Queries per second
        ttft_target_ms: TTFT SLO target
        itl_target_ms: ITL SLO target
        e2e_target_ms: E2E SLO target
        weights: Optional dict with accuracy, price, latency, complexity weights (0-10)
        include_near_miss: Whether to include near-SLO configurations
        percentile: Which percentile to use for SLO comparison (mean, p90, p95, p99)
        preferred_gpu_types: Optional list of GPU types to filter by (empty = any GPU)

    Returns:
        RankedRecommendationsResponse as dict, or None on error
    """
    import requests

    # Build request payload
    # min_accuracy=35 filters out models with 30% fallback (no AA data)
    payload = {
        "use_case": use_case,
        "user_count": user_count,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "expected_qps": expected_qps,
        "ttft_target_ms": ttft_target_ms,
        "itl_target_ms": itl_target_ms,
        "e2e_target_ms": e2e_target_ms,
        "percentile": percentile,
        "include_near_miss": include_near_miss,
        "min_accuracy": 35,
        "preferred_gpu_types": preferred_gpu_types or [],
    }

    if weights:
        payload["weights"] = weights

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/ranked-recommend-from-spec",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch ranked recommendations: {e}")
        return None


def render_weight_controls() -> None:
    """Render weight controls and configuration options in a collapsible section.

    Updates session state directly:
    - weight_accuracy, weight_cost, weight_latency (0-10 each)
    - include_near_miss (bool)
    Note: weight_complexity removed from UI (hardcoded to 0)
    """
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("""
            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>Ranking Weights</strong> (0-10) — adjust to customize balanced score
            </div>
            """, unsafe_allow_html=True)

            wcol1, wcol2, wcol3 = st.columns(3)

            with wcol1:
                accuracy = st.number_input(
                    "Accuracy",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.get("weight_accuracy", 5),
                    key="input_accuracy",
                    help="Higher = prefer more capable models"
                )
                if accuracy != st.session_state.get("weight_accuracy", 5):
                    st.session_state.weight_accuracy = accuracy

            with wcol2:
                cost = st.number_input(
                    "Cost",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.get("weight_cost", 4),
                    key="input_cost",
                    help="Higher = prefer cheaper configurations"
                )
                if cost != st.session_state.get("weight_cost", 4):
                    st.session_state.weight_cost = cost

            with wcol3:
                latency = st.number_input(
                    "Latency",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.get("weight_latency", 2),
                    key="input_latency",
                    help="Higher = prefer lower latency"
                )
                if latency != st.session_state.get("weight_latency", 2):
                    st.session_state.weight_latency = latency

        with col2:
            st.markdown("""
            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>Options</strong>
            </div>
            """, unsafe_allow_html=True)

            include_near_miss = st.checkbox(
                "Include near-miss configs",
                value=st.session_state.include_near_miss,
                key="checkbox_near_miss",
                help="Include configurations that nearly meet SLO targets"
            )
            if include_near_miss != st.session_state.include_near_miss:
                st.session_state.include_near_miss = include_near_miss

            # Re-evaluate button to apply weight changes (styled as primary/blue)
            if st.button("Re-Evaluate", key="re_evaluate_btn", type="primary", help="Apply weight changes and re-fetch recommendations"):
                st.rerun()


def render_ranked_recommendations(response: dict, show_config: bool = True):
    """Render all 5 ranked recommendation categories in a table format.

    Args:
        response: RankedRecommendationsResponse from backend
        show_config: Whether to show the configuration controls (default True)
    """
    total_configs = response.get("total_configs_evaluated", 0)
    configs_after_filters = response.get("configs_after_filters", 0)

    st.subheader("Recommended Solutions")

    # Render configuration controls right under the header
    if show_config:
        render_weight_controls()

    st.markdown(
        f'<div style="margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="font-weight: 600;">{configs_after_filters}</span> unique options</div>',
        unsafe_allow_html=True
    )

    # Define categories in the requested order
    categories = [
        ("balanced", "Balanced", ""),
        ("best_accuracy", "Best Accuracy", ""),
        ("lowest_cost", "Lowest Cost", ""),
        ("lowest_latency", "Lowest Latency", ""),
        ("simplest", "Simplest", ""),
    ]

    # Helper function to format GPU config with TP and replicas
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"

    # Get selected percentile for table display
    selected_percentile = st.session_state.get('slo_percentile', 'p95')
    percentile_label = selected_percentile.upper() if selected_percentile != 'mean' else 'Mean'
    
    # Helper function to build a table row from a recommendation
    def build_row(rec: dict, cat_name: str = "", cat_emoji: str = "", is_top: bool = False, more_count: int = 0) -> str:
        model_name = format_display_name(rec.get("model_name", "Unknown"))
        gpu_config = rec.get("gpu_config", {})
        gpu_str = format_gpu_config(gpu_config)
        # Get TTFT from benchmark_metrics using selected percentile, fallback to p95
        benchmark_metrics = rec.get("benchmark_metrics", {}) or {}
        ttft = benchmark_metrics.get(f'ttft_{selected_percentile}', rec.get("predicted_ttft_p95_ms", 0))
        cost = rec.get("cost_per_month_usd", 0)
        meets_slo = rec.get("meets_slo", False)
        scores = rec.get("scores", {})
        accuracy_score = scores.get("accuracy_score", 0) if isinstance(scores, dict) else 0
        price_score = scores.get("price_score", 0) if isinstance(scores, dict) else 0
        latency_score = scores.get("latency_score", 0) if isinstance(scores, dict) else 0
        complexity_score = scores.get("complexity_score", 0) if isinstance(scores, dict) else 0
        balanced_score = scores.get("balanced_score", 0) if isinstance(scores, dict) else 0
        slo_badge = "Yes" if meets_slo else "No"

        if is_top:
            more_badge = f'<span style="font-size: 0.8rem; margin-left: 0.5rem;">(+{more_count})</span>' if more_count > 0 else ''
            cat_cell = f'<td style="padding: 0.75rem 0.5rem;"><span style="font-weight: 600;">{cat_emoji} {cat_name}</span>{more_badge}</td>'
        else:
            cat_cell = f'<td style="padding: 0.75rem 0.5rem; padding-left: 2rem;"><span>↳</span></td>'

        return (
            f'<tr style="{"" if not is_top else ""}">'
            f'{cat_cell}'
            f'<td style="padding: 0.75rem 0.5rem; font-weight: {"500" if is_top else "400"};">{model_name}</td>'
            f'<td style="padding: 0.75rem 0.5rem; font-size: 0.85rem;">{gpu_str}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: right; ">{ttft:.0f}ms</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: right; ">${cost:,.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; ">{accuracy_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; ">{price_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; ">{latency_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; ">{complexity_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; font-weight: 600;">{balanced_score:.1f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_badge}</td>'
            f'</tr>'
        )

    # Table header styles
    th_style = 'style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;"'
    th_style_right = 'style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;"'
    th_style_center = 'style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;"'

    table_header = (
        '<thead>'
        f'<tr >'
        f'<th {th_style}>Category</th>'
        f'<th {th_style}>Model</th>'
        f'<th {th_style}>GPU Config</th>'
        f'<th {th_style_right}>TTFT ({percentile_label})</th>'
        f'<th {th_style_right}>Cost/mo</th>'
        f'<th {th_style_center}>Acc</th>'
        f'<th {th_style_center}>Cost</th>'
        f'<th {th_style_center}>Lat</th>'
        f'<th {th_style_center}>Cap</th>'
        f'<th {th_style_center}>Bal</th>'
        f'<th {th_style_center}>SLO</th>'
        '</tr>'
        '</thead>'
    )

    # Build ONE unified table with all categories
    # Add a first column for expand/collapse buttons rendered as part of the Category cell

    # Collect expansion state and category data
    category_data = []
    for cat_key, cat_name, cat_emoji in categories:
        recs = response.get(cat_key, [])
        is_expanded = cat_key in st.session_state.expanded_categories
        more_count = len(recs) - 1 if len(recs) > 1 else 0
        category_data.append({
            "key": cat_key,
            "name": cat_name,
            "emoji": cat_emoji,
            
            "recs": recs,
            "is_expanded": is_expanded,
            "more_count": more_count,
        })

    # Render expand/collapse toggle buttons in a compact row
    # These appear above the table, one per category with expandable options
    expandable_cats = [c for c in category_data if c["more_count"] > 0]
    if expandable_cats:
        st.markdown(
            '<div style="margin-bottom: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">',
            unsafe_allow_html=True
        )
        toggle_cols = st.columns(len(expandable_cats) + 1)
        toggle_cols[0].markdown(
            '<span style="font-size: 0.8rem;">Expand:</span>',
            unsafe_allow_html=True
        )
        for idx, cat in enumerate(expandable_cats):
            with toggle_cols[idx + 1]:
                btn_label = f"−{cat['emoji']}" if cat["is_expanded"] else f"+{cat['emoji']}"
                if st.button(
                    btn_label,
                    key=f"toggle_{cat['key']}",
                    help=f"{'Collapse' if cat['is_expanded'] else 'Expand'} {cat['name']} (+{cat['more_count']})"
                ):
                    if cat["is_expanded"]:
                        st.session_state.expanded_categories.discard(cat["key"])
                    else:
                        st.session_state.expanded_categories.add(cat["key"])
                    st.rerun()

    # Build all rows for the unified table
    all_rows = []

    for cat in category_data:
        cat_key = cat["key"]
        cat_name = cat["name"]
        cat_emoji = cat["emoji"]
        
        recs = cat["recs"]
        is_expanded = cat["is_expanded"]
        more_count = cat["more_count"]

        if not recs:
            # Empty category row
            all_rows.append(
                f'<tr >'
                f'<td style="padding: 0.75rem 0.5rem;"><span style="font-weight: 600;">{cat_emoji} {cat_name}</span></td>'
                f'<td colspan="10" style="padding: 0.75rem 0.5rem; font-style: italic;">No configurations found</td>'
                f'</tr>'
            )
        else:
            # Top recommendation for this category
            all_rows.append(build_row(recs[0], cat_name, cat_emoji, is_top=True, more_count=more_count))

            # Add additional rows if expanded
            if is_expanded and len(recs) > 1:
                for rec in recs[1:]:
                    all_rows.append(build_row(rec, "", "", is_top=False, more_count=0))

    # Render the single unified table
    unified_table_html = (
        f'<table style="width: 100%; border-collapse: collapse;">'
        + table_header +
        '<tbody>' + ''.join(all_rows) + '</tbody>'
        '</table>'
    )

    st.markdown(unified_table_html, unsafe_allow_html=True)


def get_workload_insights(use_case: str, qps: int, user_count: int) -> list:
    """Get workload pattern insights based on API data.

    Returns list of tuples: (icon, color, message, severity)
    """
    messages = []

    # Get workload profile from API
    workload_profile = fetch_workload_profile(use_case)
    if not workload_profile:
        return messages

    distribution = workload_profile.get('distribution', 'poisson')
    active_fraction = workload_profile.get('active_fraction', 0.2)

    messages.append((
        "", "",
        f"Pattern: {distribution.replace('_', ' ').title()} | {int(active_fraction*100)}% concurrent users",
        "info"
    ))

    return messages

# =============================================================================
# API FUNCTIONS
# =============================================================================

from typing import Optional

def extract_business_context(user_input: str) -> Optional[dict]:
    """Extract business context using the backend LLM extraction API.

    Returns None on failure — caller is responsible for showing errors.
    """
    try:
        logger.info(f"Calling LLM extraction API at {API_BASE_URL}/api/v1/extract")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract",
            json={"text": user_input},
            timeout=60,
        )
        if response.status_code == 200:
            result = response.json()
            # Map preferred_gpu_types (list) to hardware for UI compatibility
            if 'preferred_gpu_types' in result:
                gpu_list = result['preferred_gpu_types']
                result['hardware'] = ", ".join(gpu_list) if gpu_list else None
            logger.info(f"LLM extraction successful: {result.get('use_case')}")
            return result
        else:
            logger.warning(f"LLM extraction API returned status {response.status_code}: {response.text[:200]}")
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Backend not reachable: {e}")
    except requests.exceptions.Timeout as e:
        logger.warning(f"LLM extraction timed out: {e}")
    except Exception as e:
        logger.warning(f"LLM extraction failed: {type(e).__name__}: {e}")

    return None


# =============================================================================
# VISUAL COMPONENTS
# =============================================================================

def render_hero():
    """Render compact hero section with logo."""
    logo_col, title_col = st.columns([1, 11])
    with logo_col:
        st.image("ui/static/neuralnav-logo.png", width=48)
    with title_col:
        st.title("NeuralNav")
    st.caption("AI-Powered LLM Deployment Recommendations — From Natural Language to Production in Seconds")


def render_deployment_tab():
    """Tab 4: Deployment configuration and YAML generation."""
    st.subheader("Deployment Configuration")

    # Check if a configuration is selected
    selected_config = st.session_state.get("deployment_selected_config")

    if not selected_config:
        st.info("No configuration selected for deployment. Please select a configuration from the Recommendations tab.")
        st.markdown("""
        **How to select a configuration:**
        1. Go to the **Recommendations** tab
        2. Browse the available configurations
        3. Click **Select for Deployment** on your preferred configuration
        """)
        return

    # Display selected configuration summary
    model_name = selected_config.get("model_name", "Unknown Model")
    gpu_config = selected_config.get("gpu_config", {})
    gpu_type = gpu_config.get("gpu_type", "Unknown")
    gpu_count = gpu_config.get("gpu_count", 1)
    replicas = gpu_config.get("replicas", 1)

    st.markdown(f"""
    <div style="border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.85rem;">Model:</span>
                <span style="font-weight: 600; font-size: 1rem; margin-left: 0.5rem;">{model_name}</span>
            </div>
            <div>
                <span style="font-size: 0.85rem;">GPU:</span>
                <span style="font-weight: 600; font-size: 1rem; margin-left: 0.5rem;">{gpu_count}x{gpu_type}</span>
            </div>
            <div>
                <span style="font-size: 0.85rem;">Replicas:</span>
                <span style="font-weight: 600; font-size: 1rem; margin-left: 0.5rem;">{replicas}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # YAML Generation Section
    if not st.session_state.get("deployment_yaml_generated"):
        # YAML files not yet generated (edge case - shouldn't normally happen with auto-generation)
        st.subheader("Deployment Files")
        st.warning("YAML files have not been generated yet.")

        if st.button("Generate YAML Files", type="primary", key="generate_yaml_btn"):
            with st.spinner("Generating deployment files..."):
                try:
                    # Call backend API to generate YAML files
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/deploy",
                        json={
                            "recommendation": selected_config,
                            "namespace": "default"
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result.get("success"):
                        deployment_id = result.get("deployment_id")
                        st.session_state.deployment_id = deployment_id

                        # Fetch the actual YAML content
                        yaml_response = requests.get(
                            f"{API_BASE_URL}/api/v1/deployments/{deployment_id}/yaml",
                            timeout=10
                        )
                        yaml_response.raise_for_status()
                        yaml_data = yaml_response.json()

                        st.session_state.deployment_yaml_files = yaml_data.get("files", {})
                        st.session_state.deployment_yaml_generated = True
                        st.rerun()
                    else:
                        st.session_state.deployment_error = result.get('message', 'Unknown error')
                        st.rerun()

                except requests.exceptions.RequestException as e:
                    st.session_state.deployment_error = f"Failed to connect to backend API: {e}"
                    st.rerun()
                except Exception as e:
                    st.session_state.deployment_error = f"Error generating YAML files: {e}"
                    st.rerun()
    else:
        # Display generated YAML files
        deployment_id = st.session_state.get("deployment_id", "unknown")
        yaml_files = st.session_state.get("deployment_yaml_files", {})
        deployment_error = st.session_state.get("deployment_error")

        # Prepare zip file for download
        zip_buffer = None
        if yaml_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, content in yaml_files.items():
                    zf.writestr(filename, content)
            zip_buffer.seek(0)

        # Header with download icon and ID/error status
        header_col1, header_col2 = st.columns([0.07, 0.93])
        with header_col1:
            if zip_buffer:
                st.download_button(
                    label="⬇",
                    data=zip_buffer.getvalue(),
                    file_name=f"{deployment_id}.zip",
                    mime="application/zip",
                    key="download_all_yaml",
                    help="Download all YAML files as zip"
                )
            else:
                st.markdown("⬇", unsafe_allow_html=True)
        with header_col2:
            if deployment_error:
                st.markdown(f'<h3 style="font-weight: 700; margin: 0;">Deployment Files <span style="color: #ef4444; font-weight: 700;">(ERROR!)</span></h3>', unsafe_allow_html=True)
                st.error(deployment_error)
            else:
                st.markdown(f'<h3 style="font-weight: 700; margin: 0;">Deployment Files <span style="font-weight: 400; font-size: 0.8rem;">({deployment_id})</span></h3>', unsafe_allow_html=True)

        if yaml_files:
            # Sort files for consistent display order
            file_order = ["inferenceservice", "autoscaling", "servicemonitor"]
            file_labels = {
                "inferenceservice": "InferenceService (KServe)",
                "autoscaling": "Autoscaling (HPA)",
                "servicemonitor": "ServiceMonitor (Prometheus)",
            }

            for file_key in file_order:
                # Find matching file
                matching_file = None
                matching_content = None
                for filename, content in yaml_files.items():
                    if file_key in filename.lower():
                        matching_file = filename
                        matching_content = content
                        break

                if matching_content:
                    label = file_labels.get(file_key, file_key)
                    with st.expander(f"{label}", expanded=False):
                        st.code(matching_content, language="yaml")

        # Regenerate button
        if st.button("Regenerate YAML Files", key="regenerate_yaml_btn"):
            st.session_state.deployment_yaml_generated = False
            st.session_state.deployment_yaml_files = {}
            st.session_state.deployment_id = None
            st.session_state.deployment_error = None
            st.rerun()



def render_top5_table(recommendations: list, priority: str):
    """Render beautiful Top 5 recommendation leaderboard table with filtering.
    
    NOTE: The backend now implements the ACCURACY-FIRST strategy in analyzer.py.
    The UI uses the backend's pre-ranked lists directly from st.session_state.ranked_response.
    """
    
    st.subheader("Best Model Recommendations")
    
    # Show filter summary stats
    ranked_response_stats = st.session_state.get("ranked_response", {})
    total_configs = ranked_response_stats.get("total_configs_evaluated", 0)
    passed_configs = ranked_response_stats.get("configs_after_filters", 0)
    
    # Count unique models from passed configs
    all_passed = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        all_passed.extend(ranked_response_stats.get(cat, []))
    unique_models = len(set(r.get("model_name", "") for r in all_passed if r.get("model_name")))
    
    if total_configs > 0:
        filter_pct = (passed_configs / total_configs * 100) if total_configs > 0 else 0
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem; padding: 0.6rem 1rem;
                    border-radius: 8px; ">
            <span style="font-size: 0.85rem;">
                <strong style="color: #10B981;">{passed_configs:,}</strong> configs passed SLO filter 
                from <strong>{total_configs:,}</strong> total 
                <span >({filter_pct:.0f}% match)</span>
            </span>
            <span >|</span>
            <span style="font-size: 0.85rem;">
                <strong>{unique_models}</strong> unique models
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Get use case for raw accuracy lookup
    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")
    
    if not recommendations:
        st.info("No models available. Please check your requirements.")
        return
    
    # Helper function to get scores from recommendation
    # ALL scores come from BACKEND - no UI calculation
    def get_scores(rec):
        backend_scores = rec.get("scores", {}) or {}
        
        return {
            "accuracy": backend_scores.get("accuracy_score", 0),
            "latency": backend_scores.get("latency_score", 0),
            "cost": backend_scores.get("price_score", 0),
            "complexity": backend_scores.get("complexity_score", 0),
            "final": backend_scores.get("balanced_score", 0),
        }
    
    # ==========================================================================
    # USE BACKEND'S PRE-RANKED LISTS (ACCURACY-FIRST strategy applied in backend)
    # The backend's analyzer.py implements:
    # 1. Get top 5 unique models by raw accuracy (quality baseline)
    # 2. Filter all configs to only those high-quality models
    # 3. Best Latency/Cost/etc. are ranked WITHIN that quality tier
    # ==========================================================================
    ranked_response = st.session_state.get("ranked_response", {})
    
    # Get pre-ranked lists from backend (already filtered to high-quality models)
    top5_balanced = ranked_response.get("balanced", [])[:5]
    top5_accuracy = ranked_response.get("best_accuracy", [])[:5]
    top5_latency = ranked_response.get("lowest_latency", [])[:5]
    top5_cost = ranked_response.get("lowest_cost", [])[:5]
    top5_simplest = ranked_response.get("simplest", [])[:5]
    
    # Best = first in each list
    best_overall = top5_balanced[0] if top5_balanced else None
    best_accuracy = top5_accuracy[0] if top5_accuracy else None
    best_latency = top5_latency[0] if top5_latency else None
    best_cost = top5_cost[0] if top5_cost else None
    best_simplest = top5_simplest[0] if top5_simplest else None
    
    # Store top 5 for each category in session state (for explore dialogs)
    st.session_state.top5_balanced = top5_balanced
    st.session_state.top5_accuracy = top5_accuracy
    st.session_state.top5_latency = top5_latency
    st.session_state.top5_cost = top5_cost
    st.session_state.top5_simplest = top5_simplest
    
    # Helper to render a "Best" card
    def render_best_card(title, icon, color, rec, highlight_field):
        scores = get_scores(rec)
        model_name = format_display_name(rec.get('model_name', 'Unknown'))
        gpu_cfg = rec.get('gpu_config', {}) or {}
        hw_type = gpu_cfg.get('gpu_type', rec.get('hardware', 'H100'))
        hw_count = gpu_cfg.get('gpu_count', rec.get('hardware_count', 1))
        hw_display = f"{hw_count}x {hw_type}"
        
        highlight_value = scores.get(highlight_field, 0)
        final_score = scores.get("final", 0)
        
        return f'''
        <div style="border: 2px solid {color}40; border-radius: 16px; padding: 1.25rem; 
                    ">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="color: {color}; font-weight: 700; font-size: 1.1rem;">{title}</span>
        </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="flex: 1;">
                    <div style="font-weight: 700; font-size: 1.15rem;">{model_name}</div>
                    <div style="font-size: 0.85rem;">{hw_display}</div>
        </div>
                <div style="text-align: right;">
                    <div style="color: {color}; font-size: 2rem; font-weight: 800;">{highlight_value:.0f}</div>
                    <div style="font-size: 0.7rem;">SCORE</div>
        </div>
        </div>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem; padding-top: 0.75rem; ">
                <span style="font-size: 0.8rem;">Acc {scores["accuracy"]:.0f}</span>
                <span style="font-size: 0.8rem;">Lat {scores["latency"]:.0f}</span>
                <span style="font-size: 0.8rem;">Cost {scores["cost"]:.0f}</span>
                <span style="font-size: 0.8rem; font-weight: 700;">Final: {final_score:.1f}</span>
    </div>
                        </div>
        '''
    
    # Helper to render a CAROUSEL card with arrows INSIDE the card design
    def render_category_card(title, recs_list, highlight_field, category_key, col):
        """Render the #1 recommendation for a category with a Select button."""
        if not recs_list:
            return

        rec = recs_list[0]
        scores = get_scores(rec)
        model_name = format_display_name(rec.get('model_name', 'Unknown'))
        gpu_cfg = rec.get('gpu_config', {}) or {}
        hw_type = gpu_cfg.get('gpu_type', rec.get('hardware', 'H100'))
        hw_count = gpu_cfg.get('gpu_count', rec.get('hardware_count', 1))
        cost = rec.get('cost_per_month_usd', 0)
        highlight_value = scores.get(highlight_field, 0)

        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(f"{model_name}")
                c1, c2 = st.columns(2)
                c1.metric("Score", f"{highlight_value:.0f}")
                c2.metric("Cost/mo", f"${cost:,.0f}")
                st.caption(f"{hw_count}x {hw_type} | Acc {scores['accuracy']:.0f} | Lat {scores['latency']:.0f}")

                selected_category = st.session_state.get("deployment_selected_category")
                is_selected = (selected_category == category_key)

                if is_selected:
                    if st.button("Selected", key=f"selected_{category_key}", use_container_width=True, type="primary"):
                        st.session_state.deployment_selected_config = None
                        st.session_state.deployment_selected_category = None
                        st.session_state.deployment_yaml_generated = False
                        st.session_state.deployment_yaml_files = {}
                        st.session_state.deployment_id = None
                        st.session_state.deployment_error = None
                        st.rerun()
                else:
                    if st.button("Select", key=f"select_{category_key}", use_container_width=True):
                        st.session_state.deployment_selected_config = rec
                        st.session_state.deployment_selected_category = category_key
                        st.session_state.deployment_yaml_generated = False
                        st.session_state.deployment_yaml_files = {}
                        st.session_state.deployment_id = None

                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/api/v1/deploy",
                                json={"recommendation": rec, "namespace": "default"},
                                timeout=30,
                            )
                            response.raise_for_status()
                            result = response.json()
                            if result.get("success"):
                                deployment_id = result.get("deployment_id")
                                st.session_state.deployment_id = deployment_id
                                yaml_response = requests.get(
                                    f"{API_BASE_URL}/api/v1/deployments/{deployment_id}/yaml",
                                    timeout=10,
                                )
                                yaml_response.raise_for_status()
                                yaml_data = yaml_response.json()
                                st.session_state.deployment_yaml_files = yaml_data.get("files", {})
                                st.session_state.deployment_yaml_generated = True
                            else:
                                st.session_state.deployment_yaml_generated = False
                        except Exception:
                            st.session_state.deployment_yaml_generated = False
                        st.rerun()

    # Render 4 category cards: 2 on top row, 2 on bottom row
    col1, col2 = st.columns(2)
    render_category_card("Balanced", top5_balanced, "final", "balanced", col1)
    render_category_card("Best Accuracy", top5_accuracy, "accuracy", "accuracy", col2)

    col3, col4 = st.columns(2)
    render_category_card("Best Latency", top5_latency, "latency", "latency", col3)
    render_category_card("Best Cost", top5_cost, "cost", "cost", col4)

    total_available = len(recommendations)
    if total_available <= 2:
        use_case_display = use_case.replace('_', ' ').title() if use_case else "this task"
        st.info(f"Only {total_available} model(s) have benchmarks for {use_case_display}")
    


def render_score_bar(label: str, icon: str, score: float, bar_class: str, contribution: float):
    """Render a score with progress bar using native Streamlit."""
    import math
    if math.isnan(score) if isinstance(score, float) else False:
        score = 50
    if math.isnan(contribution) if isinstance(contribution, float) else False:
        contribution = 0
    score = max(0, min(100, score))
    col_label, col_bar, col_val = st.columns([2, 4, 2])
    with col_label:
        st.write(f"**{label}**")
    with col_bar:
        st.progress(int(score))
    with col_val:
        st.write(f"{score:.0f} (+{contribution:.1f})")


def render_slo_cards(use_case: str, user_count: int, priority: str = "balanced"):
    """Render SLO and workload impact cards with editable fields.
    
    SLO defaults are set to MAX (showing all configs by default).
    User drags slider LEFT to tighten SLOs and filter down configs.
    Priority affects the MAX - low_latency has tighter max values.
    """
    # Fetch SLO defaults (min, max, default) from backend
    slo_defaults = fetch_slo_defaults(use_case)
    if not slo_defaults:
        st.error(f"Failed to fetch SLO defaults for use case: {use_case}")
        return

    # Fetch expected RPS from backend using research-based workload patterns
    rps_data = fetch_expected_rps(use_case, user_count)
    if rps_data:
        estimated_qps = int(rps_data.get("expected_rps", 1))
    else:
        st.error("Failed to fetch expected RPS from backend. Please ensure the backend is running.")
        return

    # Track if use_case or user_count changed - if so, reset custom_qps to use new default
    last_use_case = st.session_state.get("_last_rps_use_case")
    last_user_count = st.session_state.get("_last_rps_user_count")
    if last_use_case != use_case or last_user_count != user_count:
        st.session_state.custom_qps = None
        st.session_state._last_rps_use_case = use_case
        st.session_state._last_rps_user_count = user_count
        if "edit_qps" in st.session_state:
            del st.session_state["edit_qps"]

    qps = int(st.session_state.custom_qps) if st.session_state.custom_qps else estimated_qps
    
    st.subheader("Set Technical Specification")
    
    
    # Create 4 columns for all cards in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**SLO Targets**")

        # Load workload profile from API for token config
        workload_profile = fetch_workload_profile(use_case)
        if not workload_profile:
            st.error(f"Failed to fetch workload profile for use case: {use_case}")
            return
        # Percentile options
        percentile_options = ["P50", "P90", "P95", "P99"]
        percentile_map = {"P50": "p50", "P90": "p90", "P95": "p95", "P99": "p99"}
        reverse_map = {"p50": "P50", "p90": "P90", "p95": "P95", "p99": "P99"}

        # Initialize percentile states for each metric
        if 'ttft_percentile' not in st.session_state:
            st.session_state.ttft_percentile = "p95"
        if 'itl_percentile' not in st.session_state:
            st.session_state.itl_percentile = "p95"
        if 'e2e_percentile' not in st.session_state:
            st.session_state.e2e_percentile = "p95"

        # Track previous SLO values to detect changes
        prev_ttft = st.session_state.get("_last_ttft")
        prev_itl = st.session_state.get("_last_itl")
        prev_e2e = st.session_state.get("_last_e2e")

        # SLO min/max ranges from backend (already fetched above)
        def get_metric_range(metric: str) -> tuple:
            metric_key = {"ttft": "ttft_ms", "itl": "itl_ms", "e2e": "e2e_ms"}[metric]
            return (
                int(slo_defaults[metric_key]["min"]),
                int(slo_defaults[metric_key]["max"]),
            )

        # === TTFT ===
        ttft_min, ttft_max = get_metric_range("ttft")
        if 'input_ttft' not in st.session_state:
            st.session_state.input_ttft = slo_defaults["ttft_ms"]["default"]

        st.markdown('<div style="margin-top: 0.5rem; margin-bottom: 0.25rem;"><span style="font-weight: 700; font-size: 0.95rem;">TTFT (Time to First Token)</span></div>', unsafe_allow_html=True)
        ttft_val_col, ttft_pct_col = st.columns([2, 1])
        with ttft_val_col:
            st.number_input("TTFT value", min_value=ttft_min, max_value=ttft_max, step=100, key="input_ttft", label_visibility="collapsed")
            st.session_state.custom_ttft = st.session_state.input_ttft
        with ttft_pct_col:
            ttft_pct_display = reverse_map.get(st.session_state.ttft_percentile, "P95")
            selected_ttft_pct = st.selectbox("TTFT percentile", percentile_options, index=percentile_options.index(ttft_pct_display), key="ttft_pct_selector", label_visibility="collapsed")
            st.session_state.ttft_percentile = percentile_map[selected_ttft_pct]
        st.caption(f"Recommended Range: {ttft_min:,} - {ttft_max:,} ms")

        # === ITL ===
        itl_min, itl_max = get_metric_range("itl")
        if 'input_itl' not in st.session_state:
            st.session_state.input_itl = slo_defaults["itl_ms"]["default"]

        st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.25rem;"><span style="font-weight: 700; font-size: 0.95rem;">ITL (Inter-Token Latency)</span></div>', unsafe_allow_html=True)
        itl_val_col, itl_pct_col = st.columns([2, 1])
        with itl_val_col:
            st.number_input("ITL value", min_value=itl_min, max_value=itl_max, step=10, key="input_itl", label_visibility="collapsed")
            st.session_state.custom_itl = st.session_state.input_itl
        with itl_pct_col:
            itl_pct_display = reverse_map.get(st.session_state.itl_percentile, "P95")
            selected_itl_pct = st.selectbox("ITL percentile", percentile_options, index=percentile_options.index(itl_pct_display), key="itl_pct_selector", label_visibility="collapsed")
            st.session_state.itl_percentile = percentile_map[selected_itl_pct]
        st.caption(f"Recommended Range: {itl_min:,} - {itl_max:,} ms")

        # === E2E ===
        e2e_min, e2e_max = get_metric_range("e2e")
        if 'input_e2e' not in st.session_state:
            st.session_state.input_e2e = slo_defaults["e2e_ms"]["default"]

        st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.25rem;"><span style="font-weight: 700; font-size: 0.95rem;">E2E (End-to-End Latency)</span></div>', unsafe_allow_html=True)
        e2e_val_col, e2e_pct_col = st.columns([2, 1])
        with e2e_val_col:
            st.number_input("E2E value", min_value=e2e_min, max_value=e2e_max, step=1000, key="input_e2e", label_visibility="collapsed")
            st.session_state.custom_e2e = st.session_state.input_e2e
        with e2e_pct_col:
            e2e_pct_display = reverse_map.get(st.session_state.e2e_percentile, "P95")
            selected_e2e_pct = st.selectbox("E2E percentile", percentile_options, index=percentile_options.index(e2e_pct_display), key="e2e_pct_selector", label_visibility="collapsed")
            st.session_state.e2e_percentile = percentile_map[selected_e2e_pct]
        st.caption(f"Recommended Range: {e2e_min:,} - {e2e_max:,} ms")

        # Check if SLO values changed - if so, clear recommendation cache
        curr_ttft = st.session_state.get("custom_ttft")
        curr_itl = st.session_state.get("custom_itl")
        curr_e2e = st.session_state.get("custom_e2e")

        if (curr_ttft != prev_ttft or curr_itl != prev_itl or curr_e2e != prev_e2e):
            # SLO values changed - invalidate recommendation cache
            if st.session_state.get("recommendation_result"):
                st.session_state.recommendation_result = None
            # Store current values for next comparison
            st.session_state._last_ttft = curr_ttft
            st.session_state._last_itl = curr_itl
            st.session_state._last_e2e = curr_e2e
    
    with col2:
        st.write("**Workload Profile**")

        # Load workload profile from API
        workload_profile = fetch_workload_profile(use_case)
        if not workload_profile:
            st.error(f"Failed to fetch workload profile for use case: {use_case}")
            return
        prompt_tokens = workload_profile['prompt_tokens']
        output_tokens = workload_profile['output_tokens']
        peak_mult = workload_profile['peak_multiplier']

        # Store workload profile values in session state for Recommendations tab
        st.session_state.spec_prompt_tokens = prompt_tokens
        st.session_state.spec_output_tokens = output_tokens
        st.session_state.spec_peak_multiplier = peak_mult

        # 1. Editable QPS - support up to 10M QPS for enterprise scale
        # Get research-based default QPS for this use case
        default_qps = estimated_qps  # This is the research-based default
        new_qps = st.number_input("Expected RPS", value=min(qps, 10000000), min_value=1, max_value=10000000, step=1, key="edit_qps", label_visibility="collapsed")
        st.markdown(f'<div style="font-size: 0.9rem; margin-top: -0.75rem; margin-bottom: 0.5rem;">Expected RPS: <span style="font-weight: 700; font-size: 1rem;">{new_qps}</span> <span style="font-size: 0.75rem;">(default: {default_qps})</span></div>', unsafe_allow_html=True)

        # Store the actual QPS value shown to user (not just custom override)
        st.session_state.spec_expected_qps = new_qps

        if new_qps != qps:
            st.session_state.custom_qps = new_qps
        
        # RPS change warning - show implications of changing from research-based default
        if new_qps > default_qps * 2:
            qps_ratio = new_qps / max(default_qps, 1)
            st.markdown(f'''
            <div style="border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 0.6rem; margin: 0.5rem 0;">
                <div style="color: #ef4444; font-weight: 600; font-size: 0.85rem;">High RPS Warning ({qps_ratio:.1f}x default)</div>
                <div style="font-size: 0.75rem; margin-top: 0.3rem;">
                    • Requires <strong >{int(qps_ratio)}x more GPU replicas</strong><br/>
                    • Estimated cost increase: <strong style="color: #ef4444;">~{int((qps_ratio-1)*100)}%</strong><br/>
                    • Consider load balancing or queue-based architecture
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif new_qps > default_qps * 1.5:
            qps_ratio = new_qps / max(default_qps, 1)
            st.markdown(f'''
            <div style="border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
                <div style="font-weight: 600; font-size: 0.8rem;">Elevated RPS ({qps_ratio:.1f}x default)</div>
                <div style="font-size: 0.7rem; margin-top: 0.2rem;">
                    May need additional replicas. Cost ~{int((qps_ratio-1)*100)}% higher.
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif new_qps < default_qps * 0.5 and default_qps > 1:
            st.markdown(f'''
            <div style="border: 1px solid rgba(56, 239, 125, 0.3); border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
                <div style="font-weight: 600; font-size: 0.8rem;">Low RPS - Cost Savings Possible</div>
                <div style="font-size: 0.7rem; margin-top: 0.2rem;">
                    Single replica may suffice. Consider smaller GPU or spot instances.
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # 2-4. Fixed workload values with inline descriptions (like datasets)
        st.markdown(f"""
        <div style="margin-top: 0.5rem; padding: 0.75rem; border-radius: 8px;">
            <div style="padding: 0.5rem 0; ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 500;">Mean Input Tokens</span>
                    <span style="font-weight: 700; font-size: 1.1rem; padding: 3px 10px; border-radius: 4px;">{prompt_tokens}</span>
            </div>
                <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Average input length per request (research-based for {use_case.replace('_', ' ')})</div>
            </div>
            <div style="padding: 0.5rem 0; ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 500;">Mean Output Tokens</span>
                    <span style="font-weight: 700; font-size: 1.1rem; padding: 3px 10px; border-radius: 4px;">{output_tokens}</span>
                </div>
                <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Average output length generated per request</div>
            </div>
            <div style="padding: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 500;">Peak Multiplier</span>
                    <span style="font-weight: 700; font-size: 1.1rem; padding: 3px 10px; border-radius: 4px;">{peak_mult}x</span>
                </div>
                <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Capacity buffer for traffic spikes (user behavior patterns)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Informational messages from research data - black background, no emojis
        workload_messages = get_workload_insights(use_case, new_qps, user_count)
        
        for icon, color, text, severity in workload_messages[:3]:  # Limit to 3 for space
            st.markdown(f'<div style="font-size: 0.85rem; padding: 0.4rem 0.5rem; line-height: 1.4; border-radius: 6px; margin: 4px 0;">{text}</div>', unsafe_allow_html=True)

    with col3:
        # Accuracy Benchmarks - show which benchmarks are used for this use case
        # Each entry: (name, weight, color, tooltip_description) - all black background
        # Updated based on USE_CASE_METHODOLOGY.md (December 2024)
        # Focus on TOP 3 benchmarks: τ²-Bench (91%), LiveCodeBench (84%), MMLU-Pro (83%), GPQA (82%)
        TASK_DATASETS = {
            "chatbot_conversational": [
                ("τ²-Bench", 45, "Conversational AI agentic workflow (114 tasks)"),
                ("MMLU-Pro", 35, "General knowledge (12,032 questions)"),
                ("GPQA", 20, "Scientific reasoning (198 questions)"),
            ],
            "code_completion": [
                ("LiveCodeBench", 45, "Code benchmark (315 questions)"),
                ("τ²-Bench", 35, "Agentic code assistance (114 tasks)"),
                ("MMLU-Pro", 20, "Knowledge for context"),
            ],
            "code_generation_detailed": [
                ("LiveCodeBench", 40, "Code generation (315 questions)"),
                ("τ²-Bench", 40, "Agentic reasoning (114 tasks)"),
                ("GPQA", 20, "Scientific reasoning for explanations"),
            ],
            "translation": [
                ("τ²-Bench", 45, "Language agentic tasks (114 tasks)"),
                ("MMLU-Pro", 35, "Language understanding (12,032 questions)"),
                ("GPQA", 20, "Reasoning"),
            ],
            "content_generation": [
                ("τ²-Bench", 45, "Creative agentic workflow (114 tasks)"),
                ("MMLU-Pro", 35, "General knowledge for facts"),
                ("GPQA", 20, "Reasoning"),
            ],
            "summarization_short": [
                ("τ²-Bench", 45, "Summarization agentic (114 tasks)"),
                ("MMLU-Pro", 35, "Comprehension (12,032 questions)"),
                ("GPQA", 20, "Reasoning"),
            ],
            "document_analysis_rag": [
                ("τ²-Bench", 50, "RAG is agentic workflow - DOMINANT (114 tasks)"),
                ("GPQA", 30, "Scientific reasoning for factual answers"),
                ("MMLU-Pro", 20, "Knowledge retrieval"),
            ],
            "long_document_summarization": [
                ("τ²-Bench", 50, "Long doc handling is agentic (114 tasks)"),
                ("MMLU-Pro", 30, "Knowledge for understanding"),
                ("GPQA", 20, "Reasoning"),
            ],
            "research_legal_analysis": [
                ("τ²-Bench", 55, "Research analysis is agentic reasoning - CRITICAL"),
                ("GPQA", 25, "Scientific reasoning (198 questions)"),
                ("MMLU-Pro", 20, "Knowledge (12,032 questions)"),
            ],
        }
        
        datasets = TASK_DATASETS.get(use_case, TASK_DATASETS["chatbot_conversational"])
        
        st.write("**Accuracy Benchmarks**")

        # Display datasets with weights
        datasets_html = '<div style="padding: 0.5rem;">'
        for item in datasets:
            name = item[0]
            weight = item[1]
            tooltip = item[2] if len(item) > 2 else ""
            datasets_html += f'''<div style="padding: 0.5rem 0; ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 500;">{name}</span>
                    <span style="font-weight: 700; font-size: 0.95rem; padding: 3px 10px; border-radius: 4px;">{weight}%</span>
            </div>
                <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">{tooltip}</div>
            </div>'''
        datasets_html += '</div>'
        datasets_html += '<div style="font-size: 0.75rem; font-weight: 600; margin-top: 0.5rem;">Weights from Artificial Analysis Intelligence Index</div>'
        st.markdown(datasets_html, unsafe_allow_html=True)

    with col4:
        # PRIORITIES card - interactive priority dropdowns with weights
        st.write("**Priorities**")

        # Load priority weights config from backend (cached)
        priority_config = fetch_priority_weights()
        # priority_config structure: {priority_weights: {accuracy: {low: 3, ...}, ...}, defaults: {...}}
        priority_weights_map = priority_config.get("priority_weights", {}) if priority_config else {}
        defaults_config = priority_config.get("defaults", {}) if priority_config else {}
        logger.debug(f"Priority config loaded: weights_map keys={list(priority_weights_map.keys())}, defaults={defaults_config}")

        # Priority options for dropdowns
        priority_options = ["Low", "Medium", "High"]
        priority_to_value = {"Low": "low", "Medium": "medium", "High": "high"}
        value_to_priority = {"low": "Low", "medium": "Medium", "high": "High"}

        # Helper to get weight for a priority level
        def get_weight_for_priority(dimension: str, priority_level: str) -> int:
            pw = priority_weights_map.get(dimension, {})
            default_weights = defaults_config.get("weights", {"accuracy": 5, "cost": 4, "latency": 2, "complexity": 2})
            weight = pw.get(priority_level, default_weights.get(dimension, 5))
            logger.info(f"get_weight_for_priority({dimension}, {priority_level}): pw={pw}, weight={weight}")
            return weight

        # Get extraction result to check for LLM-detected priorities
        extraction = st.session_state.get('extraction_result', {})

        # Check if new extraction is available - force update priorities from LLM
        # This ensures "accuracy matters" in user input updates the UI dropdowns
        if st.session_state.get('new_extraction_available', False):
            st.session_state.accuracy_priority = extraction.get('accuracy_priority', 'medium')
            st.session_state.cost_priority = extraction.get('cost_priority', 'medium')
            st.session_state.latency_priority = extraction.get('latency_priority', 'medium')
            st.session_state.weight_accuracy = get_weight_for_priority("accuracy", st.session_state.accuracy_priority)
            st.session_state.weight_cost = get_weight_for_priority("cost", st.session_state.cost_priority)
            st.session_state.weight_latency = get_weight_for_priority("latency", st.session_state.latency_priority)
            st.session_state.new_extraction_available = False  # Clear flag after processing
            logger.info(f"Updated priorities from new extraction: accuracy={st.session_state.accuracy_priority}, "
                       f"cost={st.session_state.cost_priority}, latency={st.session_state.latency_priority}")
        
        # Initialize session state for priorities if not set (first load)
        if 'accuracy_priority' not in st.session_state:
            st.session_state.accuracy_priority = extraction.get('accuracy_priority', 'medium')
        if 'cost_priority' not in st.session_state:
            st.session_state.cost_priority = extraction.get('cost_priority', 'medium')
        if 'latency_priority' not in st.session_state:
            st.session_state.latency_priority = extraction.get('latency_priority', 'medium')
        # Note: complexity_priority removed from UI (backend still extracts it)

        # Initialize weights based on priorities
        if 'weight_accuracy' not in st.session_state:
            st.session_state.weight_accuracy = get_weight_for_priority("accuracy", st.session_state.accuracy_priority)
        if 'weight_cost' not in st.session_state:
            st.session_state.weight_cost = get_weight_for_priority("cost", st.session_state.cost_priority)
        if 'weight_latency' not in st.session_state:
            st.session_state.weight_latency = get_weight_for_priority("latency", st.session_state.latency_priority)
        # Note: weight_complexity removed from UI (hardcoded to 0)

        # Container for the priority rows
        st.markdown('<div style="padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">', unsafe_allow_html=True)

        # Row 1: Accuracy
        acc_col1, acc_col2, acc_col3 = st.columns([2, 1.5, 1])
        with acc_col1:
            st.markdown('<span style="font-size: 0.9rem; font-weight: 500; line-height: 2.5;">Accuracy</span>', unsafe_allow_html=True)
        with acc_col2:
            acc_priority_display = value_to_priority.get(st.session_state.accuracy_priority, "Medium")
            new_acc_priority = st.selectbox("Accuracy", priority_options, index=priority_options.index(acc_priority_display), key="acc_priority_select", label_visibility="collapsed")
            new_acc_priority_val = priority_to_value[new_acc_priority]
            if new_acc_priority_val != st.session_state.accuracy_priority:
                st.session_state.accuracy_priority = new_acc_priority_val
                st.session_state.weight_accuracy = get_weight_for_priority("accuracy", new_acc_priority_val)
        with acc_col3:
            new_acc_weight = st.number_input("Accuracy weight", min_value=0, max_value=10, value=st.session_state.weight_accuracy, key="acc_weight_input", label_visibility="collapsed")
            if new_acc_weight != st.session_state.weight_accuracy:
                st.session_state.weight_accuracy = new_acc_weight

        # Row 2: Cost
        cost_col1, cost_col2, cost_col3 = st.columns([2, 1.5, 1])
        with cost_col1:
            st.markdown('<span style="font-size: 0.9rem; font-weight: 500; line-height: 2.5;">Cost</span>', unsafe_allow_html=True)
        with cost_col2:
            cost_priority_display = value_to_priority.get(st.session_state.cost_priority, "Medium")
            new_cost_priority = st.selectbox("Cost", priority_options, index=priority_options.index(cost_priority_display), key="cost_priority_select", label_visibility="collapsed")
            new_cost_priority_val = priority_to_value[new_cost_priority]
            if new_cost_priority_val != st.session_state.cost_priority:
                st.session_state.cost_priority = new_cost_priority_val
                st.session_state.weight_cost = get_weight_for_priority("cost", new_cost_priority_val)
        with cost_col3:
            new_cost_weight = st.number_input("Cost weight", min_value=0, max_value=10, value=st.session_state.weight_cost, key="cost_weight_input", label_visibility="collapsed")
            if new_cost_weight != st.session_state.weight_cost:
                st.session_state.weight_cost = new_cost_weight

        # Row 3: Latency
        lat_col1, lat_col2, lat_col3 = st.columns([2, 1.5, 1])
        with lat_col1:
            st.markdown('<span style="font-size: 0.9rem; font-weight: 500; line-height: 2.5;">Latency</span>', unsafe_allow_html=True)
        with lat_col2:
            lat_priority_display = value_to_priority.get(st.session_state.latency_priority, "Medium")
            new_lat_priority = st.selectbox("Latency", priority_options, index=priority_options.index(lat_priority_display), key="lat_priority_select", label_visibility="collapsed")
            new_lat_priority_val = priority_to_value[new_lat_priority]
            if new_lat_priority_val != st.session_state.latency_priority:
                st.session_state.latency_priority = new_lat_priority_val
                st.session_state.weight_latency = get_weight_for_priority("latency", new_lat_priority_val)
        with lat_col3:
            new_lat_weight = st.number_input("Latency weight", min_value=0, max_value=10, value=st.session_state.weight_latency, key="lat_weight_input", label_visibility="collapsed")
            if new_lat_weight != st.session_state.weight_latency:
                st.session_state.weight_latency = new_lat_weight

        # Note: Complexity row removed from UI but scoring logic preserved in backend
        # Complexity weight is hardcoded to 0 in fetch_ranked_recommendations()

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# WINNER DETAILS DIALOG
# =============================================================================

@st.dialog("Winner Details", width="large")
def show_winner_details_dialog():
    """Show winner details in a modal dialog."""
    winner = st.session_state.get('balanced_winner') or st.session_state.get('winner_recommendation')
    priority = st.session_state.get('winner_priority', 'balanced')
    extraction = st.session_state.get('winner_extraction', {})
    
    if not winner:
        st.warning("No winner data available.")
        return
    
    # Render the winner details
    _render_winner_details(winner, priority, extraction)
    
    # Close button
    if st.button("Close", key="close_dialog_btn", use_container_width=True):
        st.session_state.show_winner_dialog = False
        st.rerun()


# =============================================================================
# CATEGORY EXPLORATION DIALOG
# =============================================================================

@st.dialog("Top 5 Models", width="large")
def show_category_dialog():
    """Show top 5 models for a category with performance data."""
    category = st.session_state.get('explore_category', 'balanced')
    
    # Get use case for context
    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")
    
    # Category config - theme colors (no emojis)
    category_config = {
        "balanced": {"title": "Balanced - Top 5", "field": "final", "top5_key": "top5_balanced"},
        "accuracy": {"title": "Best Accuracy - Top 5", "field": "accuracy", "top5_key": "top5_accuracy"},
        "latency": {"title": "Best Latency - Top 5", "field": "latency", "top5_key": "top5_latency"},
        "cost": {"title": "Best Cost - Top 5", "field": "cost", "top5_key": "top5_cost"},
    }
    
    config = category_config.get(category, category_config["balanced"])
    top5_list = st.session_state.get(config["top5_key"], [])
    
    st.subheader(config['title'])
    st.caption(f"Use case: {format_use_case_name(use_case)}")
        
    if not top5_list:
        st.warning("No models available for this category.")
        if st.button("Close", key="close_cat_dialog_empty"):
            st.session_state.show_category_dialog = False
            st.rerun()
        return
    
    # Helper to get scores — all from backend
    def get_model_scores(rec):
        backend_scores = rec.get("scores", {}) or {}
        return {
            "accuracy": backend_scores.get("accuracy_score", 0),
            "latency": backend_scores.get("latency_score", 0),
            "cost": backend_scores.get("price_score", 0),
            "complexity": backend_scores.get("complexity_score", 0),
            "final": backend_scores.get("balanced_score", 0),
        }
    
    # Render each model in the top 5
    for i, rec in enumerate(top5_list):
        scores = get_model_scores(rec)
        model_name = format_display_name(rec.get('model_name', 'Unknown'))
        gpu_cfg = rec.get('gpu_config', {}) or {}
        hw_type = gpu_cfg.get('gpu_type', rec.get('hardware', 'H100'))
        hw_count = gpu_cfg.get('gpu_count', rec.get('hardware_count', 1))
        hw_display = f"{hw_count}x {hw_type}"
        tp = gpu_cfg.get('tensor_parallel', 1)
        
        # Get selected percentile from session state
        selected_percentile = st.session_state.get('slo_percentile', 'p95')
        percentile_suffix = selected_percentile  # mean, p90, p95, p99
        percentile_label = selected_percentile.upper() if selected_percentile != 'mean' else 'Mean'
        
        # Get benchmark metrics with all percentiles (from backend)
        benchmark_metrics = rec.get('benchmark_metrics', {}) or {}
        
        # Get values for selected percentile
        ttft = benchmark_metrics.get(f'ttft_{percentile_suffix}', rec.get('predicted_ttft_p95_ms', 'N/A'))
        itl = benchmark_metrics.get(f'itl_{percentile_suffix}', rec.get('predicted_itl_p95_ms', 'N/A'))
        e2e = benchmark_metrics.get(f'e2e_{percentile_suffix}', rec.get('predicted_e2e_p95_ms', 'N/A'))
        tps = benchmark_metrics.get(f'tps_{percentile_suffix}', 0)
        
        throughput_display = f"{tps:.0f} tok/s" if tps and tps > 0 else "N/A"
        
        highlight_score = scores.get(config["field"], 0)

        st.markdown(f"""
        <div style="border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-weight: 800; padding: 0.25rem 0.6rem;
                                border-radius: 6px; font-size: 0.85rem;">#{i+1}</span>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem;">{model_name}</div>
                        <div style="font-size: 0.8rem;">{hw_display} (TP={tp})</div>
        </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.75rem; font-weight: 800;">{highlight_score:.0f}</div>
                    <div style="font-size: 0.65rem; text-transform: uppercase;">FINAL</div>
                </div>
            </div>
            <div style="display: flex; gap: 1rem; margin-bottom: 0.75rem; padding: 0.5rem; 
                        border-radius: 8px;">
                <span style="font-size: 0.8rem;">Acc: {scores['accuracy']:.0f}</span>
                <span style="font-size: 0.8rem;">Lat: {scores['latency']:.0f}</span>
                <span style="font-size: 0.8rem;">Cost: {scores['cost']:.0f}</span>
                <span style="font-size: 0.8rem; font-weight: 600;">Final: {scores['final']:.1f}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem;">
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">TTFT ({percentile_label})</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{ttft if isinstance(ttft, str) else f'{ttft:.0f}ms'}</div>
                </div>
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">ITL ({percentile_label})</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{itl if isinstance(itl, str) else f'{itl:.0f}ms'}</div>
                </div>
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">E2E ({percentile_label})</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{e2e if isinstance(e2e, str) else f'{e2e:.0f}ms'}</div>
                </div>
                <div style="padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.65rem;">Throughput</div>
                    <div style="font-weight: 700; font-size: 0.9rem;">{throughput_display}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close button
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key="close_cat_dialog", use_container_width=True):
        st.session_state.show_category_dialog = False
        st.rerun()


@st.dialog("All Recommendation Options", width="large")
def show_full_table_dialog():
    """Show the full recommendations table with all categories."""
    ranked_response = st.session_state.get('ranked_response')
    
    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        if st.button("Close", key="close_full_table_empty"):
            st.session_state.show_full_table_dialog = False
            st.rerun()
        return
    
    st.subheader("Configuration Options")
    st.caption("All viable deployment configurations ranked by category")
    
    total_configs = ranked_response.get("total_configs_evaluated", 0)
    configs_after_filters = ranked_response.get("configs_after_filters", 0)
    
    st.markdown(f'<div style="margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="font-weight: 600;">{configs_after_filters}</span> unique options</div>', unsafe_allow_html=True)
    
    # Define categories
    categories = [
        ("balanced", "Balanced"),
        ("best_accuracy", "Best Accuracy"),
        ("lowest_cost", "Lowest Cost"),
        ("lowest_latency", "Lowest Latency"),
    ]
    
    # Helper function to format GPU config
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"
    
    # Build table rows
    all_rows = []
    for cat_key, cat_name in categories:
        recs = ranked_response.get(cat_key, [])
        if not recs:
            all_rows.append(f'<tr ><td style="padding: 0.75rem 0.5rem;"><span style="font-weight: 600;">{cat_name}</span></td><td colspan="7" style="padding: 0.75rem 0.5rem; font-style: italic;">No configurations found</td></tr>')
        else:
            for i, rec in enumerate(recs[:5]):  # Show top 5 per category
                model_name = format_display_name(rec.get("model_name", "Unknown"))
                gpu_config = rec.get("gpu_config", {})
                gpu_str = format_gpu_config(gpu_config)
                ttft = rec.get("predicted_ttft_p95_ms", 0)
                cost = rec.get("cost_per_month_usd", 0)
                scores = rec.get("scores", {}) or {}
                accuracy = scores.get("accuracy_score", 0)
                balanced = scores.get("balanced_score", 0)
                meets_slo = rec.get("meets_slo", False)
                slo_icon = "Yes" if meets_slo else "No"
                
                cat_display = f'<span style="font-weight: 600;">{cat_name}</span> (+{len(recs)-1})' if i == 0 else ""
                
                row = f'<tr ><td style="padding: 0.75rem 0.5rem;">{cat_display}</td><td style="padding: 0.75rem 0.5rem; font-weight: 500;">{model_name}</td><td style="padding: 0.75rem 0.5rem; font-size: 0.85rem;">{gpu_str}</td><td style="padding: 0.75rem 0.5rem; text-align: right; ">{ttft:.0f}ms</td><td style="padding: 0.75rem 0.5rem; text-align: right; ">${cost:,.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; ">{accuracy:.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; ">{balanced:.1f}</td><td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_icon}</td></tr>'
                all_rows.append(row)
    
    # Table header
    header = '<thead><tr ><th style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Category</th><th style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Model</th><th style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">GPU Config</th><th style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">TTFT</th><th style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Cost/mo</th><th style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Acc</th><th style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Score</th><th style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">SLO</th></tr></thead>'
    
    # Render table
    table_html = f'<table style="width: 100%; border-collapse: collapse; border-radius: 8px;">{header}<tbody>{"".join(all_rows)}</tbody></table>'
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Close button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key="close_full_table_dialog", use_container_width=True):
        st.session_state.show_full_table_dialog = False
        st.rerun()


def render_options_list_inline():
    """Render the expanded options list content inline on the page."""
    ranked_response = st.session_state.get('ranked_response')

    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        return

    # Header
    st.markdown('<div style="padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; margin-top: 1rem;"><h2 style="margin: 0; font-size: 1.5rem;">Configuration Options</h2><p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">All viable deployment configurations ranked by category</p></div>', unsafe_allow_html=True)

    total_configs = ranked_response.get("total_configs_evaluated", 0)
    configs_after_filters = ranked_response.get("configs_after_filters", 0)

    st.markdown(f'<div style="margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="font-weight: 600;">{configs_after_filters}</span> unique options</div>', unsafe_allow_html=True)

    # Define categories
    categories = [
        ("balanced", "Balanced"),
        ("best_accuracy", "Best Accuracy"),
        ("lowest_cost", "Lowest Cost"),
        ("lowest_latency", "Lowest Latency"),
    ]

    # Helper function to format GPU config
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"

    # Build table data
    table_data = []
    for cat_key, cat_name in categories:
        recs = ranked_response.get(cat_key, [])
        for rec in recs[:5]:  # Show top 5 per category
            model_name = format_display_name(rec.get("model_name", "Unknown"))
            gpu_config = rec.get("gpu_config", {})
            gpu_str = format_gpu_config(gpu_config)
            ttft = rec.get("predicted_ttft_p95_ms", 0)
            cost = rec.get("cost_per_month_usd", 0)
            scores = rec.get("scores", {}) or {}
            accuracy = scores.get("accuracy_score", 0)
            balanced = scores.get("balanced_score", 0)
            meets_slo = rec.get("meets_slo", False)
            slo_value = 1 if meets_slo else 0  # Numeric value for sorting

            table_data.append({
                "category": cat_name,
                "model": model_name,
                "gpu_config": gpu_str,
                "ttft": ttft,
                "cost": cost,
                "accuracy": accuracy,
                "balanced": balanced,
                "slo": "Yes" if meets_slo else "No",
                "slo_value": slo_value,
            })

    if table_data:
        # Build table rows HTML
        rows_html = []
        for row in table_data:
            rows_html.append(f'''
                <tr 
                    data-category="{row['category']}"
                    data-model="{row['model']}"
                    data-gpu="{row['gpu_config']}"
                    data-ttft="{row['ttft']}"
                    data-cost="{row['cost']}"
                    data-accuracy="{row['accuracy']}"
                    data-balanced="{row['balanced']}"
                    data-slo="{row['slo_value']}">
                    <td style="padding: 0.75rem 0.5rem; ">{row['category']}</td>
                    <td style="padding: 0.75rem 0.5rem; font-weight: 500;">{row['model']}</td>
                    <td style="padding: 0.75rem 0.5rem; font-size: 0.85rem;">{row['gpu_config']}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: right; ">{row['ttft']:.0f}ms</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: right; ">${row['cost']:,.0f}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: center; ">{row['accuracy']:.0f}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: center; ">{row['balanced']:.1f}</td>
                    <td style="padding: 0.75rem 0.5rem; text-align: center;">{row['slo']}</td>
                </tr>
            ''')

        # Table HTML with JavaScript sorting
        table_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            .sortable-table {{
                font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            .sortable-table th {{
                cursor: pointer;
                user-select: none;
                position: relative;
            }}
            .sortable-table th:hover {{
                !important;
            }}
            .sortable-table th.sort-asc::after {{
                content: " ▲";
                font-size: 0.7em;
                }}
            .sortable-table th.sort-desc::after {{
                content: " ▼";
                font-size: 0.7em;
                }}
            .sortable-table tbody tr:hover {{
                !important;
            }}
        </style>
        </head>
        <body>
        <table class="sortable-table" id="recsTableInline" style="width: 100%; border-collapse: collapse; border-radius: 8px;">
            <thead>
                <tr >
                    <th onclick="sortTableInline(0, 'string')" style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Category</th>
                    <th onclick="sortTableInline(1, 'string')" style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Model</th>
                    <th onclick="sortTableInline(2, 'string')" style="text-align: left; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">GPU Config</th>
                    <th onclick="sortTableInline(3, 'number')" style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">TTFT</th>
                    <th onclick="sortTableInline(4, 'number')" style="text-align: right; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Cost/mo</th>
                    <th onclick="sortTableInline(5, 'number')" style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Acc</th>
                    <th onclick="sortTableInline(6, 'number')" style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">Score</th>
                    <th onclick="sortTableInline(7, 'number')" style="text-align: center; padding: 0.75rem 0.5rem; font-size: 0.85rem; font-weight: 600;">SLO</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
        <script>
            let sortDirectionInline = {{}};

            function sortTableInline(columnIndex, type) {{
                const table = document.getElementById('recsTableInline');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const headers = table.querySelectorAll('th');

                // Toggle sort direction
                const key = columnIndex;
                sortDirectionInline[key] = sortDirectionInline[key] === 'asc' ? 'desc' : 'asc';
                const isAsc = sortDirectionInline[key] === 'asc';

                // Clear all header classes
                headers.forEach(h => {{
                    h.classList.remove('sort-asc', 'sort-desc');
                }});

                // Add class to current header
                headers[columnIndex].classList.add(isAsc ? 'sort-asc' : 'sort-desc');

                // Sort rows
                rows.sort((a, b) => {{
                    let aVal, bVal;

                    if (type === 'number') {{
                        const attrs = ['category', 'model', 'gpu', 'ttft', 'cost', 'accuracy', 'balanced', 'slo'];
                        const attr = 'data-' + attrs[columnIndex];
                        aVal = parseFloat(a.getAttribute(attr)) || 0;
                        bVal = parseFloat(b.getAttribute(attr)) || 0;
                    }} else {{
                        aVal = a.cells[columnIndex].textContent.trim();
                        bVal = b.cells[columnIndex].textContent.trim();
                    }}

                    if (aVal < bVal) return isAsc ? -1 : 1;
                    if (aVal > bVal) return isAsc ? 1 : -1;
                    return 0;
                }});

                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));
            }}
        </script>
        </body>
        </html>
        '''

        components.html(table_html, height=450, scrolling=True)
        st.markdown('<p style="font-size: 0.85rem; margin-top: 0.5rem;">Click on column headers to sort the table</p>', unsafe_allow_html=True)
    else:
        st.warning("No configurations to display")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Show dialogs if triggered
    # Only show ONE dialog at a time (Streamlit limitation)
    if st.session_state.show_winner_dialog is True and st.session_state.balanced_winner is not None:
        # Reset other dialogs
        st.session_state.show_category_dialog = False
        st.session_state.show_full_table_dialog = False
        show_winner_details_dialog()
    elif st.session_state.get('show_category_dialog') is True:
        # Reset other dialogs
        st.session_state.show_winner_dialog = False
        st.session_state.show_full_table_dialog = False
        show_category_dialog()
    elif st.session_state.get('show_full_table_dialog') is True:
        # Reset other dialogs
        st.session_state.show_winner_dialog = False
        st.session_state.show_category_dialog = False
        show_full_table_dialog()
    
    # Load models
    if st.session_state.models_df is None:
        st.session_state.models_df = load_206_models()
    models_df = st.session_state.models_df
    
    # Default priority
    priority = "balanced"
    
    # Main Content - Compact hero
    render_hero()
    
    # Tab-based navigation (4 tabs)
    tab1, tab2, tab3, tab4 = st.tabs(["Define Use Case", "Technical Specification", "Recommendations", "Deployment"])

    with tab1:
        render_use_case_input_tab(priority, models_df)

    with tab2:
        render_technical_specs_tab(priority, models_df)

    with tab3:
        render_results_tab(priority, models_df)

    with tab4:
        render_deployment_tab()



def render_use_case_input_tab(priority: str, models_df: pd.DataFrame):
    """Tab 1: Use case input interface."""

    def clear_dialog_states():
        """Clear all dialog and expanded states when starting a new use case."""
        st.session_state.show_full_table_dialog = False
        st.session_state.show_category_dialog = False
        st.session_state.show_winner_dialog = False
        st.session_state.show_options_list_expanded = False

    # Transfer pending input from button clicks before rendering the text_area widget
    if "pending_user_input" in st.session_state:
        st.session_state.user_input = st.session_state.pending_user_input
        del st.session_state.pending_user_input

    st.subheader("Describe your use case or select from 9 predefined scenarios")

    # Input area
    st.text_area(
        "Your requirements:",
        key="user_input",
        height=120,
        max_chars=2000,  # Corporate standard: limit input length
        placeholder="Describe your LLM use case in natural language...\n\nExample: I need a chatbot for customer support with 30 users. Low latency is important, and we have H100 GPUs available.",
        label_visibility="collapsed"
    )

    # Row 1: 5 task buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Chat Completion", use_container_width=True, key="task_chat"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.pending_user_input = "Customer service chatbot for 30 users."
            st.rerun()

    with col2:
        if st.button("Code Completion", use_container_width=True, key="task_code"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.pending_user_input = "IDE code completion tool for 300 developers."
            st.rerun()

    with col3:
        if st.button("Document Q&A", use_container_width=True, key="task_rag"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.pending_user_input = "Document Q&A system for enterprise knowledge base, 300 users."
            st.rerun()

    with col4:
        if st.button("Summarization", use_container_width=True, key="task_summ"):
            clear_dialog_states()
            # With priority (cost-effective) to show filtering
            st.session_state.pending_user_input = "News article summarization for 300 users, cost-effective solution preferred."
            st.rerun()

    with col5:
        if st.button("Legal Analysis", use_container_width=True, key="task_legal"):
            clear_dialog_states()
            # With priority (accuracy) to show filtering
            st.session_state.pending_user_input = "Legal document analysis for 300 lawyers, accuracy is critical."
            st.rerun()

    # Row 2: 4 more task buttons
    col6, col7, col8, col9 = st.columns(4)

    with col6:
        if st.button("Translation", use_container_width=True, key="task_trans"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.pending_user_input = "Multi-language translation service for 300 users."
            st.rerun()

    with col7:
        if st.button("Content Generation", use_container_width=True, key="task_content"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.pending_user_input = "Content generation tool for marketing team, 300 users."
            st.rerun()

    with col8:
        if st.button("Long Doc Summary", use_container_width=True, key="task_longdoc"):
            clear_dialog_states()
            # With priority (accuracy) to show filtering
            st.session_state.pending_user_input = "Long document summarization for research papers, 30 researchers, accuracy matters."
            st.rerun()

    with col9:
        if st.button("Code Generation", use_container_width=True, key="task_codegen"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.pending_user_input = "Full code generation tool for implementing features, 30 developers."
            st.rerun()
    
    # Show character count - white text
    char_count = len(st.session_state.user_input) if st.session_state.user_input else 0
    st.markdown(f'<div style="text-align: right; font-size: 0.75rem; margin-top: -0.5rem;">{char_count}/2000 characters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1, 2])
    with col1:
        # Disable button if input is too short
        analyze_disabled = len(st.session_state.user_input.strip()) < 10 if st.session_state.user_input else True
        analyze_clicked = st.button("Analyze & Recommend", type="primary", use_container_width=True, disabled=analyze_disabled)
        if analyze_disabled and st.session_state.user_input and len(st.session_state.user_input.strip()) < 10:
            st.caption("Please enter at least 10 characters")
    with col2:
        if st.button("Clear", use_container_width=True):
            # Complete session state reset
            for key in ['user_input', 'extraction_result', 'recommendation_result', 
                       'extraction_approved', 'slo_approved', 'edited_extraction',
                       'custom_ttft', 'custom_itl', 'custom_e2e', 'custom_qps', 'used_priority']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.user_input = ""
            st.rerun()
    
    # Input validation before analysis
    if analyze_clicked and st.session_state.user_input and len(st.session_state.user_input.strip()) >= 10:
        # Reset workflow state
        st.session_state.extraction_approved = None
        st.session_state.slo_approved = None
        st.session_state.recommendation_result = None
        st.session_state.edited_extraction = None
        
        # Show progress bar for better UX
        progress_container = st.empty()
        with progress_container:
            progress_bar = st.progress(0, text="Initializing extraction...")
            
        try:
            progress_bar.progress(20, text="Analyzing input text...")
            extraction = extract_business_context(st.session_state.user_input)
            progress_bar.progress(80, text="Extraction complete!")
            
            if extraction:
                # Clear old recommendation data when new extraction is done
                st.session_state.recommendation_result = None
                st.session_state.extraction_approved = None
                st.session_state.slo_approved = None
                st.session_state.edited_extraction = None
                st.session_state.ranked_response = None

                # Reset priority weights from extraction (LLM-detected priorities)
                # Note: complexity_priority and weight_complexity removed from UI
                for key in ['accuracy_priority', 'cost_priority', 'latency_priority',
                           'weight_accuracy', 'weight_cost', 'weight_latency']:
                    if key in st.session_state:
                        del st.session_state[key]

                # Store new extraction
                st.session_state.extraction_result = extraction

                # Initialize priorities and weights immediately from extraction
                # This ensures weights are set before the recommendation API is called
                priority_config = fetch_priority_weights()
                pw_map = priority_config.get("priority_weights", {}) if priority_config else {}
                defaults_cfg = priority_config.get("defaults", {}) if priority_config else {}
                default_weights = defaults_cfg.get("weights", {"accuracy": 5, "cost": 4, "latency": 2})

                # Set priorities from extraction (LLM-detected) or default to medium
                # Note: complexity_priority removed from UI (backend still extracts it)
                st.session_state.accuracy_priority = extraction.get('accuracy_priority', 'medium')
                st.session_state.cost_priority = extraction.get('cost_priority', 'medium')
                st.session_state.latency_priority = extraction.get('latency_priority', 'medium')

                # Set weights based on priorities using the JSON config mapping
                # Note: weight_complexity removed from UI (hardcoded to 0 in API call)
                st.session_state.weight_accuracy = pw_map.get("accuracy", {}).get(st.session_state.accuracy_priority, default_weights["accuracy"])
                st.session_state.weight_cost = pw_map.get("cost", {}).get(st.session_state.cost_priority, default_weights["cost"])
                st.session_state.weight_latency = pw_map.get("latency", {}).get(st.session_state.latency_priority, default_weights["latency"])

                logger.info(f"Initialized priorities from extraction: accuracy={st.session_state.accuracy_priority}, "
                           f"cost={st.session_state.cost_priority}, latency={st.session_state.latency_priority}")
                logger.info(f"Initialized weights: accuracy={st.session_state.weight_accuracy}, "
                           f"cost={st.session_state.weight_cost}, latency={st.session_state.weight_latency}")
                
                # Set flag to signal PRIORITIES UI to update from this new extraction
                st.session_state.new_extraction_available = True
                
                st.session_state.used_priority = extraction.get("priority", priority)
                st.session_state.detected_use_case = extraction.get("use_case", "chatbot_conversational")
                progress_bar.progress(100, text="Ready!")
            else:
                st.error("Could not extract business context. Please try rephrasing your input.")
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"An error occurred during analysis. Please try again.")
            progress_bar.empty()
        finally:
            # Clean up progress bar after brief delay
            import time
            time.sleep(0.5)
            progress_container.empty()
    
    # Get the priority that was actually used
    used_priority = st.session_state.get("used_priority", priority)
    
    # Show extraction with approval if extraction exists but not approved
    if st.session_state.extraction_result and st.session_state.extraction_approved is None:
        render_extraction_with_approval(st.session_state.extraction_result, models_df)
        return
    
    # If editing, show edit form
    if st.session_state.extraction_approved == False:
        render_extraction_edit_form(st.session_state.extraction_result, models_df)
        return
    
    # If approved, show message to proceed to Technical Specifications tab
    if st.session_state.extraction_approved == True:
        render_extraction_result(st.session_state.extraction_result, used_priority)
        
        # Completion banner (tab auto-advances on approval)
        st.markdown("""
        <div style="padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; margin-bottom: 0.75rem; max-width: 50%;">
            <strong>Step 1 Complete</strong> · You can now go to Technical Specification
        </div>
        """, unsafe_allow_html=True)


def render_technical_specs_tab(priority: str, models_df: pd.DataFrame):
    """Tab 2: Technical Specification (SLO targets and workload settings)."""
    used_priority = st.session_state.get("used_priority", priority)
    
    # Check if extraction is approved first
    if not st.session_state.extraction_approved:
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 8px; text-align: center; ">
            <strong style="font-size: 1.1rem;">Complete Step 1 First</strong><br>
            <span style="font-size: 0.95rem; ">Go to the <strong>Define Use Case</strong> tab to describe your use case and approve the extraction.</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    final_extraction = st.session_state.edited_extraction or st.session_state.extraction_result or {}
    
    # Show SLO section
    render_slo_with_approval(final_extraction, used_priority, models_df)
    
    # If SLO approved, show navigation message
    if st.session_state.slo_approved == True:
        # Completion banner (tab auto-advances on approval)
        st.markdown("""
        <div style="padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; margin-bottom: 0.75rem; max-width: 50%;">
            <strong>Step 2 Complete</strong> · You can now view Recommendations
        </div>
        """, unsafe_allow_html=True)


def render_results_tab(priority: str, models_df: pd.DataFrame):
    """Tab 3: Results display - Best Model Recommendations."""
    used_priority = st.session_state.get("used_priority", priority)
    
    # Check if SLO is approved
    if not st.session_state.slo_approved:
        if not st.session_state.extraction_approved:
            st.markdown("""
            <div style="padding: 1.5rem; border-radius: 8px; text-align: center; ">
                <strong style="font-size: 1.1rem;">Complete Previous Steps First</strong><br>
                <span style="font-size: 0.95rem; ">1. Go to <strong>Define Use Case</strong> tab to describe your use case<br>
                2. Then go to <strong>Technical Specification</strong> tab to set your SLO targets</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 1.5rem; border-radius: 8px; text-align: center; ">
                <strong style="font-size: 1.1rem;">Complete Step 2 First</strong><br>
                <span style="font-size: 0.95rem; ">Go to the <strong>Technical Specification</strong> tab to set your SLO targets and workload parameters.</span>
            </div>
            """, unsafe_allow_html=True)
        return
    
    final_extraction = st.session_state.edited_extraction or st.session_state.extraction_result or {}
    
    # ALWAYS regenerate recommendations to ensure fresh SLO filtering
    # Clear any cached results
    st.session_state.recommendation_result = None
    
    # Generate recommendations
    # FORCE clear cache to ensure fresh data every time
    st.session_state.recommendation_result = None
    st.session_state.pop('ranked_response', None)
    
    if True:  # Always regenerate
        # Get all specification values from session state (set in Tech Specs tab)
        # These are the EXACT values the user sees on the Technical Specification tab
        use_case = final_extraction.get("use_case", "chatbot_conversational")
        user_count = final_extraction.get("user_count", 1000)

        # Get SLO targets from session state (set by number_input widgets)
        ttft_target = st.session_state.get("custom_ttft") or st.session_state.get("input_ttft") or 500
        itl_target = st.session_state.get("custom_itl") or st.session_state.get("input_itl") or 50
        e2e_target = st.session_state.get("custom_e2e") or st.session_state.get("input_e2e") or 10000

        # Get QPS from session state - this is the value shown in the Expected RPS input
        qps_target = st.session_state.get("spec_expected_qps") or st.session_state.get("custom_qps") or 1

        # Get token config from session state (set by render_slo_cards)
        prompt_tokens = st.session_state.get("spec_prompt_tokens", 512)
        output_tokens = st.session_state.get("spec_output_tokens", 256)

        # Get percentile from session state (default to p95)
        # Note: Currently we use a single percentile for all metrics for the backend query
        # The UI allows per-metric percentiles but the backend uses one
        percentile = st.session_state.get("slo_percentile", "p95")

        # Get weights from session state (set in PRIORITIES section of Tech Spec tab)
        weights = {
            "accuracy": st.session_state.get("weight_accuracy", 5),
            "price": st.session_state.get("weight_cost", 4),
            "latency": st.session_state.get("weight_latency", 2),
            "complexity": 0,  # Complexity scoring disabled in UI
        }

        # Get GPU preference from extraction (if specified)
        preferred_gpu_types = final_extraction.get("preferred_gpu_types", [])

        with st.spinner(f"Scoring {len(models_df)} models with MCDM..."):
            recommendation = fetch_ranked_recommendations(
                use_case=use_case,
                user_count=user_count,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                expected_qps=float(qps_target),
                ttft_target_ms=int(ttft_target),
                itl_target_ms=int(itl_target),
                e2e_target_ms=int(e2e_target),
                weights=weights,
                include_near_miss=False,
                percentile=percentile,
                preferred_gpu_types=preferred_gpu_types,
            )

        if recommendation is None:
            st.error("Unable to get recommendations. Please ensure backend is running.")
        else:
            st.session_state.recommendation_result = recommendation
        
    if st.session_state.recommendation_result:
            render_recommendation_result(st.session_state.recommendation_result, used_priority, final_extraction)


def render_extraction_result(extraction: dict, priority: str):
    """Render beautiful extraction results."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")

    # Check for non-default priorities (only show if user explicitly mentioned them)
    accuracy_priority = extraction.get("accuracy_priority", "medium")
    cost_priority = extraction.get("cost_priority", "medium")
    latency_priority = extraction.get("latency_priority", "medium")

    # Build priority badges list (only for non-medium values)
    priority_badges = []
    if accuracy_priority != "medium":
        priority_badges.append(f"Accuracy: {accuracy_priority.title()}")
    if cost_priority != "medium":
        priority_badges.append(f"Cost: {cost_priority.title()}")
    if latency_priority != "medium":
        priority_badges.append(f"Latency: {latency_priority.title()}")

    # Format priorities as comma-separated display value
    priorities_display = ", ".join(priority_badges) if priority_badges else None

    # Build priorities item HTML only if there are non-default priorities
    priorities_item = "<!-- no custom priorities -->"
    if priorities_display:
        priorities_item = f'''<div class="extraction-item"><div><div class="extraction-label">Priorities</div><div class="extraction-value">{priorities_display}</div></div></div>'''

    st.markdown(f"""
    <div class="extraction-card">
        <div class="extraction-grid">
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Use Case</div>
                    <div class="extraction-value">{use_case.replace("_", " ").title() if use_case else "Unknown"}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Expected Users</div>
                    <div class="extraction-value">{user_count:,}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Hardware</div>
                    <div class="extraction-value">{hardware if hardware else "Any GPU"}</div>
                </div>
            </div>
            {priorities_item}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_extraction_with_approval(extraction: dict, models_df: pd.DataFrame):
    """Render extraction results with YES/NO approval buttons."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")

    # Check for non-default priorities (only show if user explicitly mentioned them)
    accuracy_priority = extraction.get("accuracy_priority", "medium")
    cost_priority = extraction.get("cost_priority", "medium")
    latency_priority = extraction.get("latency_priority", "medium")

    # Build priority badges list (only for non-medium values)
    priority_badges = []
    if accuracy_priority != "medium":
        priority_badges.append(f"Accuracy: {accuracy_priority.title()}")
    if cost_priority != "medium":
        priority_badges.append(f"Cost: {cost_priority.title()}")
    if latency_priority != "medium":
        priority_badges.append(f"Latency: {latency_priority.title()}")

    # Format priorities as comma-separated display value
    priorities_display = ", ".join(priority_badges) if priority_badges else None

    # Build priorities item HTML only if there are non-default priorities
    priorities_item = "<!-- no custom priorities -->"
    if priorities_display:
        priorities_item = f'''<div class="extraction-item"><div><div class="extraction-label">Priorities</div><div class="extraction-value">{priorities_display}</div></div></div>'''

    st.markdown(f"""
    <div class="extraction-card" >
        <div class="extraction-grid">
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Use Case</div>
                    <div class="extraction-value">{use_case.replace("_", " ").title() if use_case else "Unknown"}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Expected Users</div>
                    <div class="extraction-value">{user_count:,}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Hardware</div>
                    <div class="extraction-value">{hardware if hardware else "Any GPU"}</div>
                </div>
            </div>
            {priorities_item}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Approval question
    st.markdown("""
    <div style="padding: 1.25rem; border-radius: 1rem; margin: 1.5rem 0; text-align: center;
                ">
        <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">
            Is this extraction correct?
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            Verify the extracted business context before proceeding to recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Yes, Continue", type="primary", use_container_width=True, key="approve_extraction"):
            st.session_state.extraction_approved = True
            st.session_state.switch_to_tab2 = True
            st.rerun()
    with col2:
        if st.button("No, Edit", use_container_width=True, key="edit_extraction"):
            st.session_state.extraction_approved = False
            st.rerun()
    with col3:
        if st.button("Start Over", use_container_width=True, key="restart"):
            st.session_state.extraction_result = None
            st.session_state.extraction_approved = None
            st.session_state.recommendation_result = None
            st.session_state.user_input = ""
            # Reset priority weights (complexity removed from UI)
            for key in ['accuracy_priority', 'cost_priority', 'latency_priority',
                       'weight_accuracy', 'weight_cost', 'weight_latency']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def render_extraction_edit_form(extraction: dict, models_df: pd.DataFrame):
    """Render editable form for extraction correction."""
    st.subheader("Edit Business Context")
    st.info("Review and adjust the extracted values below, then click \"Apply Changes\" to continue.")
    
    # Use case dropdown
    use_cases = [
        "chatbot_conversational", "code_completion", "code_generation_detailed",
        "document_analysis_rag", "summarization_short", "long_document_summarization",
        "translation", "content_generation", "research_legal_analysis"
    ]
    use_case_labels = {
        "chatbot_conversational": "Chatbot / Conversational AI",
        "code_completion": "Code Completion (IDE autocomplete)",
        "code_generation_detailed": "Code Generation (full implementations)",
        "document_analysis_rag": "Document RAG / Q&A",
        "summarization_short": "Short Summarization (<10 pages)",
        "long_document_summarization": "Long Document Summarization (10+ pages)",
        "translation": "Translation",
        "content_generation": "Content Generation",
        "research_legal_analysis": "Research / Legal Analysis"
    }
    
    current_use_case = extraction.get("use_case", "chatbot_conversational")
    current_idx = use_cases.index(current_use_case) if current_use_case in use_cases else 0
    
    col1, col2 = st.columns(2)
    with col1:
        new_use_case = st.selectbox(
            "Use Case",
            use_cases,
            index=current_idx,
            format_func=lambda x: use_case_labels.get(x, x),
            key="edit_use_case"
        )
        
        new_user_count = st.number_input(
            "User Count",
            min_value=1,
            max_value=1000000,
            value=extraction.get("user_count", 1000),
            step=100,
            key="edit_user_count"
        )
    
    with col2:
        priorities = ["balanced", "low_latency", "cost_saving", "high_accuracy", "high_throughput"]
        priority_labels = {
            "balanced": "Balanced",
            "low_latency": "Low Latency",
            "cost_saving": "Cost Saving",
            "high_accuracy": "High Accuracy",
            "high_throughput": "High Throughput"
        }
        current_priority = extraction.get("priority", "balanced")
        priority_idx = priorities.index(current_priority) if current_priority in priorities else 0
        
        new_priority = st.selectbox(
            "Priority",
            priorities,
            index=priority_idx,
            format_func=lambda x: priority_labels.get(x, x),
            key="edit_priority"
        )
        
        hardware_options = ["Any GPU", "H100", "A100", "A10G", "L4", "T4"]
        current_hardware = extraction.get("hardware") or "Any GPU"
        hw_idx = hardware_options.index(current_hardware) if current_hardware in hardware_options else 0
        
        new_hardware = st.selectbox(
            "Hardware",
            hardware_options,
            index=hw_idx,
            key="edit_hardware"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Changes", type="primary", use_container_width=True, key="apply_edit"):
            st.session_state.edited_extraction = {
                "use_case": new_use_case,
                "user_count": new_user_count,
                "priority": new_priority,
                "hardware": new_hardware if new_hardware != "Any GPU" else None
            }
            st.session_state.used_priority = new_priority
            st.session_state.extraction_approved = True
            st.rerun()
    with col2:
        if st.button("🔙 Cancel", use_container_width=True, key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()


def render_slo_with_approval(extraction: dict, priority: str, models_df: pd.DataFrame):
    """Render SLO section with approval to proceed to recommendations."""
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)

    # SLO and Impact Cards - all 4 cards in one row
    render_slo_cards(use_case, user_count, priority)
    
    # ==========================================================================
    # VALIDATE SLO VALUES - Use case-specific ranges (same as sliders)
    # ==========================================================================
    slo_defaults = fetch_slo_defaults(use_case)
    if not slo_defaults:
        st.error(f"Failed to fetch SLO defaults for use case: {use_case}")
        return

    ttft_min = int(slo_defaults["ttft_ms"]["min"])
    ttft_max = int(slo_defaults["ttft_ms"]["max"])
    itl_min = int(slo_defaults["itl_ms"]["min"])
    itl_max = int(slo_defaults["itl_ms"]["max"])
    e2e_min = int(slo_defaults["e2e_ms"]["min"])
    e2e_max = int(slo_defaults["e2e_ms"]["max"])
    
    # Get current SLO values from session state (these are from the sliders)
    current_ttft = int(st.session_state.get('edit_ttft', ttft_max))
    current_itl = int(st.session_state.get('edit_itl', itl_max))
    current_e2e = int(st.session_state.get('edit_e2e', e2e_max))
    
    # Validate - values from sliders should always be valid since sliders enforce bounds
    # But check anyway in case of edge cases
    validation_errors = []
    if current_ttft < ttft_min or current_ttft > ttft_max:
        validation_errors.append(f"TTFT must be between {ttft_min:,}-{ttft_max:,}ms")
    if current_itl < itl_min or current_itl > itl_max:
        validation_errors.append(f"ITL must be between {itl_min}-{itl_max}ms")
    if current_e2e < e2e_min or current_e2e > e2e_max:
        validation_errors.append(f"E2E must be between {e2e_min:,}-{e2e_max:,}ms")
    
    is_valid = len(validation_errors) == 0
    
    # Proceed button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not is_valid:
            # Show validation errors
            st.markdown(f"""
            <div style="border: 1px solid rgba(239, 68, 68, 0.5); 
                        border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem; text-align: center;">
                <div style="color: #ef4444; font-weight: 600; font-size: 0.9rem;">⚠️ Invalid SLO Values</div>
                <div style="font-size: 0.8rem; margin-top: 0.25rem;">
                    {' • '.join(validation_errors)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Button disabled if invalid
        if st.button("Generate Recommendations", type="primary", use_container_width=True, key="generate_recs", disabled=not is_valid):
            st.session_state.slo_approved = True
            st.session_state.switch_to_tab3 = True
            st.rerun()


def render_recommendation_result(result: dict, priority: str, extraction: dict):
    """Render beautiful recommendation results with Top 5 table."""
    
    # === Use the ALREADY FETCHED data from result ===
    # The result was fetched with correct SLO filters from fetch_ranked_recommendations
    # DO NOT call fetch_ranked_recommendations again - it would use wrong default values!
    
    # Gather data from extraction (handle None case)
    if extraction is None:
        extraction = {}
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)

    # Use the result directly - it already has correct SLO-filtered data
    ranked_response = result
    
    if ranked_response:
        # NOTE: Table removed from main view - now only shown in "List Options" expandable section
        # render_ranked_recommendations(ranked_response)

        # Store ranked response for winner details and for the List Options section
        st.session_state.ranked_response = ranked_response
        
        # Get the Balanced winner for the Explore button
        balanced_recs = ranked_response.get("balanced", [])
        if balanced_recs:
            winner = balanced_recs[0]
            recommendations = balanced_recs
        else:
            # Fallback to any available recommendations
            for cat in ["best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
                if ranked_response.get(cat):
                    winner = ranked_response[cat][0]
                    recommendations = ranked_response[cat]
                    break
            else:
                st.warning("No recommendations found.")
                return
    else:
        st.warning("Could not fetch ranked recommendations from backend. Ensure the backend is running.")
        st.session_state.ranked_response = None
        recommendations = result.get("recommendations", [])
        if not recommendations:
            st.warning("No recommendations found. Try adjusting your requirements.")
            return
        winner = recommendations[0]
    
    # Store winner for explore dialog
    st.session_state.winner_recommendation = winner
    st.session_state.winner_priority = priority
    st.session_state.winner_extraction = extraction
    
    # Render the 4 "Best" cards with Explore button
    st.markdown("---")
    
    # Get all recommendations for the cards
    all_recs = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        cat_recs = st.session_state.ranked_response.get(cat, []) if st.session_state.ranked_response else []
        all_recs.extend(cat_recs)
    
    # Remove duplicates by model+hardware
    seen = set()
    unique_recs = []
    for rec in all_recs:
        model = rec.get("model_name", "")
        gpu_cfg = rec.get("gpu_config", {}) or {}
        hw = f"{gpu_cfg.get('gpu_type', 'H100')}x{gpu_cfg.get('gpu_count', 1)}"
        key = f"{model}_{hw}"
        if key not in seen:
            seen.add(key)
            unique_recs.append(rec)
    
    if unique_recs:
        render_top5_table(unique_recs, priority)
    
    # === LIST OPTIONS SECTION (inline expandable) ===
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        # Toggle button text based on expanded state
        is_expanded = st.session_state.get('show_options_list_expanded', False)
        button_text = "Hide Option List" if is_expanded else "List Options"
        if st.button(button_text, key="list_options_btn", use_container_width=True):
            # Toggle the expanded state
            st.session_state.show_options_list_expanded = not is_expanded
            st.rerun()

    # Render inline expanded content if expanded
    if st.session_state.get('show_options_list_expanded', False):
        render_options_list_inline()

    # === MODIFY SLOs & RE-RUN SECTION ===
    st.markdown("---")
    st.markdown("""
    <div style="padding: 1rem; border-radius: 0.75rem; margin-top: 1rem; ">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-weight: 700; font-size: 1rem;">Want Different Results?</span>
        </div>
        <p style="font-size: 0.85rem; margin: 0;">
            Adjust SLO targets above to find models with different latency/performance trade-offs. 
            Stricter SLOs = fewer models, Relaxed SLOs = more options.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Show button to start new case
    col_new, col_spacer = st.columns([1, 3])

    with col_new:
        if st.button("New Case", key="new_case_btn", type="secondary", use_container_width=True):
            # Complete session state reset - start fresh from Define Use Case tab
            keys_to_clear = [
                'user_input', 'extraction_result', 'recommendation_result',
                'extraction_approved', 'slo_approved', 'edited_extraction',
                'custom_ttft', 'custom_itl', 'custom_e2e', 'custom_qps', 'used_priority',
                'ranked_response', 'show_category_dialog', 'show_full_table_dialog',
                'show_winner_dialog', 'explore_category', 'detected_use_case',
                'top5_balanced', 'top5_accuracy', 'top5_latency', 'top5_cost', 'top5_simplest',
                'show_options_list_expanded'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Set flag to switch to Tab 1 after rerun
            st.session_state.switch_to_tab1 = True
            st.rerun()


def _render_winner_details(winner: dict, priority: str, extraction: dict):
    """Render detailed winner information inside the expander."""
    
    # Handle both backend format (scores) and UI format (score_breakdown)
    backend_scores = winner.get("scores", {}) or {}
    ui_breakdown = winner.get("score_breakdown", {}) or {}
    breakdown = {
        "quality_score": backend_scores.get("accuracy_score", ui_breakdown.get("quality_score", 0)),
        "latency_score": backend_scores.get("latency_score", ui_breakdown.get("latency_score", 0)),
        "cost_score": backend_scores.get("price_score", ui_breakdown.get("cost_score", 0)),
        "capacity_score": backend_scores.get("complexity_score", ui_breakdown.get("capacity_score", 0)),
    }
    
    # === 📋 FINAL RECOMMENDATION BOX (Schema-Aligned Clean Format) ===
    st.subheader("Final Recommendation")
    
    # Extract data for clean display - handle both backend and UI formats
    model_name = winner.get("model_name", "Unknown Model")
    
    # Get hardware config - backend returns gpu_config object
    gpu_config = winner.get("gpu_config", {}) or {}
    hardware = gpu_config.get("gpu_type", winner.get("hardware", "H100"))
    hw_count = gpu_config.get("gpu_count", winner.get("hardware_count", 1))
    tp = gpu_config.get("tensor_parallel", 1)
    replicas = gpu_config.get("replicas", 1)
    
    # Get final score - backend uses balanced_score in scores
    backend_scores = winner.get("scores", {}) or {}
    final_score = backend_scores.get("balanced_score", winner.get("final_score", 0))
    quality_score = breakdown.get("quality_score", 0)
    
    # Get SLO data - backend returns predicted_* fields directly on winner
    # Try backend format first (predicted_ttft_p95_ms), then benchmark_slo format
    ttft_p95 = winner.get("predicted_ttft_p95_ms", 0)
    itl_p95 = winner.get("predicted_itl_p95_ms", 0)
    e2e_p95 = winner.get("predicted_e2e_p95_ms", 0)
    throughput_qps = winner.get("predicted_throughput_qps", 0)
    
    # Fallback to benchmark_slo if backend fields empty
    if not ttft_p95:
        benchmark_slo = winner.get("benchmark_slo", {})
        slo_actual = benchmark_slo.get("slo_actual", {}) if benchmark_slo else {}
        throughput_data = benchmark_slo.get("throughput", {}) if benchmark_slo else {}
        ttft_p95 = slo_actual.get("ttft_p95_ms", slo_actual.get("ttft_mean_ms", 0))
        itl_p95 = slo_actual.get("itl_p95_ms", slo_actual.get("itl_mean_ms", 0))
        e2e_p95 = slo_actual.get("e2e_p95_ms", slo_actual.get("e2e_mean_ms", 0))
        throughput_qps = throughput_data.get("tokens_per_sec", 0) / 100 if throughput_data.get("tokens_per_sec") else 0
    
    # Format for display
    ttft_display = f"{int(ttft_p95)}" if ttft_p95 and ttft_p95 > 0 else "—"
    itl_display = f"{int(itl_p95)}" if itl_p95 and itl_p95 > 0 else "—"
    e2e_display = f"{int(e2e_p95)}" if e2e_p95 and e2e_p95 > 0 else "—"
    max_rps = f"{throughput_qps:.1f}" if throughput_qps and throughput_qps > 0 else "—"
    
    # Schema-aligned recommendation box - Build HTML without comments
    # All models now have benchmark data (filtered to valid models only)
    benchmark_status = "<strong style=''>Verified</strong> - Real benchmark data"
    priority_text = priority.replace('_', ' ').title()
    
    # Build hardware display text
    hw_display = f"{hw_count}x {hardware}"
    if tp > 1 and replicas > 1:
        hw_display += f" (TP={tp}, R={replicas})"
    
    rec_html = f'''<div style="padding: 2rem; border-radius: 1.25rem; margin-bottom: 1.5rem; ">
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 1rem; ">
        <span style="font-size: 2.5rem;"></span>
        <div>
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">RECOMMENDATION</h2>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem;">Based on {priority_text} optimization</p>
                    </div>
                    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
        <div style="display: flex; flex-direction: column; gap: 1.25rem;">
            <div style="padding: 1rem; border-radius: 0.75rem; ">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Model</p>
                <p style="margin: 0; font-size: 1.25rem; font-weight: 700;">{model_name}</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem;">Quality Score: <span style="font-weight: 700;">{quality_score:.0f}%</span></p>
                    </div>
            <div style="padding: 1rem; border-radius: 0.75rem; ">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Hardware Configuration</p>
                <p style="margin: 0; font-size: 1.25rem; font-weight: 700;">{hw_display}</p>
                    </div>
                </div>
        <div style="padding: 1.25rem; border-radius: 0.75rem; ">
            <p style="margin: 0 0 1rem 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">Expected SLO (p95)</p>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; ">
                    <span style="font-size: 0.85rem;">Max RPS</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{max_rps}</span>
            </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; ">
                    <span style="font-size: 0.85rem;">TTFT (p95)</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{ttft_display}<span style="font-size: 0.7rem; "> ms</span></span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; ">
                    <span style="font-size: 0.85rem;">ITL (p95)</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{itl_display}<span style="font-size: 0.7rem; "> ms</span></span>
                    </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0;">
                    <span style="font-size: 0.85rem;">E2E (p95)</span>
                    <span style="font-weight: 800; font-size: 1.1rem;">{e2e_display}<span style="font-size: 0.7rem; "> ms</span></span>
                    </div>
                    </div>
                    </div>
                </div>
    <div style="margin-top: 1.5rem; padding-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size: 0.85rem;"><strong >Verified</strong> - Real benchmark data</div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 0.9rem;">Final Score:</span>
            <span style="padding: 0.5rem 1.25rem; border-radius: 0.5rem; font-weight: 900; font-size: 1.5rem; ">{final_score:.1f}</span>
            </div>
        </div>
</div>'''
    
    st.markdown(rec_html, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Winner Details: Score Breakdown")
    
    # Add explanation for the score notation - golden styled
    st.markdown("""
    <div style="padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; ">
        <p style="margin: 0; font-size: 0.9rem;">
            <strong >Score Format:</strong> <code style="padding: 0.1rem 0.4rem; border-radius: 0.25rem;">87 → +17.4</code> means the model scored <strong>87/100</strong> in this category, contributing <strong >+17.4 points</strong> to the final weighted score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f'<h3 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; ">{winner.get("model_name", "Unknown")}</h3>', unsafe_allow_html=True)
        
        # Get actual weights from session state (0-10 scale, set by user in Tech Spec tab)
        w_accuracy = st.session_state.get("weight_accuracy", 5)
        w_latency = st.session_state.get("weight_latency", 2)
        w_cost = st.session_state.get("weight_cost", 4)
        w_total = w_accuracy + w_latency + w_cost
        # Normalize to fractions (complexity weight is 0 in UI)
        weights = {
            "accuracy": w_accuracy / w_total if w_total > 0 else 0.33,
            "latency": w_latency / w_total if w_total > 0 else 0.33,
            "cost": w_cost / w_total if w_total > 0 else 0.33,
        }

        # Calculate contributions
        q_score = breakdown.get("quality_score", 0)
        l_score = breakdown.get("latency_score", 0)
        c_score = breakdown.get("cost_score", 0)
        cap_score = breakdown.get("capacity_score", 0)

        q_contrib = q_score * weights["accuracy"]
        l_contrib = l_score * weights["latency"]
        c_contrib = c_score * weights["cost"]
        cap_contrib = 0  # Complexity weight is 0 in UI
        
        render_score_bar("Accuracy", "", q_score, "score-bar-accuracy", q_contrib)
        render_score_bar("Latency", "", l_score, "score-bar-latency", l_contrib)
        render_score_bar("Cost", "", c_score, "score-bar-cost", c_contrib)
        render_score_bar("Capacity", "", cap_score, "score-bar-capacity", cap_contrib)
    
    with col2:
        st.subheader("Why This Model?")

        model_name = winner.get('model_name', 'Unknown')
        use_case = extraction.get('use_case', 'chatbot_conversational')
        use_case_display = format_use_case_name(use_case)

        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.75rem; margin-bottom: 1rem; ">
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.6;">
                <strong>{model_name}</strong> ranked highest for <strong>{use_case_display}</strong>
                with a balanced score of <strong>{breakdown.get('quality_score', 0):.0f}</strong> accuracy,
                <strong>{breakdown.get('latency_score', 0):.0f}</strong> latency,
                and <strong>{breakdown.get('cost_score', 0):.0f}</strong> cost efficiency.
            </p>
        </div>
        """, unsafe_allow_html=True)

        pros = winner.get("pros", ["Top Quality", "Fast Responses"])
        cons = winner.get("cons", [])
        
        st.markdown('<p style="font-weight: 600; margin-bottom: 0.5rem;">Strengths:</p>', unsafe_allow_html=True)
        pros_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1rem;">'
        for pro in pros:
            pros_html += f'<span class="tag tag-pro">{pro}</span>'
        pros_html += '</div>'
        st.markdown(pros_html, unsafe_allow_html=True)
        
        if cons:
            st.markdown('<p style="font-weight: 600; margin-bottom: 0.5rem;">Trade-offs:</p>', unsafe_allow_html=True)
            cons_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem;">'
            for con in cons:
                cons_html += f'<span class="tag tag-con">{con}</span>'
            cons_html += '</div>'
            st.markdown(cons_html, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(212, 175, 55, 0.3); margin: 1rem 0;">', unsafe_allow_html=True)
        st.markdown(f"""
        <p ><strong >Final Score:</strong> <code style="padding: 0.35rem 0.75rem; border-radius: 0.25rem; font-weight: 700; font-size: 1.1rem;">{winner.get('final_score', 0):.1f}/100</code></p>
        <p style="color: rgba(212, 175, 55, 0.8); font-style: italic;">Based on {priority.replace('_', ' ').title()} priority weighting</p>
        """, unsafe_allow_html=True)
    
    # Display benchmark SLO data - use backend fields or benchmark_slo
    # Get predicted values from backend OR from benchmark_slo
    benchmark_slo = winner.get("benchmark_slo", {}) or {}
    gpu_config = winner.get("gpu_config", {}) or {}
    
    # Get SLO values - prioritize backend's predicted_* fields, fallback to benchmark_slo
    ttft_p95_val = winner.get("predicted_ttft_p95_ms") or benchmark_slo.get("slo_actual", {}).get("ttft_p95_ms", 0)
    itl_p95_val = winner.get("predicted_itl_p95_ms") or benchmark_slo.get("slo_actual", {}).get("itl_p95_ms", 0)
    e2e_p95_val = winner.get("predicted_e2e_p95_ms") or benchmark_slo.get("slo_actual", {}).get("e2e_p95_ms", 0)
    throughput_qps_val = winner.get("predicted_throughput_qps") or (benchmark_slo.get("throughput", {}).get("tokens_per_sec", 0) / 100 if benchmark_slo.get("throughput", {}).get("tokens_per_sec") else 0)
    
    # Get traffic profile from winner or result
    traffic_profile = winner.get("traffic_profile", {}) or {}
    prompt_tokens_val = traffic_profile.get("prompt_tokens", benchmark_slo.get("token_config", {}).get("prompt", 512))
    output_tokens_val = traffic_profile.get("output_tokens", benchmark_slo.get("token_config", {}).get("output", 256))
    
    # Get hardware info
    hw_type_val = gpu_config.get("gpu_type", benchmark_slo.get("hardware", "H100"))
    hw_count_val = gpu_config.get("gpu_count", benchmark_slo.get("hardware_count", 1))
    tp_val = gpu_config.get("tensor_parallel", 1)
    replicas_val = gpu_config.get("replicas", 1)
    
    # Show benchmark box if we have any SLO data
    if ttft_p95_val or itl_p95_val or e2e_p95_val:
        st.markdown("---")
        st.subheader("Real Benchmark SLOs (Actual Achievable Performance)")
        
        st.markdown(f"""
        <div style="padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; ">
            <p style="margin: 0; font-size: 0.9rem;">
                <strong >Benchmarks:</strong> Real measured values from vLLM simulation.
                Hardware: <strong >{hw_count_val}x {hw_type_val}</strong> | 
                Token Config: <strong >{prompt_tokens_val}→{output_tokens_val}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use the values we already extracted
        slo_actual = benchmark_slo.get("slo_actual", {})
        throughput = benchmark_slo.get("throughput", {})
        token_config = benchmark_slo.get("token_config", {})
        hardware = hw_type_val
        hw_count = hw_count_val
        
        col1, col2, col3 = st.columns(3)
        
        # Use our extracted values with fallback to slo_actual
        ttft_p95_show = ttft_p95_val or slo_actual.get('ttft_p95_ms', 0)
        itl_p95_show = itl_p95_val or slo_actual.get('itl_p95_ms', 0)
        e2e_p95_show = e2e_p95_val or slo_actual.get('e2e_p95_ms', 0)
        tps_show = throughput_qps_val * 100 if throughput_qps_val else throughput.get('tokens_per_sec', 0)
        
        with col1:
            st.markdown(f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">⏱️</span>
                    <span style="font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">TTFT</span>
                </div>
                <div style="text-align: center;">
                    <p style="font-weight: 800; font-size: 2rem; margin: 0;">{int(ttft_p95_show) if ttft_p95_show else 'N/A'}<span style="font-size: 1rem; ">ms</span></p>
                    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">p95 latency</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1rem; font-weight: 700;">TTFT</span>
                    <span style="font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">ITL</span>
                </div>
                <div style="text-align: center;">
                    <p style="font-weight: 800; font-size: 2rem; margin: 0;">{int(itl_p95_show) if itl_p95_show else 'N/A'}<span style="font-size: 1rem; ">ms</span></p>
                    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">inter-token latency</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="padding: 1.25rem; border-radius: 0.75rem; ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">🏁</span>
                    <span style="font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">E2E</span>
                </div>
                <div style="text-align: center;">
                    <p style="font-weight: 800; font-size: 2rem; margin: 0;">{int(e2e_p95_show) if e2e_p95_show else 'N/A'}<span style="font-size: 1rem; ">ms</span></p>
                    <p style="font-size: 0.75rem; margin: 0.25rem 0 0 0;">end-to-end</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Throughput and Config row
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="padding: 1rem; border-radius: 0.75rem; text-align: center; ">
                <span style="font-size: 1.25rem;">🚀</span>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Throughput</p>
                <p style="font-weight: 800; font-size: 1.5rem; margin: 0;">{int(tps_show) if tps_show else 'N/A'} <span style="font-size: 0.8rem;">tok/s</span></p>
            </div>
            <div style="padding: 1rem; border-radius: 0.75rem; text-align: center; ">
                <span style="font-size: 1rem; font-weight: 700;">HW</span>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Hardware</p>
                <p style="font-weight: 800; font-size: 1.25rem; margin: 0;">{hw_count_val}x {hw_type_val}</p>
            </div>
            <div style="padding: 1rem; border-radius: 0.75rem; text-align: center; ">
                <span style="font-size: 1rem; font-weight: 700;">Tokens</span>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Token Config</p>
                <p style="font-weight: 700; font-size: 1rem; margin: 0;">{prompt_tokens_val} → {output_tokens_val}</p>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 0.75rem; border-radius: 0.5rem; ">
            <p style="margin: 0; font-size: 0.8rem; text-align: center;">
                <strong >Data Source:</strong> vLLM Simulation Benchmarks | 
                <strong >Model:</strong> {winner.get('model_name', 'Unknown')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    # All recommendations now come from valid models with both AA + benchmark data
    # No need to show "No benchmark data" warning


if __name__ == "__main__":
    main()
