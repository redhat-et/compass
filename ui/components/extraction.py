"""Extraction display components for the NeuralNav UI.

Extraction result display, approval workflow, and edit form.
"""

import streamlit as st

from helpers import format_use_case_name


def render_extraction_result(extraction: dict, priority: str):
    """Render extraction results (read-only, after approval)."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")

    accuracy_priority = extraction.get("accuracy_priority", "medium")
    cost_priority = extraction.get("cost_priority", "medium")
    latency_priority = extraction.get("latency_priority", "medium")

    priority_badges = []
    if accuracy_priority != "medium":
        priority_badges.append(f"Accuracy: {accuracy_priority.title()}")
    if cost_priority != "medium":
        priority_badges.append(f"Cost: {cost_priority.title()}")
    if latency_priority != "medium":
        priority_badges.append(f"Latency: {latency_priority.title()}")

    priorities_display = ", ".join(priority_badges) if priority_badges else None

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


def render_extraction_with_approval(extraction: dict, models_df):
    """Render extraction results with YES/NO approval buttons."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")

    accuracy_priority = extraction.get("accuracy_priority", "medium")
    cost_priority = extraction.get("cost_priority", "medium")
    latency_priority = extraction.get("latency_priority", "medium")

    priority_badges = []
    if accuracy_priority != "medium":
        priority_badges.append(f"Accuracy: {accuracy_priority.title()}")
    if cost_priority != "medium":
        priority_badges.append(f"Cost: {cost_priority.title()}")
    if latency_priority != "medium":
        priority_badges.append(f"Latency: {latency_priority.title()}")

    priorities_display = ", ".join(priority_badges) if priority_badges else None

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
            for key in [
                "accuracy_priority", "cost_priority", "latency_priority",
                "weight_accuracy", "weight_cost", "weight_latency",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def render_extraction_edit_form(extraction: dict, models_df):
    """Render editable form for extraction correction."""
    st.subheader("Edit Business Context")
    st.info('Review and adjust the extracted values below, then click "Apply Changes" to continue.')

    use_cases = [
        "chatbot_conversational", "code_completion", "code_generation_detailed",
        "document_analysis_rag", "summarization_short", "long_document_summarization",
        "translation", "content_generation", "research_legal_analysis",
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
        "research_legal_analysis": "Research / Legal Analysis",
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
            key="edit_use_case",
        )

        new_user_count = st.number_input(
            "User Count",
            min_value=1,
            max_value=1000000,
            value=extraction.get("user_count", 1000),
            step=100,
            key="edit_user_count",
        )

    with col2:
        priorities = ["balanced", "low_latency", "cost_saving", "high_accuracy", "high_throughput"]
        priority_labels = {
            "balanced": "Balanced",
            "low_latency": "Low Latency",
            "cost_saving": "Cost Saving",
            "high_accuracy": "High Accuracy",
            "high_throughput": "High Throughput",
        }
        current_priority = extraction.get("priority", "balanced")
        priority_idx = priorities.index(current_priority) if current_priority in priorities else 0

        new_priority = st.selectbox(
            "Priority",
            priorities,
            index=priority_idx,
            format_func=lambda x: priority_labels.get(x, x),
            key="edit_priority",
        )

        hardware_options = ["Any GPU", "H100", "A100", "A10G", "L4", "T4"]
        current_hardware = extraction.get("hardware") or "Any GPU"
        hw_idx = hardware_options.index(current_hardware) if current_hardware in hardware_options else 0

        new_hardware = st.selectbox(
            "Hardware",
            hardware_options,
            index=hw_idx,
            key="edit_hardware",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Changes", type="primary", use_container_width=True, key="apply_edit"):
            st.session_state.edited_extraction = {
                "use_case": new_use_case,
                "user_count": new_user_count,
                "priority": new_priority,
                "hardware": new_hardware if new_hardware != "Any GPU" else None,
            }
            st.session_state.used_priority = new_priority
            st.session_state.extraction_approved = True
            st.rerun()
    with col2:
        if st.button("🔙 Cancel", use_container_width=True, key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()
