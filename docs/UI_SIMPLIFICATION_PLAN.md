# UI Simplification Plan

## Context

The UI (`ui/app.py`) is a 5,774-line monolithic Streamlit file that has accumulated significant technical debt:

1. **Backend logic in the UI** — business logic (mock extraction, benchmark filtering, accuracy lookups, score calculations) that duplicates or should live in the backend
2. **Silent fallback code** — `try/except` blocks that hide backend failures behind hardcoded defaults and mock data, masking bugs
3. **Excessive styling** — ~1,300 lines of custom CSS with animations, gradients, backdrop filters, and custom fonts overriding Streamlit's native look
4. **No modular structure** — everything in one file with 67 session state variables, repeated patterns, and functions exceeding 500 lines

The goal is to strip this down to a clean, native-Streamlit UI that trusts the backend, surfaces errors honestly, and is easy to maintain. Work is broken into 7 independently testable and committable steps.

**File under modification**: `ui/app.py` (all phases until Phase 7)

---

## Phase 1: Remove Dead Code and the About Tab -- DONE (5,774 → 5,430 lines)

**What was removed:**

- `get_benchmark_ranges_for_token_config()` — read from non-existent `benchmarks_redhat_performance.json`. All 3 call sites updated to use `slo_defaults` from backend `/api/v1/slo-defaults/{use_case}`.
- `priority_factors` dict and factor calculations — adjusted defaults based on the broken benchmark data.
- Redundant `workload_profile` fetch and dead `ttft_default`/`itl_default`/`e2e_default`/`ttft`/`itl`/`e2e` variables at top of `render_slo_cards()` — all values now read directly from `slo_defaults` dict where needed.
- `get_metric_range()` helper — rewritten to pull min/max from `slo_defaults` (3 lines vs 16).
- About tab: `render_about_section()`, `render_catalog_content()`, `render_how_it_works_content()` + tab declaration.
- Unused CSS: 5 keyframe animations (`pulse-glow`, `gradient-shift`, `shimmer`, `scale-in`, `border-glow`), `.hero-badges`/`.hero-badge`, `.extraction-icon-*`.
- SLO range labels changed from "Range" to "Recommended Range".

---

## Phase 2: Remove Fallback Logic and `mock_extraction()` -- DONE (5,430 → 5,217 lines)

**What was removed:**

- `mock_extraction()` (~191 lines) — keyword-based intent extraction duplicating the backend's LLM extraction. `extract_business_context()` now returns `None` on failure; caller shows `st.error()`.
- `fetch_priority_weights()` hardcoded fallback — was returning a hardcoded weight dict on API failure. Now returns `None`; callers already handle `None` via `if priority_config else {}`.
- QPS fallback heuristic `estimated_qps = max(1, user_count // 50)` — `render_slo_cards()` now shows `st.error()` and returns early when `fetch_expected_rps()` fails.
- Local re-sorting fallback (5 blocks) — UI was re-sorting recommendations locally when backend ranked lists were empty. Removed; if backend returns empty lists the UI shows the empty state.
- SLO defaults fallback was already addressed in Phase 1 (`st.error()` and early return).

---

## Phase 3: Remove Duplicated Backend Logic -- DONE (5,217 → 5,039 lines)

**What was removed:**

- `get_raw_aa_accuracy()`, `load_weighted_scores()`, and `BENCHMARK_TO_AA_NAME_MAP` (~115 lines) — complex model name mapping + CSV matching for accuracy scores. All callers now use `scores.accuracy_score` from the backend.
- `calc_throughput()` and `USE_CASE_OUTPUT_TOKENS` dict (~18 lines) — throughput calculation duplicating backend's `benchmark_metrics.tps_*`. Throughput display now uses backend TPS directly, showing "N/A" if unavailable.
- Hardcoded `priority_weights` dict in winner details dialog — replaced with actual user weights from `st.session_state.weight_accuracy/cost/latency`.
- `use_case_context` and `model_traits` marketing copy dicts (~30 lines) — replaced with a simple score-based summary using the model name and backend scores.
- `get_model_scores()` in category dialog simplified to use backend scores only (removed `ui_breakdown` fallback and `raw_aa_accuracy` lookup).

---

## Phase 4: Strip Custom CSS — Switch to Native Streamlit -- DONE (5,039 → 3,224 lines)

**What was changed:**

- Deleted the 1,259-line main CSS block (Google Fonts, custom properties, animations, all component styles). Replaced with a 1-line `.stDeployButton { display: none; }` override.
- Deleted 6 inline CSS blocks scattered in render functions (filter controls, SLO inputs, dialogs, edit form, winner details).
- `render_hero()` → `st.title()` + `st.caption()`.
- `render_score_bar()` → `st.progress()` + `st.write()` using native columns.
- Carousel component (204 lines) → simple category cards using `st.container(border=True)`, `st.metric()`, and `st.caption()`. Removed prev/next buttons and index management.
- Replaced 11 `section-header` divs and inline HTML headings with `st.subheader()`.
- Removed 3 JavaScript tab-switching blocks (fragile `<script>` hacks).
- `unsafe_allow_html=True` usage reduced from 95 to ~64. Remaining uses are in table/dialog rendering functions (will be addressed in later phases).
- `slo-card` HTML wrappers → `st.write("**Title**")`.

---

## Phase 5: Simplify Dialogs and Repeated Patterns -- DONE (3,218 → 3,117 lines)

**What was changed:**

- **Session state initialization**: Replaced 27 individual `if "key" not in st.session_state:` blocks with a single `SESSION_DEFAULTS` dict + loop (~45 lines replaced by ~20).
- **Consolidated `get_scores()`**: Extracted identical `get_scores()` / `get_model_scores()` into one module-level function. Removed both local duplicates.
- **Simplified dialog mutual-exclusion**: Removed redundant reset of other dialog flags — Streamlit's `@st.dialog` handles exclusivity natively.
- **Deduplicated SLO inputs**: Extracted `render_slo_input()` helper to replace 3 near-identical TTFT/ITL/E2E blocks (~45 lines → 3 calls + 15-line helper).
- **Removed dead tab-switching writes**: Removed 3 `switch_to_tab` session state assignments (JS tab-switching blocks were removed in Phase 4, making these dead writes).
- **Moved per-metric percentile init** (`ttft_percentile`, `itl_percentile`, `e2e_percentile`) into `SESSION_DEFAULTS`.
- **Name functions**: Kept `format_display_name()` / `normalize_model_name()` — both actively used (6+ call sites), backend does not normalize names.

---

## Phase 6: Break `render_slo_cards()` and `render_top5_table()` -- DONE (3,117 → 3,035 lines)

**What was changed:**

- **`render_top5_table()`** broken into 3 focused functions:
  - `_render_filter_summary()` — SLO filter stats (configs passed, unique models)
  - `_render_category_card()` — single category card with prev/next navigation and Select button (moved from nested to module level)
  - `render_top5_table()` — now a slim orchestrator (~30 lines)

- **`render_slo_cards()`** broken into 4 focused functions + data moved to module level:
  - `TASK_DATASETS` dict — moved from inside function to module level
  - `_render_slo_targets(slo_defaults)` — 3 SLO inputs with percentile selectors + change detection
  - `_render_workload_profile(use_case, workload_profile, estimated_qps, qps, user_count)` — QPS input, RPS warnings, token counts, workload insights
  - `_render_accuracy_benchmarks(use_case)` — benchmark weight display per use case
  - `_render_priorities()` — priority dropdowns and weight inputs (accuracy, cost, latency), deduplicated 3 near-identical rows into a loop
  - `render_slo_cards()` — now a slim orchestrator (~25 lines)

- **Dead code removed:**
  - `render_best_card()` (36 lines) — defined but never called
  - Unused `best_*` variables (5 lines) — set but never read
  - Duplicate `fetch_workload_profile()` call in SLO Targets column — result was unused
  - Unused `priority` parameter cascading through `render_slo_cards()`, `render_slo_with_approval()`, `render_technical_specs_tab()`
  - Unused `models_df` parameter in `render_slo_with_approval()` and `render_technical_specs_tab()`

- **Feature restored:** Prev/next navigation (`< #1 of 5 >`) added to category cards, allowing users to browse and select from all top-5 recommendations per category.

---

## Phase 7: Split into Multi-File Module -- DONE (3,035 → 2,775 lines across 10 files)

**Context:** `ui/app.py` was 3,035 lines with 52 functions clearly grouped by domain. All prior phases cleaned and organized the code. Split into focused modules with no behavior change.

### Target Structure

```
ui/
  app.py              # Entry point + tab routing (447 lines)
  api_client.py       # All backend API calls (276 lines)
  helpers.py          # Pure utility functions (149 lines)
  state.py            # Session state defaults + init (65 lines)
  components/
    __init__.py        # Empty package marker
    extraction.py      # Extraction display/approval/edit (245 lines)
    slo.py             # SLO targets, workload, benchmarks, priorities (464 lines)
    recommendations.py # Category cards, top-5 table, options list (447 lines)
    deployment.py      # Deployment tab (167 lines)
    dialogs.py         # Winner details, category/full-table dialogs (515 lines)
  static/
    neuralnav-logo.png # (unchanged)
```

### Import Path Strategy

`streamlit run ui/app.py` runs from repo root, so `ui/` is not on `sys.path` by default. At the top of `app.py` (before local imports):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```
Then all modules use flat imports: `from api_client import ...`, `from components.slo import ...`.

### Dead Code to Remove (not moved — deleted)

- `render_weight_controls()` (lines 465-539) — only called from dead `render_ranked_recommendations()`
- `render_ranked_recommendations()` (lines 542-732) — call site commented out
- `DATA_DIR` (line 30) — declared, never used
- `build_row()` and first `format_gpu_config()` — local functions inside dead code

~260 lines removed.

### File Assignments

**`ui/helpers.py`** — pure functions, no Streamlit dependency:
- `PROVIDER_MAPPING` dict
- `normalize_model_name()`
- `format_display_name()`
- `format_use_case_name()`
- `get_scores()`
- `format_gpu_config()` — NEW module-level function (currently defined identically as local function in 3 places; consolidate into one)

**`ui/state.py`** — session state initialization:
- `SESSION_DEFAULTS` dict
- `init_session_state()` function wrapping the init loop

**`ui/api_client.py`** — all HTTP communication with backend:
- `API_BASE_URL`, `logger`
- `load_206_models()` (@st.cache_data)
- `fetch_slo_defaults()` (@st.cache_data)
- `fetch_expected_rps()` (@st.cache_data)
- `fetch_workload_profile()` (@st.cache_data)
- `fetch_priority_weights()` (@st.cache_data)
- `fetch_ranked_recommendations()`
- `extract_business_context()`
- `deploy_and_generate_yaml(recommendation)` — NEW, extracted from inline requests in `_render_category_card()`

**`ui/components/slo.py`**:
- `TASK_DATASETS` dict
- `get_workload_insights()`
- `_render_slo_targets()`, `_render_workload_profile()`, `_render_accuracy_benchmarks()`, `_render_priorities()`
- `render_slo_cards()`, `render_slo_with_approval()`
- Imports from: `api_client` (fetch_*), no cross-component imports

**`ui/components/recommendations.py`**:
- `_render_filter_summary()`, `_render_category_card()`, `render_top5_table()`
- `render_options_list_inline()`
- `render_recommendation_result()`
- Imports from: `api_client` (deploy_and_generate_yaml), `helpers` (format_display_name, get_scores, format_gpu_config)

**`ui/components/extraction.py`**:
- `render_extraction_result()`, `render_extraction_with_approval()`, `render_extraction_edit_form()`
- Imports from: `api_client` (fetch_priority_weights), `helpers` (format_use_case_name)

**`ui/components/deployment.py`**:
- `render_deployment_tab()`
- Imports from: `api_client` (API_BASE_URL) — keeps inline requests for now (deployment-specific flow)

**`ui/components/dialogs.py`**:
- `render_score_bar()`, `_render_winner_details()`
- `show_winner_details_dialog()`, `show_category_dialog()`, `show_full_table_dialog()` (all @st.dialog)
- Imports from: `helpers` (format_display_name, format_use_case_name, get_scores, format_gpu_config)

**`ui/app.py`** — slim entry point:
- `st.set_page_config()` (must be first Streamlit call)
- CSS override (3 lines)
- `init_session_state()` call
- `render_hero()` (8 lines, kept inline)
- `main()` — dialog dispatch + tab routing
- `render_use_case_input_tab()` (~220 lines, orchestrates Tab 1 flow)
- `render_technical_specs_tab()` (25 lines)
- `render_results_tab()` (~90 lines)
- `if __name__ == "__main__": main()`

### Dependency Graph (no cycles)

```
app.py → state, api_client, helpers, components/*
components/extraction.py → api_client, helpers
components/slo.py → api_client
components/recommendations.py → api_client, helpers
components/deployment.py → api_client
components/dialogs.py → helpers
state.py → (none)
api_client.py → (none)
helpers.py → (none)
```

### Implementation Sequence

1. Create `ui/helpers.py` — no dependencies
2. Create `ui/state.py` — no dependencies
3. Create `ui/api_client.py` — no internal dependencies
4. Create `ui/components/__init__.py` — empty
5. Create `ui/components/dialogs.py` — depends on helpers
6. Create `ui/components/slo.py` — depends on api_client
7. Create `ui/components/extraction.py` — depends on api_client, helpers
8. Create `ui/components/recommendations.py` — depends on api_client, helpers
9. Create `ui/components/deployment.py` — depends on api_client
10. Rewrite `ui/app.py` — imports from all above, delete dead code

Verify syntax (`python -m py_compile`) after each file. Full workflow test after step 10.

### Verification

1. `python -m py_compile ui/app.py` — syntax check
2. `streamlit run ui/app.py` — full workflow test
3. All 4 tabs should function identically to before the split
4. Prev/next navigation on category cards should still work
5. All dialogs (winner details, category explore, full table) should open/close correctly

**Actual removal**: ~260 lines (dead code)
**Risk**: Medium — touches every function (import paths change)
**Test**: `streamlit run ui/app.py` — full workflow, all tabs, all interactions. Nothing should change visually or functionally.

---

## Summary

| Phase | Description | Lines Removed | Risk |
|-------|-------------|---------------|------|
| 1 | Remove dead code + About tab | ~300 | Very Low |
| 2 | Remove fallbacks + mock_extraction | ~230 | Low |
| 3 | Remove duplicated backend logic | ~150 | Low |
| 4 | Strip CSS, switch to native Streamlit | ~1,500 | Medium |
| 5 | Simplify dialogs + deduplicate patterns | ~200 | Medium |
| 6 | Break apart large functions | 0 (reorg) | Low |
| 7 | Split into multi-file module | ~260 (dead code) + reorg into 10 files | Medium |

**Total reduction**: ~3,000 lines removed from original 5,774 → 2,775 lines across 10 focused files.

## Verification (after each phase)

1. Start backend: `make start-backend` (or `uv run uvicorn neuralnav.api.app:create_app --factory`)
2. Start UI: `streamlit run ui/app.py`
3. Test full workflow:
   - Tab 1: Enter a use case description, verify extraction works
   - Tab 2: Verify SLO defaults load, inputs work, approve specs
   - Tab 3: Verify recommendations display with scores
   - Tab 4: Select a config, generate YAML, verify download
4. Test error handling (Phase 2+): Stop backend, verify UI shows clear errors
