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

## Phase 5: Simplify Dialogs and Repeated Patterns

**What to change:**

- **Replace manual dialog state management** (lines 4424-4440) with Streamlit's `@st.dialog` decorator. Remove `show_winner_dialog`, `show_category_dialog`, `show_full_table_dialog` session state variables and the mutual-exclusion logic.

- **~~Remove JavaScript tab switching~~** — Done in Phase 4.

- **Deduplicate SLO input fields** — the TTFT/ITL/E2E input pattern (lines 3566-3612) is repeated 3 times with only labels and keys differing. Extract a helper function.

- **Consolidate `get_scores()` / `get_model_scores()`** — two near-identical helpers in different functions. Unify into one module-level function.

- **Simplify session state initialization** — replace 67 individual `if "key" not in st.session_state:` blocks with a single defaults dict + loop.

- **Remove `format_display_name()` and `normalize_model_name()`** complexity if the backend already returns normalized names. If not, keep but simplify.

**Estimated removal**: ~200 lines
**Risk**: Medium — dialog behavior changes slightly
**Test**: Full workflow. Verify dialogs open/close correctly. Verify SLO inputs work. Verify tab navigation works (manually clicking).

---

## Phase 6: Break `render_slo_cards()` and `render_top5_table()`

**What to change:**

These two functions are the largest remaining (577 and 461 lines respectively after prior phases). Break them into focused sub-functions:

- **`render_slo_cards()`** → split into:
  - `_render_slo_inputs(use_case, slo_defaults)` — the 3 SLO number inputs
  - `_render_workload_profile(use_case, user_count)` — workload info display
  - `_render_weight_controls()` — already exists, just call it

- **`render_top5_table()`** → split into:
  - `_render_category_card(category, recs, ...)` — single category display (was repeated 5x as carousel, now simplified in Phase 4)
  - `_render_recommendation_row(rec)` — single recommendation display

**Estimated removal**: 0 (reorganization)
**Risk**: Low — pure refactor, no behavior change
**Test**: Full workflow, verify all recommendation displays and SLO inputs work identically.

---

## Phase 7: Split into Multi-File Module

**What to change:**

Convert the single `app.py` into a focused module structure:

```
ui/
  app.py              # Entry point: page config, session state init, main(), tab routing (~100-150 lines)
  api_client.py       # All fetch_* functions, extract_business_context() (~150 lines)
  helpers.py          # normalize_model_name(), format_use_case_name(), format_gpu_config(), get_scores() (~80 lines)
  state.py            # Session state defaults dict + init function (~40 lines)
  components/
    __init__.py
    extraction.py     # Extraction result display, approval workflow, edit form
    slo.py            # SLO inputs, workload profile, weight controls
    recommendations.py # Recommendation tables, category displays
    deployment.py     # YAML generation, deployment tab
    dialogs.py        # Winner details, category exploration, full table dialogs
  static/
    neuralnav-logo.png  # Already exists
```

**Estimated removal**: 0 (pure reorganization)
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
| 7 | Split into multi-file module | 0 (reorg) | Medium |

**Total estimated reduction**: ~2,380 lines removed from original 5,774, with the remainder reorganized into ~8 focused files.

## Verification (after each phase)

1. Start backend: `make start-backend` (or `uv run uvicorn neuralnav.api.app:create_app --factory`)
2. Start UI: `streamlit run ui/app.py`
3. Test full workflow:
   - Tab 1: Enter a use case description, verify extraction works
   - Tab 2: Verify SLO defaults load, inputs work, approve specs
   - Tab 3: Verify recommendations display with scores
   - Tab 4: Select a config, generate YAML, verify download
4. Test error handling (Phase 2+): Stop backend, verify UI shows clear errors
