# Refactor: Remove Obsolete Recommendation API Fields

## Goal

Remove dead code from the recommendation API. The UI has two functions that call the same endpoint, and both include obsolete fields that have no effect on recommendations.

## Why These Fields Are Obsolete

| Field | Status | Reason |
|-------|--------|--------|
| `latency_requirement` | **REMOVE** | Used only in dead code (`model_evaluator.score_model()` is never called) or bypassed when explicit SLOs are provided |
| `budget_constraint` | **REMOVE** | Used only in dead code (`model_evaluator.score_model()`) |
| `throughput_priority` | **REMOVE** | Not used; throughput is handled by explicit `expected_qps` |
| `weights` | **KEEP** | Drives MCDM scoring in `capacity_planner.py` |

The `weights` dict (accuracy, price, latency, complexity) controls all recommendation ranking. The other fields are remnants of an older design.

---

## Implementation Steps

### Step 1: Simplify `fetch_ranked_recommendations()` in `ui/app.py`

Remove the `priority_mapping` dict and the `priority` parameter. Remove these fields from the request payload:
- `latency_requirement`
- `budget_constraint`
- `throughput_priority`

### Step 2: Replace `get_enhanced_recommendation()` call site

Replace:
```python
recommendation = get_enhanced_recommendation(business_context)
```

With:
```python
weights = {
    "accuracy": st.session_state.get("weight_accuracy", 5),
    "price": st.session_state.get("weight_cost", 4),
    "latency": st.session_state.get("weight_latency", 2),
    "complexity": 0,
}

recommendation = fetch_ranked_recommendations(
    use_case=business_context.get("use_case", "chatbot_conversational"),
    user_count=business_context.get("user_count", 1000),
    prompt_tokens=business_context.get("prompt_tokens", 512),
    output_tokens=business_context.get("output_tokens", 256),
    expected_qps=business_context.get("expected_qps", 1),
    ttft_target_ms=business_context.get("ttft_p95_target_ms", 500),
    itl_target_ms=business_context.get("itl_p95_target_ms", 50),
    e2e_target_ms=business_context.get("e2e_p95_target_ms", 10000),
    weights=weights,
    include_near_miss=False,
    percentile=business_context.get("percentile", "p95"),
)

if recommendation is None:
    st.error("Unable to get recommendations. Please ensure backend is running.")
    return
```

### Step 3: Delete `get_enhanced_recommendation()`

### Step 4: Delete fallback code chain (~400 lines)

- `benchmark_recommendation()`
- `VALID_BENCHMARK_MODELS`
- `BENCHMARK_TO_QUALITY_MODEL_MAP`
- `HARDWARE_COSTS`
- `mock_recommendation_fallback()`
- `get_selection_reason()`

### Step 5: Check if helper functions are still needed

Delete if only used by fallback code:
- `load_weighted_scores()`
- `load_performance_benchmarks()`

### Step 6: Clean up backend API schema

Remove from `RankedRecommendationFromSpecRequest` in `backend/src/api/routes.py`:
```python
latency_requirement: str = "high"       # REMOVE
budget_constraint: str = "moderate"     # REMOVE
throughput_priority: str = "medium"     # REMOVE
```

### Step 7: Clean up dead backend code (optional)

- `backend/src/recommendation/model_evaluator.py` - `score_model()` is never called
- `backend/src/context_intent/traffic_profile.py` - `_adjust_slo_for_latency()` is bypassed

---

## Files to Modify

**UI:**
- `ui/app.py` - Remove fallback code, update API call

**Backend:**
- `backend/src/api/routes.py` - Remove obsolete fields from request schema
- `backend/src/recommendation/model_evaluator.py` - Clean up dead code (optional)
- `backend/src/context_intent/traffic_profile.py` - Simplify bypassed logic (optional)

## Estimated Lines Removed

- **UI**: ~450 lines
- **Backend**: ~20-50 lines (optional)

---

## Verification

1. Start backend: `make backend`
2. Start UI: `make ui`
3. Test recommendation flow - should work normally
4. Stop backend and test again - should show graceful error message
5. Verify no Python errors (missing imports/functions)
