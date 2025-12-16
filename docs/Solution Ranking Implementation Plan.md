# Solution Ranking POC Implementation Plan

## Overview

Implement Phase 1 of the Solution Ranking feature from [Compass Solution Ranking.md](Compass%20Solution%20Ranking.md). This adds multi-criteria scoring (0-100 scale) for all deployment configurations and presents users with 5 ranked views.

## Scope

**In Scope (Phase 1 POC):**
- 4-criteria scoring: Accuracy, Price, Latency, Complexity
- 5 ranked views: Best Accuracy, Lowest Cost, Lowest Latency, Simplest, Balanced
- Filters: minimum accuracy threshold, maximum cost ceiling
- Near-SLO configurations with warnings (within 20% of targets)
- Top 5 configurations per ranking view

**Out of Scope:**
- Benchmark-based accuracy scores (using model size tiers as proxy)
- User-adjustable weights (using defaults: 40% accuracy, 40% price, 10% latency, 10% complexity)
- Pareto frontier visualization
- Quantization variants

---

## Implementation Steps

### Step 1: Create SolutionScorer Class

**New file:** `backend/src/recommendation/solution_scorer.py`

```python
class SolutionScorer:
    """Score deployment configurations on 4 criteria (0-100 scale)."""

    def score_accuracy(self, model_size_str: str) -> int
    def score_price(self, cost: float, min_cost: float, max_cost: float) -> int
    def score_latency(self, predicted: dict, targets: dict) -> int
    def score_complexity(self, gpu_count: int, replicas: int) -> int
    def score_balanced(self, accuracy: int, price: int, latency: int, complexity: int) -> float
```

Scoring rules:
- **Accuracy**: Model size tier (7B=55, 8B=60, 70B=85, 405B=95)
- **Price**: `100 * (max_cost - cost) / (max_cost - min_cost)`
- **Latency**: 90-100 for SLO-compliant (headroom bonus), 70-89 for near-miss
- **Complexity**: 1 GPU=100, 2 GPUs=90, 4 GPUs=75, 8+ GPUs=60

---

### Step 2: Extend Schema with Scores

**Modify:** `backend/src/context_intent/schema.py`

Add:
```python
class ConfigurationScores(BaseModel):
    accuracy_score: int
    price_score: int
    latency_score: int
    complexity_score: int
    balanced_score: float
    slo_status: Literal["compliant", "near_miss", "exceeds"]

# Add to DeploymentRecommendation:
scores: ConfigurationScores | None = None
```

Add new response model:
```python
class RankedRecommendationsResponse(BaseModel):
    min_accuracy_threshold: int | None
    max_cost_ceiling: float | None
    include_near_miss: bool
    best_accuracy: list[DeploymentRecommendation]
    lowest_cost: list[DeploymentRecommendation]
    lowest_latency: list[DeploymentRecommendation]
    simplest: list[DeploymentRecommendation]
    balanced: list[DeploymentRecommendation]
    total_configs_evaluated: int
```

---

### Step 3: Modify CapacityPlanner

**Modify:** `backend/src/recommendation/capacity_planner.py`

Add new method:
```python
def plan_all_capacities(
    self,
    models: list[ModelInfo],
    traffic_profile: TrafficProfile,
    slo_targets: SLOTargets,
    intent: DeploymentIntent,
    include_near_miss: bool = True,
    near_miss_tolerance: float = 0.2,
) -> list[DeploymentRecommendation]:
    """Return ALL viable configurations with scores attached."""
```

Key changes:
1. Query PostgreSQL with relaxed SLO (1.2x targets) for near-miss support
2. Loop through all models (not just best)
3. Use SolutionScorer to calculate all 4 scores
4. Mark slo_status as "compliant" or "near_miss"
5. Return full list (not just best + 2 alternatives)

---

### Step 4: Create RankingService

**New file:** `backend/src/recommendation/ranking_service.py`

```python
class RankingService:
    def generate_ranked_lists(
        self,
        configurations: list[DeploymentRecommendation],
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        top_n: int = 5,
    ) -> dict[str, list[DeploymentRecommendation]]:
        """Generate 5 ranked lists from configurations."""
```

Logic:
1. Apply filters (min_accuracy, max_cost)
2. Sort by each criterion (descending score)
3. Return dict with keys: best_accuracy, lowest_cost, lowest_latency, simplest, balanced

---

### Step 5: Add Workflow Method

**Modify:** `backend/src/orchestration/workflow.py`

Add:
```python
def generate_ranked_recommendations(
    self,
    user_message: str,
    conversation_history: list | None = None,
    min_accuracy: int | None = None,
    max_cost: float | None = None,
    include_near_miss: bool = True,
) -> RankedRecommendationsResponse:
```

Flow:
1. Extract intent, generate traffic profile and SLO targets
2. Get top 3 model candidates
3. Call `capacity_planner.plan_all_capacities()` for all models
4. Call `ranking_service.generate_ranked_lists()` with filters
5. Return RankedRecommendationsResponse

---

### Step 6: Add API Endpoint

**Modify:** `backend/src/api/routes.py`

Add:
```python
@app.post("/api/ranked-recommend")
async def get_ranked_recommendations(request: RankedRecommendationRequest):
    """Return 5 ranked lists of recommendations."""
```

Keep existing `/api/recommend` unchanged for backward compatibility.

---

### Step 7: Update UI

**Modify:** `ui/app.py`

Add to Overview tab:
1. Ranking view selector (dropdown or tabs)
2. Filter controls (min accuracy slider, max cost input, near-miss checkbox)
3. Ranked configurations table with scores and status icons
4. Select button for each configuration

---

## Files to Modify

| File | Changes |
|------|---------|
| `backend/src/recommendation/solution_scorer.py` | **NEW** - Scoring logic |
| `backend/src/recommendation/ranking_service.py` | **NEW** - Ranking logic |
| `backend/src/recommendation/capacity_planner.py` | Add `plan_all_capacities()` |
| `backend/src/context_intent/schema.py` | Add `ConfigurationScores`, `RankedRecommendationsResponse` |
| `backend/src/orchestration/workflow.py` | Add `generate_ranked_recommendations()` |
| `backend/src/api/routes.py` | Add `/api/ranked-recommend` endpoint |
| `ui/app.py` | Add ranking selector, filters, scored table |

---

## Testing

1. **Unit tests** for SolutionScorer (known inputs/outputs)
2. **Unit tests** for RankingService (filtering, sorting)
3. **Integration test** with demo scenario from `data/demo_scenarios.json`
4. **Manual UI testing** of ranking views and filters

---

## Estimated Effort

| Step | Complexity |
|------|------------|
| 1. SolutionScorer | Small - isolated, testable |
| 2. Schema changes | Small - adding fields |
| 3. CapacityPlanner changes | Medium - core logic |
| 4. RankingService | Small - sorting/filtering |
| 5. Workflow integration | Medium - orchestration |
| 6. API endpoint | Small - thin layer |
| 7. UI changes | Medium - new components |

Total: ~1-2 days of focused work for a working POC.
