# PostgreSQL Migration Plan

## Overview

This document tracks the 6-phase migration to support PostgreSQL benchmarks with traffic-aware fuzzy matching.

**Goal**: Enable Compass to use real benchmark data from PostgreSQL (downstream/production) while maintaining JSON support (upstream/demos).

**Status**: Phase 1 complete, ready to begin Phase 2 (2025-10-29)

---

## Phase 1: Schema & Data Updates ✅ COMPLETED

**Objective**: Update benchmarks.json to SQL-aligned schema with diverse traffic profiles

**Changes Made**:
- ✅ Migrated schema to match PostgreSQL `exported_summaries` table
- ✅ Renamed fields:
  - `model_id` → `model_hf_repo`
  - `gpu_type` → `hardware`
  - `tensor_parallel` → `hardware_count`
  - `tpot_*` → `itl_*` (Inter-Token Latency)
  - `ttft_p50_ms` → `ttft_mean`
  - `throughput_tokens_per_sec` → `tokens_per_second`
  - `max_qps` → `requests_per_second`
- ✅ Added new fields:
  - `framework`, `framework_version`
  - `e2e_mean`, `e2e_p90`, `e2e_p99` (pre-calculated E2E latency)
  - Traffic characteristics: `mean_input_tokens`, `mean_output_tokens`, `prompt_tokens`, `prompt_tokens_stdev`, etc.
- ✅ Expanded dataset: 24 → 120 entries
- ✅ Added 5 traffic profiles per model+GPU config:
  - (150, 200) - Default chatbot
  - (512, 256) - Medium prompt/response
  - (1024, 512) - Long documents
  - (2048, 1024) - Summarization
  - (100, 50) - Short chat

**Files Changed**:
- `data/benchmarks.json` - Migrated to new schema
- `scripts/migrate_benchmarks_schema.py` - Migration script (kept for reference)

**Commit**: `9f378aa` - "Update benchmarks.json to SQL-aligned schema with traffic profiles"

---

## Phase 2: Rename TPOT to ITL Throughout Codebase

**Objective**: Update all code to use ITL (Inter-Token Latency) instead of TPOT

**Decision**: Use ITL terminology to match SQL database. Report TPOT in performance predictions if available for backward compatibility.

**Files to Update**:
1. `backend/src/context_intent/schema.py`
   - `SLOTargets.tpot_p90_target_ms` → `itl_p90_target_ms`

2. `backend/src/knowledge_base/benchmarks.py`
   - Update field mappings to read new schema
   - Map `itl_*` fields from benchmarks

3. `backend/src/knowledge_base/slo_templates.py`
   - All `tpot` references → `itl`

4. `backend/src/recommendation/capacity_planner.py`
   - All `tpot` variables → `itl`
   - Update benchmark field access

5. `ui/app.py`
   - Display labels: "TPOT" → "ITL" or "Inter-Token Latency"
   - Update all references in specs display and monitoring

6. `data/slo_templates.json`
   - All `tpot_*` → `itl_*`

7. Tests:
   - `tests/test_recommendation_workflow.py`
   - `tests/test_yaml_generation.py`
   - Any other test files referencing tpot

**Validation**:
- [ ] All tests pass
- [ ] UI displays ITL correctly
- [ ] Recommendations use new field names

---

## Phase 3: Use Pre-Calculated E2E Latency

**Objective**: Use `e2e_p90` from benchmarks instead of dynamic calculation

**Decision**: Use pre-calculated E2E values from both SQL and JSON sources.

**Changes Needed**:
1. `backend/src/recommendation/capacity_planner.py`
   - Remove or deprecate `_estimate_e2e_latency()` function
   - Use `benchmark.e2e_p90` directly
   - Update SLO compliance checks to use pre-calculated values

**Validation**:
- [ ] Capacity planning uses benchmark E2E values
- [ ] SLO compliance checks work correctly
- [ ] No breaking changes to recommendations

---

## Phase 4: Implement Fuzzy Matching with Distance Logging

**Objective**: Find closest benchmark match based on traffic characteristics

**Decision**:
- Always use closest match (no rejection threshold)
- Log distance metric for debugging

**Algorithm**:
```python
def find_closest_benchmark(
    model: str,
    hardware: str,
    hardware_count: int,
    target_input_tokens: int,
    target_output_tokens: int
) -> Benchmark:
    # 1. Filter by model + hardware + count
    candidates = filter_benchmarks(model, hardware, hardware_count)

    # 2. Find closest traffic match using distance metric
    best_match = min(candidates, key=lambda b: traffic_distance(
        (b.mean_input_tokens, b.mean_output_tokens),
        (target_input_tokens, target_output_tokens)
    ))

    # 3. Log distance for debugging
    distance = traffic_distance(...)
    logger.info(f"Using benchmark with traffic ({best_match.mean_input_tokens}, "
                f"{best_match.mean_output_tokens}) for requested "
                f"({target_input_tokens}, {target_output_tokens}), "
                f"distance={distance:.3f}")

    return best_match

def traffic_distance(benchmark_traffic, target_traffic):
    """Weighted Euclidean distance between traffic profiles"""
    input_diff = (benchmark_traffic[0] - target_traffic[0]) / target_traffic[0]
    output_diff = (benchmark_traffic[1] - target_traffic[1]) / target_traffic[1]
    return (input_diff ** 2 + output_diff ** 2) ** 0.5
```

**Files to Update**:
1. `backend/src/knowledge_base/benchmarks.py`
   - Add `find_closest_benchmark()` function
   - Add `traffic_distance()` helper

2. `backend/src/recommendation/capacity_planner.py`
   - Update benchmark lookup to use fuzzy matching
   - Pass traffic characteristics to lookup

**Validation**:
- [ ] Fuzzy matching finds reasonable matches
- [ ] Distance logging works
- [ ] Handles edge cases (no candidates, exact match, etc.)

---

## Phase 5: Update Architecture Documentation

**Objective**: Document parametric models as Phase 3+ enhancement

**Files to Update**:
1. `docs/ARCHITECTURE.md` - Add to "Possible Future Enhancements":
   ```markdown
   ### Knowledge Base - Parametric Performance Models (Phase 3+)

   Develop regression models for each (model, GPU, tensor_parallel) configuration
   that predict TTFT, ITL, and E2E latency for arbitrary input/output token lengths.

   **Approach**:
   - Train models on benchmark data: f(input_tokens, output_tokens) → (ttft, itl, e2e)
   - Model types: Linear regression, polynomial regression, or neural networks
   - Per-configuration models account for hardware-specific behaviors
   - Interpolation for in-range predictions, extrapolation with confidence bounds

   **Benefits**:
   - Precise predictions for any traffic profile
   - No need for exact benchmark matches
   - Confidence intervals for uncertainty quantification
   - Continuous improvement as more benchmarks are collected
   ```

---

## Phase 6: PostgreSQL Backend Support

**Objective**: Add PostgreSQL as optional data source for production benchmarks

**Architecture**:
```
┌─────────────────────────────────────────────────┐
│          BenchmarkRepository (Abstract)         │
│           Interface with SQL schema             │
└────────────┬────────────────────────────────────┘
             │
       ┌─────┴─────┐
       │           │
   ┌───▼───┐   ┌──▼─────┐
   │ JSON  │   │  SQL   │
   │Backend│   │Backend │
   └───────┘   └────────┘
   Default     Optional
   (upstream)  (downstream)
```

**Configuration**:
```python
# Environment variable controls data source
BENCHMARK_SOURCE = os.getenv("BENCHMARK_SOURCE", "json")  # "json" or "postgresql"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/integ")
```

### Phase 6a: Add Dependencies

**Files to Update**:
1. `requirements.txt`:
   ```
   # PostgreSQL support (optional - for production benchmarks)
   psycopg2-binary==2.9.9
   ```

### Phase 6b: Create Abstraction Layer

**Files to Create**:
1. `backend/src/knowledge_base/repository.py` - Abstract interface
2. `backend/src/knowledge_base/json_repository.py` - JSON implementation (current)
3. `backend/src/knowledge_base/sql_repository.py` - PostgreSQL implementation

**Example Interface**:
```python
from abc import ABC, abstractmethod

class BenchmarkRepository(ABC):
    """Abstract interface for benchmark data sources"""

    @abstractmethod
    def find_benchmarks(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int
    ) -> list[dict]:
        """Find all benchmarks matching model and hardware"""
        pass

    @abstractmethod
    def find_closest_benchmark(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        target_input_tokens: int,
        target_output_tokens: int
    ) -> dict:
        """Find closest benchmark by traffic characteristics"""
        pass

class JSONBenchmarkRepository(BenchmarkRepository):
    """JSON file-based benchmark repository"""
    def __init__(self, json_path: str = "data/benchmarks.json"):
        ...

class PostgreSQLBenchmarkRepository(BenchmarkRepository):
    """PostgreSQL-based benchmark repository"""
    def __init__(self, database_url: str):
        ...
```

**Factory Pattern**:
```python
# backend/src/knowledge_base/__init__.py
def get_benchmark_repository() -> BenchmarkRepository:
    source = os.getenv("BENCHMARK_SOURCE", "json")
    if source == "postgresql":
        db_url = os.getenv("DATABASE_URL")
        return PostgreSQLBenchmarkRepository(db_url)
    else:
        return JSONBenchmarkRepository()
```

### Phase 6c: Update Makefile

**File**: `Makefile`

Add PostgreSQL support:
```makefile
# PostgreSQL (optional - for production benchmarks)
.PHONY: postgres-start
postgres-start:
	@echo "Starting PostgreSQL for benchmark data..."
	docker run --name compass-postgres -d \
	  -e POSTGRES_PASSWORD=compass \
	  -e POSTGRES_DB=integ \
	  -p 5432:5432 \
	  postgres:16
	@echo "PostgreSQL started on port 5432"

.PHONY: postgres-stop
postgres-stop:
	@echo "Stopping PostgreSQL..."
	docker stop compass-postgres || true
	docker rm compass-postgres || true

.PHONY: postgres-load
postgres-load:
	@echo "Loading benchmark data into PostgreSQL..."
	@echo "Note: Place integ-oct-29.sql in data/ directory (downstream only)"
	pg_restore -h localhost -U postgres -d integ data/integ-oct-29.sql

.PHONY: postgres-shell
postgres-shell:
	docker exec -it compass-postgres psql -U postgres -d integ

# Add to help target
help:
	...
	@echo "  postgres-start    - Start PostgreSQL container for benchmarks"
	@echo "  postgres-stop     - Stop PostgreSQL container"
	@echo "  postgres-load     - Load benchmark SQL dump (downstream only)"
```

### Phase 6d: Update Documentation

**Files to Update**:

1. `docs/DEVELOPER_GUIDE.md` - Add PostgreSQL setup section:
   ```markdown
   ## Using PostgreSQL for Benchmarks (Optional)

   By default, Compass uses JSON files for benchmark data. For production
   deployments or when using real benchmark data, you can configure PostgreSQL.

   ### Prerequisites
   - Docker (for local PostgreSQL)
   - Real benchmark SQL dump (downstream only - not in public repo)

   ### Setup

   1. Start PostgreSQL:
      ```bash
      make postgres-start
      ```

   2. Load benchmark data (if you have the SQL dump):
      ```bash
      # Place integ-oct-29.sql in data/ directory
      make postgres-load
      ```

   3. Configure Compass to use PostgreSQL:
      ```bash
      export BENCHMARK_SOURCE=postgresql
      export DATABASE_URL=postgresql://postgres:compass@localhost:5432/integ
      ```

   4. Run Compass normally:
      ```bash
      make run
      ```

   ### Switching Back to JSON

   ```bash
   unset BENCHMARK_SOURCE
   # or
   export BENCHMARK_SOURCE=json
   ```
   ```

2. `README.md` - Add note about data sources:
   ```markdown
   ## Data Sources

   Compass supports two benchmark data sources:

   - **JSON** (default): Synthetic benchmark data included in the repository
   - **PostgreSQL** (optional): Real benchmark data for production use

   See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for PostgreSQL setup.
   ```

3. `.gitignore` - Ensure SQL files are ignored:
   ```
   # PostgreSQL dumps (downstream only)
   *.sql
   data/*.sql
   ```

### Phase 6e: Integration

**Files to Update**:
1. `backend/src/recommendation/capacity_planner.py`
   - Use `get_benchmark_repository()` instead of direct JSON access

2. `backend/src/knowledge_base/benchmarks.py`
   - Refactor to use repository pattern

3. All other files that access benchmark data

**Validation**:
- [ ] JSON backend works (default)
- [ ] PostgreSQL backend works with real data
- [ ] Environment variable switching works
- [ ] All tests pass with both backends
- [ ] Performance is acceptable

---

## Key Design Decisions

### 1. Field Naming: TPOT → ITL
**Decision**: Use ITL (Inter-Token Latency) everywhere to match SQL database.

**Rationale**:
- SQL database uses `itl_*` fields
- ITL is more technically accurate (measures time between tokens during decode)
- Standardizing on one term reduces confusion

### 2. E2E Latency Calculation
**Decision**: Use pre-calculated E2E values from benchmarks.

**Rationale**:
- E2E is already calculated in SQL database
- Faster than dynamic calculation
- More consistent across data sources
- E2E = TTFT + (output_tokens × ITL) is simple enough to pre-calculate

### 3. Fuzzy Matching Strategy
**Decision**: Always use closest match, log distance for debugging.

**Rationale**:
- Better UX than rejecting requests
- Distance logging enables debugging
- Users can see how close the match is
- Future: Parametric models will eliminate need for fuzzy matching

### 4. Data Source Architecture
**Decision**: Abstract repository pattern with JSON default, PostgreSQL optional.

**Rationale**:
- JSON works for upstream/open-source demos
- PostgreSQL required for production/downstream with real data
- Abstraction allows easy switching
- No breaking changes to existing code

---

## Testing Strategy

### Phase 2-4 Testing (JSON only)
- [ ] Unit tests for ITL rename
- [ ] Unit tests for fuzzy matching algorithm
- [ ] Integration tests with new schema
- [ ] UI testing for ITL display

### Phase 6 Testing (PostgreSQL)
- [ ] Unit tests for repository abstraction
- [ ] Integration tests with PostgreSQL
- [ ] Integration tests with JSON (ensure no regression)
- [ ] Environment variable switching tests
- [ ] Performance tests (JSON vs PostgreSQL)

---

## Rollout Plan

### For Upstream (Open Source)
1. Merge Phases 1-5 (JSON improvements)
2. Merge Phase 6 (PostgreSQL support as optional)
3. Document PostgreSQL as "optional for production"

### For Downstream (Production)
1. Pull upstream changes
2. Add `integ-oct-29.sql` to `data/` directory (gitignored)
3. Configure PostgreSQL via environment variables
4. Run `make postgres-start && make postgres-load`
5. Test with real data

---

## Migration Checklist

- [x] Phase 1: Schema migration
- [ ] Phase 2: TPOT → ITL rename
- [ ] Phase 3: Pre-calculated E2E
- [ ] Phase 4: Fuzzy matching
- [ ] Phase 5: Architecture docs
- [ ] Phase 6a: Add PostgreSQL dependencies
- [ ] Phase 6b: Abstraction layer
- [ ] Phase 6c: Makefile updates
- [ ] Phase 6d: Documentation
- [ ] Phase 6e: Integration
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Ready for downstream testing

---

## Context for Future Work

### Current State (After Phase 1)
- benchmarks.json has 120 entries with SQL-aligned schema
- Fields renamed: model_hf_repo, hardware, hardware_count, itl_*
- Traffic characteristics added: mean_input_tokens, mean_output_tokens
- E2E latency pre-calculated: e2e_mean, e2e_p90, e2e_p99

### Next Steps
Start with Phase 2: Rename TPOT to ITL in all code files (see file list above).

### Important Notes
- Real SQL data file (integ-oct-29.sql) is 760KB, ~1500-2500 entries
- Cannot be published upstream due to legal restrictions
- Must be kept downstream only (.gitignore)
- JSON data scaled to similar magnitudes for realistic demos

---

*Last Updated: 2025-10-29*
*Status: Phase 1 Complete*
