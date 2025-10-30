# PostgreSQL Migration Plan

## Overview

This document tracks the migration to support PostgreSQL benchmarks with traffic profile-based exact matching.

**Goal**: Enable Compass to use real benchmark data from PostgreSQL for both upstream and downstream, using exact matching on GuideLLM traffic profiles (prompt_tokens, output_tokens).

**Status**: Phase 1 complete, PostgreSQL analysis complete, ready to begin implementation (2025-10-30)

**Key Discovery**: The real database contains benchmarks organized around 4 GuideLLM traffic profiles (512→256, 1024→1024, 4096→512, 10240→1536), eliminating the need for fuzzy matching. We can use exact matching on `prompt_tokens` and `output_tokens` fields.

---

## Phase 1: Infrastructure Setup ✅ COMPLETED

**Objective**: Set up PostgreSQL infrastructure and analyze real benchmark data

**Changes Made**:
- ✅ Added PostgreSQL dependency to `requirements.txt` (psycopg2-binary==2.9.9)
- ✅ Created database schema in `scripts/schema.sql` matching `exported_summaries` table
- ✅ Added Makefile targets for PostgreSQL management:
  - `postgres-start` - Start PostgreSQL container
  - `postgres-stop` - Stop PostgreSQL container
  - `postgres-load-real` - Load real benchmark dump (integ-oct-29.sql)
  - `postgres-load-synthetic` - Load synthetic data from JSON
  - `postgres-query-traffic` - Query unique traffic patterns
  - `postgres-query-models` - Query available models
  - `postgres-shell` - Open PostgreSQL shell
- ✅ Loaded real benchmark data (2,662 entries)
- ✅ Analyzed traffic patterns - discovered 4 GuideLLM profiles:
  - (512, 256) - 1,865 benchmarks - Interactive workloads
  - (1024, 1024) - 153 benchmarks - Balanced workloads
  - (4096, 512) - 611 benchmarks - Prefill-dominated workloads
  - (10240, 1536) - 33 benchmarks - Edge/enterprise workloads
- ✅ Updated `.gitignore` to exclude SQL files (NDA-restricted)

**Files Changed**:
- `requirements.txt` - Added psycopg2-binary
- `scripts/schema.sql` - Database schema
- `Makefile` - PostgreSQL management targets
- `.gitignore` - Exclude *.sql files

**Commits**:
- `c917c35` - "Add PostgreSQL migration plan and ignore SQL files"

---

## Phase 2: Update SLO Templates and Use Case Definitions

**Objective**: Update slo_templates.json to align with traffic_and_slos.md framework

**Decision**: Map 9 use cases to 4 GuideLLM traffic profiles with experience-driven SLO targets (p95).

**Reference**: See `docs/traffic_and_slos.md` for detailed framework.

**Traffic Profile Mapping**:
- **(512, 256)** → Chatbot/Q&A, Code Completion
- **(1024, 1024)** → Detailed Code Generation, Translation, Content Generation
- **(4096, 512)** → Summarization, Document Analysis/RAG
- **(10240, 1536)** → Long Document Summarization, Research/Legal Analysis

**Files to Update**:
1. `data/slo_templates.json`
   - Replace 7 use cases with 9 from traffic_and_slos.md
   - Add `traffic_profile` field: `{"prompt_tokens": int, "output_tokens": int}`
   - Add `experience_class` field: "instant", "conversational", "interactive", "deferred", "batch"
   - Update SLO targets to use p95 (not p90):
     - `ttft_p95_ms`, `itl_p95_ms`, `e2e_p95_ms`
   - Include rationale/notes for each use case

**Example Structure**:
```json
{
  "chatbot_conversational": {
    "use_case": "Chatbot / Q&A",
    "description": "Conversational assistants, customer support, knowledge search",
    "traffic_profile": {
      "prompt_tokens": 512,
      "output_tokens": 256
    },
    "experience_class": "conversational",
    "slo_targets": {
      "ttft_p95_ms": 150,
      "itl_p95_ms": 25,
      "e2e_p95_ms": 7000
    },
    "rationale": "Highly interactive; perceived responsiveness drives satisfaction"
  }
}
```

**Validation**:
- [ ] All 9 use cases from traffic_and_slos.md are represented
- [ ] Traffic profiles match GuideLLM configurations
- [ ] SLO targets use p95 percentiles
- [ ] Experience classes are correctly assigned

---

## Phase 3: Update Synthetic Benchmark Data

**Objective**: Update benchmarks.json to match the 4 GuideLLM traffic profiles

**Decision**: Replace existing traffic profiles with the 4 real profiles from the database.

**Changes Needed**:
1. `data/benchmarks.json`
   - Update all benchmark entries to use one of 4 traffic profiles:
     - (512, 256) - Most common
     - (1024, 1024) - Balanced
     - (4096, 512) - Prefill-dominated
     - (10240, 1536) - Edge workloads (optional for Phase 1)
   - Set `prompt_tokens` and `output_tokens` fields
   - Keep `mean_input_tokens` ≈ `prompt_tokens` ± small variance
   - Keep `mean_output_tokens` ≈ `output_tokens` ± small variance
   - Ensure p95 fields are populated (ttft_p95, itl_p95, e2e_p95)

**Approach**:
- For each (model, hardware, hardware_count) combination, create benchmarks for 3-4 traffic profiles
- Focus on (512, 256), (1024, 1024), and (4096, 512) for Phase 1
- Skip (10240, 1536) initially (edge case, only 33 real benchmarks)

**Validation**:
- [ ] All benchmarks have prompt_tokens and output_tokens fields
- [ ] Traffic profiles match GuideLLM configurations
- [ ] Sufficient coverage across models and hardware types

---

## Phase 4: Create PostgreSQL Data Loader

**Objective**: Create script to load synthetic benchmarks into PostgreSQL

**Files to Create**:
1. `scripts/load_benchmarks.py`
   - Read benchmarks.json
   - Connect to PostgreSQL
   - Truncate and reload exported_summaries table
   - Report statistics

**Algorithm**:
```python
def load_benchmarks():
    # 1. Load JSON data
    benchmarks = load_json("data/benchmarks.json")

    # 2. Connect to PostgreSQL
    conn = psycopg2.connect(DATABASE_URL)

    # 3. Clear existing synthetic data (keep real data separate)
    # Option: Use different database or table for synthetic vs real

    # 4. Insert benchmarks
    for benchmark in benchmarks:
        INSERT INTO exported_summaries (...)
        VALUES (...)

    # 5. Report statistics
    print(f"Loaded {len(benchmarks)} benchmarks")
    print(f"Traffic profiles: {unique_profiles}")
```

**Validation**:
- [ ] Script loads data successfully
- [ ] Data queryable via postgres-query-traffic
- [ ] No conflicts with real data (if both present)

---

## Phase 5: Update Code to Use PostgreSQL and Traffic Profiles

**Objective**: Migrate code from JSON to PostgreSQL with traffic profile-based queries

**Decision**: Use PostgreSQL for all environments (upstream and downstream). Query by exact match on `(model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens)`, then filter by p95 SLO compliance.

**Files to Update**:

1. **`backend/src/knowledge_base/benchmarks.py`**
   - Replace JSON file reading with PostgreSQL queries
   - Add `get_benchmarks_by_traffic_profile()` function:
     ```python
     def get_benchmarks_by_traffic_profile(
         model: str,
         hardware: str,
         hardware_count: int,
         prompt_tokens: int,
         output_tokens: int
     ) -> list[dict]:
         """Find benchmarks matching exact traffic profile"""
         query = """
             SELECT * FROM exported_summaries
             WHERE model_hf_repo = %s
               AND hardware = %s
               AND hardware_count = %s
               AND prompt_tokens = %s
               AND output_tokens = %s
         """
         return execute_query(query, params)
     ```
   - Add connection pooling for performance
   - Add environment variable for DATABASE_URL

2. **`backend/src/context_intent/schema.py`**
   - Update `SLOTargets` to use p95:
     - `ttft_p95_target_ms` (was `ttft_p90_target_ms`)
     - `itl_p95_target_ms` (was `tpot_p90_target_ms`)
     - `e2e_p95_target_ms` (was `e2e_p90_target_ms`)
   - Add `traffic_profile` field to DeploymentIntent
   - Add `experience_class` field

3. **`backend/src/recommendation/capacity_planner.py`**
   - Update to query PostgreSQL via benchmarks.py
   - Use exact match on traffic profile (prompt_tokens, output_tokens)
   - Filter results by p95 SLO compliance:
     ```python
     matching_benchmarks = get_benchmarks_by_traffic_profile(...)
     compliant = [b for b in matching_benchmarks
                  if b['ttft_p95'] <= slo.ttft_p95_target_ms
                  and b['itl_p95'] <= slo.itl_p95_target_ms
                  and b['e2e_p95'] <= slo.e2e_p95_target_ms]
     ```
   - Use pre-calculated e2e_p95 from benchmarks (no dynamic calculation)

4. **`backend/src/knowledge_base/slo_templates.py`**
   - Update to read new slo_templates.json structure
   - Extract traffic_profile and experience_class
   - Map to p95 SLO targets

5. **`ui/app.py`**
   - Update use case selector to show new use cases
   - Display traffic profile in UI
   - Update SLO display to show p95 (not p90)
   - Rename "TPOT" → "ITL" throughout UI

6. **Configuration**
   - Add to `.env` or environment:
     ```
     DATABASE_URL=postgresql://postgres:compass@localhost:5432/compass
     ```

**Validation**:
- [ ] Code connects to PostgreSQL successfully
- [ ] Benchmark queries return correct results
- [ ] SLO filtering works with p95 values
- [ ] No performance regressions
- [ ] All tests pass

---

## Phase 6: Testing and Documentation

**Objective**: Comprehensive testing and documentation updates

**Testing Strategy**:

1. **Unit Tests**
   - Test PostgreSQL connection and queries
   - Test traffic profile matching logic
   - Test p95 SLO filtering
   - Test slo_templates.json parsing

2. **Integration Tests**
   - End-to-end workflow with PostgreSQL
   - Use case → traffic profile → benchmark lookup
   - SLO compliance checking
   - Recommendation generation

3. **Files to Update**:
   - `tests/test_recommendation_workflow.py` - Update for new SLO structure
   - `tests/test_yaml_generation.py` - Update for traffic profiles
   - Add `tests/test_postgresql_integration.py` - New PostgreSQL tests

**Documentation Updates**:

1. **`docs/ARCHITECTURE.md`**
   - Update Knowledge Base section to describe PostgreSQL
   - Update SLO-driven capacity planning with p95 targets
   - Add traffic profile concept
   - Reference traffic_and_slos.md

2. **`README.md`**
   - Add PostgreSQL setup to Quick Start
   - Update architecture overview
   - Add traffic profile explanation

3. **`docs/DEVELOPER_GUIDE.md`** (if exists, or create)
   - PostgreSQL setup instructions
   - How to load synthetic vs real data
   - How to query traffic patterns
   - Development workflow

**Validation**:
- [ ] All unit tests pass
- [ ] Integration tests pass with synthetic data
- [ ] Integration tests pass with real data (downstream)
- [ ] Documentation is clear and complete
- [ ] UI correctly displays new use cases and traffic profiles

---

## Key Design Decisions

### 1. PostgreSQL for All Environments
**Decision**: Use PostgreSQL for both upstream (open-source) and downstream (production), not JSON.

**Rationale**:
- Single code path - simpler implementation
- More realistic testing upstream
- PostgreSQL is free/open-source, easy to run via Docker
- Eliminates need for abstraction layer
- Easier to share real (unencumbered) benchmark data in future

### 2. Traffic Profile-Based Exact Matching
**Decision**: Match on exact `(prompt_tokens, output_tokens)` values, not fuzzy matching on `mean_*` values.

**Rationale**:
- GuideLLM benchmarks use 4 fixed traffic profiles (512→256, 1024→1024, 4096→512, 10240→1536)
- 1,990 "unique" patterns are just natural variance around these 4 targets
- Exact matching on `prompt_tokens`/`output_tokens` is simpler and faster than fuzzy distance calculations
- Eliminates complexity of distance metrics and logging
- Can add more profiles in future if needed

### 3. Experience-Driven SLOs with p95
**Decision**: Use p95 percentiles for SLO targets, organized by experience class.

**Rationale**:
- p95 is more conservative than p90, provides better UX guarantees
- Experience classes (instant, conversational, interactive, deferred, batch) capture user expectations better than arbitrary thresholds
- Same traffic profile can have different SLO requirements (e.g., code completion needs tighter latency than content generation)
- Aligns with industry best practices for user-facing services

### 4. Use Case → Traffic Profile Mapping
**Decision**: Map 9 use cases to 4 traffic profiles, allowing multiple use cases per profile.

**Rationale**:
- Traffic profiles are computational patterns (what the workload *is*)
- Use cases are user intentions (what the workload is *for*)
- This separation allows flexible SLO assignment
- Users think in terms of use cases; system optimizes by traffic profile

### 5. Pre-Calculated E2E Latency
**Decision**: Use pre-calculated `e2e_p95` from benchmarks instead of dynamic calculation.

**Rationale**:
- E2E already calculated in database from real benchmark runs
- More accurate than TTFT + (tokens × ITL) formula (accounts for batching, scheduling, etc.)
- Faster query performance (no computation needed)
- Consistent across all data sources

---

## Testing Strategy

### Unit Tests
- [ ] PostgreSQL connection and query functions
- [ ] Traffic profile matching logic
- [ ] p95 SLO filtering
- [ ] slo_templates.json parsing
- [ ] Benchmark data loader script

### Integration Tests
- [ ] End-to-end workflow with synthetic data
- [ ] Use case selection → traffic profile → benchmark lookup → recommendation
- [ ] SLO compliance checking with p95 values
- [ ] YAML generation with new schema
- [ ] Deployment workflow

### System Tests
- [ ] UI displays new use cases correctly
- [ ] All 9 use cases work end-to-end
- [ ] PostgreSQL performance is acceptable
- [ ] Real data integration (downstream only)

---

## Rollout Plan

### For Upstream (Open Source)
1. Complete Phases 2-6
2. Load synthetic data into PostgreSQL
3. Update all documentation
4. Ensure `make dev` works with PostgreSQL
5. Provide clear setup instructions in README

### For Downstream (Production)
1. Pull upstream changes
2. Run `make postgres-start`
3. Load real data: `make postgres-load-real`
4. Verify traffic profiles match expectations
5. Run integration tests
6. Deploy to production

---

## Migration Checklist

- [x] Phase 1: Infrastructure setup and PostgreSQL analysis
- [ ] Phase 2: Update SLO templates and use case definitions
- [ ] Phase 3: Update synthetic benchmark data
- [ ] Phase 4: Create PostgreSQL data loader
- [ ] Phase 5: Update code to use PostgreSQL and traffic profiles
- [ ] Phase 6: Testing and documentation
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Ready for production use

---

## Future Enhancements (Phase 2+)

### Parametric Performance Models
Develop regression models for each (model, GPU, hardware_count) configuration to predict TTFT, ITL, and E2E latency for arbitrary traffic profiles.

**Approach**:
- Train models: f(prompt_tokens, output_tokens) → (ttft_p95, itl_p95, e2e_p95)
- Use existing benchmark data as training set
- Interpolate for in-range predictions
- Provide confidence intervals

**Benefits**:
- Support arbitrary traffic profiles beyond the 4 GuideLLM defaults
- More accurate predictions for edge cases
- Continuous improvement as more benchmarks collected

### Additional Traffic Profiles
As more benchmark data becomes available, add support for:
- (256, 128) - Very short interactions
- (2048, 512) - Medium-long summarization
- Custom user-defined profiles

---

## Summary

This migration plan transitions Compass to use PostgreSQL with traffic profile-based exact matching:

- **4 GuideLLM traffic profiles** replace arbitrary token lengths
- **9 experience-driven use cases** map to these profiles
- **p95 SLO targets** provide better UX guarantees
- **PostgreSQL for all environments** simplifies architecture
- **Exact matching** eliminates fuzzy logic complexity

The result is a cleaner, faster, more maintainable system aligned with real-world benchmark data and user experience expectations.

---

*Last Updated: 2025-10-30*
*Status: Phase 1 Complete, Ready for Phase 2*
