## Ranking and Presenting Compass Recommendations

### Executive Summary

The Compass project aims to recommend optimal LLM and GPU configurations that satisfy users' specific use cases and priorities. This document outlines a strategic framework for ranking and presenting configuration options based on multiple decision criteria, acknowledging that users have varying priorities and may benefit from understanding tradeoffs between competing objectives.

**Priority Hierarchy:**
- **Primary Criteria**: Accuracy and Price are the most important factors for most users
- **Secondary Criteria**: Latency Performance (SLO headroom) and Operational Complexity provide additional differentiation among viable options
- **Note**: Since all recommendations are filtered to meet SLO requirements, latency becomes a secondary factor for ranking among compliant options

***

### Primary Optimization Criteria

**Note**: Accuracy and Price are considered the primary ranking criteria, while Latency Performance and Operational Complexity are secondary factors that help differentiate among options that meet basic requirements.

#### 1. **Accuracy (Model Quality)**

**Definition**: The capability of the LLM to perform the intended task effectively.

**Compass Approach**:
- Current: Uses model quality indicators from catalog metadata
- Planned: Integration with benchmark-based metrics (MMLU, HumanEval, MT-Bench, etc.)

**Key Tradeoff**: Larger, more accurate models have higher latency and cost. Benchmark data captures this relationship for informed decision-making.

***

#### 2. **Price (Total Cost of Ownership)**

**Definition**: The financial cost of deploying and operating the configuration.

**Deployment Models**:
- **Cloud GPU rental**: Hourly/monthly rates from cloud providers (AWS, GCP, Azure)
- **On-premise GPU purchase**: Capital expenditure + operational costs (power, cooling, maintenance)
- **Existing GPU infrastructure**: Utilization-based costing for already-owned hardware

**Compass Approach**: Current code supports cloud GPU pricing. On-premise and existing infrastructure cost models are planned.

***

#### 3. **Latency Performance** (Secondary)

**Definition**: Response time characteristics measured through SLO metrics:
- **TTFT** (Time to First Token) - Critical for interactive applications
- **ITL** (Inter-Token Latency) - Important for streaming responses
- **E2E** (End-to-End Latency) - Overall response time

**Compass Approach**: Uses p95 percentile targets from benchmark data to filter configurations that meet user-specified SLO requirements.

**Role in Ranking**: Since all recommended configurations already meet SLO targets, latency becomes a secondary differentiator. Configurations with better SLO headroom (e.g., 120ms TTFT vs. 150ms target) may be preferred for additional reliability margin, but this is less critical than accuracy and price differences.

***

#### 4. **Operational Complexity** (Secondary)

**Definition**: The difficulty of deploying and managing the configuration.

**Key Factors**:
- Number of GPU instances (fewer is simpler)
- Infrastructure coordination (multi-node vs. single-node)
- Deployment topology (tensor parallelism vs. replicas)

**Example Tradeoff**: 2x H100 GPUs may be preferred over 8x L4 GPUs even at higher cost due to reduced networking complexity and management overhead.

**Compass Approach**: Balances tensor parallelism and replica scaling to minimize operational burden while meeting SLO targets.

**Role in Ranking**: Complexity can be an important differentiator when choosing between configurations with similar accuracy and cost profiles. Simpler deployments are generally more reliable and easier to troubleshoot.

***

### Configuration Variants

Some factors affect recommendations but are treated as **configuration variants** rather than independent ranking criteria:

#### **Quantization Options**

Quantization (FP16, INT8, INT4) creates multiple deployment options for each model-GPU combination:

**Impact on Primary Criteria**:
- **Accuracy**: Lower precision may reduce quality (benchmark-dependent)
- **Price**: Reduced memory requirements → fewer GPUs needed
- **Latency**: Faster inference due to smaller compute requirements
- **Complexity**: Simpler deployment with smaller memory footprint

**Compass Approach**: Present quantization as selectable options within each recommendation, with clear indication of tradeoffs based on benchmark data.

**Example**:
```
Recommended: Llama-3-70B on 2x H100
├─ FP16 (Standard):  $450/month, TTFT=120ms, Reference accuracy
└─ INT8 (Optimized): $290/month, TTFT=95ms,  ~1-2% accuracy reduction
```

***

### Secondary Considerations

The following factors may be addressed in future releases based on team priorities:

- **GPU Availability**: Procurement constraints, lead times, and regional availability
- **Reliability**: Uptime SLAs, redundancy options, and failover capabilities

*These are documented for completeness but not prioritized for initial implementation.*

***

### Scoring and Ranking Framework

#### Scoring Each Configuration (0-100 Scale)

Each viable configuration receives scores across the four primary criteria:

**1. Accuracy Score** (0-100):
- Based on model capability tier or benchmark scores
- Example: 7B model = 60, 70B model = 85, 405B model = 95
- Minimum threshold filter (e.g., user requires score ≥ 70)

**2. Price Score** (0-100):
- Inverse of cost: Lower cost = higher score
- Normalized: `100 * (max_cost - config_cost) / (max_cost - min_cost)`
- Maximum cost filter (e.g., user has $500/month budget ceiling)

**3. Latency Score** (0-100):
- Composite of TTFT, ITL, and E2E performance vs. targets
- Configurations meeting all SLOs score ≥ 90
- Below-SLO configurations score proportionally lower
- Near-miss configurations (10-20% over SLO) score 70-89

**4. Complexity Score** (0-100):
- Based on GPU count and deployment topology
- Example: 1 GPU = 100, 2 GPUs = 90, 4 GPUs = 75, 8+ GPUs = 60
- Factors: Single-node (simpler) vs. multi-node, tensor parallelism overhead

#### Ranking Strategies

**Primary Approach: Single-Criterion Ranking**

Show top 5 configurations for each optimization priority:

1. **Best Accuracy** - Sort by accuracy score (descending), filter by cost ceiling
2. **Lowest Cost** - Sort by price score (descending), filter by minimum accuracy
3. **Lowest Latency** - Sort by latency score (descending), filter by cost and accuracy
4. **Simplest** - Sort by complexity score (descending), filter by cost and accuracy
5. **Balanced** - Sort by composite score using equal weights (25% each criterion)

**Handling "Balanced" Recommendation**:
- **Weighted composite score** allows flexible prioritization of different criteria
- **Default weights**: Can be set based on general user preferences (e.g., 40% accuracy, 40% price, 10% latency, 10% complexity to reflect primary vs. secondary criteria)
- **User-adjustable weights**: Allow users to customize weights based on their specific priorities
- **Priority-based weighting**: Alternatively, derive weights from higher-level user priorities (e.g., "Quality first" → higher accuracy weight, "Budget conscious" → higher price weight)
- **Pareto frontier approach**: Alternative method that finds configurations where no other option is better on all criteria (more complex but mathematically optimal)

#### Filtering Logic

**Hard Constraints** (Applied before ranking):
- Minimum accuracy score (user-specified or default)
- Maximum cost ceiling (optional user constraint)
- SLO compliance (optional: include/exclude near-miss configurations)

**Example Workflow**:
```
1. Filter all viable configs by minimum accuracy ≥ 70
2. Filter by maximum cost ≤ $500/month (if specified)
3. Score remaining configs on all 4 criteria
4. Generate ranked lists for each optimization criterion:
   - Best Accuracy (top 5)
   - Lowest Cost (top 5)
   - Lowest Latency (top 5)
   - Simplest (top 5)
   - Balanced (top 5, using weighted composite score)
5. Present as ordered lists with selectable views (tabs, dropdown, or list filters)
6. Optionally display graphical representations (e.g., Pareto frontier chart)
```

**Display Options**:
- **Ordered Lists**: Primary display method showing top N configurations for each ranking criterion
- **Pareto Frontier Chart**: Graphical visualization plotting configurations on 2D/3D space (e.g., Cost vs. Accuracy, with point size representing complexity)
- **Interactive Filtering**: Allow users to adjust weights and see rankings update in real-time

#### Near-SLO Options Integration

Include configurations that slightly exceed SLO targets:
- Score latency as 70-89 (vs. 90-100 for SLO-compliant)
- Clearly mark with warning indicators
- Highlight cost/accuracy benefits compared to SLO-compliant alternatives

**Example Display**:
```
🎯 Lowest Cost (Top 5):

1. ✅ Llama-3-8B on 1x L4    - $120/month (Score: 100) [Meets SLOs]
2. ✅ Llama-3-8B on 1x A100  - $180/month (Score: 95)  [Meets SLOs]
3. ⚠️ Llama-3-70B on 2x A100 - $290/month (Score: 85)  [TTFT +10% over SLO]
4. ✅ Llama-3-70B on 2x H100 - $450/month (Score: 70)  [Meets SLOs]
5. ✅ Llama-3-405B on 4x H100 - $900/month (Score: 40) [Meets SLOs]
```

***

### Implementation Phasing

**Phase 1**: Single-criterion ranking
- Implement 0-100 scoring for all 4 criteria
- Support minimum accuracy and maximum cost filters
- Show top 5 for "Lowest Cost" (primary view)
- Show top 5 for other criteria (Accuracy, Latency, Simplest, Balanced) as alternative views

**Phase 2**: Multi-priority views
- Add tabs/buttons for all 5 ranking modes (Accuracy, Cost, Latency, Simplest, Balanced)
- Include near-SLO configurations with clear warnings
- Show top 5 for each mode (consistent across all views)

**Phase 3**: Interactive refinement and visualization
- User-adjustable weights for balanced score (slider controls or direct weight input)
- Visual Pareto frontier chart for multi-dimensional tradeoff exploration
  - 2D charts: Cost vs. Accuracy (most common)
  - 3D charts: Cost vs. Accuracy vs. Complexity (advanced view)
  - Interactive point selection to view full configuration details
- Sensitivity analysis (e.g., "How does ranking change if budget increases to $600?")
- Priority-driven weight selection (e.g., user selects "Quality first" and system sets weights automatically)

***

### Summary: Decision Framework

**Ranking Criteria**:
1. **Accuracy** (Primary) - Model capability (metadata → benchmark scores)
2. **Price** (Primary) - TCO across deployment models (cloud, on-prem, existing)
3. **Latency** (Secondary) - SLO headroom beyond compliance threshold (TTFT, ITL, E2E at p95)
4. **Operational Complexity** (Secondary) - GPU count and deployment topology

**Configuration Variants**:
- Quantization (FP16/INT8/INT4) as selectable options within recommendations

**Secondary Considerations**:
- Throughput, GPU availability, reliability, scalability (future releases)

***

### Key Decisions for Team Discussion

1. **Scoring Scale**: Use 0-100 or 0-1 normalization?
   - **Recommendation**: 0-100 scale for better interpretability ("85/100" vs "0.85")
   - Easier to explain to users and debug during development

2. **Balanced Score Calculation**: How to determine the "Balanced" recommendation?
   - **Option A**: Default weights reflecting primary vs. secondary criteria (e.g., 40% accuracy, 40% price, 10% latency, 10% complexity)
   - **Option B**: User-adjustable weights via UI controls (sliders or direct input)
   - **Option C**: Priority-driven weights derived from high-level user preferences ("Quality first", "Budget conscious", etc.)
   - **Option D**: Pareto frontier - mathematically optimal but more complex
   - **Recommendation**: Start with Option A for Phase 1, add Option B in Phase 3

3. **Number of Options per View**: Show top 5 for each ranking criterion?
   - **Recommendation**: Yes - provides good variety without overwhelming users
   - UI can display as expandable list or tabbed views

4. **Minimum Accuracy Threshold**: Should there be a default minimum, or always user-specified?
   - **Option A**: No default, show all configurations
   - **Option B**: Default minimum based on use case (e.g., production = 70, development = 50)
   - **Recommendation**: Option B - prevents showing clearly inadequate models

5. **Near-SLO Configurations**: Include by default or require opt-in?
   - **Recommendation**: Include by default with clear ⚠️ warnings
   - Significant cost savings justify showing, but must be clearly marked
   - Allow users to toggle "Hide near-miss options" if desired

6. **Multi-Cost Model Support**: How to handle cloud vs. on-prem vs. existing GPU pricing?
   - **Option A**: Cloud GPU pricing only (standardized rates)
   - **Option B**: Also support user-provided cost parameters for on-prem/existing infrastructure.
   - **Option C (Future)**: Compare cloud vs. on-prem vs. existing GPU options
   - Scoring logic remains the same, just the cost input source changes

7. **Accuracy Scoring Method**: Model tiers vs. benchmark scores?
   - **Phase 1 (Current Approach)**: Simple capability tiers (7B=60, 70B=85, 405B=95)
   - **Phase 2 (WIP)**: Integrate actual benchmark scores (MMLU, HumanEval normalized to 0-100)
   - Allows smooth evolution without changing scoring framework

8. **Visualization Approaches**: How to present recommendations graphically?
   - **Primary**: Ordered lists for each ranking criterion (simplest, always available)
   - **Phase 3**: Pareto frontier charts for multi-dimensional tradeoff exploration
     - 2D scatter plots: Cost vs. Accuracy (most intuitive)
     - Interactive elements: Click points to view full configuration details
     - Point styling: Size/color to represent secondary criteria (complexity, latency)
   - **Benefits**: Graphical views help users understand tradeoff space and identify optimal regions
   - **Note**: Lists remain primary interface; charts supplement for advanced users

