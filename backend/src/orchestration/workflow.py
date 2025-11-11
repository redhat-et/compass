"""Workflow orchestration for end-to-end recommendation flow."""

import logging

from ..context_intent.extractor import IntentExtractor
from ..context_intent.schema import ConversationMessage, DeploymentRecommendation
from ..llm.ollama_client import OllamaClient
from ..recommendation.capacity_planner import CapacityPlanner
from ..recommendation.model_recommender import ModelRecommender
from ..context_intent.traffic_profile import TrafficProfileGenerator

logger = logging.getLogger(__name__)


class RecommendationWorkflow:
    """Orchestrate the full recommendation workflow."""

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        intent_extractor: IntentExtractor | None = None,
        traffic_generator: TrafficProfileGenerator | None = None,
        model_recommender: ModelRecommender | None = None,
        capacity_planner: CapacityPlanner | None = None,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            llm_client: Ollama client (creates default if not provided)
            intent_extractor: Intent extractor
            traffic_generator: Traffic profile generator
            model_recommender: Model recommender
            capacity_planner: Capacity planner
        """
        self.llm_client = llm_client or OllamaClient()
        self.intent_extractor = intent_extractor or IntentExtractor(self.llm_client)
        self.traffic_generator = traffic_generator or TrafficProfileGenerator()
        self.model_recommender = model_recommender or ModelRecommender()
        self.capacity_planner = capacity_planner or CapacityPlanner()

    def generate_specification(
        self, user_message: str, conversation_history: list[ConversationMessage] | None = None
    ) -> tuple:
        """
        Generate deployment specification from user message.

        This always succeeds and returns the specification even if no viable
        configurations exist.

        Returns:
            Tuple of (DeploymentSpecification, intent, traffic_profile, slo_targets, model_candidates)
        """
        from ..context_intent.schema import DeploymentSpecification

        logger.info("Step 1: Extracting deployment intent")
        intent = self.intent_extractor.extract_intent(user_message, conversation_history)
        intent = self.intent_extractor.infer_missing_fields(intent)
        logger.info(
            f"Intent extracted: {intent.use_case}, {intent.user_count} users, {intent.latency_requirement} latency"
        )

        logger.info("Step 2: Generating traffic profile and SLO targets")
        traffic_profile = self.traffic_generator.generate_profile(intent)
        slo_targets = self.traffic_generator.generate_slo_targets(intent)
        logger.info(
            f"Traffic profile: ({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens}), "
            f"{traffic_profile.expected_qps} QPS"
        )
        logger.info(
            f"SLO targets (p95): TTFT={slo_targets.ttft_p95_target_ms}ms, "
            f"ITL={slo_targets.itl_p95_target_ms}ms, E2E={slo_targets.e2e_p95_target_ms}ms"
        )

        logger.info("Step 3: Recommending models")
        model_candidates = self.model_recommender.recommend_models(intent, top_k=3)

        if not model_candidates:
            logger.warning(f"No suitable models found for use case: {intent.use_case}")
            model_candidates = []

        models_to_evaluate = [m.name for m, _ in model_candidates] if model_candidates else []

        specification = DeploymentSpecification(
            intent=intent,
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            models_to_evaluate=models_to_evaluate if models_to_evaluate else None,
        )

        return specification, intent, traffic_profile, slo_targets, model_candidates

    def _recommend_from_specs(
        self,
        intent: "DeploymentIntent",
        traffic_profile: "TrafficProfile",
        slo_targets: "SLOTargets",
    ) -> DeploymentRecommendation:
        """
        Shared logic to generate recommendation from specifications.

        This is the core recommendation algorithm used by both:
        - generate_recommendation() - after extracting specs from conversation
        - generate_recommendation_from_specs() - using user-edited specs

        Args:
            intent: Deployment intent
            traffic_profile: Traffic characteristics
            slo_targets: SLO targets

        Returns:
            DeploymentRecommendation

        Raises:
            ValueError: If no viable configurations found
        """
        # Step 1: Recommend models based on intent
        logger.info("Recommending models")
        model_candidates = self.model_recommender.recommend_models(intent, top_k=3)

        if not model_candidates:
            raise ValueError(f"No suitable models found for use case: {intent.use_case}")

        logger.info(f"Found {len(model_candidates)} model candidates")

        # Step 2: Plan capacity for each model
        logger.info("Planning GPU capacity")
        viable_recommendations = []

        for model, score in model_candidates:
            logger.info(f"Planning capacity for {model.name} (score: {score:.1f})")

            recommendation = self.capacity_planner.plan_capacity(
                model, traffic_profile, slo_targets, intent
            )

            if recommendation:
                viable_recommendations.append((recommendation, score))
                logger.info(
                    f"  ✓ Viable: {recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}, "
                    f"${recommendation.cost_per_month_usd:.0f}/month"
                )
            else:
                logger.info("  ✗ No viable configuration found")

        if not viable_recommendations:
            # Build helpful error message with context
            error_msg = (
                f"No viable deployment configurations found meeting SLO targets.\n\n"
                f"**Requirements:**\n"
                f"- Use case: {intent.use_case} ({intent.experience_class} experience)\n"
                f"- Scale: {intent.user_count:,} users\n"
                f"- Latency requirement: {intent.latency_requirement}\n"
                f"- Budget: {intent.budget_constraint}\n\n"
                f"**Traffic profile:**\n"
                f"- {traffic_profile.prompt_tokens} prompt tokens → {traffic_profile.output_tokens} output tokens\n"
                f"- Expected load: {traffic_profile.expected_qps} queries/second\n\n"
                f"**SLO targets (p95):**\n"
                f"- TTFT ≤ {slo_targets.ttft_p95_target_ms}ms\n"
                f"- ITL ≤ {slo_targets.itl_p95_target_ms}ms\n"
                f"- E2E ≤ {slo_targets.e2e_p95_target_ms}ms\n\n"
                f"**Models evaluated:** {', '.join(m.name for m, _ in model_candidates)}\n\n"
                f"None of these models can meet the SLO targets with available hardware configurations. "
                f"Try relaxing latency requirements or considering a different use case."
            )
            raise ValueError(error_msg)

        # Step 3: Sort and select best recommendation
        # Sort by:
        # 1. Model score (higher is better)
        # 2. Throughput capacity (higher QPS is better for same cost)
        # 3. Cost (lower is better)
        viable_recommendations.sort(
            key=lambda x: (-x[1], -x[0].predicted_throughput_qps, x[0].cost_per_month_usd)
        )
        best_recommendation = viable_recommendations[0][0]

        logger.info(
            f"Selected: {best_recommendation.model_name} on "
            f"{best_recommendation.gpu_config.gpu_count}x {best_recommendation.gpu_config.gpu_type}"
        )

        # Step 4: Filter and add valuable alternative options
        if len(viable_recommendations) > 1:
            # Filter alternatives to provide distinct value propositions
            # Avoid showing strictly dominated options (higher cost, same/lower performance)
            best = best_recommendation
            valuable_alternatives = []

            for rec, score in viable_recommendations[1:]:
                # Always include if different model (provides choice)
                if rec.model_id != best.model_id:
                    valuable_alternatives.append((rec, score))
                    continue

                # Same model - only include if it provides distinct value:
                # 1. Different GPU type (e.g., H100 vs H200)
                if rec.gpu_config.gpu_type != best.gpu_config.gpu_type:
                    valuable_alternatives.append((rec, score))
                    continue

                # 2. Significantly better performance for modest cost increase
                #    (e.g., 50% more throughput for <30% more cost)
                cost_ratio = rec.cost_per_month_usd / best.cost_per_month_usd
                throughput_ratio = rec.predicted_throughput_qps / best.predicted_throughput_qps
                if throughput_ratio >= 1.5 and cost_ratio < 1.3:
                    valuable_alternatives.append((rec, score))
                    continue

                # 3. Significantly better latency (20%+ improvement) if cost is similar
                latency_improvement = (
                    (best.predicted_ttft_p95_ms - rec.predicted_ttft_p95_ms)
                    / best.predicted_ttft_p95_ms
                )
                if latency_improvement >= 0.2 and cost_ratio <= 1.2:
                    valuable_alternatives.append((rec, score))
                    continue

                # Otherwise skip - this is a dominated option
                logger.debug(
                    f"Skipping dominated alternative: {rec.model_name} on "
                    f"{rec.gpu_config.gpu_count}x {rec.gpu_config.gpu_type} "
                    f"(cost={cost_ratio:.2f}x, throughput={throughput_ratio:.2f}x)"
                )

            # Convert to dict format for alternatives
            best_recommendation.alternative_options = [
                rec.to_alternative_dict() for rec, _ in valuable_alternatives[:3]  # Limit to 3 alternatives
            ]

            logger.info(
                f"Added {len(best_recommendation.alternative_options)} valuable alternative options "
                f"(filtered from {len(viable_recommendations)-1} total options)"
            )

        return best_recommendation

    def generate_recommendation(
        self, user_message: str, conversation_history: list[ConversationMessage] | None = None
    ) -> DeploymentRecommendation:
        """
        Generate deployment recommendation from user message.

        This is the main workflow that orchestrates all components:
        1. Extract intent from conversation
        2. Generate traffic profile and SLO targets
        3. Recommend models
        4. Plan GPU capacity
        5. Return best recommendation

        Args:
            user_message: User's deployment request
            conversation_history: Optional conversation context

        Returns:
            DeploymentRecommendation

        Raises:
            ValueError: If recommendation cannot be generated
        """
        logger.info("Starting recommendation workflow")

        # Generate specification from conversation
        specification, intent, traffic_profile, slo_targets, model_candidates = (
            self.generate_specification(user_message, conversation_history)
        )

        # Generate recommendation from specification (shared logic)
        return self._recommend_from_specs(intent, traffic_profile, slo_targets)

    def generate_recommendation_from_specs(self, specifications: dict) -> DeploymentRecommendation:
        """
        Generate recommendation from user-edited specifications (exploration mode).

        This bypasses intent extraction and uses the provided specifications directly.

        Args:
            specifications: Dict with keys: intent, traffic_profile, slo_targets

        Returns:
            DeploymentRecommendation

        Raises:
            ValueError: If recommendation cannot be generated
        """
        from ..context_intent.schema import DeploymentIntent, SLOTargets, TrafficProfile

        logger.info("Starting re-recommendation workflow with edited specifications")

        # Infer experience_class if not provided in intent
        intent_data = specifications["intent"].copy()
        if "experience_class" not in intent_data or not intent_data.get("experience_class"):
            # Use the same inference logic as the extractor
            use_case = intent_data.get("use_case", "")
            if use_case == "code_completion":
                intent_data["experience_class"] = "instant"
            elif use_case in [
                "chatbot_conversational",
                "code_generation_detailed",
                "translation",
                "content_generation",
                "summarization_short",
            ]:
                intent_data["experience_class"] = "conversational"
            elif use_case == "document_analysis_rag":
                intent_data["experience_class"] = "interactive"
            elif use_case == "long_document_summarization":
                intent_data["experience_class"] = "deferred"
            elif use_case == "research_legal_analysis":
                intent_data["experience_class"] = "batch"
            else:
                intent_data["experience_class"] = "conversational"  # Default

        # Parse specifications into proper schema objects
        intent = DeploymentIntent(**intent_data)
        traffic_profile = TrafficProfile(**specifications["traffic_profile"])
        slo_targets = SLOTargets(**specifications["slo_targets"])

        logger.info(
            f"Specs: {intent.use_case}, {intent.user_count} users, "
            f"{traffic_profile.expected_qps} QPS, "
            f"TTFT target={slo_targets.ttft_p95_target_ms}ms (p95)"
        )

        # Generate recommendation from specification (shared logic)
        return self._recommend_from_specs(intent, traffic_profile, slo_targets)

    def validate_recommendation(self, recommendation: DeploymentRecommendation) -> bool:
        """
        Validate that recommendation meets all requirements.

        Args:
            recommendation: Deployment recommendation to validate

        Returns:
            True if valid
        """
        # Check SLO targets are met
        if not recommendation.meets_slo:
            logger.warning("Recommendation does not meet SLO targets")
            return False

        # Check TTFT
        if recommendation.predicted_ttft_p95_ms > recommendation.slo_targets.ttft_p95_target_ms:
            logger.warning(
                f"TTFT {recommendation.predicted_ttft_p95_ms}ms exceeds target "
                f"{recommendation.slo_targets.ttft_p95_target_ms}ms"
            )
            return False

        # Check ITL
        if recommendation.predicted_itl_p95_ms > recommendation.slo_targets.itl_p95_target_ms:
            logger.warning(
                f"ITL {recommendation.predicted_itl_p95_ms}ms exceeds target "
                f"{recommendation.slo_targets.itl_p95_target_ms}ms"
            )
            return False

        # Check E2E
        if recommendation.predicted_e2e_p95_ms > recommendation.slo_targets.e2e_p95_target_ms:
            logger.warning(
                f"E2E {recommendation.predicted_e2e_p95_ms}ms exceeds target "
                f"{recommendation.slo_targets.e2e_p95_target_ms}ms"
            )
            return False

        # Check throughput
        if recommendation.predicted_throughput_qps < recommendation.traffic_profile.expected_qps:
            logger.warning(
                f"Throughput {recommendation.predicted_throughput_qps} QPS below required "
                f"{recommendation.traffic_profile.expected_qps} QPS"
            )
            return False

        return True
