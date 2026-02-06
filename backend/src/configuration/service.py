"""Configuration Service facade.

Provides a high-level interface for generating and validating deployment configurations.
"""

import logging
from pathlib import Path

from ..shared.schemas import DeploymentRecommendation
from .generator import DeploymentGenerator
from .validator import YAMLValidator

logger = logging.getLogger(__name__)


class ConfigurationService:
    """High-level service for generating deployment configurations."""

    def __init__(
        self,
        output_dir: str | None = None,
        simulator_mode: bool = False,
    ):
        """
        Initialize the Configuration Service.

        Args:
            output_dir: Directory to write generated YAML files
            simulator_mode: If True, use vLLM simulator instead of real vLLM
        """
        self.generator = DeploymentGenerator(
            output_dir=output_dir,
            simulator_mode=simulator_mode,
        )
        self.validator = YAMLValidator()

    def generate_and_validate(
        self,
        recommendation: DeploymentRecommendation,
        namespace: str = "default",
    ) -> dict:
        """
        Generate all deployment configurations and validate them.

        This is the primary method for creating production-ready configurations.

        Args:
            recommendation: The deployment recommendation
            namespace: Kubernetes namespace for deployment

        Returns:
            Dictionary with deployment_id, files, and validation results

        Raises:
            ValueError: If validation fails
        """
        # Generate all configuration files
        result = self.generator.generate_all(recommendation, namespace)

        # Validate the generated files
        validation_errors = []
        for config_type, file_path in result["files"].items():
            is_valid, errors = self.validator.validate_file(file_path)
            if not is_valid:
                validation_errors.extend(errors)

        if validation_errors:
            error_msg = "\n".join(validation_errors)
            raise ValueError(f"Generated configurations failed validation:\n{error_msg}")

        result["validated"] = True
        logger.info(
            f"Generated and validated {len(result['files'])} configuration files "
            f"for deployment {result['deployment_id']}"
        )

        return result

    def generate_yaml(
        self,
        recommendation: DeploymentRecommendation,
        namespace: str = "default",
    ) -> dict:
        """
        Generate deployment configurations without validation.

        Use this for quick generation when validation is not needed.

        Args:
            recommendation: The deployment recommendation
            namespace: Kubernetes namespace for deployment

        Returns:
            Dictionary with deployment_id and file paths
        """
        return self.generator.generate_all(recommendation, namespace)

    def validate_yaml(self, yaml_content: str) -> tuple[bool, list[str]]:
        """
        Validate YAML content.

        Args:
            yaml_content: YAML content string to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        return self.validator.validate_yaml(yaml_content)
