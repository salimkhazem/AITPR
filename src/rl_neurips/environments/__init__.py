"""
Environment Suite for AITPR Evaluation

This module contains environment configurations and utilities for testing
AITPR across various RL domains.
"""

from .env_suite import (
    EnvironmentSuite, 
    NormalizedEnv,
    create_evaluation_environments,
    run_environment_suite_test
)

__all__ = [
    "EnvironmentSuite",
    "NormalizedEnv", 
    "create_evaluation_environments",
    "run_environment_suite_test"
]