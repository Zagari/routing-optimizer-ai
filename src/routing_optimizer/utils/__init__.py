"""
Utility functions for the routing optimizer.
"""

from routing_optimizer.utils.secrets import (
    get_openai_api_key,
    get_secret_from_aws,
    is_aws_environment,
)

__all__ = ["get_openai_api_key", "get_secret_from_aws", "is_aws_environment"]
