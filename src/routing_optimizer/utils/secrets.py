"""
AWS Secrets Manager integration for secure credential management.

This module provides utilities for fetching secrets from AWS Secrets Manager
and falling back to environment variables. It does NOT depend on Streamlit.
"""

import os
from typing import Optional

# Tentar importar boto3 (opcional - pode nao estar instalado localmente)
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def get_secret_from_aws(secret_name: str, region_name: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret from AWS Secrets Manager.

    Args:
        secret_name: Name of the secret in Secrets Manager
        region_name: AWS region (defaults to AWS_REGION env var or us-east-1)

    Returns:
        Secret value as string, or None if not found/available
    """
    if not BOTO3_AVAILABLE:
        return None

    region = region_name or os.getenv("AWS_REGION", "us-east-1")

    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        return response.get("SecretString")
    except ClientError:
        return None
    except Exception:
        return None


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from server-side sources.

    Priority:
    1. AWS Secrets Manager (when running on EC2)
    2. Environment variable (OPENAI_API_KEY)

    Note: This function does NOT check Streamlit session state.
    For Streamlit apps, use routing_optimizer.app.utils.get_openai_api_key()
    which also checks user-provided keys in session state.

    Returns:
        API key string or None
    """
    # 1. Try AWS Secrets Manager
    secret_name = os.getenv("SECRET_NAME", "routing-optimizer/openai-api-key")
    aws_key = get_secret_from_aws(secret_name)
    if aws_key:
        return aws_key

    # 2. Environment variable / .env file
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    return None


def is_aws_environment() -> bool:
    """
    Check if running in an AWS environment.

    Returns:
        True if AWS credentials are configured, False otherwise.
    """
    if not BOTO3_AVAILABLE:
        return False

    try:
        # Try to get caller identity - will fail if not in AWS
        import boto3
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return True
    except Exception:
        return False
