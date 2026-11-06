"""Модуль для работы с S3-совместимым хранилищем."""

import os
from typing import Any

import boto3
from botocore.config import Config


def get_s3_client() -> Any:
    """Создает и возвращает клиент boto3 для S3."""
    endpoint_url = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("S3_ACCESS_KEY", "admin")
    secret_key = os.getenv("S3_SECRET_KEY", "admin123")

    # Для локальных S3 (RustFS) нужна адресация в стиле пути
    config = Config(s3={"addressing_style": "path"})

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=config,
        region_name="us-east-1",  # Заглушка, часто требуется для boto3
    )
