"""Скрипт для скачивания CIFAR-10 и загрузки сырого архива в S3."""

import os
import sys
from pathlib import Path

import torchvision
from botocore.exceptions import ClientError

from my_ds_project.s3.client import get_s3_client


def main() -> None:
    """Скачивает CIFAR-10 и загружает архив в S3 (идемпотентно)."""
    bucket = os.getenv("S3_BUCKET", "my-ds-bucket")
    s3_key = "raw/cifar-10-python.tar.gz"
    local_dir = Path("data/raw")
    local_dir.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()

    # 1. Проверяем идемпотентность: есть ли файл в S3?
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        print(f"Файл {s3_key} уже существует в S3. Пропускаем.")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            print(f"Ошибка при проверке S3: {e}")
            sys.exit(1)

    # 2. Скачиваем архив через torchvision
    print("Скачиваем CIFAR-10...")
    # torchvision скачает архив в local_dir и распакует его рядом
    torchvision.datasets.CIFAR10(root=str(local_dir), train=True, download=True)

    # 3. Ищем скачанный архив
    archive_path = local_dir / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        print(f"Ошибка: архив не найден по пути {archive_path}")
        sys.exit(1)

    # 4. Загружаем в S3
    print(f"Загружаем {archive_path} в S3 как {s3_key}...")
    s3.upload_file(str(archive_path), bucket, s3_key)
    print("Загрузка завершена успешно!")


if __name__ == "__main__":
    main()
