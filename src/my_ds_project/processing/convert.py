"""Модуль для конвертации сырых данных CIFAR-10 в PNG и JSON."""

import argparse
import hashlib
import json
import pickle
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image

from my_ds_project.s3.client import get_s3_client


def download_from_s3(bucket: str, s3_key: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Файл {local_path} уже существует локально.")
        return
    print(f"Скачиваем {s3_key} из S3...")
    s3 = get_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(local_path))


def convert_cifar_to_images(archive_path: Path, output_dir: Path) -> None:
    extract_dir = output_dir / "raw_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not (extract_dir / "cifar-10-batches-py").exists():
        print("Распаковка архива...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    data_dir = extract_dir / "cifar-10-batches-py"

    with open(data_dir / "batches.meta", "rb") as f:
        meta: dict[bytes, Any] = pickle.load(f, encoding="bytes")
        classes: list[str] = [c.decode("utf-8") for c in meta[b"label_names"]]

    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    for split in ["train", "test"]:
        print(f"Обработка {split}...")
        if split == "train":
            batch_files = [data_dir / f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = [data_dir / "test_batch"]

        # Явно указываем тип списка для mypy
        images_meta: list[dict[str, Any]] = []

        for batch_file in batch_files:
            with open(batch_file, "rb") as f:
                batch: dict[bytes, Any] = pickle.load(f, encoding="bytes")

            data = batch[b"data"]
            labels = batch[b"labels"]

            # Переместили импорт numpy наверх
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

            start_id = len(images_meta)
            # Добавили strict=False, чтобы Ruff не ругался
            for i, (img_array, label_idx) in enumerate(zip(data, labels, strict=False)):
                image_id = f"{split}_{start_id + i:05d}"
                img_path = images_dir / f"{image_id}.png"

                img = Image.fromarray(img_array)
                img.save(img_path)

                sha256_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()

                images_meta.append(
                    {
                        "image_id": image_id,
                        "file_name": f"images/{image_id}.png",
                        "label": int(label_idx),
                        "label_name": classes[int(label_idx)],
                        "sha256": sha256_hash,
                    }
                )

        json_path = annotations_dir / f"{split}.json"
        with open(json_path, "w") as f:
            json.dump({"classes": classes, "images": images_meta}, f, indent=2)
        print(f"Сохранено {len(images_meta)} изображений в {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвертация CIFAR-10")
    parser.add_argument("--bucket", default="my-ds-bucket")
    parser.add_argument("--s3-key", default="raw/cifar-10-python.tar.gz")
    parser.add_argument("--local-archive", default="data/raw/cifar-10-python.tar.gz")
    parser.add_argument("--output-dir", default="data/processed/v1")
    args = parser.parse_args()

    download_from_s3(args.bucket, args.s3_key, Path(args.local_archive))
    convert_cifar_to_images(Path(args.local_archive), Path(args.output_dir))


if __name__ == "__main__":
    main()
