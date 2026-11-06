"""Модуль для выгрузки processed данных в S3."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from my_ds_project.s3.client import get_s3_client


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "my-ds-bucket"))
    parser.add_argument("--input-dir", default="data/processed/v1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    s3_prefix = f"processed/cifar10/{args.version}"

    s3 = get_s3_client()

    # Проверка на существование версии
    try:
        s3.head_object(Bucket=args.bucket, Key=f"{s3_prefix}/manifest.json")
        print(f"Ошибка: Версия {args.version} уже существует в S3. Укажите другую версию.")
        return
    except Exception:
        pass  # Версии нет, можно загружать

    # Собираем данные для manifest
    annotations = {}
    for json_file in ["train.json", "val.json", "test.json"]:
        path = input_dir / "annotations" / json_file
        if path.exists():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            annotations[json_file] = {"size": path.stat().st_size, "sha256": sha}

    stats_path = input_dir / "stats.json"
    stats = {}
    if stats_path.exists():
        sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        stats = {"stats.json": {"size": stats_path.stat().st_size, "sha256": sha}}

    manifest = {
        "version": args.version,
        "git_commit": get_git_commit(),
        "val_size": args.val_size,
        "seed": args.seed,
        "num_images": {
            "train": len(json.loads((input_dir / "annotations/train.json").read_text())["images"]),
            "val": len(json.loads((input_dir / "annotations/val.json").read_text())["images"]),
            "test": len(json.loads((input_dir / "annotations/test.json").read_text())["images"]),
        },
        "files": {**annotations, **stats},
    }

    # Сохраняем manifest локально
    manifest_path = input_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Загрузка в S3: {s3_prefix}/")
    # Загружаем картинки (параллельно)
    import concurrent.futures

    images_dir = input_dir / "images"

    def upload_file(path: Path) -> None:
        s3_key = f"{s3_prefix}/{path.relative_to(input_dir).as_posix()}"
        s3.upload_file(str(path), args.bucket, s3_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(upload_file, images_dir.glob("*.png")))

    # Загружаем JSON-файлы
    for file in input_dir.rglob("*.json"):
        s3_key = f"{s3_prefix}/{file.relative_to(input_dir).as_posix()}"
        s3.upload_file(str(file), args.bucket, s3_key)

    print("Выгрузка завершена успешно!")


if __name__ == "__main__":
    main()
