"""Скрипт для обучения модели на CIFAR-10 с логированием в MLflow."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import boto3
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from botocore.config import Config
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Загружаем переменные окружения из .env
load_dotenv(override=True)

# Настройки S3
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "admin123")
S3_BUCKET = os.getenv("S3_BUCKET", "my-ds-bucket")


def get_git_commit() -> str:
    """Возвращает хеш текущего коммита."""
    build_commit = os.getenv("GIT_COMMIT")
    if build_commit:
        return build_commit
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


class CIFAR10Dataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, images_dir: Path, annotations_file: Path, transform: Any = None):
        with open(annotations_file) as f:
            data = json.load(f)
        self.images_dir = images_dir
        self.images_meta = data["images"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_meta)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_meta = self.images_meta[idx]
        img_path = self.images_dir / img_meta["file_name"]
        image = Image.open(img_path).convert("RGB")
        label = int(img_meta["label"])

        if self.transform:
            tensor_image: torch.Tensor = self.transform(image)
            return tensor_image, label

        return transforms.ToTensor()(image), label


class SimpleCNN(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.fc = nn.Linear(channels[1] * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config values into MLflow-compatible scalar parameters."""
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_config(value, name))
        elif isinstance(value, list | tuple):
            flattened[name] = json.dumps(value)
        else:
            flattened[name] = value
    return flattened


def get_s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )


def download_data_from_s3(s3_path: str, local_dir: Path) -> None:
    if local_dir.exists() and any(local_dir.iterdir()):
        print("Данные уже скачаны локально.")
        return
    print(f"Скачиваем данные из S3 ({s3_path})...")
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_path):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = local_dir / key[len(s3_path) :].lstrip("/")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, str(local_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 1. Установка seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 2. Скачивание данных
    version = config["data"]["s3_path"].rstrip("/").split("/")[-1]
    local_data_dir = Path("data/processed") / version
    download_data_from_s3(config["data"]["s3_path"], local_data_dir)

    # 3. Загрузка статистики нормализации
    with open(local_data_dir / "stats.json") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # 4. Датасеты
    train_dataset = CIFAR10Dataset(
        local_data_dir, local_data_dir / "annotations/train.json", transform
    )
    val_dataset = CIFAR10Dataset(
        local_data_dir, local_data_dir / "annotations/val.json", transform
    )
    test_dataset = CIFAR10Dataset(
        local_data_dir, local_data_dir / "annotations/test.json", transform
    )

    # Исправляем типизацию для Subset
    train_ds: Any = train_dataset
    if config["data"].get("max_train_samples"):
        indices = list(range(config["data"]["max_train_samples"]))
        train_ds = torch.utils.data.Subset(train_dataset, indices)

    bs = config["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # 5. Модель
    model_type = config["model"].get("type", "simple_cnn")
    if model_type == "linear":
        model: nn.Module = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    elif model_type == "simple_cnn":
        model = SimpleCNN(channels=config["model"]["params"]["channels"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # 6. MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config["run_name"]):
        # Логируем git-коммит
        git_commit = get_git_commit()
        mlflow.log_param("git_commit", git_commit)

        mlflow.log_params(flatten_config(config))
        mlflow.log_param("data_version", config["data"]["s3_path"])
        mlflow.log_param("seed", config["seed"])

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_path = Path("best_model.pt")
        start_time = time.time()

        # Для сохранения матрицы ошибок val с лучшей эпохи
        best_val_preds = []
        best_val_labels = []
        best_val_acc = 0.0
        best_val_f1 = 0.0

        for epoch in range(config["train"]["epochs"]):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Валидация
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            val_loss /= len(val_loader)
            correct = sum(1 for p, t in zip(all_preds, all_labels, strict=False) if p == t)
            val_acc = correct / len(all_labels)
            val_f1 = f1_score(all_labels, all_preds, average="macro")

            print(
                f"Epoch {epoch+1}: val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
            )

            mlflow.log_metrics(
                {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}, step=epoch
            )

            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_val_preds = all_preds
                best_val_labels = all_labels
                best_val_acc = val_acc
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"Новая лучшая модель сохранена (loss={val_loss:.4f})")

        total_time = time.time() - start_time
        mlflow.log_metric("training_time_sec", total_time)
        mlflow.log_param("best_epoch", best_epoch)
        print(f"Обучение завершено за {total_time:.2f} сек.")

        # --- Матрица ошибок на VAL (для лучшей эпохи) ---
        print("Построение матрицы ошибок для val...")
        cm_val = confusion_matrix(best_val_labels, best_val_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Val - Best Epoch)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("confusion_matrix_val.png")
        mlflow.log_artifact("confusion_matrix_val.png")
        plt.close()

        # --- Оценка на тесте ---
        print("Оценка на тестовой выборке...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.eval()
        test_loss = 0.0
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                test_preds.extend(preds.tolist())
                test_labels.extend(labels.tolist())

        test_loss /= len(test_loader)
        correct_test = sum(1 for p, t in zip(test_preds, test_labels, strict=False) if p == t)
        test_acc = correct_test / len(test_labels)
        test_f1 = f1_score(test_labels, test_preds, average="macro")

        print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}")
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})

        # --- Сохранение в S3 ---
        s3_client = get_s3_client()
        s3_prefix = f"models/{config['experiment_name']}/{config['run_name']}"

        print(f"Загрузка модели в S3: {s3_prefix}")
        s3_client.upload_file(str(best_model_path), S3_BUCKET, f"{s3_prefix}/model.pt")
        s3_client.upload_file(args.config, S3_BUCKET, f"{s3_prefix}/config.yaml")

        # Сохраняем метрики в JSON
        metrics = {
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "val_f1": best_val_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        s3_client.upload_file("metrics.json", S3_BUCKET, f"{s3_prefix}/metrics.json")

        # Логируем артефакты в MLflow
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(args.config)


if __name__ == "__main__":
    main()
