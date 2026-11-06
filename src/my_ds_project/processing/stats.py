"""Модуль для подсчета статистики нормализации."""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default="data/processed/v1/annotations/train.json")
    parser.add_argument("--images-dir", default="data/processed/v1")
    parser.add_argument("--output", default="data/processed/v1/stats.json")
    args = parser.parse_args()

    with open(args.train_json) as f:
        data = json.load(f)

    # Инициализируем переменные ДО цикла
    sum_pixels: np.ndarray = np.zeros((3,), dtype=np.float64)
    sum_sq_pixels: np.ndarray = np.zeros((3,), dtype=np.float64)
    count_pixels: int = 0

    print(f"Обработка {len(data['images'])} изображений...")
    for img_meta in data["images"]:
        img_path = Path(args.images_dir) / img_meta["file_name"]
        img = Image.open(img_path).convert("RGB")

        raw_arr = np.array(img)
        arr = cast(np.ndarray, raw_arr / 255.0)

        # Считаем сумму и сумму квадратов для картинки
        img_sum: np.ndarray = np.sum(arr, axis=(0, 1))
        img_sq_sum: np.ndarray = np.sum(arr**2, axis=(0, 1))

        # Прибавляем к общим суммам
        sum_pixels = np.add(sum_pixels, img_sum)
        sum_sq_pixels = np.add(sum_sq_pixels, img_sq_sum)
        count_pixels += int(arr.shape[0] * arr.shape[1])

    # Считаем среднее и стандартное отклонение
    mean: np.ndarray = sum_pixels / count_pixels
    std: np.ndarray = np.sqrt((sum_sq_pixels / count_pixels) - (mean**2))

    stats = {"mean": mean.tolist(), "std": std.tolist(), "channels": ["R", "G", "B"]}

    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Статистика сохранена в {args.output}")
    print(f"mean: {mean}")
    print(f"std:  {std}")


if __name__ == "__main__":
    main()
