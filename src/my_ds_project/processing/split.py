"""Модуль для разбиения train на train и val."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", default="data/processed/v1/annotations/train.json")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Нет файла {input_path}")

    with open(input_path) as f:
        data: dict[str, Any] = json.load(f)

    images = data["images"]
    classes = data["classes"]

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for img in images:
        by_class[img["label"]].append(img)

    rng = random.Random(args.seed)
    train_imgs = []
    val_imgs = []

    for _label, class_imgs in by_class.items():
        rng.shuffle(class_imgs)
        num_val = int(len(class_imgs) * (args.val_size / len(images)))
        if num_val == 0:
            num_val = 1
        val_imgs.extend(class_imgs[:num_val])
        train_imgs.extend(class_imgs[num_val:])

    rng.shuffle(train_imgs)
    rng.shuffle(val_imgs)

    with open(input_path.parent / "train.json", "w") as f:
        json.dump({"classes": classes, "images": train_imgs}, f, indent=2)
    with open(input_path.parent / "val.json", "w") as f:
        json.dump({"classes": classes, "images": val_imgs}, f, indent=2)

    print(f"Разбиение завершено: train={len(train_imgs)}, val={len(val_imgs)}")


if __name__ == "__main__":
    main()
