"""Скрипт для запуска серии экспериментов (Grid Search)."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def set_nested_value(config: dict[str, Any], keys: list[str], value: Any) -> None:
    """Устанавливает значение во вложенном словаре по списку ключей."""
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-config", default="configs/grid.yaml")
    args = parser.parse_args()

    with open(args.grid_config) as f:
        grid_config = yaml.safe_load(f)

    base_config_path = grid_config["base"]
    grid = grid_config["grid"]

    # Генерируем все комбинации параметров
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    print(f"Найдено {len(combinations)} комбинаций для запуска.")

    for i, combo in enumerate(combinations):
        # Загружаем базовый конфиг
        with open(base_config_path) as f:
            config = yaml.safe_load(f)

        # Обновляем конфиг значениями из комбинации
        combo_str = []
        for key, val in zip(keys, combo, strict=True):
            keys_path = key.split(".")
            set_nested_value(config, keys_path, val)
            combo_str.append(f"{key}={val}")

        # Формируем уникальное имя для запуска
        run_name = f"grid-{i+1}-" + "-".join(combo_str)
        config["run_name"] = run_name

        # Сохраняем временный конфиг
        temp_config_path = Path(f"configs/temp_{i}.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        print(f"\n--- Запуск {i+1}/{len(combinations)}: {run_name} ---")

        # Запускаем обучение напрямую через системный python (без uv)
        try:
            subprocess.run(
                [sys.executable, "-m", "my_ds_project.train", "--config", str(temp_config_path)],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"Ошибка в запуске {run_name}")

    print("\nВсе эксперименты завершены.")


if __name__ == "__main__":
    main()
