"""Run and aggregate a reproducible multi-seed augmentation comparison.

Smoke check:
    python -m scripts.sweep_augmentations --epochs 1 --max-train-batches 2 --max-val-batches 1 --max-test-batches 1

Formal run:
    python -m scripts.sweep_augmentations
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION_CONFIGS = {
    "none": PROJECT_ROOT / "configs/experiment/cosine_no_augmentation.yaml",
    "crop": PROJECT_ROOT / "configs/experiment/cosine_crop_only.yaml",
    "crop_flip": PROJECT_ROOT / "configs/experiment/cosine_crop_flip.yaml",
}
SEEDS = (42, 123, 2024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


def run_command(arguments: list[str]) -> None:
    print("$", " ".join(arguments))
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def run_one(
    augmentation_name: str,
    config_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_name = f"{augmentation_name}_seed_{seed}"
    output_dir = PROJECT_ROOT / "outputs" / "augmentation_sweeps_formal" / run_name
    train_command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.epochs is not None:
        train_command += ["--epochs", str(args.epochs)]
    if args.max_train_batches is not None:
        train_command += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_val_batches is not None:
        train_command += ["--max-val-batches", str(args.max_val_batches)]
    run_command(train_command)

    evaluate_command = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(output_dir / "best.pt"),
    ]
    if args.max_test_batches is not None:
        evaluate_command += ["--max-test-batches", str(args.max_test_batches)]
    run_command(evaluate_command)

    with (output_dir / "test_report.json").open("r", encoding="utf-8") as file:
        report = json.load(file)
    return {
        "augmentation": augmentation_name,
        "seed": seed,
        "test_accuracy": report["overall_accuracy"],
        "samples": report["samples"],
        "output_dir": str(output_dir),
    }


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize each augmentation policy by mean, std, best, and worst seed."""
    summary: list[dict[str, object]] = []
    for augmentation_name in AUGMENTATION_CONFIGS:
        group = [record for record in records if record["augmentation"] == augmentation_name]
        if not group:
            continue
        scores = [float(record["test_accuracy"]) for record in group]
        best = max(group, key=lambda record: float(record["test_accuracy"]))
        worst = min(group, key=lambda record: float(record["test_accuracy"]))
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        summary.append(
            {
                "augmentation": augmentation_name,
                "seeds": len(scores),
                "mean_test_accuracy": mean,
                "std_test_accuracy": variance**0.5,
                "best_seed": best["seed"],
                "best_accuracy": best["test_accuracy"],
                "worst_seed": worst["seed"],
                "worst_accuracy": worst["test_accuracy"],
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    records = [
        run_one(augmentation, config_path, seed, args)
        for augmentation, config_path in AUGMENTATION_CONFIGS.items()
        for seed in args.seeds
    ]
    summary = aggregate(records)
    benchmark_dir = PROJECT_ROOT / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    with (benchmark_dir / "augmentation_seed_records.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (benchmark_dir / "augmentation_seed_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
