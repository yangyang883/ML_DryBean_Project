from __future__ import annotations

import argparse

from src.config import MODEL_NAMES, ensure_dirs
from src.eda import run_eda
from src.evaluate import evaluate_all
from src.robustness import run_robustness
from src.speed_test import run_speed_test
from src.train import train_all, train_model
from src.validation import run_sanity_check, validate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry Bean Dataset multi-class ML system")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["eda", "train", "train_all", "evaluate_all", "robustness", "speed", "validate", "all"],
        help="Pipeline step to run.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, help="Model used with --mode train.")
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()

    if args.mode == "eda":
        run_eda()
    elif args.mode == "train":
        if not args.model:
            raise SystemExit("--model is required when --mode train")
        train_model(args.model)
    elif args.mode == "train_all":
        train_all()
    elif args.mode == "evaluate_all":
        evaluate_all()
    elif args.mode == "robustness":
        run_robustness()
    elif args.mode == "speed":
        run_speed_test()
    elif args.mode == "validate":
        run_sanity_check()
        validate_metrics()
    elif args.mode == "all":
        run_eda()
        train_all()
        run_robustness()
        run_speed_test()
        evaluate_all()
        run_sanity_check()
        validate_metrics()
        print("[OK] Full pipeline finished.")


if __name__ == "__main__":
    main()
