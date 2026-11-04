"""Separate five-article Base smoke test using the shared experiment runner."""
import sys
from .experiment import main as experiment_main
from .inference import ROOT


def main():
    experiment_main([
        "--config", str(ROOT / "experiments/base_smoke.json"),
        "--output-dir", str(ROOT / "results/base_smoke"),
        "--verbose", *sys.argv[1:],
    ])


if __name__ == "__main__":
    main()
