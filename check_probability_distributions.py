"""Check that each learned next-token distribution sums to approximately 1."""

import argparse
import math
from pathlib import Path

from first_order_language_model import SecondOrderLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("language_dataset.txt"),
        help="training file containing one tokenised sentence per line",
    )
    args = parser.parse_args()

    sentences = [
        line.split()
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = SecondOrderLanguageModel(sentences)

    all_valid = True
    for previous, distribution in sorted(model.probabilities.items()):
        total = sum(distribution.values())
        valid = math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9)
        status = "OK" if valid else "ERROR"
        print(f"{previous}: {total:.12f} ({status})")
        all_valid = all_valid and valid

    if not all_valid:
        raise SystemExit("At least one distribution does not sum to 1.")
    print("All conditional distributions sum to approximately 1.")


if __name__ == "__main__":
    main()
