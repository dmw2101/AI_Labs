"""Display next-token distributions and top predictions for five contexts."""

from pathlib import Path

from first_order_language_model import SecondOrderLanguageModel


def main() -> None:
    dataset_path = Path(__file__).with_name("language_dataset.txt")
    sentences = [
        line.split()
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = SecondOrderLanguageModel(sentences)

    contexts = (
        ("<START>", "the"),
        ("the", "cat"),
        ("the", "dog"),
        ("sat", "on"),
        ("on", "the"),
    )
    for context in contexts:
        print(f"\nPrevious words: {context[0]} {context[1]}")
        for following, probability in sorted(model.distribution(context).items()):
            print(
                f"  P({following} | {context[0]}, {context[1]}) "
                f"= {probability:.6f}"
            )
        print("  Most probable next word:", model.predict_next(context))


if __name__ == "__main__":
    main()
