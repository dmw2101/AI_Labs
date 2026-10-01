"""Generate and save 20 sentences from the first-order language model."""

from pathlib import Path

from first_order_language_model import SecondOrderLanguageModel


def main() -> None:
    folder = Path(__file__).parent
    dataset_path = folder / "language_dataset.txt"
    output_path = folder / "generated_sentences.txt"

    sentences = [
        line.split()
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = SecondOrderLanguageModel(sentences)

    # Use a different fixed seed for each sample, making the result
    # reproducible while still sampling from the learned distributions.
    generated = [
        model.generate_sentence(seed=42 + index)
        for index in range(20)
    ]

    output = "\n".join(
        f"{index}. {' '.join(sentence)}"
        for index, sentence in enumerate(generated, start=1)
    )
    output_path.write_text(output + "\n", encoding="utf-8")
    print(f"Saved {len(generated)} sentences to {output_path}")


if __name__ == "__main__":
    main()
