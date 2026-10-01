"""Compare strict greedy generation with probabilistic sampling."""

from pathlib import Path

from first_order_language_model import SecondOrderLanguageModel


SENTENCE_COUNT = 5


def main() -> None:
    folder = Path(__file__).parent
    dataset_path = folder / "language_dataset.txt"
    output_path = folder / "generation_modes.txt"
    sentences = [
        line.split()
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = SecondOrderLanguageModel(sentences)
    report = [
        "Mode A: Greedy (always choose the most probable next token)",
        "The alphabetical tie-breaking rule makes greedy output deterministic:",
    ]

    for index in range(1, SENTENCE_COUNT + 1):
        try:
            sentence = model.generate_sentence(mode="greedy")
            report.append(f"{index}. {' '.join(sentence)}")
        except RuntimeError as error:
            report.append(f"{index}. No complete sentence: {error}")

    report.extend(
        [
            "",
            "Mode B: Sampling (five complete sampled sentences)",
        ]
    )
    for index in range(SENTENCE_COUNT):
        sentence = model.generate_sentence(seed=100 + index, mode="sampling")
        report.append(f"{index + 1}. {' '.join(sentence)}")

    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"), end="")
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
