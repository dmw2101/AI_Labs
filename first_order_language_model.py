"""A second-order language model built from observed token triples.

Examples:
    python first_order_language_model.py
    python first_order_language_model.py --context the cat --seed 42
"""

import argparse
from collections import Counter
from pathlib import Path
import random


START = "<START>"
END = "<END>"


class SecondOrderLanguageModel:
    def __init__(self, sentences: list[list[str]]) -> None:
        """Learn P(next | previous two tokens) from tokenised sentences."""
        self.triple_counts: dict[tuple[str, str], Counter[str]] = {}

        for sentence in sentences:
            if not sentence:
                continue
            tokens = list(sentence)
            if tokens[:2] != [START, START]:
                if tokens[0] == START:
                    tokens.insert(0, START)
                else:
                    tokens[:0] = [START, START]
            if tokens[-1] != END:
                tokens.append(END)

            # Each key is a two-token context; its Counter stores third tokens.
            for index in range(2, len(tokens)):
                context = (tokens[index - 2], tokens[index - 1])
                following = tokens[index]
                counts = self.triple_counts.setdefault(context, Counter())
                counts[following] += 1

        if not self.triple_counts:
            raise ValueError("Training data must contain a nonempty sentence.")

        # P(next | previous two) = triple count / context's outgoing count.
        self.probabilities: dict[tuple[str, str], dict[str, float]] = {}
        for context, counts in self.triple_counts.items():
            total = sum(counts.values())
            self.probabilities[context] = {
                following: count / total
                for following, count in counts.items()
            }

    def distribution(self, context: tuple[str, str]) -> dict[str, float]:
        """Return P(next token | the supplied pair of previous tokens)."""
        if context not in self.probabilities:
            raise ValueError(f"No observed triples for context {context!r}.")
        return dict(self.probabilities[context])

    def display_probabilities(self, context: tuple[str, str]) -> None:
        for following, probability in sorted(self.distribution(context).items()):
            print(
                f"P({following} | {context[0]}, {context[1]}) "
                f"= {probability:.6f}"
            )
        print("All unlisted next tokens have probability 0 (no smoothing).")

    def predict_next(self, context: tuple[str, str]) -> str:
        """Return the most probable next token; break ties alphabetically."""
        distribution = self.distribution(context)
        return max(sorted(distribution), key=lambda token: distribution[token])

    def generate_sentence(
        self,
        seed: int | None = None,
        max_tokens: int = 100,
        mode: str = "sampling",
    ) -> list[str]:
        """Generate until END using greedy decoding or weighted sampling.

        Raise an error if the safety limit is reached before END, rather than
        silently returning an unfinished sentence.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if mode not in {"greedy", "sampling"}:
            raise ValueError("mode must be 'greedy' or 'sampling'.")

        rng = random.Random(seed)
        context = (START, START)
        sentence: list[str] = []
        seen_greedy_contexts: set[tuple[str, str]] = set()

        for _ in range(max_tokens):
            distribution = self.distribution(context)
            if mode == "greedy":
                if context in seen_greedy_contexts:
                    raise RuntimeError(
                        f"Greedy decoding entered a cycle at {context!r} "
                        f"before generating {END}."
                    )
                seen_greedy_contexts.add(context)
                following = self.predict_next(context)
            else:
                following = rng.choices(
                    list(distribution), weights=list(distribution.values()), k=1
                )[0]

            if following == END:
                return sentence
            sentence.append(following)
            context = (context[1], following)

        raise RuntimeError(f"No {END} generated within {max_tokens} samples.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).with_name("language_dataset.txt"),
        help="training file containing one tokenised sentence per line",
    )
    parser.add_argument(
        "--context",
        nargs=2,
        metavar=("PREVIOUS_2", "PREVIOUS_1"),
        default=[START, "the"],
        help="the two preceding tokens to inspect (default: <START> the)",
    )
    parser.add_argument("--seed", type=int, default=None, help="sampling seed")
    parser.add_argument(
        "--mode",
        choices=("greedy", "sampling"),
        default="sampling",
        help="next-token selection method (default: sampling)",
    )
    args = parser.parse_args()

    sentences = [
        line.split()
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = SecondOrderLanguageModel(sentences)
    context = tuple(args.context)
    model.display_probabilities(context)
    print("Most probable next token:", model.predict_next(context))
    try:
        sentence = model.generate_sentence(seed=args.seed, mode=args.mode)
        print(f"Generated sentence ({args.mode}):", " ".join(sentence))
    except RuntimeError as error:
        print(f"Could not generate a complete sentence ({args.mode}): {error}")


if __name__ == "__main__":
    main()
