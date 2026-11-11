"""Classify every directed single-token move on the synthetic training score."""
from collections import Counter

from experiment import EPS, LENGTHS, VOCAB, embed_prompt, enumerate_space, score, score_gradient


def audit(length):
    counts = Counter()
    examples = {}
    for prompt in enumerate_space(length):
        origin = embed_prompt(prompt)
        gradient = score_gradient(origin)
        value = score(prompt)
        for index, old in enumerate(prompt):
            for token in VOCAB:
                if token == old:
                    continue
                neighbor = prompt[:index] + (token,) + prompt[index + 1:]
                target = embed_prompt(neighbor)
                estimate = sum(g * (y - x) for g, x, y in zip(gradient, origin, target))
                change = score(neighbor) - value
                key = (estimate > EPS, change > EPS)
                counts[key] += 1
                examples.setdefault(key, (prompt, neighbor, estimate, change))
    expected = len(VOCAB) ** length * length * (len(VOCAB) - 1)
    assert sum(counts.values()) == expected
    return counts, examples


if __name__ == "__main__":
    print("length true_positive false_positive false_negative true_negative")
    for length in LENGTHS:
        counts, examples = audit(length)
        print(length, counts[True, True], counts[True, False],
              counts[False, True], counts[False, False])
        print("First false positive:", examples.get((True, False)))
