"""Enumerate one-substitution stopping points on the fixed synthetic objective."""
from experiment import EPS, LENGTHS, VOCAB, enumerate_space, score


def landscape(length):
    scores = {prompt: score(prompt) for prompt in enumerate_space(length)}
    optimum = max(scores.values())
    terminal = []
    comparisons = 0
    for prompt, value in scores.items():
        improves = False
        for index, old in enumerate(prompt):
            for token in VOCAB:
                if old == token:
                    continue
                neighbor = prompt[:index] + (token,) + prompt[index + 1:]
                comparisons += 1
                if scores[neighbor] > value + EPS:
                    improves = True
        if not improves:
            terminal.append(prompt)
    assert comparisons == len(scores) * length * (len(VOCAB) - 1)
    optimal = {p for p, v in scores.items() if abs(v - optimum) <= EPS}
    return len(scores), comparisons, terminal, set(terminal) - optimal


if __name__ == "__main__":
    for length in LENGTHS:
        size, comparisons, terminal, suboptimal = landscape(length)
        print(length, size, comparisons, len(terminal), len(suboptimal), terminal)
