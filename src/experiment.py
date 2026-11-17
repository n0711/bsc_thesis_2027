"""Reproducible, self-contained *synthetic* prompt-search experiment.

NOT a language-model experiment. No network, model calls, or external datasets.
Python 3.10+ standard library only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

VOCAB = ("clear", "short", "formal", "example", "json", "verify")
EMBED = {
    "clear": (1.0, 0.0), "short": (0.6, -0.4),
    "formal": (0.3, 0.8), "example": (-0.3, 0.9),
    "json": (-0.8, 0.2), "verify": (0.8, 0.7),
}
# Independent fixed synthetic task vectors. These are NOT natural-language examples.
TRAIN_TARGETS = ((0.7, 0.2), (-0.1, 0.8), (0.3, 0.6), (-0.4, 0.1))
HOLDOUT_TARGETS = ((0.4, 0.3), (0.1, 0.9), (0.9, -0.1), (-0.2, 0.1))
LENGTHS = (2, 3, 4)
BUDGETS = (12, 24, 36, 72)
SEEDS = 50
EPS = 1e-12


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def embed_prompt(prompt: tuple[str, ...]) -> tuple[float, float]:
    if not prompt or any(token not in EMBED for token in prompt):
        raise ValueError("prompt must be nonempty and contain known vocabulary tokens")
    return tuple(sum((1 + 0.1 * i) * EMBED[token][j]
                     for i, token in enumerate(prompt)) / len(prompt)
                 for j in range(2))


def score(prompt: tuple[str, ...], targets=TRAIN_TARGETS) -> float:
    """Mean binary-class probability, not classification accuracy."""
    x = embed_prompt(prompt)
    return sum(sigmoid(3 * sum(a*b for a, b in zip(x, u)))
               for u in targets) / len(targets)


def score_gradient(x: tuple[float, float], targets=TRAIN_TARGETS):
    """Analytic gradient of the continuous surrogate score w.r.t. x."""
    ans = [0.0, 0.0]
    for u in targets:
        p = sigmoid(3 * sum(a*b for a, b in zip(x, u)))
        for j in range(2):
            ans[j] += 3 * u[j] * p * (1 - p) / len(targets)
    return tuple(ans)


def enumerate_space(length: int):
    if length < 1:
        raise ValueError("length must be positive")
    yield from itertools.product(VOCAB, repeat=length)


def exact_oracle(length: int):
    candidates = list(enumerate_space(length))
    ranked = [(score(p), p) for p in candidates]
    optimum = max(v for v, _ in ranked)
    ties = [p for v, p in ranked if abs(v - optimum) <= EPS]
    return {"score": optimum, "prompts": ties, "space_size": len(candidates),
            "scores": [v for v, _ in ranked]}


@dataclass(frozen=True)
class SearchResult:
    prompt: tuple[str, ...]
    train_score: float
    calls: int
    unique_candidates: int


def random_search(length: int, budget: int, seed: int) -> SearchResult:
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    best = None
    best_score = -1.0
    seen = set()
    for _ in range(budget):
        candidate = tuple(rng.choice(VOCAB) for _ in range(length))
        val = score(candidate)
        seen.add(candidate)
        if val > best_score + EPS:
            best, best_score = candidate, val
    return SearchResult(best, best_score, budget, len(seen))


def coordinate_search(length: int, budget: int, seed: int) -> SearchResult:
    """Restarting first-improvement hill climb with EXACTLY budget score calls.

    A new initial candidate consumes one call. Candidate substitutions are visited
    in seeded random order; improvements update the incumbent immediately. At a
    local maximum, restart. No score evaluations are uncounted. Unlike the prior
    draft, this algorithm uses its full allotted budget even after convergence.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    calls, seen = 0, set()
    overall, overall_score = None, -1.0
    current, current_score = None, -1.0
    neighbors, next_idx = [], 0
    while calls < budget:
        if current is None:
            current = tuple(rng.choice(VOCAB) for _ in range(length))
            current_score = score(current)
            calls += 1
            seen.add(current)
            if current_score > overall_score + EPS:
                overall, overall_score = current, current_score
            neighbors = []
            next_idx = 0
            continue
        if not neighbors:
            neighbors = [(i, t) for i in range(length) for t in VOCAB if t != current[i]]
            rng.shuffle(neighbors)
            next_idx = 0
        if next_idx >= len(neighbors):
            current = None  # sweep without improvement: new restart
            neighbors = []
            continue
        i, t = neighbors[next_idx]
        next_idx += 1
        candidate = current[:i] + (t,) + current[i + 1:]
        val = score(candidate)
        calls += 1
        seen.add(candidate)
        if val > overall_score + EPS:
            overall, overall_score = candidate, val
        if val > current_score + EPS:
            current, current_score = candidate, val
            neighbors = []
    return SearchResult(overall, overall_score, calls, len(seen))


def exact_random_optimal_probability(space_size: int, optimal_count: int, budget: int):
    """For iid uniformly sampled prompts *with replacement*."""
    if not (0 < optimal_count <= space_size and budget >= 1):
        raise ValueError("invalid probability parameters")
    if optimal_count == space_size:
        return 1.0
    miss_probability = Fraction(space_size - optimal_count, space_size) ** budget
    return float(1 - miss_probability)


def exact_random_expected_best(scores: list[float], budget: int):
    """Exact expectation of maximum score of `budget` iid uniform draws."""
    counts = Counter(scores)
    total = len(scores)
    cumulative = 0
    mean = 0.0
    for val in sorted(counts):
        prev = cumulative / total
        cumulative += counts[val]
        now = cumulative / total
        mean += val * (now ** budget - prev ** budget)
    return mean


def bootstrap_mean_interval(values: list[float], seed: int = 2027, nboot: int = 2000):
    """Descriptive 2.5--97.5% percentile interval over heuristic seeds only."""
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values)
                   for _ in range(nboot))
    return [boots[int(0.025*nboot)], boots[int(0.975*nboot)]]


def run(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "evidence_type": "synthetic_surrogate_not_LLM",
        "evaluation_protocol": "score calls matched exactly; random sampling with replacement; fresh heldout vectors",
        "vocab_size": len(VOCAB), "lengths": list(LENGTHS),
        "budgets": list(BUDGETS), "seeds_per_method": SEEDS,
        "train_target_count": len(TRAIN_TARGETS),
        "holdout_target_count": len(HOLDOUT_TARGETS), "experiments": [],
    }
    for length in LENGTHS:
        oracle = exact_oracle(length)
        for budget in BUDGETS:
            entry = {
                "length": length, "budget": budget,
                "search_space": oracle["space_size"],
                "oracle_train_score": oracle["score"],
                "oracle_prompt": list(oracle["prompts"][0]),
                "oracle_tie_count": len(oracle["prompts"]),
                "random_exact_optimum_hit_probability": exact_random_optimal_probability(
                    oracle["space_size"], len(oracle["prompts"]), budget),
                "random_exact_expected_best_score": exact_random_expected_best(oracle["scores"], budget),
                "methods": {},
            }
            for method_name, method in (("random", random_search), ("coordinate", coordinate_search)):
                block = []
                for seed in range(SEEDS):
                    outcome = method(length, budget, seed)
                    assert outcome.calls == budget
                    r = {
                        "length": length, "budget": budget, "method": method_name,
                        "seed": seed, "calls": outcome.calls,
                        "unique_candidates": outcome.unique_candidates,
                        "train_score": outcome.train_score,
                        "heldout_score": score(outcome.prompt, HOLDOUT_TARGETS),
                        "train_regret": max(0., oracle["score"] - outcome.train_score),
                        "optimal_hit": int(abs(outcome.train_score - oracle["score"]) <= EPS),
                        "prompt": " ".join(outcome.prompt),
                    }
                    rows.append(r)
                    block.append(r)
                regrets = [r["train_regret"] for r in block]
                entry["methods"][method_name] = {
                    "mean_train_score": sum(r["train_score"] for r in block) / SEEDS,
                    "mean_heldout_score": sum(r["heldout_score"] for r in block) / SEEDS,
                    "mean_regret": sum(regrets) / SEEDS,
                    "regret_bootstrap_95pct_seed_interval": bootstrap_mean_interval(
                        regrets, seed=2027 + length*100 + budget + (method_name=="coordinate")),
                    "optimal_hits": sum(r["optimal_hit"] for r in block),
                    "mean_unique_candidates": sum(r["unique_candidates"] for r in block) / SEEDS,
                }
            summary["experiments"].append(entry)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()
    result = run(args.out)
    print(f"Wrote {len(result['experiments'])} configurations and "
          f"{len(result['experiments'])*SEEDS*2} rows to {args.out}")


if __name__ == "__main__":
    main()
