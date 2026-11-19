"""Executable referee for the fixed BSc thesis SYNTHETIC evidence archive.

Read-only: recomputes every stored prompt score, verifies budgets, keys, oracle
regrets, aggregate tables, and exploratory sign-test counts. Adds a Holm
multiple-testing sensitivity analysis and held-out differences. It cannot
prove the method represents real LLM prompting or detect unobserved leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from experiment import (BUDGETS, EPS, HOLDOUT_TARGETS, LENGTHS, SEEDS,
                        TRAIN_TARGETS, VOCAB, enumerate_space, score)


def approximately_equal(left, right, tolerance=1e-10):
    return math.isfinite(float(left)) and math.isfinite(float(right)) and (
        abs(float(left) - float(right)) <= tolerance
    )


def require_close(actual, reported, description):
    if not approximately_equal(actual, reported):
        raise ValueError(
            f"{description}: recomputed={actual!r}, reported={reported!r}"
        )


def exact_sign_test(wins, losses):
    """Two-sided exact binomial test conditional on non-tied signs."""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(
        math.comb(n, k) for k in range(min(wins, losses) + 1)
    ) / (2 ** n))


def holm_adjust(p_values):
    """Step-down Holm familywise adjustment, with monotone ordering."""
    if not p_values:
        return []
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise ValueError("p-values must be finite and lie in [0,1]")
    order = sorted(range(len(p_values)), key=lambda i: (p_values[i], i))
    answer = [0.0] * len(p_values)
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        answer[index] = running
    return answer


def audit(rows, summary, paired):
    """Raises on broken evidence; returns independent analysis for the archive."""
    if summary.get("evidence_type") != "synthetic_surrogate_not_LLM":
        raise ValueError("summary evidence type changed")
    if paired.get("evidence_type") != "synthetic_surrogate_not_LLM":
        raise ValueError("paired evidence type changed")
    if len(rows) != len(LENGTHS) * len(BUDGETS) * 2 * SEEDS:
        raise ValueError("unexpected number of raw search observations")

    # Re-enumerate the finite score space; do not trust archived oracle values.
    oracles = {
        length: max(score(prompt) for prompt in enumerate_space(length))
        for length in LENGTHS
    }
    records = {}
    groups = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        try:
            length = int(row["length"])
            budget = int(row["budget"])
            seed = int(row["seed"])
            method = row["method"]
            calls = int(row["calls"])
            unique_candidates = int(row["unique_candidates"])
            tokens = tuple(row["prompt"].split())
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"invalid row {index}") from err
        key = (length, budget, seed, method)
        if key in records:
            raise ValueError(f"duplicate (length, budget, seed, method): {key}")
        if (length not in LENGTHS or budget not in BUDGETS or
                seed not in range(SEEDS) or method not in ("random", "coordinate")):
            raise ValueError(f"unexpected configuration in row {index}")
        if calls != budget or not (1 <= unique_candidates <= budget):
            raise ValueError(f"invalid objective-call accounting in row {index}")
        if len(tokens) != length or any(token not in VOCAB for token in tokens):
            raise ValueError(f"invalid selected prompt in row {index}")
        train = score(tokens, TRAIN_TARGETS)
        heldout = score(tokens, HOLDOUT_TARGETS)
        regret = max(0.0, oracles[length] - train)
        hit = int(abs(oracles[length] - train) <= EPS)
        for name, computed in (
            ("train_score", train),
            ("heldout_score", heldout),
            ("train_regret", regret),
            ("optimal_hit", hit),
        ):
            require_close(computed, row[name], f"row {index} {name}")
        records[key] = {
            "train_score": train,
            "heldout_score": heldout,
            "train_regret": regret,
            "hit": hit,
            "unique_candidates": unique_candidates,
        }
        groups[length, budget, method].append(records[key])

    archived_summary = {
        (int(entry["length"]), int(entry["budget"])): entry
        for entry in summary["experiments"]
    }
    archived_paired = {
        (int(entry["length"]), int(entry["budget"])): entry
        for entry in paired["configurations"]
    }
    expected_keys = {(length, budget) for length in LENGTHS for budget in BUDGETS}
    if set(archived_summary) != expected_keys or set(archived_paired) != expected_keys:
        raise ValueError("missing, duplicate or unexpected configurations")

    results = []
    for length, budget in sorted(expected_keys):
        archive = archived_summary[length, budget]
        require_close(oracles[length], archive["oracle_train_score"],
                      f"oracle {length}/{budget}")
        method_means = {}
        for method in ("random", "coordinate"):
            group = groups[length, budget, method]
            if len(group) != SEEDS:
                raise ValueError(f"incomplete {length}/{budget}/{method} group")
            observed = {
                "mean_train_score": sum(r["train_score"] for r in group) / SEEDS,
                "mean_heldout_score": sum(r["heldout_score"] for r in group) / SEEDS,
                "mean_regret": sum(r["train_regret"] for r in group) / SEEDS,
                "mean_unique_candidates": sum(
                    r["unique_candidates"] for r in group
                ) / SEEDS,
                "optimal_hits": sum(r["hit"] for r in group),
            }
            for field, value in observed.items():
                require_close(value, archive["methods"][method][field],
                              f"summary {length}/{budget}/{method}/{field}")
            method_means[method] = observed

        train_diffs, heldout_diffs = [], []
        for seed in range(SEEDS):
            rnd = records[length, budget, seed, "random"]
            coord = records[length, budget, seed, "coordinate"]
            train_diffs.append(coord["train_score"] - rnd["train_score"])
            heldout_diffs.append(coord["heldout_score"] - rnd["heldout_score"])
        mean_train = sum(train_diffs) / SEEDS
        mean_heldout = sum(heldout_diffs) / SEEDS
        wins = sum(d > EPS for d in train_diffs)
        losses = sum(d < -EPS for d in train_diffs)
        ties = SEEDS - wins - losses
        old = archived_paired[length, budget]
        for key, val in (("pairs", SEEDS), ("wins", wins),
                         ("losses", losses), ("ties", ties)):
            if old[key] != val:
                raise ValueError(f"paired {length}/{budget}/{key}: counts differ")
        require_close(mean_train, old["mean_coordinate_minus_random"],
                      f"paired mean {length}/{budget}")
        p = exact_sign_test(wins, losses)
        require_close(p, old["two_sided_sign_p_unadjusted"],
                      f"paired sign p {length}/{budget}")
        results.append({
            "length": length,
            "budget": budget,
            "paired_seeds": SEEDS,
            "train_coordinate_minus_random": mean_train,
            "heldout_coordinate_minus_random": mean_heldout,
            "heldout_difference_positive": mean_heldout > 0,
            "train_sign_test_p_unadjusted_exploratory": p,
            "train_wins": wins, "train_losses": losses, "train_ties": ties,
        })

    adjusted = holm_adjust([
        row["train_sign_test_p_unadjusted_exploratory"] for row in results
    ])
    for row, p in zip(results, adjusted):
        row["train_sign_test_p_holm_exploratory"] = p

    return {
        "evidence_type": "synthetic_surrogate_not_LLM",
        "referee": "read_only_recalculation_of_existing_dataset",
        "record_count_verified": len(records),
        "configurations_checked": len(results),
        "all_training_mean_differences_positive": all(
            r["train_coordinate_minus_random"] > 0 for r in results
        ),
        "configurations_with_positive_heldout_difference": sum(
            r["heldout_difference_positive"] for r in results
        ),
        "holm_family_size": len(results),
        "holm_adjusted_train_sign_p_below_0_05": sum(
            r["train_sign_test_p_holm_exploratory"] < 0.05
            for r in results
        ),
        "configurations": results,
        "caution": (
            "Seed-level comparisons are exploratory on one fixed hand-chosen "
            "surrogate. Holm adjustment protects a stated 12-comparison family "
            "under the sign-test null assumptions; it does not make seeds "
            "independent real tasks. Held-out vectors are fixed rather than "
            "a sampled task population. No LLM inference was executed."
        ),
    }


def render_latex(report):
    """Build the manuscript's statistical audit table from verified raw rows."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{Executable referee audit of twelve synthetic configurations. "
        r"Each mean is coordinate minus random over 50 indexed seeds. "
        r"Holm values adjust the exploratory training-score sign tests.}",
        r"\label{tab:independent-referee}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$m$ & $B$ & Train difference & Held-out difference & Holm $p$ \\",
        r"\midrule",
    ]
    for r in report["configurations"]:
        lines.append(
            f'{r["length"]} & {r["budget"]} & '
            f'{r["train_coordinate_minus_random"]:+.6f} & '
            f'{r["heldout_coordinate_minus_random"]:+.6f} & '
            + r"\texttt{"
            + f'{r["train_sign_test_p_holm_exploratory"]:.2e}'
            + r"}\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-tex", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    with (root / "results/runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((root / "results/summary.json").read_text(
        encoding="utf-8"
    ))
    paired = json.loads((root / "results/paired_analysis.json").read_text(
        encoding="utf-8"
    ))
    report = audit(rows, summary, paired)
    if args.out_tex is not None:
        args.out_tex.write_text(render_latex(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
