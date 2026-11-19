"""Exact finite-space transfer audit for the FIXED synthetic objectives.

No real model or statistical population is involved. This is a
deterministic mathematical comparison between two prescribed sets of
two-dimensional target vectors; not a claim about natural-language transfer.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from experiment import (EMBED, EPS, HOLDOUT_TARGETS, LENGTHS, TRAIN_TARGETS,
                        enumerate_space, embed_prompt, score)

ROOT = Path(__file__).resolve().parents[1]


def target_pairing_distance(train=TRAIN_TARGETS, heldout=HOLDOUT_TARGETS):
    """Smallest mean Euclidean distance over all equal-size target pairings.

    The mean-score objective is invariant under reordering targets.
    Exhaustive assignment is used only for the four-target research fixture.
    """
    if not train or len(train) != len(heldout):
        raise ValueError("same positive target count required")
    if len(train) > 8:
        raise ValueError("this exact pairing helper supports at most 8 targets")
    if any(len(a) != len(b) for a in train for b in heldout):
        raise ValueError("target dimension mismatch")
    return min(
        sum(math.dist(a, b) for a, b in zip(train, ordering)) / len(train)
        for ordering in itertools.permutations(heldout)
    )


def uniform_embedding_radius(length):
    """Triangle-inequality norm bound for the weighted embedding average."""
    if length < 1:
        raise ValueError("positive prompt length required")
    max_token_norm = max(math.sqrt(sum(v * v for v in vector))
                         for vector in EMBED.values())
    mean_position_weight = sum(1 + 0.1 * i for i in range(length)) / length
    return mean_position_weight * max_token_norm


def audit_length(length):
    """Exhaustive objectives, regrets, and pairwise ordering certificates."""
    prompts = list(enumerate_space(length))
    values = [(p, score(p, TRAIN_TARGETS), score(p, HOLDOUT_TARGETS))
              for p in prompts]
    n = len(values)
    train_oracle = max(record[1] for record in values)
    heldout_oracle = max(record[2] for record in values)
    train_best = next(r for r in values if abs(r[1] - train_oracle) <= EPS)
    heldout_best = next(r for r in values if abs(r[2] - heldout_oracle) <= EPS)

    gaps = [heldout - train for _, train, heldout in values]
    gap_min, gap_max = min(gaps), max(gaps)
    gap_oscillation = gap_max - gap_min
    uniform_gap_bound = (
        0.75 * uniform_embedding_radius(length) *
        target_pairing_distance()
    )
    # |sigma(a)-sigma(b)| <= |a-b|/4 implies this global upper bound
    # on |S_H(p)-S_T(p)| for every fixed discrete prompt p.
    if max(abs(gap_min), abs(gap_max)) > uniform_gap_bound + 1e-12:
        raise AssertionError("analytic target-shift gap bound violated")

    max_regret_difference = 0.0
    for _, t, h in values:
        t_regret = train_oracle - t
        h_regret = heldout_oracle - h
        max_regret_difference = max(max_regret_difference,
                                    abs(h_regret - t_regret))
        if abs(h_regret - t_regret) > gap_oscillation + 1e-12:
            raise AssertionError("regret transfer theorem violated")

    comparable = reversals = certified = certificate_errors = 0
    for index, (_, t1, h1) in enumerate(values):
        for _, t2, h2 in values[index+1:]:
            train_difference = t1 - t2
            if abs(train_difference) <= EPS:
                continue
            comparable += 1
            heldout_difference = h1 - h2
            if train_difference * heldout_difference < 0 and (
                    abs(heldout_difference) > EPS):
                reversals += 1
            if abs(train_difference) > gap_oscillation + EPS:
                certified += 1
                if train_difference * heldout_difference <= 0:
                    certificate_errors += 1
    if certificate_errors:
        raise AssertionError("finite-pair ordering certificate failed")
    return {
        "length": length,
        "candidate_count": n,
        "comparable_unordered_pairs": comparable,
        "observed_reversed_order_pairs": reversals,
        "certified_order_preserving_pairs": certified,
        "certified_order_errors": certificate_errors,
        "training_oracle_score": train_oracle,
        "heldout_oracle_score": heldout_oracle,
        "training_oracle_prompt": list(train_best[0]),
        "heldout_oracle_prompt": list(heldout_best[0]),
        "heldout_regret_of_training_oracle": heldout_oracle - train_best[2],
        "training_regret_of_heldout_oracle": train_oracle - heldout_best[1],
        "maximum_absolute_score_gap": max(abs(gap_min), abs(gap_max)),
        "score_gap_minimum": gap_min,
        "score_gap_maximum": gap_max,
        "exact_gap_oscillation": gap_oscillation,
        "analytic_uniform_gap_upper_bound": uniform_gap_bound,
        "maximum_absolute_regret_difference": max_regret_difference,
    }


def build_report():
    reports = [audit_length(length) for length in LENGTHS]
    return {
        "schema_version": 1,
        "evidence_type": "synthetic_surrogate_not_LLM",
        "claim": "exact_target_shift_and_regret_transfer_on_finite_space",
        "target_pairing_mean_euclidean_distance": target_pairing_distance(),
        "lengths": reports,
        "limitations": (
            "The two target sets are hand-picked, fixed and not iid samples "
            "from real-world tasks. Certificates are for these enumerated "
            "prompts and this logistic surrogate only. Not evidence of "
            "language-model prompt generalization."
        ),
    }


def latex_table(report):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Exact synthetic target-shift audit. 'Ordered' counts "
        r"unordered prompt pairs with distinct development scores. "
        r"'Cert.' counts provably preserved development orderings; "
        r"'Reversed' counts observed held-out rank inversions.}",
        r"\label{tab:target-transfer}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$m$ & $|P_m|$ & Gap span & Train-oracle $r_H$ & Ordered & Cert. & Reversed \\",
        r"\midrule",
    ]
    for result in report["lengths"]:
        lines.append(
            f'{result["length"]} & {result["candidate_count"]} & '
            f'{result["exact_gap_oscillation"]:.6f} & '
            f'{result["heldout_regret_of_training_oracle"]:.6f} & '
            f'{result["comparable_unordered_pairs"]} & '
            f'{result["certified_order_preserving_pairs"]} & '
            f'{result["observed_reversed_order_pairs"]}' + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _round_floats(value):
    if isinstance(value, float):
        val = round(value, 12)
        return 0.0 if val == 0.0 else val
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    return value


def write_outputs(report, root=ROOT):
    root = Path(root)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "tex").mkdir(parents=True, exist_ok=True)
    (root / "results/target_transfer_audit.json").write_text(
        json.dumps(_round_floats(report), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    (root / "tex/target_transfer_audit.tex").write_text(
        latex_table(report), encoding="utf-8", newline="\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = build_report()
    write_outputs(report, root=args.root)
    for result in report["lengths"]:
        print(result["length"], "candidates", result["candidate_count"],
              "heldout_regret", result["heldout_regret_of_training_oracle"],
              "certified_pairs", result["certified_order_preserving_pairs"],
              "reversals", result["observed_reversed_order_pairs"])


if __name__ == "__main__":
    main()
