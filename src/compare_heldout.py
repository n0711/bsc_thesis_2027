"""Paired held-out analysis for externally provided model predictions.

This script NEVER invokes a model or fetches data. It checks that the
baseline and candidate were evaluated on IDENTICAL example IDs with the
same reference outputs. Its output does not authenticate prediction origin
or prove the test split was not used during prompt development.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from score_outputs import summarize


def exact_discordance_p(wins, losses):
    """Conditional exact two-sided McNemar/binomial p-value."""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(
        math.comb(n, k) for k in range(min(wins, losses) + 1)
    ) / 2**n)


def compare(records, baseline_id, candidate_id):
    if not baseline_id or not candidate_id or baseline_id == candidate_id:
        raise ValueError("distinct nonempty baseline and candidate IDs required")
    # Reuse the offline scorer's schema/type/duplicate validation. It does
    # NOT authenticate the provider or run a model.
    summarize(records)
    groups = defaultdict(lambda: {"baseline": {}, "candidate": {}})
    for row in records:
        prompt_id = row["prompt_id"]
        if prompt_id not in (baseline_id, candidate_id):
            raise ValueError("unexpected prompt ID in input")
        if not all(isinstance(row[name], str) and row[name]
                   for name in ("model", "task", "id", "prompt_id")):
            raise ValueError("model/task/example/prompt identifiers must be nonempty strings")
        key = (row["model"], row["task"])
        label = "baseline" if prompt_id == baseline_id else "candidate"
        example = row["id"]
        groups[key][label][example] = row

    comparisons = []
    for (model, task), pairs in sorted(groups.items()):
        baseline, candidate = pairs["baseline"], pairs["candidate"]
        if not baseline or not candidate or set(baseline) != set(candidate):
            raise ValueError(f"unmatched examples for model={model}, task={task}")
        n = len(baseline)
        wins = losses = ties = 0
        sums = {"base_correct": 0, "candidate_correct": 0,
                "base_exact": 0, "candidate_exact": 0,
                "base_valid": 0, "candidate_valid": 0,
                "base_input": 0, "candidate_input": 0,
                "base_output": 0, "candidate_output": 0,
                "base_latency": 0., "candidate_latency": 0.}
        for ex_id in sorted(baseline):
            a, b = baseline[ex_id], candidate[ex_id]
            if a["expected"] != b["expected"]:
                raise ValueError(f"different reference outputs for {model}/{task}/{ex_id}")
            ea, eb = int(a["prediction"] == a["expected"]), int(b["prediction"] == b["expected"])
            ca, cb = int(bool(a["output_valid"]) and ea), int(bool(b["output_valid"]) and eb)
            sums["base_correct"] += ca
            sums["candidate_correct"] += cb
            sums["base_exact"] += ea
            sums["candidate_exact"] += eb
            sums["base_valid"] += int(a["output_valid"])
            sums["candidate_valid"] += int(b["output_valid"])
            sums["base_input"] += a["input_tokens"]
            sums["candidate_input"] += b["input_tokens"]
            sums["base_output"] += a["output_tokens"]
            sums["candidate_output"] += b["output_tokens"]
            sums["base_latency"] += a["latency_ms"]
            sums["candidate_latency"] += b["latency_ms"]
            wins += int(cb > ca)
            losses += int(cb < ca)
            ties += int(cb == ca)
        comparisons.append({
            "model": model, "task": task, "n_paired_examples": n,
            "baseline_valid_exact_match": sums["base_correct"] / n,
            "candidate_valid_exact_match": sums["candidate_correct"] / n,
            "paired_valid_exact_match_difference": (
                sums["candidate_correct"] - sums["base_correct"]
            ) / n,
            "baseline_plain_exact_match": sums["base_exact"] / n,
            "candidate_plain_exact_match": sums["candidate_exact"] / n,
            "valid_output_rate_difference": (
                sums["candidate_valid"] - sums["base_valid"]
            ) / n,
            "input_tokens_mean_difference": (
                sums["candidate_input"] - sums["base_input"]
            ) / n,
            "output_tokens_mean_difference": (
                sums["candidate_output"] - sums["base_output"]
            ) / n,
            "latency_ms_mean_difference": (
                sums["candidate_latency"] - sums["base_latency"]
            ) / n,
            "candidate_only_correct": wins,
            "baseline_only_correct": losses,
            "both_same_correctness": ties,
            "exact_discordance_p_exploratory": exact_discordance_p(wins, losses),
        })
    return {
        "evidence_type": "externally_supplied_model_predictions_origin_unverified",
        "baseline_prompt_id": baseline_id,
        "candidate_prompt_id": candidate_id,
        "comparisons": comparisons,
        "limitations": (
            "Offline scorer does not call a model or verify origins, split "
            "integrity, prompt freezing, decoding reproducibility, or "
            "population representativeness. Statistical p-values are "
            "exploratory unless sampling assumptions and multiplicity are "
            "specified in advance."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    data = [json.loads(line) for line in args.input.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    result = compare(data, args.baseline_id, args.candidate_id)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
