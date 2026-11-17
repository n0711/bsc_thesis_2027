"""Exhaustively compare gradient predictions with discrete synthetic moves."""
from collections import Counter
import json
import math
from pathlib import Path

from experiment import EPS, LENGTHS, VOCAB, embed_prompt, enumerate_space, score, score_gradient

ROOT = Path(__file__).resolve().parents[1]


def _average_ranks(values):
    """Return one-based midranks, grouping values within the audit tolerance."""
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] - ordered[start][1] <= EPS:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left, right):
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    covariance = sum((x - left_mean) * (y - right_mean)
                     for x, y in zip(left_ranks, right_ranks))
    left_variance = sum((x - left_mean) ** 2 for x in left_ranks)
    right_variance = sum((y - right_mean) ** 2 for y in right_ranks)
    if left_variance == 0 or right_variance == 0:
        return None
    return covariance / math.sqrt(left_variance * right_variance)


def _maxima(values):
    maximum = max(values)
    return {index for index, value in enumerate(values)
            if maximum - value <= EPS}


def audit(length):
    counts = Counter()
    examples = {}
    states = []
    for prompt in enumerate_space(length):
        origin = embed_prompt(prompt)
        gradient = score_gradient(origin)
        gradient_norm = math.sqrt(sum(component * component for component in gradient))
        value = score(prompt)
        moves = []
        for index, old in enumerate(prompt):
            for token in VOCAB:
                if token == old:
                    continue
                neighbor = prompt[:index] + (token,) + prompt[index + 1:]
                target = embed_prompt(neighbor)
                estimate = sum(g * (y - x)
                               for g, x, y in zip(gradient, origin, target))
                change = score(neighbor) - value
                predicted_improvement = estimate > EPS
                actual_improvement = change > EPS
                counts[predicted_improvement, actual_improvement] += 1
                examples.setdefault((predicted_improvement, actual_improvement),
                                    (prompt, neighbor, estimate, change))
                moves.append({
                    "neighbor": list(neighbor),
                    "predicted_change": estimate,
                    "actual_change": change,
                })

        estimates = [move["predicted_change"] for move in moves]
        changes = [move["actual_change"] for move in moves]
        predicted_best = _maxima(estimates)
        actual_best = _maxima(changes)
        maximum_change = max(changes)
        states.append({
            "prompt": list(prompt),
            "neighbor_count": len(moves),
            "top1_agreement": bool(predicted_best & actual_best),
            "predicted_top_neighbors": [
                moves[index]["neighbor"] for index in sorted(predicted_best)
            ],
            "actual_top_neighbors": [
                moves[index]["neighbor"] for index in sorted(actual_best)
            ],
            "spearman_rank_correlation": _spearman(estimates, changes),
            "max_discrete_change": maximum_change,
            "has_strictly_improving_neighbor": maximum_change > EPS,
            "gradient_norm": gradient_norm,
            "nonzero_gradient_without_improvement": (
                maximum_change <= EPS and gradient_norm > EPS
            ),
        })

    expected = len(VOCAB) ** length * length * (len(VOCAB) - 1)
    assert sum(counts.values()) == expected
    defined_correlations = [
        state["spearman_rank_correlation"] for state in states
        if state["spearman_rank_correlation"] is not None
    ]
    summary = {
        "length": length,
        "prompt_count": len(states),
        "directed_move_count": expected,
        "sign_counts": {
            "true_positive": counts[True, True],
            "false_positive": counts[True, False],
            "false_negative": counts[False, True],
            "true_negative": counts[False, False],
        },
        "global_sign_agreement_count": counts[True, True] + counts[False, False],
        "global_sign_agreement_rate": (
            (counts[True, True] + counts[False, False]) / expected
        ),
        "top1_agreement_count": sum(state["top1_agreement"] for state in states),
        "top1_agreement_rate": (
            sum(state["top1_agreement"] for state in states) / len(states)
        ),
        "mean_spearman_rank_correlation": (
            sum(defined_correlations) / len(defined_correlations)
            if defined_correlations else None
        ),
        "spearman_defined_state_count": len(defined_correlations),
        "nonimproving_nonzero_gradient_state_count": sum(
            state["nonzero_gradient_without_improvement"] for state in states
        ),
        "states": states,
        "first_false_positive": examples.get((True, False)),
    }
    return summary


def build_report():
    return {
        "schema_version": 1,
        "evidence_type": "synthetic_surrogate_not_LLM",
        "objective": "mean logistic training score on four fixed synthetic vectors",
        "improvement_tolerance": EPS,
        "ranking_ties": (
            "top-1 maxima within EPS are tied; Spearman ties group values within EPS"
        ),
        "lengths": [audit(length) for length in LENGTHS],
    }


def _latex(report):
    lines = [
        r"\section{Does the gradient identify improving token replacements?}",
        r"The continuous score gradient gives a directional derivative at the current",
        r"embedding. A token replacement takes a finite step. The audit evaluates every",
        r"directed one-token replacement on the fixed synthetic training objective.",
        r"",
        r"For each prompt, the five-$m$ immediate neighbours are ranked by the",
        r"first-order estimate and by the measured finite score change. Top-1 agreement",
        r"counts tied maxima as agreement when the two tied sets intersect. The mean",
        r"Spearman coefficient is computed across defined per-prompt rankings using",
        r"midranks, grouping rank ties within the audit tolerance; the table reports",
        r"the mean over prompts with a defined coefficient.",
        r"",
        r"\begin{center}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$m$ & Moves & Sign agreement & Top-1 & Mean Spearman & Terminal, nonzero $\nabla F$ \\",
        r"\midrule",
    ]
    totals = {
        "moves": 0,
        "agreements": 0,
        "top1": 0,
        "prompts": 0,
        "terminal_nonzero": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }
    for result in report["lengths"]:
        counts = result["sign_counts"]
        totals["moves"] += result["directed_move_count"]
        totals["agreements"] += result["global_sign_agreement_count"]
        totals["top1"] += result["top1_agreement_count"]
        totals["prompts"] += result["prompt_count"]
        totals["terminal_nonzero"] += result["nonimproving_nonzero_gradient_state_count"]
        for name in ("true_positive", "false_positive", "false_negative", "true_negative"):
            totals[name] += counts[name]
        lines.append(
            f"{result['length']} & {result['directed_move_count']} & "
            f"{result['global_sign_agreement_rate']:.4f} & "
            f"{result['top1_agreement_rate']:.4f} & "
            f"{result['mean_spearman_rank_correlation']:.4f} & "
            f"{result['nonimproving_nonzero_gradient_state_count']} "
            + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        (
            f"Across {totals['moves']:,} directed moves, sign agreement was "
            f"{totals['agreements'] / totals['moves']:.2%}. Per-state top-1 "
            f"agreement was {totals['top1'] / totals['prompts']:.2%}. Among "
            f"{totals['prompts']:,} prompt states, "
            f"{totals['terminal_nonzero']} had no strictly improving discrete "
            r"neighbour ($\max\Delta F\leq 10^{-12}$) despite a nonzero continuous "
            r"gradient (norm $>10^{-12}$). A nonzero gradient therefore does not "
            r"guarantee an improving move in the permitted discrete neighbourhood."
        ),
        "",
        (
            f"The sign confusion counts were {totals['true_positive']:,} true "
            f"positive, {totals['false_positive']:,} false positive, "
            f"{totals['false_negative']:,} false negative and "
            f"{totals['true_negative']:,} true negative. These exhaustive counts "
            r"apply only to the enumerated surrogate spaces and do not establish "
            r"the same behavior for longer prompts or language models."
        ),
        "",
        r"\paragraph{Reproduction.}",
        r"From the repository root run \texttt{python3 src/gradient\_audit.py}.",
        r"The script writes the per-state JSON results to",
        r"\texttt{results/gradient\_audit.json} and regenerates this table.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report, root=ROOT):
    results_dir = root / "results"
    tex_dir = root / "tex"
    results_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(
        _round_floats(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    (results_dir / "gradient_audit.json").write_text(
        json_text, encoding="utf-8", newline="\n"
    )
    (tex_dir / "gradient_moves.tex").write_text(
        _latex(report), encoding="utf-8", newline="\n"
    )


def _round_floats(value):
    if isinstance(value, float):
        rounded = round(value, 12)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def main():
    report = build_report()
    write_outputs(report)
    for result in report["lengths"]:
        print(
            result["length"],
            result["directed_move_count"],
            result["global_sign_agreement_rate"],
            result["top1_agreement_rate"],
            result["mean_spearman_rank_correlation"],
            result["nonimproving_nonzero_gradient_state_count"],
        )
    print("Wrote results/gradient_audit.json and tex/gradient_moves.tex")


if __name__ == "__main__":
    main()
