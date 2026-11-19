"""Taylor certificates for discrete moves on the fixed SYNTHETIC surrogate.

The result is mathematical only for the specified logistic mean with fixed
target vectors. It is not a certificate for a real LLM or for all prompt spaces.
"""
from __future__ import annotations

import json
import math

from experiment import (EPS, LENGTHS, TRAIN_TARGETS, VOCAB, embed_prompt,
                        enumerate_space, score, score_gradient)


def gradient_lipschitz_bound(targets=TRAIN_TARGETS):
    """Global Euclidean operator-norm upper bound on the objective Hessian.

    sigma''(z) = p(1-p)(1-2p), p=sigma(z), and
    sup_z |sigma''(z)| = sqrt(3)/18.
    For mean sigma(3*x dot u), L <= sqrt(3)/2 * mean ||u||^2.
    """
    if not targets:
        raise ValueError("targets must be nonempty")
    return math.sqrt(3) / 2 * sum(
        sum(float(component) ** 2 for component in target)
        for target in targets
    ) / len(targets)


def move_certificate(origin, destination, gradient, hessian_bound):
    """Return first-order estimate and a sound global Taylor interval.

    For d=destination-origin:
      |F(origin+d)-F(origin)-grad(F)(origin) dot d| <= L*||d||^2/2.
    This does NOT require the function to be convex.
    """
    if len(origin) != len(destination) or len(origin) != len(gradient):
        raise ValueError("dimension mismatch")
    if not math.isfinite(hessian_bound) or hessian_bound < 0:
        raise ValueError("invalid Hessian bound")
    delta = tuple(end - start for start, end in zip(origin, destination))
    estimate = sum(g * d for g, d in zip(gradient, delta))
    radius = hessian_bound * sum(d * d for d in delta) / 2
    return {"estimate": estimate, "radius": radius,
            "lower": estimate - radius, "upper": estimate + radius}


def audit_length(length):
    """Exhaustively evaluate ALL directed one-token replacements."""
    bound = gradient_lipschitz_bound()
    counts = {
        "moves": 0, "actual_improvements": 0, "predicted_improvements": 0,
        "first_order_false_positives": 0, "certified_improvements": 0,
        "certified_decreases": 0, "uncertified_moves": 0,
        "certificate_sign_errors": 0, "taylor_bound_violations": 0,
    }
    max_abs_remainder = 0.0
    for prompt in enumerate_space(length):
        origin = embed_prompt(prompt)
        origin_score = score(prompt)
        gradient = score_gradient(origin)
        for position, original in enumerate(prompt):
            for replacement in VOCAB:
                if replacement == original:
                    continue
                neighbor = prompt[:position] + (replacement,) + prompt[position + 1:]
                interval = move_certificate(
                    origin, embed_prompt(neighbor), gradient, bound
                )
                actual_change = score(neighbor) - origin_score
                remainder = abs(actual_change - interval["estimate"])
                max_abs_remainder = max(max_abs_remainder, remainder)
                counts["moves"] += 1
                predicted = interval["estimate"] > EPS
                actual = actual_change > EPS
                counts["predicted_improvements"] += int(predicted)
                counts["actual_improvements"] += int(actual)
                counts["first_order_false_positives"] += int(predicted and not actual)
                improves = interval["lower"] > EPS
                decreases = interval["upper"] < -EPS
                counts["certified_improvements"] += int(improves)
                counts["certified_decreases"] += int(decreases)
                counts["uncertified_moves"] += int(not (improves or decreases))
                counts["certificate_sign_errors"] += int(
                    (improves and not actual) or
                    (decreases and actual_change >= -EPS)
                )
                counts["taylor_bound_violations"] += int(
                    remainder > interval["radius"] + 1e-12
                )
    expected = len(VOCAB) ** length * length * (len(VOCAB) - 1)
    if counts["moves"] != expected:
        raise AssertionError("directed move enumeration incomplete")
    if counts["certificate_sign_errors"] or counts["taylor_bound_violations"]:
        raise AssertionError("certificate violated on an enumerated move")
    counts["length"] = length
    counts["certified_coverage"] = (
        counts["certified_improvements"] + counts["certified_decreases"]
    ) / expected
    counts["max_observed_abs_taylor_remainder"] = max_abs_remainder
    return counts


def build_report():
    bound = gradient_lipschitz_bound()
    length_reports = [audit_length(length) for length in LENGTHS]
    keys = ("moves", "actual_improvements", "predicted_improvements",
            "first_order_false_positives", "certified_improvements",
            "certified_decreases", "uncertified_moves",
            "certificate_sign_errors", "taylor_bound_violations")
    totals = {key: sum(report[key] for report in length_reports) for key in keys}
    totals["certified_coverage"] = (
        totals["certified_improvements"] + totals["certified_decreases"]
    ) / totals["moves"]
    return {
        "evidence_type": "synthetic_surrogate_not_LLM",
        "claim": "global_second_order_taylor_bound_on_fixed_logistic_surrogate",
        "improvement_threshold": EPS,
        "hessian_operator_norm_upper_bound": bound,
        "taylor_remainder_coefficient": bound / 2,
        "lengths": length_reports,
        "total": totals,
        "limitations": (
            "Exhaustive for token lengths 2/3/4 only; certification relies on a "
            "globally bounded synthetic Hessian and known fixed embeddings. "
            "No guarantee is asserted for a transformer model."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), sort_keys=True, indent=2, allow_nan=False))
