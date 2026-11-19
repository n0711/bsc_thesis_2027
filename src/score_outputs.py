"""Offline scoring of *user-supplied* LLM prediction records.

No model invocations are made; if no prediction JSONL is supplied, no claim of
real-model performance is made. Exact-match is appropriate only for preselected
classification/extraction tasks with canonicalized expected outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REQUIRED = {"id", "task", "model", "prompt_id", "expected", "prediction",
            "output_valid", "input_tokens", "output_tokens", "latency_ms"}


def summarize(records: list[dict]):
    if not records:
        raise ValueError("empty record set")
    groups = defaultdict(list)
    seen = set()
    for idx, r in enumerate(records, start=1):
        missing = REQUIRED - set(r)
        if missing:
            raise ValueError(f"record {idx} missing: {sorted(missing)}")
        if any(not isinstance(r[name], str) or not r[name].strip()
               for name in ("id", "task", "model", "prompt_id")):
            raise ValueError(f"record {idx}: identifiers must be nonempty strings")
        if not isinstance(r["expected"], str) or not isinstance(r["prediction"], str):
            raise ValueError(f"record {idx}: expected and prediction must be strings")
        if type(r["output_valid"]) is not bool:
            raise ValueError(f"record {idx}: output_valid must be Boolean")
        if any(type(r[k]) is not int or r[k] < 0 for k in ("input_tokens", "output_tokens")):
            raise ValueError(f"record {idx}: token counts must be nonnegative integers")
        if (type(r["latency_ms"]) not in (int, float)
                or not math.isfinite(r["latency_ms"])
                or r["latency_ms"] < 0):
            raise ValueError(f"record {idx}: latency must be nonnegative")
        key = (r["model"], r["prompt_id"], r["task"], r["id"])
        if key in seen:
            raise ValueError(f"record {idx}: repeated task example ID within configuration")
        seen.add(key)
        groups[key[:3]].append(r)
    outcome = {}
    for key in sorted(groups):
        vals = groups[key]
        n = len(vals)
        outcome["/".join(key)] = {
            "n": n,
            "exact_match": sum(r["prediction"] == r["expected"] for r in vals) / n,
            "valid_output_rate": sum(r["output_valid"] for r in vals) / n,
            "mean_input_tokens": sum(r["input_tokens"] for r in vals) / n,
            "mean_output_tokens": sum(r["output_tokens"] for r in vals) / n,
            "mean_latency_ms": sum(r["latency_ms"] for r in vals) / n,
        }
    return {"evidence_type": "user_supplied_real_model_outputs_unverified_by_this_script",
            "groups": outcome, "n_records": len(records)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="User-provided JSONL of actual predictions")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.write_text(json.dumps(summarize(data), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
