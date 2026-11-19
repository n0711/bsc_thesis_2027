"""Frozen local-Ollama pilot runner. No synthetic benchmark data are imported.

The runner makes actual inference requests ONLY when the CLI 'run' command
is invoked against a configured local Ollama instance. CI supplies an
in-memory fake transport and cannot produce real-model evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from compare_heldout import compare
from score_outputs import summarize

SCHEMA_VERSION = 1
MODEL_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
TASK_KINDS = {"classification", "json_field"}
SPLITS = {"dev", "test"}
ALLOWED_SPEC = {
    "schema_version", "protocol_id", "dataset_source", "dataset_license",
    "model", "model_digest", "tasks", "prompts", "decoding",
}
ALLOWED_ROW = {"id", "task", "split", "input", "expected"}


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def object_hash(data):
    return digest(canonical(data).encode("utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == ALLOWED_SPEC,
            "spec must contain exactly the documented fields")
    require(type(spec["schema_version"]) is int and
            spec["schema_version"] == SCHEMA_VERSION, "unsupported spec version")
    for key in ("protocol_id", "dataset_source", "dataset_license", "model"):
        value = spec[key]
        require(isinstance(value, str) and len(value.strip()) >= 3 and
                "REPLACE" not in value.upper() and "EXAMPLE" not in value.upper(),
                f"configure a real {key} before freezing")
    require(isinstance(spec["model_digest"], str) and
            bool(MODEL_DIGEST.fullmatch(spec["model_digest"])),
            "model_digest must be sha256: plus 64 lowercase hex digits")
    tasks = spec["tasks"]
    require(isinstance(tasks, dict) and len(tasks) == 2,
            "pilot must declare two tasks")
    kinds = []
    for task_name, task_spec in tasks.items():
        require(isinstance(task_name, str) and task_name.strip() and
                isinstance(task_spec, dict), "invalid task")
        kind = task_spec.get("kind")
        require(kind in TASK_KINDS, "unsupported task kind")
        kinds.append(kind)
        if kind == "classification":
            require(set(task_spec) == {"kind", "labels"} and
                    isinstance(task_spec["labels"], list) and
                    2 <= len(task_spec["labels"]) <= 10 and
                    all(isinstance(s, str) and s and s == s.strip()
                        for s in task_spec["labels"]) and
                    len(set(task_spec["labels"])) == len(task_spec["labels"]),
                    "classification requires distinct nonempty labels")
        else:
            require(set(task_spec) == {"kind"}, "json_field takes no labels")
    require(set(kinds) == TASK_KINDS,
            "declare one classification and one json_field task")
    prompts = spec["prompts"]
    require(isinstance(prompts, dict) and set(prompts) ==
            {"author_written", "coordinate_selected"},
            "frozen baseline and development-selected prompt required")
    require(all(isinstance(p, str) and p.strip() and len(p) <= 8000
                for p in prompts.values()), "prompt bodies must be nonempty")
    decoding = spec["decoding"]
    require(isinstance(decoding, dict) and
            set(decoding) == {"temperature", "seed", "num_predict"},
            "decoding settings must be pinned")
    require(type(decoding["temperature"]) in (float, int) and
            decoding["temperature"] == 0, "pilot uses temperature zero")
    require(type(decoding["seed"]) is int and decoding["seed"] >= 0,
            "seed must be nonnegative integer")
    require(type(decoding["num_predict"]) is int and
            1 <= decoding["num_predict"] <= 1024,
            "num_predict must be between 1 and 1024")
    return spec


def read_dataset(path):
    raw = Path(path).read_bytes()
    require(raw.strip(), "empty dataset")
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        require(bool(line.strip()), f"blank JSONL line at {number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL line {number}") from exc
        rows.append(row)
    return rows, digest(raw)


def _unique_json_keys(pairs):
    """Reject ambiguous duplicate fields rather than silently using last key."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON object key")
        result[name] = value
    return result


def normalize(task_spec, response):
    """Return canonical answer and task-validity, without guessing meaning."""
    require(isinstance(response, str), "response must be a string")
    raw = response.strip()
    if task_spec["kind"] == "classification":
        return raw, raw in task_spec["labels"]
    try:
        result = json.loads(raw, object_pairs_hook=_unique_json_keys)
    except (json.JSONDecodeError, ValueError):
        return raw, False
    if (not isinstance(result, dict) or set(result) != {"value"} or
            not isinstance(result["value"], str)):
        return raw, False
    return canonical(result), True


def validate_rows(rows, spec):
    require(isinstance(rows, list) and rows, "dataset must contain records")
    ids = set()
    counts = {(task, split): 0 for task in spec["tasks"] for split in SPLITS}
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict) and set(row) == ALLOWED_ROW,
                f"invalid fields at dataset row {index}")
        require(isinstance(row["id"], str) and row["id"].strip() and
                isinstance(row["input"], str) and row["input"].strip() and
                isinstance(row["expected"], str),
                f"invalid data types at dataset row {index}")
        require(row["task"] in spec["tasks"] and row["split"] in SPLITS,
                f"undeclared task or split at dataset row {index}")
        require(row["id"] not in ids, f"duplicate or cross-split ID: {row['id']}")
        ids.add(row["id"])
        expected, valid = normalize(spec["tasks"][row["task"]],
                                    row["expected"])
        require(valid and expected == row["expected"],
                f"expected answer must be canonical and valid at row {index}")
        counts[row["task"], row["split"]] += 1
    require(all(n > 0 for n in counts.values()),
            "each task requires at least one dev and one test example")
    return counts


def partition_hashes(rows):
    return {
        split: object_hash(sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: row["id"]))
        for split in ("dev", "test")
    }


def freeze(spec, rows, dataset_sha256, timestamp=None):
    validate_spec(spec)
    validate_rows(rows, spec)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "protocol_freeze_not_model_evidence",
        "frozen_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        "spec": spec,
        "dataset_sha256": dataset_sha256,
        "split_sha256": partition_hashes(rows),
        "prompt_sha256": {
            key: digest(body.encode("utf-8"))
            for key, body in spec["prompts"].items()
        },
        "limitation": "File hashes cannot prove held-out answers were unseen.",
    }
    frozen["manifest_sha256"] = object_hash(frozen)
    return frozen


def verify_manifest(manifest, rows, dataset_sha256):
    require(isinstance(manifest, dict) and
            manifest.get("schema_version") == SCHEMA_VERSION,
            "unsupported manifest")
    claimed = manifest.get("manifest_sha256")
    require(isinstance(claimed, str) and len(claimed) == 64 and
            claimed == object_hash({k: v for k, v in manifest.items()
                                   if k != "manifest_sha256"}),
            "manifest checksum mismatch")
    spec = validate_spec(manifest["spec"])
    validate_rows(rows, spec)
    require(manifest["dataset_sha256"] == dataset_sha256,
            "dataset changed after freeze")
    require(manifest["split_sha256"] == partition_hashes(rows),
            "split commitment mismatch")
    require(manifest["prompt_sha256"] == {
        name: digest(body.encode("utf-8"))
        for name, body in spec["prompts"].items()
    }, "prompt commitment mismatch")
    return spec


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and
            parsed.port == 11434 and not parsed.username and
            not parsed.password and parsed.path in ("", "/") and
            not parsed.query and not parsed.fragment,
            "only local http://127.0.0.1:11434 is permitted")
    return "http://127.0.0.1:11434"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """No 3xx hops, especially off-loopback while submitting test inputs."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_json(url, payload=None, timeout=120):
    body = None if payload is None else canonical(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET")
    try:
        # Deliberately ignore environment HTTP proxies and deny redirects.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local Ollama request failed: {type(exc).__name__}") from exc


def verify_ollama_digest(spec, endpoint, get_json=http_json):
    response = get_json(endpoint + "/api/tags")
    require(isinstance(response, dict) and
            isinstance(response.get("models"), list),
            "invalid Ollama tags response")
    matches = [m for m in response["models"]
               if isinstance(m, dict) and m.get("name") == spec["model"]]
    require(len(matches) == 1 and
            matches[0].get("digest") == spec["model_digest"],
            "requested model missing or its digest changed")
    return True


def _actual_transport(spec, task_spec, body, endpoint):
    payload = {
        "model": spec["model"],
        "system": body["system"],
        "prompt": body["input"],
        "stream": False,
        "options": spec["decoding"],
    }
    if task_spec["kind"] == "json_field":
        payload["format"] = "json"
    return http_json(endpoint + "/api/generate", payload)


def run_evaluation(manifest, rows, dataset_sha256, split,
                   transport=None, endpoint="http://127.0.0.1:11434"):
    """Run frozen baseline and candidate on same IDs; abort on missing calls."""
    spec = verify_manifest(manifest, rows, dataset_sha256)
    require(split in SPLITS, "split must be dev or test")
    selected = sorted((row for row in rows if row["split"] == split),
                      key=lambda row: (row["task"], row["id"]))
    real_call = transport is None
    if real_call:
        endpoint = validate_endpoint(endpoint)
        verify_ollama_digest(spec, endpoint)
    responses = []
    for prompt_id in ("author_written", "coordinate_selected"):
        for row in selected:
            request = {
                "system": spec["prompts"][prompt_id],
                "input": row["input"],
            }
            start = time.perf_counter()
            response = (_actual_transport(
                spec, spec["tasks"][row["task"]], request, endpoint)
                if real_call else transport(spec, row, prompt_id))
            latency_ms = (time.perf_counter() - start) * 1000
            require(isinstance(response, dict) and
                    response.get("model") == spec["model"] and
                    response.get("done") is True and
                    isinstance(response.get("response"), str),
                    "incomplete or wrong-model generation response")
            for key in ("prompt_eval_count", "eval_count"):
                require(type(response.get(key)) is int and
                        response[key] >= 0,
                        f"missing/inconsistent Ollama token count {key}")
            prediction, valid = normalize(
                spec["tasks"][row["task"]], response["response"])
            responses.append({
                "id": row["id"], "task": row["task"],
                "model": spec["model"], "prompt_id": prompt_id,
                "expected": row["expected"], "prediction": prediction,
                "output_valid": valid,
                "input_tokens": response["prompt_eval_count"],
                "output_tokens": response["eval_count"],
                "latency_ms": latency_ms,
                "split": split,
            })
    require(len(responses) == len(selected) * 2,
            "partial run cannot become an evaluation result")
    result = compare(responses, "author_written", "coordinate_selected")
    summary = summarize(responses)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": ("local_model_runner_outputs_not_independently_verified"
                          if real_call else "mock_transport_fixture_not_real_model"),
        "protocol_id": spec["protocol_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": dataset_sha256,
        "split_sha256": manifest["split_sha256"][split],
        "prompt_sha256": manifest["prompt_sha256"],
        "model": spec["model"],
        "model_digest": spec["model_digest"],
        "decoding": spec["decoding"],
        "split": split,
        "n_predictions": len(responses),
        "endpoint": endpoint if real_call else "in_memory_test_double",
        "warning": ("This does not independently prove model provenance, "
                    "test-set blindness or generalization."),
    }
    return responses, summary, result, provenance


def write_outputs(path, responses, summary, comparison, provenance):
    """Refuse to replace an existing run and keep real evidence untracked."""
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    (path / "predictions.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in responses), encoding="utf-8")
    for filename, item in (
        ("scores.json", summary), ("paired.json", comparison),
        ("run_manifest.json", provenance)
    ):
        (path / filename).write_text(
            json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Freeze/verify/run strictly local Ollama pilot evidence")
    sub = parser.add_subparsers(dest="action", required=True)
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--dataset", type=Path, required=True)
    p_freeze.add_argument("--spec", type=Path, required=True)
    p_freeze.add_argument("--manifest", type=Path, required=True)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--dataset", type=Path, required=True)
    p_verify.add_argument("--manifest", type=Path, required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", type=Path, required=True)
    p_run.add_argument("--manifest", type=Path, required=True)
    p_run.add_argument("--split", choices=("dev", "test"), required=True)
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument("--endpoint", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    records, data_sha = read_dataset(args.dataset)
    if args.action == "freeze":
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        frozen = freeze(spec, records, data_sha)
        with args.manifest.open("x", encoding="utf-8") as output:
            output.write(json.dumps(frozen, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
        print("Created frozen manifest; this is not a model evaluation.")
    else:
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        verify_manifest(frozen, records, data_sha)
        if args.action == "verify":
            print("Manifest, prompt, dataset and split commitments verified.")
        else:
            prediction, scores, paired, metadata = run_evaluation(
                frozen, records, data_sha, args.split,
                endpoint=args.endpoint)
            write_outputs(args.output_dir, prediction, scores, paired, metadata)
            print(f"Recorded {len(prediction)} inference outputs to "
                  f"{args.output_dir}; verify provenance independently.")


if __name__ == "__main__":
    main()
