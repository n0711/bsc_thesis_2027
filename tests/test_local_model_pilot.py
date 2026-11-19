"""Local-model protocol tests use in-memory fixtures, never an LLM."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_model_pilot import (
    _actual_transport, canonical, digest, freeze, normalize, object_hash, partition_hashes,
    read_dataset, run_evaluation, validate_endpoint, validate_rows,
    validate_spec, verify_manifest, verify_ollama_digest, write_outputs,
)


def fixture_spec():
    return {
        "schema_version": 1,
        "protocol_id": "controlled-fixture-001",
        "dataset_source": "handwritten-test-fixtures-only",
        "dataset_license": "internal-fixture-not-experimental-data",
        "model": "fake-local:1",
        "model_digest": "sha256:" + "a" * 64,
        "tasks": {
            "sentiment": {"kind": "classification", "labels": ["yes", "no"]},
            "field": {"kind": "json_field"},
        },
        "prompts": {
            "author_written": "Answer using the required format.",
            "coordinate_selected": "Return one exact valid answer.",
        },
        "decoding": {"temperature": 0, "seed": 2027, "num_predict": 64},
    }


def fixture_rows():
    return [
        {"id": "d1", "task": "sentiment", "split": "dev",
         "input": "A training fixture", "expected": "yes"},
        {"id": "d2", "task": "field", "split": "dev",
         "input": "Value is AA", "expected": '{"value":"AA"}'},
        {"id": "t1", "task": "sentiment", "split": "test",
         "input": "A held-out fixture", "expected": "no"},
        {"id": "t2", "task": "field", "split": "test",
         "input": "Value is BB", "expected": '{"value":"BB"}'},
    ]


def mock_transport(spec, row, prompt_id):
    raw = row["expected"]
    if row["task"] == "sentiment" and prompt_id == "author_written":
        raw = "yes" if raw == "no" else "no"
    if row["task"] == "field" and prompt_id == "author_written":
        raw = "not valid json"
    return {
        "model": spec["model"],
        "done": True,
        "response": raw,
        "prompt_eval_count": 11,
        "eval_count": 3,
    }


class LocalModelPilotTests(unittest.TestCase):
    def setUp(self):
        self.spec = fixture_spec()
        self.rows = fixture_rows()
        self.dataset_sha = object_hash(self.rows)
        self.manifest = freeze(self.spec, self.rows, self.dataset_sha,
                               timestamp="2026-11-19T00:00:00+00:00")

    def test_frozen_manifest_and_split_hashes_are_deterministic(self):
        self.assertEqual(
            self.manifest,
            freeze(self.spec, self.rows, self.dataset_sha,
                   timestamp="2026-11-19T00:00:00+00:00")
        )
        self.assertNotEqual(
            self.manifest["split_sha256"]["dev"],
            self.manifest["split_sha256"]["test"]
        )
        self.assertEqual(verify_manifest(self.manifest, self.rows,
                                         self.dataset_sha), self.spec)
        self.assertEqual(set(partition_hashes(self.rows)), {"dev", "test"})

    def test_modified_dataset_or_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            verify_manifest(self.manifest, self.rows, "0" * 64)
        altered = copy.deepcopy(self.manifest)
        altered["spec"]["prompts"]["coordinate_selected"] = "altered"
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_manifest(altered, self.rows, self.dataset_sha)

    def test_recomputed_manifest_checksum_does_not_hide_modified_data(self):
        altered = copy.deepcopy(self.rows)
        altered[0]["input"] = "different private fixture"
        with self.assertRaisesRegex(ValueError, "split commitment"):
            verify_manifest(self.manifest, altered, self.dataset_sha)

    def test_no_overlap_and_both_tasks_in_each_split(self):
        altered = copy.deepcopy(self.rows)
        altered[0]["id"] = altered[2]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(altered, self.spec)
        with self.assertRaisesRegex(ValueError, "each task"):
            validate_rows(self.rows[:-1], self.spec)

    def test_spec_rejects_placeholder_invalid_digest_and_wrong_tasks(self):
        spec = copy.deepcopy(self.spec)
        spec["model"] = "REPLACE"
        with self.assertRaisesRegex(ValueError, "model"):
            validate_spec(spec)
        spec = copy.deepcopy(self.spec)
        spec["model_digest"] = "notpinned"
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_spec(spec)
        spec = copy.deepcopy(self.spec)
        spec["tasks"]["field"] = {"kind": "classification", "labels": ["A", "B"]}
        with self.assertRaisesRegex(ValueError, "one classification"):
            validate_spec(spec)
        spec = copy.deepcopy(self.spec)
        spec["decoding"]["temperature"] = 0.7
        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_spec(spec)

    def test_rejects_noncanonical_references(self):
        altered = copy.deepcopy(self.rows)
        altered[1]["expected"] = '{"value": "AA"}'
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_rows(altered, self.spec)
        altered[1]["expected"] = '{"value":1}'
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_rows(altered, self.spec)

    def test_output_normalization_and_schema_validation(self):
        self.assertEqual(normalize(self.spec["tasks"]["sentiment"],
                                   " no \n"), ("no", True))
        self.assertEqual(normalize(self.spec["tasks"]["sentiment"],
                                   "probably no"), ("probably no", False))
        self.assertEqual(normalize(self.spec["tasks"]["field"],
                                   '{ "value": "AA" }'),
                         ('{"value":"AA"}', True))
        self.assertFalse(normalize(self.spec["tasks"]["field"],
                                   '{"value":"AA","note":"x"}')[1])
        self.assertFalse(normalize(self.spec["tasks"]["field"],
                                   '{"value":"AA","value":"BB"}')[1])

    def test_fixture_run_is_paired_and_not_model_evidence(self):
        with patch("local_model_pilot.http_json",
                   side_effect=AssertionError("mock test used the network")):
            predictions, scores, paired, provenance = run_evaluation(
                self.manifest, self.rows, self.dataset_sha, "test",
                transport=mock_transport)
        self.assertEqual(len(predictions), 4)
        self.assertEqual(scores["n_records"], 4)
        self.assertEqual(len(paired["comparisons"]), 2)
        self.assertEqual(provenance["evidence_type"],
                         "mock_transport_fixture_not_real_model")
        self.assertEqual(provenance["split"], "test")
        self.assertEqual({r["split"] for r in predictions}, {"test"})
        for group in paired["comparisons"]:
            self.assertEqual(group["n_paired_examples"], 1)
            self.assertEqual(group["paired_valid_exact_match_difference"], 1)

    def test_incomplete_or_wrong_model_response_fails_closed(self):
        def wrong_model(spec, row, prompt):
            response = mock_transport(spec, row, prompt)
            response["model"] = "other-model"
            return response
        with self.assertRaisesRegex(ValueError, "wrong-model"):
            run_evaluation(self.manifest, self.rows, self.dataset_sha,
                           "dev", transport=wrong_model)
        def missing_tokens(spec, row, prompt):
            response = mock_transport(spec, row, prompt)
            del response["prompt_eval_count"]
            return response
        with self.assertRaisesRegex(ValueError, "token count"):
            run_evaluation(self.manifest, self.rows, self.dataset_sha,
                           "dev", transport=missing_tokens)

    def test_rejects_remote_inference_endpoint(self):
        self.assertEqual(validate_endpoint("http://127.0.0.1:11434"),
                         "http://127.0.0.1:11434")
        for url in ("https://example.com", "http://192.168.0.2:11434",
                    "http://127.0.0.1:4444", "file:///tmp/local.json"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_endpoint(url)

    def test_ollama_digest_is_checked_before_real_inference(self):
        spec = self.spec
        self.assertTrue(verify_ollama_digest(
            spec, "http://127.0.0.1:11434",
            get_json=lambda url: {"models": [{"name": spec["model"],
                                               "digest": spec["model_digest"]}]}))
        with self.assertRaisesRegex(ValueError, "digest changed"):
            verify_ollama_digest(
                spec, "http://127.0.0.1:11434",
                get_json=lambda url: {"models": [{"name": spec["model"],
                                                   "digest": "unexpected"}]})

    def test_output_directory_is_not_overwritten(self):
        data = run_evaluation(self.manifest, self.rows, self.dataset_sha,
                              "dev", transport=mock_transport)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "real-model-private-output"
            write_outputs(target, *data)
            self.assertTrue((target / "predictions.jsonl").exists())
            metadata = json.loads((target / "run_manifest.json").read_text())
            self.assertEqual(metadata["evidence_type"],
                             "mock_transport_fixture_not_real_model")
            with self.assertRaises(FileExistsError):
                write_outputs(target, *data)

    def test_read_dataset_preserves_exact_file_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.jsonl"
            data = "".join(canonical(row) + "\n" for row in self.rows)
            path.write_text(data, encoding="utf-8")
            parsed, hashed = read_dataset(path)
            self.assertEqual(parsed, self.rows)
            self.assertEqual(hashed, digest(data.encode("utf-8")))
            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                read_dataset(path)


    def test_local_generation_request_and_task_json_mode(self):
        captured = []
        def fake_http(url, payload):
            captured.append((url, payload))
            return {"response": "yes"}
        with patch("local_model_pilot.http_json", side_effect=fake_http):
            _actual_transport(self.spec, self.spec["tasks"]["sentiment"],
                              {"system": "A", "input": "B"},
                              "http://127.0.0.1:11434")
            _actual_transport(self.spec, self.spec["tasks"]["field"],
                              {"system": "A", "input": "B"},
                              "http://127.0.0.1:11434")
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0][0],
                         "http://127.0.0.1:11434/api/generate")
        self.assertEqual(captured[0][1]["prompt"], "B")
        self.assertEqual(captured[0][1]["system"], "A")
        self.assertEqual(captured[0][1]["options"],
                         self.spec["decoding"])
        self.assertFalse(captured[0][1]["stream"])
        self.assertNotIn("format", captured[0][1])
        self.assertEqual(captured[1][1]["format"], "json")

    def test_freeze_and_verify_cli_without_any_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            dataset = directory / "data.jsonl"
            spec_file = directory / "spec.json"
            manifest = directory / "frozen.json"
            dataset.write_text("".join(
                canonical(row) + "\n" for row in self.rows
            ), encoding="utf-8")
            spec_file.write_text(
                json.dumps(self.spec), encoding="utf-8")
            cmd = [sys.executable, str(ROOT / "src/local_model_pilot.py")]
            freeze_cmd = cmd + [
                "freeze", "--dataset", str(dataset), "--spec",
                str(spec_file), "--manifest", str(manifest),
            ]
            result = subprocess.run(
                freeze_cmd, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("not a model evaluation", result.stdout)
            self.assertTrue(manifest.is_file())
            verify_cmd = cmd + [
                "verify", "--dataset", str(dataset),
                "--manifest", str(manifest),
            ]
            result = subprocess.run(
                verify_cmd, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("verified", result.stdout)
            result = subprocess.run(
                freeze_cmd, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)



if __name__ == "__main__":
    unittest.main()
