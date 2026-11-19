# Local-model pilot, reproducible but not yet executed

## Purpose and scientific boundary

This milestone makes the **prospective** real-model experiment executable
through a local Ollama HTTP service. The repository contains only the
runner, a protocol template, and **mock-transport unit tests**. It contains
**no actual model predictions** and makes no claim that the synthetic
Taylor theorem applies to an LLM.

The pilot compares one frozen author-written prompt with one separately
chosen development-selected prompt. **It does not implement the random and
coordinate candidate-search algorithms on a real model yet.** A full
equal-budget comparison will require a separately auditable development
search log, including every attempt and token cost. Do not call this pilot
an end-to-end optimization study until that exists.

## Inputs and task definitions

The input JSONL file has exactly five fields in each record:

\`\`\`json
{"id":"dev-c1","task":"classification","split":"dev","input":"A licensed example text","expected":"yes"}
\`\`\`

The remaining rows include non-overlapping IDs for the \`test\` split, plus
a second task, \`extraction\`, whose reference response is a canonical JSON
string such as \`"{\\"value\\":\\"ABC\\"}"\`. These descriptions are **schema
examples**, not a sample dataset or measured results. No private or
licensed dataset content has been committed.

The initial pilot uses exactly:

- **Classification**: a fixed closed set of labels, with surrounding
  whitespace stripped from the model prediction.
- **Structured extraction**: a single JSON object containing exactly one
  string-valued key, \`value\`, converted to compact sorted-key JSON for
  comparison.

Unexpected text, unrecognized labels, extra JSON keys, wrong value types
and invalid JSON are marked invalid. The primary metric is exact match
**conditioned on output validity**, defined as
\`1[valid prediction and normalized prediction == expected]\`.

Each task must have at least one development and one held-out test record;
real research should use enough examples for meaningful uncertainty
estimation and a documented sample-size rationale.

## Protocol preparation (no model calls)

1. Obtain suitable licensed, non-sensitive classification and structured
   extraction datasets. Record sources, versions, permissions and exact
   split procedures, and obtain supervisor approval before conducting an
   experiment.
2. Make \`private_pilot/data.jsonl\` from those records. **Do not commit it.**
   Freeze development and held-out IDs before testing, and keep held-out
   answers inaccessible to prompt selection in the actual workflow.
3. Copy \`docs/local_model_spec.template.json\` into
   \`private_pilot/spec.json\`. Replace every \`REPLACE...\` value with real
   details, including the **exact** local model tag and the SHA-256 digest
   reported by Ollama's \`/api/tags\`.
4. Set both frozen prompt bodies. The \`coordinate_selected\` label requires
   separate evidence of development-only prompt selection. Merely naming
   a prompt this way does not establish its origin.
5. Freeze the dataset, prompts and settings. The command refuses to
   overwrite an existing manifest:

\`\`\`bash
python3 src/local_model_pilot.py freeze \
  --dataset private_pilot/data.jsonl \
  --spec private_pilot/spec.json \
  --manifest private_pilot/frozen_manifest.json

python3 src/local_model_pilot.py verify \
  --dataset private_pilot/data.jsonl \
  --manifest private_pilot/frozen_manifest.json
\`\`\`

The manifest includes raw-file SHA-256, independent per-split commitments,
prompt hashes, the pinned model digest and a manifest checksum. Altering
the dataset, prompts, task definitions or decoding parameters invalidates
the commitment. However, a file hash cannot prove that the held-out answers
were not previously viewed or used. A separate reviewer must inspect
split provenance and the time the protocol was frozen.

## Run on the local device only (requires real Ollama service)

An installed, running Ollama server must expose
\`http://127.0.0.1:11434\`. Confirm the installed model and the exact
digest from \`http://127.0.0.1:11434/api/tags\` before freezing.
The runner rejects other endpoints and fails if the model digest differs.

\`\`\`bash
python3 src/local_model_pilot.py run \
  --dataset private_pilot/data.jsonl \
  --manifest private_pilot/frozen_manifest.json \
  --split dev \
  --output-dir private_pilot/runs/development_001

# Only after development-only prompt selection and freezing it:
python3 src/local_model_pilot.py run \
  --dataset private_pilot/data.jsonl \
  --manifest private_pilot/frozen_manifest.json \
  --split test \
  --output-dir private_pilot/runs/heldout_001
\`\`\`

**Important:** An independent experimental run should not use the same
already-tested held-out set for repeated prompt changes. The commands
demonstrate evaluation mechanics only, not an approved full study.

The runner makes \`POST /api/generate\` calls with \`stream=false\`,
\`temperature=0\`, pinned seed and output-token limit. A locally
installed model is still subject to backend nondeterminism and hardware
differences. It validates the response model identity, \`done=true\`,
and the returned input/output token counts. Missing or failed calls abort
the complete evaluation before output files are created.

A successful run saves, in the **ignored private output directory**:

- \`predictions.jsonl\`: paired per-example responses and consumption,
- \`scores.json\`: separate task and prompt metrics,
- \`paired.json\`: paired correctness, validity, token and latency differences,
- \`run_manifest.json\`: manifest hashes, model digest, decoding parameters,
  dataset split, endpoint and evidence-origin caveat.

A normal local-run evidence tag is
\`local_model_runner_outputs_not_independently_verified\`. In-memory mock
tests instead produce \`mock_transport_fixture_not_real_model\`. Never
combine these with the \`synthetic_surrogate_not_LLM\` evidence register.

## Verification and remaining blockers

\`\`\`bash
python3 -m unittest discover -s tests -v
\`\`\`

CI exercises the protocol with an injected fake transport; it does not
download a model, contact Ollama, run inference or manufacture empirical
scores. Failed checksum, wrong model digest, changed split, wrong reference,
noncanonical extraction, duplicate example ID, wrong endpoint or
incomplete token counts are deliberately rejected.

**Still outstanding:** approved task datasets and licenses, dataset
selection and sample-size rationale, hardware/model availability, a
traceable and budget-matched development search, one-time held-out
evaluation, independent checking of run provenance and an explicit
supervisor review. The mathematical certificates remain specific to the
small fixed logistic surrogate, not to this future model evaluation.
