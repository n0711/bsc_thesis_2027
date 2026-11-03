# Proposed held-out real-model study - NOT EXECUTED

1. Preselect two objective tasks (classification and deterministic structured extraction) and their public dataset licenses.
2. Pin train/development/test IDs, never optimize a prompt on held-out test answers.
3. Fix model versions, inference parameters, decoding, prompt templates, output schemas, and candidate-generation methods.
4. Compare a frozen author-written baseline with random and coordinate-based search under **equal development model-call budgets**. Track token totals, wall time, retries, errors and normalized costs.
5. Evaluate selected prompts on held-out test data. Use exactly specified normalization and task metrics, report failure modes and paired uncertainty intervals.
6. Save actual predictions in JSONL and analyze with `src/score_outputs.py`. The scorer does **not** invoke a model and cannot verify the origin of externally supplied records.
7. Require supervisor approval of the final experiment design and any relevant research ethics/data-use checks.

No placeholder performance values are presented as observations. The provided synthetic study is an engineering/mathematical validation stage only.
