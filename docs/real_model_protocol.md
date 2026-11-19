# Proposed held-out real-model study - NOT EXECUTED

This protocol specifies future scientific measurements. No model inference,
dataset access, human ratings or real-model performance numbers have been
collected in this repository. The synthetic results are not LLM evidence.

## Stage A - Preregister the design before reading test labels

1. Choose two objective tasks, classification with finite labels and
   deterministic structured extraction. Record each dataset's source,
   version, license and allowed processing methods, and consult the
   thesis supervisor before execution.
2. Save a manifest containing fixed example identifiers for development
   and held-out test sets, partition hashes and the declared normalization
   of reference and predicted outputs. Prohibit test-set access during
   prompt optimization.
3. Pin model, tokenizer/version if available, decoding configuration,
   prompt templates, stop strings, model-call budget, candidate generation
   and any retry policy. Fix the primary endpoint, valid exact-match
   accuracy, before executing test inference.
4. Freeze one author-written baseline, a seeded uniform-random search and
   the coordinate-style search method. Compare the latter two with equal
   development model-call budgets, record all requests, response metadata,
   token totals, latency and retry/error events. If another method uses
   gradient/embedding operations, declare that extra cost separately.
5. Select a prompt by development-only metrics; freeze the selected prompt
   identifiers and evaluation code. Test them on exactly the same
   held-out example IDs. Any change after inspecting held-out results
   must be disclosed as exploratory and cannot be presented as a fresh,
   confirmatory test of the same hypotheses.

## Stage B - Report paired metrics on actual model records

For example \(i\) and prompt method \(a\), let the declared reference
be \(y_i\), prediction be \(\hat y_{ai}\), and output-validity flag
be \(v_{ai}\). Define correctness:

\[
c_{ai}=\mathbf{1}[v_{ai}=1\;\land\;\hat y_{ai}=y_i].
\]

For a selected prompt \(s\) versus a baseline \(b\) over the SAME
held-out IDs, the paired metric is:

\[
\Delta=\frac{1}{n}\sum_{i=1}^{n}(c_{si}-c_{bi}).
\]

Report plain exact match separately (without applying validity),
valid-output rate, input/output token use and latency. Report the
number of examples where only the selected prompt was correct, only the
baseline was correct, and where both had the same correctness. A
conditional exact McNemar/sign test on discordant examples can be
reported as exploratory, with its assumptions and planned multiplicity
correction. For interval estimates, resample paired *example IDs*
(not algorithm seed indices), with tasks reported separately.

Use the following existing read-only tools once actual prediction JSONL
files have been obtained from a separately documented model runner:

```sh
python3 src/score_outputs.py \
  --input actual_predictions.jsonl \
  --output actual_scores.json
python3 src/compare_heldout.py \
  --input actual_predictions.jsonl \
  --baseline-id author_written \
  --candidate-id coordinate_selected \
  --output actual_paired_scores.json
```

Each record follows `src/score_outputs.py`:
`id, task, model, prompt_id, expected, prediction, output_valid,
input_tokens, output_tokens, latency_ms`. Record IDs and references
must match across the compared methods. The comparator rejects missing
examples, changed references and duplicate configuration/ID records.
Both scripts score *externally supplied predictions* only. They cannot
establish that any prediction actually came from the declared model,
that held-out answers were previously unseen, or that a run was
independent of the experimental designer.

## Stage C - Executable referee objections

- **Leakage objection:** verify the split manifest and frozen prompt
  timestamp preceded testing. Static scoring code cannot prove this.
- **Baseline objection:** check both methods' call budgets and
  extra search overhead before interpreting improvements.
- **Selection objection:** preserve negative and null results,
  not only the configuration most favorable to the chosen optimizer.
- **Metric objection:** verify example-ID pairing, exact output
  normalization, schema validity and non-finite/missing values.
- **Multiple testing objection:** preregister a primary comparison
  and define the family of secondary comparisons. Adjust if the
  inferential family is greater than one, but do not treat a Holm
  adjustment as proof of independent task generalization.
- **Provenance objection:** retain prediction JSONL, dataset source,
  model/provider revision, prompt texts, software commit SHA, run logs
  and hashes. Have a separate reviewer examine which claims remain
  unsupported.

This plan is informed by the executable criticism workflow of
[A Society of AI Scientists (alphaXiv, 2026)](
https://www.alphaxiv.org/abs/2611.society-ai-scientists).
The paper is a methodology reference only. The dissertation's
Taylor guarantee remains confined to the fixed logistic surrogate.

**Status: proposed, not executed. Supervisor approval, permitted datasets,
frozen test protocol and actual model runs remain outstanding.**
