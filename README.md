# BSc Thesis 2027 - Prompt Engineering as a Discrete-Continuous Optimization Problem

**Author:** Charalampos Nadiotis · **Frederick University** · **Status:** working dissertation, November 2026

This project extends the author's **submitted October 2026 theoretical article** into a mathematical study with an executable **synthetic surrogate**. It does **not** claim measured performance on real large language models.

## What is completed

- XeLaTeX A4 dissertation draft (10 chapters, two appendices, references, evidence register).
- Exact finite-space optimization (6-symbol alphabet, prompt lengths 2/3/4).
- Equal-budget comparison: random search and restarting first-improvement coordinate search, 12/24/36/72 score evaluations, 50 seeds each.
- 1,200 synthetic observations, JSON/CSV evidence, exact random-search hit probabilities and expected best-score formula.
- 52 local Python unit tests including gradient ranking, discrete stopping states, finite differences and strict evaluation accounting.
- Offline output scorer for **future externally supplied actual model predictions**, does not perform model inference.

## Reproduction

Python 3.10+ is required. Experiments and tests have no third-party runtime dependencies.

```sh
python3 -m unittest discover -s tests -v
python3 src/experiment.py
python3 src/make_tables.py
python3 src/paired_analysis.py
python3 src/gradient_audit.py
python3 src/landscape_audit.py
python3 src/target_transfer_audit.py
python3 src/taylor_certificate.py
python3 src/independent_referee.py --out-tex tex/independent_referee.tex
cd tex
xelatex -halt-on-error -interaction=nonstopmode thesis.tex
xelatex -halt-on-error -interaction=nonstopmode thesis.tex
```

Data files are `results/runs.csv`, `results/summary.json`. All printed result tables are generated from the JSON. For real-model predictions, see `src/score_outputs.py` and `docs/real_model_protocol.md`.

## Evidence boundary

The tested surrogate is **not an LLM**. Scores represent probabilities from fixed two-dimensional vectors and a logistic function, **not** task accuracy, model quality or human ratings. A full LLM study, verified bibliography, supervisor review and department format approval are **pending**. Do not describe this draft as an accepted thesis or the submitted article as a published paper without independent confirmation.

## Layout

`tex/` dissertation source and generated tables/plot, `src/` reproducible experiments and scoring, `tests/` unit tests, `results/` synthetic outputs, `docs/` protocol, status and claim-to-evidence ledger, `.github/workflows/` verification configuration.

## Paired-seed analysis milestone

Run `python3 src/paired_analysis.py` after `python3 src/experiment.py`. The gradient and landscape audit scripts also emit per-state JSON evidence under `results/` and matching LaTeX tables in `tex/`; their outputs use stable ordering and normalized numeric precision. The 52-test suite additionally checks the exact second-order Taylor certificate and off-grid inequalities. These analyses operate only on the synthetic objective; the paired-seed sign-test p-values are unadjusted across twelve configurations and must not be interpreted as real-LLM evidence.

## January 2027 preliminary presentation and MATLAB figures

- `presentation/ACSC489_January_2027_Preliminary_Report.pptx` is edited from the supplied eight-slide template.
- `matlab/synthetic_summary.csv` contains only the twelve confirmed synthetic configurations.
- `matlab/generate_thesis_figures.m` is the MATLAB R2020a+ plotting source for the four charts.
- `evidence/native-matlab/` contains four native MATLAB R2026b PNG exports and the render log. The dissertation uses these committed evidence images; comparison against the older Python-rendered copies has not been performed.
- The new dissertation chapter describes the figures, exact scoring definitions and evidence limitations.
- The preliminary slide deck is not an approved or graded university presentation. The academic supervisor is Dr. Stephania Loizidou, as identified by the student. Review and approval are still pending.

The document and slides use comma punctuation and short hyphens in prose, as requested.

## Proof-backed discrete-move certification (2026-11-19)

The dissertation now proves a **global gradient-Lipschitz / second-order Taylor
bound** for the fixed mean-logistic surrogate, stronger than using the
first-order gradient merely as a rank heuristic. Applying this proof to all
29,520 directed one-token replacements certifies **10,479 strict increases**
and **9,103 strict decreases**, with **9,938 inconclusive replacements**
(66.33% certified coverage). The previous 1,004 first-order false-positive
predictions are distinguished explicitly from certified predictions. A
deterministic exhaustive script and four regression tests reproduce the
counts and check the bound at off-grid vectors.

The methodology is informed by *A Society of AI Scientists* (Ahmad et al.,
alphaXiv, 2026): see `docs/claim_ledger.md` for precise claim, code, check
and limitation mappings. The cited preprint is methodological inspiration,
not a source for the Taylor theorem or surrogate measurements. The
certificate **does not** apply to deployed LLMs without new assumptions
and evidence. Supervisor review remains pending.

## Independent executable referee (2026-11-19)

The read-only `src/independent_referee.py` recomputes **all 1,200 committed
synthetic observations**, including development and held-out scores,
oracle regret, optimal-hit flags, equal budgets, paired sign counts, and
aggregate summaries. It detects duplicate observations and modified
reported values. Holm-adjusted exploratory sign-test p-values are
computed for the family of twelve development-score comparisons.

**Key limitation:** coordinate search has higher mean development score
in all twelve conditions, but higher mean held-out score in only five
conditions. These are fixed hand-chosen synthetic vectors and seeded
algorithm runs, not real LLM outcomes, independent tasks or proof of
prompt-search generalization.

Run `python3 src/independent_referee.py --out-tex tex/independent_referee.tex`
to reproduce the independent audit JSON and the dissertation's statistical
table. CI verifies committed table parity. Failure modes are covered by
the separate test file `tests/test_independent_referee.py`.

## Paired held-out metrics interface (NOT EXECUTED)

For a future licensed real-model study, `src/compare_heldout.py` now
compares frozen baseline and candidate prompts only after validating that
both share the same model, task, example IDs and reference answers.
It separately reports validity-adjusted exact match, plain exact match,
invalid output rate, input and output tokens, latency and exploratory
discordant-pair counts. `src/score_outputs.py` rejects nonfinite latency
and missing identifiers. Seven regression tests use invented unit-test
fixtures, **not actual model observations**.

The complete proposed protocol and commands are at
`docs/real_model_protocol.md`. No test performance has been measured on
a real LLM.

## Exact target-shift and transfer-regret theorem (2026-11-19)

The fixed synthetic development objective and held-out objective have
**different globally optimal token tuples** for all three enumerated lengths:
repeated `formal` is best on development, repeated `verify` is best
on the held-out vectors. These are labels for fixed vectors, not tested
English instructions. The development oracle's held-out regret is
approximately 0.057181, 0.057159, and 0.056911 for lengths 2, 3, and 4.

The analytical theorem proves, for two finite objectives `F` and `G`,
that **their simple regrets differ by at most the oscillation of
`G-F` across the common candidate set**, and that a development
score gap greater than this oscillation certifies preservation of
the pairwise ordering under the held-out objective. A second analytic
bound uses sigmoid Lipschitz continuity and optimal matching of the
four fixed target vectors. Full enumeration audits 36, 216, and 1,296
candidates and their pairwise rankings, finding 149, 5,487, and
205,554 reversed orderings, with no certificate violations.

Reproduce with `python3 src/target_transfer_audit.py`.
It regenerates `results/target_transfer_audit.json` and
`tex/target_transfer_audit.tex`. Six tests check input assumptions,
oracles, bounds, and deterministic output. The theorem and discussion
are in `tex/target_transfer_theorem.tex`.

**Important limitation:** this is fixed synthetic target-shift analysis,
not real-model performance, out-of-distribution testing with a sampled
task population, or an LLM generalization guarantee.
