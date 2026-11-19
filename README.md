# BSc Thesis 2027 - Prompt Engineering as a Discrete-Continuous Optimization Problem

**Author:** Charalampos Nadiotis · **Frederick University** · **Status:** working dissertation, November 2026

This project extends the author's **submitted October 2026 theoretical article** into a mathematical study with an executable **synthetic surrogate**. It does **not** claim measured performance on real large language models.

## What is completed

- XeLaTeX A4 dissertation draft (10 chapters, two appendices, references, evidence register).
- Exact finite-space optimization (6-symbol alphabet, prompt lengths 2/3/4).
- Equal-budget comparison: random search and restarting first-improvement coordinate search, 12/24/36/72 score evaluations, 50 seeds each.
- 1,200 synthetic observations, JSON/CSV evidence, exact random-search hit probabilities and expected best-score formula.
- 31 local Python unit tests including gradient ranking, discrete stopping states, finite differences and strict evaluation accounting.
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
python3 src/taylor_certificate.py
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

Run `python3 src/paired_analysis.py` after `python3 src/experiment.py`. The gradient and landscape audit scripts also emit per-state JSON evidence under `results/` and matching LaTeX tables in `tex/`; their outputs use stable ordering and normalized numeric precision. The 31-test suite additionally checks the exact second-order Taylor certificate and off-grid inequalities. These analyses operate only on the synthetic objective; the paired-seed sign-test p-values are unadjusted across twelve configurations and must not be interpreted as real-LLM evidence.

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
