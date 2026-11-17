# BSc Thesis 2027 - Prompt Engineering as a Discrete-Continuous Optimization Problem

**Author:** Charalampos Nadiotis · **Frederick University** · **Status:** working dissertation, November 2026

This project extends the author's **submitted October 2026 theoretical article** into a mathematical study with an executable **synthetic surrogate**. It does **not** claim measured performance on real large language models.

## What is completed

- XeLaTeX A4 dissertation draft (10 chapters, two appendices, references, evidence register).
- Exact finite-space optimization (6-symbol alphabet, prompt lengths 2/3/4).
- Equal-budget comparison: random search and restarting first-improvement coordinate search, 12/24/36/72 score evaluations, 50 seeds each.
- 1,200 synthetic observations, JSON/CSV evidence, exact random-search hit probabilities and expected best-score formula.
- 27 local Python unit tests including gradient ranking, discrete stopping states, finite differences and strict evaluation accounting.
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
cd tex
xelatex -halt-on-error -interaction=nonstopmode thesis.tex
xelatex -halt-on-error -interaction=nonstopmode thesis.tex
```

Data files are `results/runs.csv`, `results/summary.json`. All printed result tables are generated from the JSON. For real-model predictions, see `src/score_outputs.py` and `docs/real_model_protocol.md`.

## Evidence boundary

The tested surrogate is **not an LLM**. Scores represent probabilities from fixed two-dimensional vectors and a logistic function, **not** task accuracy, model quality or human ratings. A full LLM study, verified bibliography, supervisor review and department format approval are **pending**. Do not describe this draft as an accepted thesis or the submitted article as a published paper without independent confirmation.

## Layout

`tex/` dissertation source and generated tables/plot, `src/` reproducible experiments and scoring, `tests/` unit tests, `results/` synthetic outputs, `docs/` protocol and status, `.github/workflows/` verification configuration.

## Paired-seed analysis milestone

Run `python3 src/paired_analysis.py` after `python3 src/experiment.py`. The gradient and landscape audit scripts also emit per-state JSON evidence under `results/` and matching LaTeX tables in `tex/`; their outputs use stable ordering and normalized numeric precision. The 27-test suite checks these audit behaviors. These analyses operate only on the synthetic objective; the paired-seed sign-test p-values are unadjusted across twelve configurations and must not be interpreted as real-LLM evidence.

## January 2027 preliminary presentation and MATLAB figures

- `presentation/ACSC489_January_2027_Preliminary_Report.pptx` is edited from the supplied eight-slide template.
- `matlab/synthetic_summary.csv` contains only the twelve confirmed synthetic configurations.
- `matlab/generate_thesis_figures.m` is the MATLAB R2020a+ plotting source for the four charts.
- `evidence/native-matlab/` contains four native MATLAB R2026b PNG exports and the render log. The dissertation uses these committed evidence images; comparison against the older Python-rendered copies has not been performed.
- The new dissertation chapter describes the figures, exact scoring definitions and evidence limitations.
- The preliminary slide deck is not an approved or graded university presentation. The academic supervisor is Dr. Stephania Loizidou, as identified by the student. Review and approval are still pending.

The document and slides use comma punctuation and short hyphens in prose, as requested.
