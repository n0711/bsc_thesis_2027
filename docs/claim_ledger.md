# Claim-to-evidence ledger - 2026-11-19

This register adapts the **claim -> executable code -> result -> criticism**
workflow discussed in [Ahmad et al., *A Society of AI Scientists*
(alphaXiv preprint, November 2026)](https://www.alphaxiv.org/abs/2611.society-ai-scientists).
It does not inherit any experimental result from that preprint.

| Claim ID | Exact claim | Derivation / evidence | Verification | Boundary |
|---|---|---|---|---|
| P1 | Mean logistic surrogate has gradient bounded by $\frac{3}{4|T|}\sum_{u\in T}\|u\|_2$. | `tex/thesis.tex`, Chapter 4 | Central finite difference in `tests/test_experiment.py` | Smooth fixed-vector surrogate only |
| P2 | Its gradient is globally Lipschitz with constant at most $\frac{\sqrt3}{2|T|}\sum_{u\in T}\|u\|_2^2$. | `tex/taylor_certificates.tex`, Proposition and Hessian proof | `tests/test_taylor_certificate.py` analytic critical points and off-grid samples | Not a gradient Lipschitz bound for LLMs |
| P3 | Every move's actual score change lies inside the global second-order interval. | Taylor's integral remainder, Equation (second-order-certificate) | `src/taylor_certificate.py`, exhaustive 29,520 directed moves | Does not certify whether uncertified moves help |
| M1 | Of 29,520 directed moves, 10,479 are certified rises and 9,103 certified falls (66.33% coverage). | Frozen `src/experiment.py` parameters and `src/taylor_certificate.py` | Run the script; regression counts in `tests/test_taylor_certificate.py` | Lengths 2, 3 and 4 only; synthetic |
| M2 | 1,004 positive first-order predictions are not real improvements. | `results/gradient_audit.json` and `tex/gradient_moves.tex` | Independent recount in `src/taylor_certificate.py` | Does not assert gradient signs for natural language |
| M3 | The equal-score-call search comparison uses 1,200 deterministic surrogate records. | `results/runs.csv`, `results/summary.json` | `src/experiment.py`, CI regeneration and `tests/test_experiment.py` | These are fixed-seed algorithmic trials, not independent tasks |
| M4 | Paired coordinate-minus-random results are descriptive. | `results/paired_analysis.json`, `src/paired_analysis.py` | `tests/test_paired_analysis.py` | Sign-test p-values unadjusted; no LLM inference |
| U1 | Search improves real LLM task accuracy. | **No evidence yet** | Requires `docs/real_model_protocol.md` and actual held-out model runs | **Unsupported** |
| U2 | The discrete move theorem applies to transformers. | **No evidence yet** | Would need model-specific mathematical assumptions and proof | **Unsupported** |
| U3 | The work is independently peer reviewed or novel. | **No evidence yet** | Supervisor assessment and literature review | **Unsupported** |

## Reproduce and challenge the claims

```bash
python3 -m unittest discover -s tests -v
python3 src/experiment.py
python3 src/gradient_audit.py
python3 src/taylor_certificate.py
cd tex
xelatex -halt-on-error -interaction=nonstopmode thesis.tex
xelatex -halt-on-error -interaction=nonstopmode thesis.tex
```

The new certification script prints a JSON report to standard output and is
also executed as a CI step with its report retained as a workflow artifact.
It does **not** mutate the existing training results or claim an external replication.

A review must separately check (a) the Hessian derivation, (b) that the fixed
vectors are the intended benchmark, (c) the implementation's exact directed
neighborhood and numerical tolerance, and (d) whether conclusions are correctly
restricted to the synthetic benchmark. Independent review is pending.

| R1 | All 1,200 saved observations and 12 aggregates reconcile with recomputation from selected tokens. | `src/independent_referee.py`, `results/runs.csv`, `results/summary.json` | `tests/test_independent_referee.py` and CI audit report | Shared specified scorer is reused, so this does not independently validate the objective itself |
| R2 | Positive mean development advantage in 12/12 conditions, positive mean held-out advantage in only 5/12. | `results/runs.csv`, `tex/independent_referee.tex` | Re-scored, paired by length, budget and seed in `src/independent_referee.py` | Only fixed synthetic development and held-out targets |
| R3 | The twelve training-score sign tests have exploratory Holm-adjusted values below 0.05. | `tex/independent_referee.tex`, `src/independent_referee.py` | Exact binomial sign tests and step-down adjustment tested against reference examples | Not evidence of an advantage across independently sampled real tasks; no preregistered claim |

| F1 | Same-example held-out accuracy and usage comparison interface is implemented. | `src/compare_heldout.py`, `src/score_outputs.py` | `tests/test_compare_heldout.py`, `tests/test_score_outputs.py` | Unit-test fixtures only. Model invocation, provenance and real-task performance remain unsupported |

| T1 | For two fixed finite objectives, simple regret differs by no more than the oscillation of their score gap. | Proposition and proof in `tex/target_transfer_theorem.tex` | `src/target_transfer_audit.py` checks all 1,548 prompts; `tests/test_target_transfer_audit.py` | General finite-objective theorem, but computed values are specific to the fixed surrogate |
| T2 | A development score margin exceeding the exact score-gap oscillation certifies preserved held-out ordering. | Same proposition | Exhaustive unordered-pair audit with zero violations and preserved certified counts in `results/target_transfer_audit.json` | Sufficient condition, not a necessary one; a non-certified pair need not reverse |
| T3 | The development optimum is repeated `formal`, held-out optimum repeated `verify` for all three lengths, with approximate held-out regret 0.057. | Frozen `src/experiment.py` vectors and `results/target_transfer_audit.json` | Exhaustive oracles in `src/target_transfer_audit.py`, unit tests and CI parity | Labels are abstract vectors, not claims about natural-language prompting |
| T4 | Fixed target changes reverse 149, 5,487 and 205,554 strict development ranking pairs for lengths 2, 3 and 4. | `results/target_transfer_audit.json`, `tex/target_transfer_audit.tex` | Direct enumeration under tolerance 1e-12 and CI regeneration | Numerical enumeration at binary-float precision; not sampled tasks |

| H1 | A two-task local-model pilot can reject changes to frozen data, prompt text and model digest before inference. | `src/local_model_pilot.py`, `docs/local_model_pilot.md` | `tests/test_local_model_pilot.py` covers hashes, schema, duplicate/split IDs and digest failures | Tamper-evident protocol files do not prove test blindness or independent provenance |
| H2 | A local runner can produce two matched method records per example and classify invalid outputs separately from wrong answers. | `src/local_model_pilot.py`, `src/compare_heldout.py` | In-memory transport fixture tests and CI | Fake transport is NOT an LLM; no model result exists until actual inference |
| H3 | A complete real-model random-versus-coordinate search comparison is available. | **Not yet supported** | Requires development-only search implementation, matched call/token budgets, full call records and actual model inference | Candidate is only a frozen user-supplied prompt and must not be attributed to coordinate search without provenance |
