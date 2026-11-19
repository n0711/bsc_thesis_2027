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
