# Evidence and completion register (2026-11-03)

| Item | Status | Basis or blocker |
|---|---|---|
| Submitted October theoretical article | Author-reported submission | Paper provided to this conversation, acceptance not verified |
| Mixed discrete/continuous mathematical formulation | Drafted | A4 XeLaTeX manuscript, supervisor proofreading required |
| Analytic surrogate gradient and Lipschitz bound | Derived and numerical gradient checked | `tex/thesis.tex`, `src/experiment.py`, `tests/` |
| Exact oracle for lengths 2, 3, 4 | Completed locally | 36, 216, 1296 candidates, `results/summary.json` |
| Matched-budget random vs coordinate | Completed locally | 12 configurations, 50 seeds x 2 methods = 1,200 rows |
| Unit tests | 17 passed locally | Python 3.13, CI still needs post-push verification |
| XeLaTeX compilation | Completed locally | 33-page working draft, A4, PDF and source in local downloadable package |
| Real-model held-out study | NOT DONE | Requires model access, dataset selection and approved plan |
| Human preference validation | NOT DONE | Not necessary for current synthetic milestone |
| Institutional final submission | NOT DONE | Academic supervisor: Dr. Stephania Loizidou, review and department approval pending, verified references/format |

Scores here are fixed-surrogate sigmoid probabilities, not real LLM accuracies. Re-run code after any model or algorithm edits. Never infer novelty or historical precedence solely from the submitted article.

## Additional milestone: paired-seed analysis (2026-11-03)

Implemented `src/paired_analysis.py` and five additional unit tests (22 total). The analysis pairs coordinate and random search by configuration and seed, validates exact objective-call budgets, computes mean differences, paired bootstrap percentile intervals, and exploratory exact two-sided sign-test p-values. Twelve paired configurations are exported to `results/paired_analysis.json` and `tex/paired_results.tex`, included in the compiled dissertation. These comparisons use a fixed synthetic surrogate, they are not independent LLM task replications. Unadjusted p-values are descriptive only.

## Additional milestone: formal second-order certificate audit (2026-11-19)

A new globally valid Hessian operator-norm bound is derived for the **fixed
two-dimensional logistic surrogate**, not for LLM scoring. The resulting
Taylor interval certifies 19,582 of 29,520 directed one-token moves as strict
improvements or decreases. The remaining 9,938 moves are inconclusive.
The exhaustive checker finds zero sign or remainder-bound violations on
these finite spaces. Four additional regression tests probe the inequality
at off-grid points and verify exact per-length counts.

Files: `tex/taylor_certificates.tex`, `src/taylor_certificate.py`,
`tests/test_taylor_certificate.py`, `docs/claim_ledger.md`.
The existing experiment outputs have not been edited. CI runs the
certificate audit and uploads the JSON report. These are author-side
verification steps and not independent peer review, external task
generalization or a real-model benchmark. Supervisor review and CI
completion must be checked separately.
