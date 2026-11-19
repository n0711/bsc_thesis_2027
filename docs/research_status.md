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

## Additional milestone: executable statistical referee (2026-11-19)

A separate read-only reviewer now rescored all 1,200 committed synthetic
candidate prompts, checked the 12-condition dataset for duplicated keys,
score-call accounting, oracle regret, mean-score aggregates and paired
sign-test summaries, and applied Holm's procedure to the twelve
development-score sign tests. All 12 exploratory Holm-adjusted values are
below 0.05, but **only 5 of 12 configurations have a positive held-out
mean difference**, despite all 12 having positive development differences.

Eight new regression tests deliberately alter stored scores, p-values,
row identity and call budgets to ensure the referee raises errors.
New files: `src/independent_referee.py`,
`tests/test_independent_referee.py`,
`tex/independent_referee.tex`, `tex/referee_interpretation.tex`.
The peer-referee principle is methodologically inspired by *A Society
of AI Scientists*, not a source of either mathematical or numerical
findings. No external models were queried.

## Additional milestone: offline paired real-model evaluation interface (2026-11-19)

The prospective held-out evaluation can now reject unequal example IDs
and reference labels when comparing baseline and selected prompts,
calculate validity-adjusted and plain exact match separately, and
record paired input/output-token and latency differences. All examples
in the tests are invented fixtures; `src/compare_heldout.py` neither
invokes a model nor verifies external prediction provenance. The offline
scorer additionally rejects non-finite latency and empty identifiers.
Seven unit tests cover the new comparison and validation paths.
The protocol in `docs/real_model_protocol.md` has been expanded with
predeclared splits, input provenance and executable referee objections.

## Additional milestone: finite synthetic target-transfer theorem (2026-11-19)

Added analytic uniform target-shift bound for a fixed mean-logistic
surrogate, finite-objective simple-regret transfer theorem, and sufficient
pairwise ranking-preservation certificate. The exhaustive audit enumerates
all 36, 216 and 1296 prompts for lengths 2, 3 and 4, and the unordered
prompt pairs with distinct development scores. It finds 149, 5,487 and
205,554 held-out ordering reversals, and 64, 940 and 14,374 certifiably
preserved orderings. The development optimum differs from the held-out
optimum at every length, with positive held-out regret of roughly 0.057.

Evidence and source: `src/target_transfer_audit.py`,
`results/target_transfer_audit.json`,
`tex/target_transfer_theorem.tex`,
`tex/target_transfer_audit.tex`,
`tests/test_target_transfer_audit.py`. CI will regenerate
and compare outputs. There is no real-model inference, sampled task
distribution, causal claim, or independent peer review.
