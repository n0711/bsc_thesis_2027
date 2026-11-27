# M1/M3 - International literature review and comparison: screening pass 1

Date: 2026-11-27. This is a preliminary scoping review, not a completed systematic review. The twelve records below were identified via targeted web queries and checked against publisher abstracts or the original preprint record. Full-text claims, novelty, numerical comparisons and implementation details remain to be verified.

## Screening method

Search terms: automatic prompt optimization survey; gradient discrete continuous prompt replacement; prompt tuning; instruction optimization; cost-aware prompt optimization; tight evaluation budget; direct prompt optimization continuous representations. Search via ACL Anthology, PMLR, ICLR proceedings, NeurIPS proceedings and arXiv. Expand via backward/forward citation tracking. Include methods for frozen-model task adaptation, discrete/continuous search, evaluation efficiency or theory; exclude unrelated generic prompting guides and duplicates. Record publication status, model access, representation, objective, guarantees, budget accounting and held-out evaluation. Do not pool reported results across incompatible tasks or access regimes.

## Initial comparison matrix

| Reference | Representation and method | Access | Relevance and unresolved comparison |
|---|---|---|---|
| [Shin et al., AutoPrompt, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.346/) | Discrete trigger-token search guided by gradients | Gradient access to masked LM | Closest early token replacement precedent; compare candidate ranking and verification |
| [Lester et al., Prompt Tuning, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.243/) | Continuous learned soft embeddings | Backpropagation | Distinguish soft vectors from deployable hard text |
| [Pryzant et al., ProTeGi, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.494/) | Textual critiques and beam search | LLM API | Textual gradients are not numerical derivatives |
| [Wen et al., Hard Prompts Made Easy, NeurIPS 2023](https://papers.nips.cc/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html) | Gradient-based hard prompt optimization | Differentiable model | Close discrete-continuous precedent; check projection and guarantees |
| [Yang et al., OPRO, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3339f19c5fcee3ad74502947a32be9e6-Abstract-Conference.html) | LLM proposes candidates using previous scores | Score-based LLM access | Compare budget, including optimizer model cost |
| [Chen et al., InstructZero, ICML 2024](https://proceedings.mlr.press/v235/chen24e.html) | Bayesian optimization over soft prompts converted into text | Black-box target plus open-model generator | Hybrid continuous-to-discrete methods already exist |
| [Wang et al., Direct Prompt Optimization with Continuous Representations, ACL 2025](https://aclanthology.org/2025.acl-long.133/) | Prompt distributions with greedy strategies | Continuous optimization | Closest recent representation overlap; check theoretical results |
| [Zehle et al., CAPO, AutoML 2025](https://proceedings.mlr.press/v293/zehle25a.html) | Cost-aware search using AutoML | Score-based | Cost-aware optimization already exists; compare accounting |
| [Ramnath et al., Systematic Survey, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1681/) | Review of automatic prompt optimization | Not applicable | Citation expansion and taxonomy |
| [GEPA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) | Reflective evolutionary prompt search | Feedback and model outputs | Compare compatible tasks and total calls |
| [Gradient-Guided Multi-Judge Prompt Optimization, ACL 2026](https://aclanthology.org/2026.acl-long.1089/) | First-order continuous masking to score segments | Differentiable access for gradient computation | Check relationship to first-order and curvature-based move screening |
| [Liu et al., BudgetAPO, arXiv 2026-10-05](https://arxiv.org/abs/2610.05671) | Noise-adaptive allocation of scarce evaluations | Score-based | Preprint, not established peer-reviewed result; tight-budget optimization already studied |

## Candidate research gap - NOT established novelty

The combination of discrete prompts, continuous representations, gradient guidance and cost budgets is not independently novel. A narrower hypothesis is whether curvature-aware certification can improve the tradeoff between reliable move selection and true evaluation cost, and how such decisions transfer to a held-out objective. Search next for certified discrete optimization, smoothness-based screening, safe screening, Lipschitz bounds, trust regions, and prompt generalization. A standard Taylor bound alone is not a novel theorem.

## Proposed falsifiable hypotheses

H1: For the declared differentiable surrogate, certified moves have no sign violations when a valid curvature bound is used.
H2: Relative to gradient-only ranking and equal-budget coordinate search, certified screening improves development objective or reduces actual total cost at matched quality. Negative results count.
H3: Development improvement does not guarantee held-out improvement; quantify transfer and compare across prespecified splits.

## Next steps

- Read full text of AutoPrompt, Hard Prompts Made Easy, Direct Prompt Optimization, InstructZero, CAPO, GMPO and BudgetAPO first.
- Expand to 20–30 screened records, including citation chains and local research, with documented inclusion and exclusion.
- Extract exact assumptions and costs before choosing baselines; compare methods within model-access regimes, not as interchangeable competitors.
- Verify candidate research gap with supervisor; do not claim originality before this review.
- No real-model evidence is generated by this literature-review update.
