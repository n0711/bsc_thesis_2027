# ACSC489 January 2027 research roadmap

Updated 2026-11-27. Working plan, not an official university rubric or supervisor approval.

## Objective
January: clearly state the thesis idea, related international and Cyprus research, and planned work. Publication: pursue a specific, defensible contribution with proof assumptions, fair comparisons and held-out evidence.

Working title: Prompt Engineering as a Discrete–Continuous Optimization Problem.
Candidate question: Can curvature-aware certification of discrete prompt replacements improve search efficiency under fixed evaluation budgets while preserving held-out quality? Novelty is unverified.

## Priorities until January
35% literature and gap, 30% mathematical formulation, 20% algorithm comparison and protocol, 10% resource constraints, 5% preliminary evidence. Defer expanded implementation and distributed optimization.

| Milestone | Target | Status | Acceptance criterion |
|---|---|---|---|
| M1 International literature review | Dec 6 | In progress | 20–30 screened papers, source links, critical notes, evidence of search coverage |
| M2 Cyprus related research | Dec 6 | Pending | Verify primary institutional/publication sources, distinguish adjacent work from exact overlap |
| M3 Algorithm comparison | Dec 13 | In progress | Compare access, representation, candidate selection, guarantees, budget and held-out evidence |
| M4 Research gap | Dec 13 | Pending | Document closest prior work and what is not yet established; supervisor review |
| M5 Mathematical framework | Dec 20 | Pending review | Formal objectives, assumptions, proof scope, complexity, falsifiable hypotheses |
| M6 Experimental methodology | Dec 27 | Pending review | Frozen splits, equal budgets, baselines, metrics, replication, statistical plan |
| M7 Preliminary evidence | Dec 27 | Existing evidence, review pending | Link reproducible synthetic results and limitations, no real-model claims |
| M8 XeLaTeX proposal | Dec 31 | Pending | Idea, related work, planned methodology, references |
| M9 Presentation and oral defence | January | Pending | Match supplied template and confirmed supervisor expectations; rehearse |
| M10 Publication preparation | Spring 2027 | Pending | Novelty check, actual experiments, reproducibility, supervisor-approved manuscript |

Dates are internal targets, not official deadlines. M1 and M3 starter: [literature review](january_2027_literature_review.md).

## Existing evidence (repository README, not independently rerun in this update)
- 1,200 synthetic observations; 29,520 directed replacements, 10,479 certified increases, 9,103 certified decreases, 9,938 inconclusive.
- Independent referee recomputes synthetic results. Coordinate search improves mean development score in 12/12 configurations, mean held-out score in only 5/12.
- Target-shift regret bound and finite enumeration, native MATLAB figures.
- No measured real LLM results. Preserve all evidence separately from future model results.
- Review whether the Taylor certificate yields fewer actual evaluations when gradient and curvature estimation costs are included. The underlying smoothness inequality is standard, not itself a novelty claim.

## Spring evaluation gates
Compare random and first-improvement coordinate search, gradient-only ranking and certified screening under common candidate sets and budgets; compare compatible published methods when feasible. Count objective calls, model examples/tokens, gradient computation, optimizer-model calls and latency separately. Freeze test data before search; report uncertainty and negative findings. Real-model certification requires justified smoothness assumptions for a specified differentiable objective; no guarantee for discrete accuracy or opaque APIs. Distributed optimization remains future work.
