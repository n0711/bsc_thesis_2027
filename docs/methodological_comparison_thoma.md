# Research note - Thoma and Hadjicostis as a methodological comparison

Status: literature/context note for supervisor review, 18 November 2026. This is **not** a new experiment or an endorsement of importing cybersecurity models into prompt engineering.

## Verified source and scope

- M. Thoma and C. N. Hadjicostis, "Detection of collaborative misbehaviour in distributed cyber-attacks", *Computer Communications* **174**, 28-41 (2021). DOI: https://doi.org/10.1016/j.comcom.2021.04.013
- M. Thoma, "Detection of Collaborative Cyber-Attacks through Correlation and Time Dependency Analysis", University of Cyprus ECE seminar abstract (2 March 2016). https://www.ucy.ac.cy/ece/wp-content/uploads/sites/41/2021/06/Marios_Thoma_-_02.03.2016.pdf

The work concerns detection of coordinated network behaviour. The University of Cyprus seminar abstract explicitly refers to hidden Markov models and mathematical/simulation-based analysis. It does **not** investigate language-model prompts or discrete-continuous prompt optimization.

## Narrow methodological analogy

| Question | Thoma/Hadjicostis research | Current BSc thesis |
|---|---|---|
| Target phenomenon | Coordinated patterns in distributed network traffic | Score changes after discrete prompt-token replacements |
| Modelled object | Correlated behaviour and time dependency | Differentiable embedding-space scoring surrogate |
| Validation principle | Compare a formal detection model against relevant observations | Compare gradient-predicted finite changes against exact finite changes |
| Supported conclusion | Only results about the specified network-detection setting | Only results about the specified finite synthetic objective |

In the current thesis, the prediction is the first-order directional estimate
`predicted_delta = gradient · (new_embedding - old_embedding)`,
and the observation is `actual_delta = score(new_prompt) - score(old_prompt)`.
The existing audit tests sign agreement, top-ranked neighbour agreement and rank correlation.

**Important boundary:** this comparison is interpretive, not evidence that a hidden Markov model would improve prompt search. HMM research should not be inserted into the experiments without an independently specified sequential model, an evaluation hypothesis and appropriate data.

## Suggested discussion with the supervisor

1. Is a two-paragraph cross-domain methodology precedent appropriate in Chapter 2, or should it stay in this research note?
2. Should the real-model protocol predeclare a primary paired effect estimate and a method for computing uncertainty, rather than only naming paired intervals?
3. Can the final manuscript distinguish deterministic surrogate error, search-seed variability and real-model inference stochasticity more explicitly?

## Evidence status

- **Implemented in this proposal:** literature context and references only.
- **Already in the repository:** finite-space synthetic audit, paired-seed analyses, and a prospective real-model protocol.
- **Not performed here:** new simulations, new real-model inference, HMM experiments, independent reproduction of the 2021 paper, or XeLaTeX compilation.

No numerical outcomes or original claims are attributed to Thoma's work beyond what the cited sources establish.
