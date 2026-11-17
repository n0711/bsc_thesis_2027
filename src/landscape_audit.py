"""Enumerate discrete stopping points on the fixed synthetic objective."""
import json
import math
from pathlib import Path

from experiment import EPS, LENGTHS, VOCAB, embed_prompt, enumerate_space, score, score_gradient

ROOT = Path(__file__).resolve().parents[1]


def landscape(length):
    scores = {prompt: score(prompt) for prompt in enumerate_space(length)}
    optimum = max(scores.values())
    terminal = []
    comparisons = 0
    for prompt, value in scores.items():
        neighbor_changes = []
        for index, old in enumerate(prompt):
            for token in VOCAB:
                if old == token:
                    continue
                neighbor = prompt[:index] + (token,) + prompt[index + 1:]
                comparisons += 1
                neighbor_changes.append(scores[neighbor] - value)
        maximum_change = max(neighbor_changes)
        if maximum_change <= EPS:
            gradient = score_gradient(embed_prompt(prompt))
            gradient_norm = math.sqrt(sum(component * component for component in gradient))
            terminal.append({
                "prompt": list(prompt),
                "score": value,
                "max_neighbor_change": maximum_change,
                "suboptimal": value < optimum - EPS,
                "gradient_norm": gradient_norm,
                "nonzero_gradient": gradient_norm > EPS,
            })
    expected = len(scores) * length * (len(VOCAB) - 1)
    assert comparisons == expected
    return {
        "length": length,
        "prompt_count": len(scores),
        "directed_comparison_count": comparisons,
        "terminal_count": len(terminal),
        "suboptimal_terminal_count": sum(state["suboptimal"] for state in terminal),
        "nonimproving_nonzero_gradient_count": sum(
            state["nonzero_gradient"] for state in terminal
        ),
        "terminal_states": terminal,
    }


def build_report():
    return {
        "schema_version": 1,
        "evidence_type": "synthetic_surrogate_not_LLM",
        "improvement_tolerance": EPS,
        "lengths": [landscape(length) for length in LENGTHS],
    }


def _latex(report):
    lines = [
        r"\section{Where single-token search can stop}",
        r"\label{sec:coordinate-stopping}",
        r"A finite budget can end a run while an improving move is still available.",
        r"A different limitation arises when every permitted move fails the improvement",
        r"test. We enumerated the training-score landscape independently of the fifty search seeds.",
        r"",
        r"For a prompt $p$ of length $m$, let $N(p)$ contain every tuple obtained by",
        r"replacing exactly one token. With six symbols, each prompt has $5m$ neighbours.",
        r"Writing $S$ for the implemented mean logistic training score, we count $p$",
        r"as terminal when",
        r"\[",
        r"  \max_{q\in N(p)} S(q) \leq S(p)+10^{-12}.",
        r"\]",
        r"This is the improvement threshold in \texttt{coordinate\_search}. A terminal",
        r"prompt is suboptimal when its score lies more than $10^{-12}$ below the",
        r"enumerated maximum at the same length.",
        r"",
        r"\begin{center}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$m$ & Prompts & Directed comparisons & Terminal & Suboptimal terminal & Terminal, nonzero $\nabla F$ \\",
        r"\midrule",
    ]
    total_terminal = 0
    total_suboptimal = 0
    total_nonzero = 0
    for result in report["lengths"]:
        lines.append(
            f"{result['length']} & {result['prompt_count']:,} & "
            f"{result['directed_comparison_count']:,} & "
            f"{result['terminal_count']} & {result['suboptimal_terminal_count']} & "
            f"{result['nonimproving_nonzero_gradient_count']} "
            + r"\\"
        )
        total_terminal += result["terminal_count"]
        total_suboptimal += result["suboptimal_terminal_count"]
        total_nonzero += result["nonimproving_nonzero_gradient_count"]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        (
            f"Across these lengths, {total_terminal} states are terminal, of which "
            f"{total_suboptimal} are suboptimal. In {total_nonzero} terminal states, "
            r"the continuous gradient norm exceeds $10^{-12}$. Thus the absence of "
            r"an improving discrete neighbor does not imply a zero continuous gradient."
        ),
        "",
        r"\paragraph{Reproduction.}",
        r"From the repository root run \texttt{python3 src/landscape\_audit.py}.",
        r"The script writes terminal-state details to",
        r"\texttt{results/landscape\_audit.json} and regenerates this table.",
        "",
        r"\paragraph{What still requires validation.}",
        r"These counts give no visit frequencies. Attributing a failed trial to a trap",
        r"requires its seed, length, budget, evaluated moves and scores in order,",
        r"incumbent changes, completed neighbourhood sweeps, restarts and final",
        r"evaluation index. Measuring the benefit of restarting requires a matched",
        r"no-restart baseline with a declared rule for unused calls, the same starts,",
        r"budgets and scorer, and paired per-seed outcomes. Extending the result to",
        r"language-model tasks requires a pinned model and tokenizer, fixed task and",
        r"split IDs, a frozen prompt space and move rule, decoding settings, a scoring",
        r"rule, and retained per-example predictions and call logs. No model inference",
        r"enters the enumeration above.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report, root=ROOT):
    results_dir = root / "results"
    tex_dir = root / "tex"
    results_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(
        _round_floats(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    (results_dir / "landscape_audit.json").write_text(
        json_text, encoding="utf-8", newline="\n"
    )
    (tex_dir / "coordinate_landscape.tex").write_text(
        _latex(report), encoding="utf-8", newline="\n"
    )


def _round_floats(value):
    if isinstance(value, float):
        rounded = round(value, 12)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def main():
    report = build_report()
    write_outputs(report)
    for result in report["lengths"]:
        print(
            result["length"],
            result["prompt_count"],
            result["directed_comparison_count"],
            result["terminal_count"],
            result["suboptimal_terminal_count"],
            result["nonimproving_nonzero_gradient_count"],
        )
    print("Wrote results/landscape_audit.json and tex/coordinate_landscape.tex")


if __name__ == "__main__":
    main()
