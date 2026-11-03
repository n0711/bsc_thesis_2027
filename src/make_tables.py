"""Generate manuscript tables and PGFPlots data from recorded summary.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results" / "summary.json").read_text())
lines = [r"\begin{table}[htbp]", r"\centering\small", r"\caption{Equal-budget synthetic search results for $m=3$ over 50 seeds. Scores are surrogate probabilities, not LLM accuracy.}", r"\label{tab:main_results}", r"\begin{tabular}{rrrrrr}", r"\toprule", r"$B$ & Exact oracle & Random mean & Coordinate mean & Random hits & Coordinate hits \\", r"\midrule"]
for item in summary["experiments"]:
    if item['length'] != 3: continue
    a=item['methods']['random']; b=item['methods']['coordinate']
    lines.append(f"{item['budget']} & {item['oracle_train_score']:.6f} & {a['mean_train_score']:.6f} & {b['mean_train_score']:.6f} & {a['optimal_hits']}/50 & {b['optimal_hits']}/50 " + r"\\")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
lines += [r"\begin{table}[htbp]",r"\centering\small",r"\caption{Length dependence at 36 score calls. Each entry reports the mean training regret over 50 seeds.}",r"\label{tab:length}",r"\begin{tabular}{rrrrr}",r"\toprule",r"$m$ & $6^m$ & Oracle score & Random regret & Coordinate regret \\",r"\midrule"]
for item in summary["experiments"]:
    if item['budget'] != 36: continue
    a=item['methods']['random']; b=item['methods']['coordinate']
    lines.append(f"{item['length']} & {item['search_space']} & {item['oracle_train_score']:.6f} & {a['mean_regret']:.6f} & {b['mean_regret']:.6f} " + r"\\")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
(ROOT / "tex" / "generated_results.tex").write_text('\n'.join(lines)+'\n')
rows=['length budget random coordinate expected_random']
for item in summary['experiments']:
    rows.append(f"{item['length']} {item['budget']} {item['methods']['random']['mean_train_score']:.12f} {item['methods']['coordinate']['mean_train_score']:.12f} {item['random_exact_expected_best_score']:.12f}")
(ROOT/'tex'/'plot_data.tsv').write_text('\n'.join(rows)+'\n')
print('Generated evidence-backed manuscript tables and plot data')
