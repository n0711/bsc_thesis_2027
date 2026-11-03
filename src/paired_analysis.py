"""Paired-seed descriptive analysis of SYNTHETIC search results, no LLM calls."""
import csv
import json
import math
import random
from pathlib import Path


def exact_two_sided_sign_p(wins, losses):
    """Exact two-sided binomial sign test, ignoring tied pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(differences, nboot=4000, seed=1701):
    if not differences or nboot < 1:
        raise ValueError('nonempty differences and positive nboot required')
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(nboot))
    return [samples[int(.025*nboot)], samples[min(nboot-1, int(.975*nboot))]]


def analyze(rows):
    groups = {}
    for r in rows:
        key = (int(r['length']), int(r['budget']), int(r['seed']))
        slot = groups.setdefault(key, {})
        method = r['method']
        if method in slot:
            raise ValueError(f'duplicate method for {key}')
        slot[method] = r
    by_config = {}
    for (length, budget, seed), pair in groups.items():
        if set(pair) != {'random', 'coordinate'}:
            raise ValueError(f'missing method for {(length,budget,seed)}')
        if int(pair['random']['calls']) != budget or int(pair['coordinate']['calls']) != budget:
            raise ValueError('unmatched objective call budgets')
        by_config.setdefault((length,budget), []).append((seed, pair))
    result = []
    for (length,budget), entries in sorted(by_config.items()):
        entries.sort()
        diffs = [float(pair['coordinate']['train_score']) - float(pair['random']['train_score']) for _,pair in entries]
        wins = sum(d > 1e-12 for d in diffs)
        losses = sum(d < -1e-12 for d in diffs)
        result.append({'length':length,'budget':budget,'pairs':len(diffs),
                       'mean_coordinate_minus_random':sum(diffs)/len(diffs),
                       'paired_bootstrap_95pct':paired_bootstrap(diffs,seed=1701+length*100+budget),
                       'wins':wins,'losses':losses,'ties':len(diffs)-wins-losses,
                       'two_sided_sign_p_unadjusted':exact_two_sided_sign_p(wins,losses)})
    return {'evidence_type':'synthetic_surrogate_not_LLM',
            'caution':'Exploratory paired-seed comparisons; 12 configurations; p-values unadjusted for multiple comparisons; seeds are algorithmic replicates, not independent tasks.',
            'configurations':result}


def main():
    root = Path(__file__).resolve().parents[1]
    with (root/'results/runs.csv').open(newline='',encoding='utf-8') as f:
        output = analyze(list(csv.DictReader(f)))
    (root/'results/paired_analysis.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    lines = [r'\begin{table}[htbp]',r'\centering\small',r'\caption{Paired-seed differences on the synthetic surrogate (coordinate minus random; 50 paired seeds per row).}',r'\begin{tabular}{rrrrrr}',r'\toprule',r'$m$ & Budget & Mean difference & Wins & Losses & Ties\\',r'\midrule']
    for r in output['configurations']:
        lines.append(f"{r['length']} & {r['budget']} & {r['mean_coordinate_minus_random']:+.6f} & {r['wins']} & {r['losses']} & {r['ties']}\\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\label{tab:paired}',r'\end{table}']
    (root/'tex/paired_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Generated 12 paired-seed configurations (synthetic only)')

if __name__ == '__main__':
    main()
