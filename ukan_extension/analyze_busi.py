"""
BUSI (U-KAN): control (weight 1) -> experts (Psi x3)

  --psi_from : baseline used to compute Psi, build the weight lists and define the low-Dice group
  --control  : control model (weight 1; identical to the original U-KAN)
  --noise    : another baseline with the same settings (noise reference)

Training set (weighted_cases_BUSI*.csv): did the weighted training images themselves improve?

Test set (130 images):
  prominent : Psi passes the threshold of the training list (label-free, from the --psi_from prediction)
  low_dice  : --psi_from baseline Dice < 0.5 (uses labels; analysis only, never used for selection)
              defined with a model other than the control to avoid regression to the mean
  hit = prominent & low_dice, false_pick = prominent & not low_dice (images that were already fine)
  coverage  = hit / low_dice      precision = hit / prominent
  threshold : minimum Psi in the training weight list (= list_threshold in proxy_selection_busi.csv)

Outputs (outputs/analysis_busi/):
  test_BUSI_selection.csv  per expert: threshold, how many images Psi selects, how many of them are low-Dice
  test_BUSI_summary.csv    per expert x group (all / prominent / low_dice / false_pick): control -> expert
  test_BUSI_percase.csv    per image
  test_BUSI_noise.csv      two baselines with identical settings (best; the first baseline has no last)
  test_BUSI_dice_hist.png  test-set Dice distribution of the baseline, with the low_dice cut-off
Only one checkpoint is analysed (--checkpoint, default last = fixed epoch).

Usage (in Seg_UKAN): python analyze_busi.py --lists <list folder>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--psi_from', default='busi_UKAN_seed2981')
ap.add_argument('--control', default='busi_UKAN_base2')
ap.add_argument('--noise', default='busi_UKAN_seed2981')
ap.add_argument('--lists', default='/mnt/data2t/BUSI/psi_lists')
ap.add_argument('--expert_fmt', default='busi_UKAN_{short}w3')
ap.add_argument('--low_t', default=0.5, type=float, help='low_dice = baseline Dice below this value')
ap.add_argument('--checkpoint', default='last', choices=['last', 'best'])
args = ap.parse_args()
O = args.output_dir
OUT = f'{O}/analysis_busi'
os.makedirs(OUT, exist_ok=True)


def load(name, which):
    f = f'{O}/{name}/percase_{which}.csv'
    return pd.read_csv(f).set_index('case') if os.path.exists(f) else None


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else np.nan


def mwu(a, b):
    return float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan


def change(g):
    """control -> expert for one group of images (columns: control, expert, delta)"""
    return dict(n=len(g), before_control=g.control.mean(), after_expert=g.expert.mean(), delta=g.delta.mean(),
                n_up=int((g.delta > 0).sum()), n_down=int((g.delta < 0).sum()), wilcoxon_p=wil(g.delta))


PSI = load(args.psi_from, 'best')
PSI is not None or sys.exit(f'missing {O}/{args.psi_from}/percase_best.csv (run predict_busi.py --psi first)')
sel = pd.read_csv(f'{args.lists}/proxy_selection_busi.csv')
sel = sel[sel.status.isin(['significant', '顯著'])]   # older lists were written in Chinese
EXPERTS = []
for _, r in sel.iterrows():
    lf = r.list_file if os.path.isabs(str(r.list_file)) else os.path.join(args.lists, os.path.basename(r.list_file))
    wl = pd.read_csv(lf)
    col = r.proxy
    if r.rule == 'top30':
        t = wl[col].min(); f = (lambda t: lambda v: v >= t)(t); desc = f'>= {t:.6g}'
    elif r.rule == 'bottom30':
        t = wl[col].max(); f = (lambda t: lambda v: v <= t)(t); desc = f'<= {t:.6g}'
    else:
        f = lambda v: v > 0; desc = '> 0'
    EXPERTS.append(dict(psi=col, model=args.expert_fmt.format(short=col.split('_')[0]),
                        wset=set(wl.case.astype(str)), sel=f, rule=desc))

# ---------- test-set selection (depends only on the --psi_from baseline, not on checkpoint) ----------
VAL = PSI[PSI.split == 'val']
LOW = VAL.dice < args.low_t
# Dice histogram of the baseline (why 0.5: the distribution is bimodal)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.arange(0, 1.0001, 0.05)
cnt, _, bars = ax.hist(VAL.dice, bins=bins, color='#7a9cc6', edgecolor='white')
for b, c in zip(bars, cnt):
    if b.get_x() < args.low_t:
        b.set_color('#d9534f')
    if c:
        ax.text(b.get_x() + 0.025, c + 0.5, int(c), ha='center', fontsize=8)
ax.axvline(args.low_t, color='k', ls='--', lw=1)
ax.text(args.low_t - 0.01, cnt.max() * 0.9, f'low_dice: Dice < {args.low_t}\n{int(LOW.sum())} images '
        f'({int((VAL.dice < 0.1).sum())} below 0.1)', ha='right', fontsize=9)
ax.set_xlabel('Dice'); ax.set_ylabel('images'); ax.set_xlim(0, 1)
ax.set_title(f'BUSI test set ({len(VAL)} images): Dice of {args.psi_from}', fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/test_BUSI_dice_hist.png', dpi=150); plt.close(fig)

SELROWS, PROM = [], {}
for e in EXPERTS:
    p = VAL[e['psi']].apply(lambda v: bool(not pd.isna(v) and e['sel'](v)))
    PROM[e['model']] = p
    hit, n_low, n_p = int((p & LOW).sum()), int(LOW.sum()), int(p.sum())
    SELROWS.append(dict(expert=e['model'], psi=e['psi'], threshold=e['rule'], n_test=len(VAL),
                        n_prominent=n_p, n_low_dice=n_low, n_hit=hit, n_false_pick=n_p - hit,
                        n_missed=n_low - hit, coverage=hit / max(n_low, 1), precision=hit / max(n_p, 1)))

TR, TRS, TRC, TE, SUM, NOISE = [], [], [], [], [], []
# noise reference: the first baseline only has best
ctl, nz = load(args.control, 'best'), load(args.noise, 'best')
if ctl is not None and nz is not None and args.noise != args.control:
    for sp, nm in (('train', 'train'), ('val', 'test')):
        cs = ctl[ctl.split == sp].index.intersection(nz.index)
        d = nz.loc[cs, 'dice'] - ctl.loc[cs, 'dice']
        NOISE.append(dict(checkpoint='best', split=nm, model_a=args.control, model_b=args.noise, n=len(cs),
                          dice_a=ctl.loc[cs, 'dice'].mean(), dice_b=nz.loc[cs, 'dice'].mean(), delta=d.mean(),
                          sd_delta=d.std(), n_up=int((d > 0).sum()), n_down=int((d < 0).sum()),
                          wilcoxon_p=wil(d)))
for which in (args.checkpoint,):
    ctl = load(args.control, which)
    if ctl is None:
        print(f'skip {which}: missing {args.control}/percase_{which}.csv')
        continue
    for e in EXPERTS:
        ex = load(e['model'], which)
        if ex is None:
            print(f'skip {e["model"]} {which}: missing percase_{which}.csv')
            continue
        # ---------- training set: weighted images ----------
        cs = ctl[ctl.split == 'train'].index.intersection(ex.index)
        t = pd.DataFrame({'control': ctl.loc[cs, 'dice'], 'expert': ex.loc[cs, 'dice']})
        t['delta'] = t.expert - t.control
        t['weighted'] = [c in e['wset'] for c in t.index]
        for c, r in t.iterrows():
            TR.append(dict(checkpoint=which, expert=e['model'], case=c, weighted='yes' if r.weighted else 'no',
                           before_control=r.control, after_expert=r.expert, delta=r.delta))
        W, U = t[t.weighted], t[~t.weighted]
        TRS.append(dict(checkpoint=which, expert=e['model'], **change(W)))
        TRC.append(dict(checkpoint=which, expert=e['model'], n_weighted=len(W), delta_weighted=W.delta.mean(),
                        n_unweighted=len(U), delta_unweighted=U.delta.mean(),
                        weighted_minus_unweighted=W.delta.mean() - U.delta.mean(),
                        mannwhitney_p=mwu(W.delta, U.delta)))
        # ---------- test set ----------
        cs = VAL.index.intersection(ctl.index).intersection(ex.index)
        g = pd.DataFrame({'control': ctl.loc[cs, 'dice'], 'expert': ex.loc[cs, 'dice']})
        g['delta'] = g.expert - g.control
        p, low = PROM[e['model']].loc[cs], LOW.loc[cs]
        for c, r in g.iterrows():
            TE.append(dict(checkpoint=which, expert=e['model'], case=c, psi_value=VAL.loc[c, e['psi']],
                           prominent='yes' if p[c] else 'no', low_dice='yes' if low[c] else 'no',
                           baseline_dice=VAL.loc[c, 'dice'], before_control=r.control, after_expert=r.expert,
                           delta=r.delta))
        for name, m in [('all', p | ~p), ('prominent', p), ('low_dice', low), ('false_pick', p & ~low)]:
            SUM.append(dict(checkpoint=which, expert=e['model'], group=name, **change(g[m])))

for name, rows in [('weighted_cases_BUSI', TR), ('weighted_cases_BUSI_summary', TRS),
                   ('weighted_cases_BUSI_compare', TRC), ('test_BUSI_selection', SELROWS),
                   ('test_BUSI_summary', SUM), ('test_BUSI_percase', TE), ('test_BUSI_noise', NOISE)]:
    if rows:
        pd.DataFrame(rows).to_csv(f'{OUT}/{name}.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
for title, rows in [('noise (two baselines)', NOISE), ('training set: weighted vs unweighted', TRC),
                    ('test set: selection', SELROWS),
                    (f'test set ({args.checkpoint}): control -> expert', SUM)]:
    if rows:
        print(f'\n[{title}]')
        print(pd.DataFrame(rows).to_string(index=False, float_format='%.4f'))
print(f'\nlow_dice = {args.psi_from} test Dice < {args.low_t}; output: {OUT}')
