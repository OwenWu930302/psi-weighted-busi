"""
BUSI 搬移實驗：把測試集中 Dice < 0.2 的影像移到訓練集，看它們在訓練集上能不能被學起來
  make   ：選出兩個 baseline（seed2981 best、base2 last）在測試集都 Dice < 0.2 的影像 → 清單 csv
  analyze：對照組（base2，影像在測試集）→ 搬移後的模型（影像在訓練集），last
用法（在 Seg_UKAN 資料夾）：
  python moved_busi.py make
  python train.py ... --name busi_UKAN_moved --move_to_train /mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv
  python moved_busi.py analyze
輸出（outputs/analysis_busi/）：
  moved_BUSI_percase.csv  搬移的每張影像：before_control（在測試集）→ after_moved（在訓練集）
  moved_BUSI_summary.csv  moved / remaining_test / original_train 三組的前後變化
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('step', choices=['make', 'analyze'])
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--list', default='/mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv')
ap.add_argument('--threshold', default=0.2, type=float)
ap.add_argument('--control', default='busi_UKAN_base2')
ap.add_argument('--models', default='busi_UKAN_moved,busi_UKAN_movedw3n', help='搬移後訓練的模型（不存在的略過）')
args = ap.parse_args()
O = args.output_dir


def load(name, which):
    f = f'{O}/{name}/percase_{which}.csv'
    os.path.exists(f) or sys.exit(f'missing {f}')
    return pd.read_csv(f).set_index('case')


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else np.nan


if args.step == 'make':
    a = load('busi_UKAN_seed2981', 'best')
    b = load(args.control, 'last')
    a = a[a.split == 'val'].dice
    b = b.loc[a.index, 'dice']
    m = (a < args.threshold) & (b < args.threshold)
    out = pd.DataFrame({'case': a.index[m], 'dice_seed2981_best': a[m].values, 'dice_base2_last': b[m].values})
    out = out.sort_values('dice_base2_last')
    out.to_csv(args.list, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(out.to_string(index=False, float_format='%.4f'))
    print(f'\n{len(out)} 張測試影像 Dice < {args.threshold}（兩個 baseline 皆是）→ {args.list}')
    sys.exit()

mv = set(pd.read_csv(args.list, encoding='utf-8-sig').case.astype(str))
ctl = load(args.control, 'last')
P, S = [], []
for name in args.models.split(','):
    if not os.path.exists(f'{O}/{name}/percase_last.csv'):
        print(f'skip {name}: missing percase_last.csv')
        continue
    ex = load(name, 'last')
    bad = [c for c in mv if ex.loc[c, 'split'] != 'train']
    bad and sys.exit(f'{name}: {bad} 不在訓練集，模型不是用 --move_to_train 訓練的？')
    g = pd.DataFrame({'control': ctl.dice, 'moved': ex.loc[ctl.index, 'dice'], 'split': ctl.split})
    g['delta'] = g.moved - g.control
    for c in sorted(mv, key=lambda c: g.loc[c, 'control']):
        P.append(dict(model=name, case=c, before_control=g.loc[c, 'control'], after_moved=g.loc[c, 'moved'],
                      delta=g.loc[c, 'delta']))
    for grp, m in [('moved (test -> train)', g.index.isin(mv)),
                   ('remaining_test', (g.split == 'val') & ~g.index.isin(mv)),
                   ('original_train', g.split == 'train')]:
        x = g[m]
        S.append(dict(model=name, group=grp, n=len(x), before_control=x.control.mean(), after_moved=x.moved.mean(),
                      delta=x.delta.mean(), n_up=int((x.delta > 0).sum()), n_down=int((x.delta < 0).sum()),
                      wilcoxon_p=wil(x.delta)))
P or sys.exit('沒有搬移後的模型可分析')
os.makedirs(f'{O}/analysis_busi', exist_ok=True)
for fn, rows in (('moved_BUSI_percase', P), ('moved_BUSI_summary', S)):
    pd.DataFrame(rows).to_csv(f'{O}/analysis_busi/{fn}.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
pd.set_option('display.width', 200)
print(pd.DataFrame(P).to_string(index=False, float_format='%.4f'))
print()
print(pd.DataFrame(S).to_string(index=False, float_format='%.4f'))
print(f'\ncontrol = {args.control} (last); output: {O}/analysis_busi/moved_BUSI_*.csv')