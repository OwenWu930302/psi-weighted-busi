"""
BUSI：用 baseline 的訓練集預測篩選 Ψ，並產生加權清單（規則同 nnU-Net 專案 docs/METHODS.md 5.4）
  低分組 = 訓練集逐張 Dice 最低 30%（含並列）
  篩選：每個 Ψ 對低分組做 Mann-Whitney，AUC 與 p（p < 0.05 = 顯著，成為專家）
  清單規則：
    AUC ≥ 0.5、連續值               → 前 30%（並列全部納入）
    AUC ≥ 0.5、整數值且前 30% 門檻 ≤ 0 → > 0（大量為 0 的離散值，否則前 30% 會選到全部）
    AUC < 0.5（反向）                → 最低 30%（並列全部納入）
  清單超過訓練集一半（離散值大量並列，例如腫瘤的 Ψ7 幾乎都是 1）時不建立專家
  只用訓練集（模型訓練過的 517 張；同 ACDC 的 all 模式），完全不看測試集
輸出：<輸出資料夾>/proxy_selection_busi.csv、<輸出資料夾>/wcases_busi_<Ψ>.csv
用法（在 Seg_UKAN 資料夾）：python screen_busi.py outputs/busi_UKAN_seed2981/percase_best.csv /mnt/data2t/BUSI/psi_lists
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(SRC)
d = d[d.split == 'train'].reset_index(drop=True)
thr = d.dice.quantile(0.30)
low = d.dice <= thr
k = int(round(len(d) * 0.30))
print(f'訓練集 {len(d)} 張；低分組 = Dice ≤ {thr:.4f}（{int(low.sum())} 張）；前 30% 名額 = {k}\n')
NOTE = {'psi2_shadow': 'undefined', 'psi3_inclusion': 'single structure', 'psi4_adjacency': 'single structure',
        'psi5_exclusion': 'single structure', 'psi9_nesting': 'single structure', 'psi10_centroid': 'single structure'}
rows = []
print(f'{"Ψ":<16}{"AUC":>7}{"p":>10}  {"狀態":<6}{"方向":<10}{"規則":<10}{"門檻":<14}{"清單":>5}')
for c in [c for c in d.columns if c.startswith('psi')]:
    x = d[c]
    if x.isna().all() or np.ptp(x.dropna()) < 1e-6:
        st = NOTE.get(c, '恆定或全 NaN')
        rows.append(dict(proxy=c, n=len(d), low_n=int(low.sum()), status=st))
        print(f'{c:<16}{"—":>7}{"—":>10}  {st}')
        continue
    g1, g0 = x[low].dropna(), x[~low].dropna()
    u, p = mannwhitneyu(g1, g0, alternative='two-sided')
    auc = u / (len(g1) * len(g0))
    status = 'significant' if p < 0.05 else ('marginal' if p < 0.10 else 'not significant')
    v = x.dropna()
    integer = np.allclose(v, np.round(v))
    if auc < 0.5:
        rule = 'bottom30'
        t = v.sort_values().iloc[k - 1]
        sel, desc = d[x <= t], f'<= {t:.6g}'
    else:
        t = v.sort_values(ascending=False).iloc[k - 1]
        if integer and t <= 0:
            rule, sel, desc = 'gt0', d[x > 0], '> 0'
        else:
            rule, sel, desc = 'top30', d[x >= t], f'>= {t:.6g}'
    r = dict(proxy=c, n=len(d), low_n=int(low.sum()), dice_threshold=thr, auc=auc, p=p,
             direction='high = hard' if auc >= 0.5 else 'low = hard', low_mean=g1.mean(), rest_mean=g0.mean(),
             status=status, rule=rule, list_threshold=desc, list_n=len(sel))
    if status == 'significant' and len(sel) > 0.5 * len(d):            # 離散值大量並列，清單超過一半就不是「特徵明顯」
        r['status'] = status = 'significant, list too large (ties)'
    if status == 'significant':
        f = f'{OUT}/wcases_busi_{c}.csv'
        sel[['case', c, 'dice']].sort_values(c, ascending=(rule == 'bottom30')).to_csv(f, index=False)
        r['list_file'] = f
    rows.append(r)
    print(f'{c:<16}{auc:>7.3f}{p:>10.2e}  {status:<6}{r["direction"]:<10}{rule:<10}{desc:<14}{len(sel):>5}'
          + ('  -> expert' if status == 'significant' else ''))
pd.DataFrame(rows).to_csv(f'{OUT}/proxy_selection_busi.csv', index=False, encoding='utf-8-sig')
print(f'\n→ {OUT}/proxy_selection_busi.csv；顯著的 Ψ 已寫出 wcases_busi_<Ψ>.csv')