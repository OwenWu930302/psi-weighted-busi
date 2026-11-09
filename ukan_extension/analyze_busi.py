"""
BUSI（U-KAN）分析：對照組（權重 1）→ 專家（Ψ ×3），格式同 nnU-Net 專案的 weighted_cases / test_prominent
  對照組：權重全為 1 時與原版 U-KAN 完全相同，所以對照組就是 baseline（--control）
  雜訊基準（--noise）：同設定再訓練一次的 baseline，兩次 baseline 的差 = 重訓本身的波動
  Ψ 與門檻：由 --psi_from 的 baseline 預測計算（screen_busi.py 產生的清單），不用專家的預測
  1. 訓練集（517 張）：被加權的影像本身有沒有變好（weighted_cases）
  2. 測試集（130 張，論文報告分數的那份）：
       特徵明顯 = 套用訓練清單的門檻（前 30%：Ψ ≥ 清單最小值；反向：Ψ ≤ 清單最大值；> 0）
       比較明顯組、其他影像、整個測試集（test_prominent）
  兩種 checkpoint 各算一次：best（原作者用法：驗證集最佳 epoch）、last（最後一個 epoch，不經挑選）
輸出：outputs/analysis_busi/*.csv
用法（在 Seg_UKAN 資料夾）：python analyze_busi.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--psi_from', default='busi_UKAN_seed2981', help='算 Ψ 與清單用的 baseline')
ap.add_argument('--control', default='busi_UKAN_base2', help='對照組（權重 1）')
ap.add_argument('--noise', default='busi_UKAN_seed2981', help='雜訊基準：另一次 baseline')
ap.add_argument('--lists', default='/mnt/data2t/BUSI/psi_lists')
ap.add_argument('--expert_fmt', default='busi_UKAN_{short}w3', help='{short} = psi1、psi6 …')
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


PSI = load(args.psi_from, 'best')
PSI is not None or sys.exit(f'找不到 {O}/{args.psi_from}/percase_best.csv（先執行 predict_busi.py --psi）')
sel = pd.read_csv(f'{args.lists}/proxy_selection_busi.csv')
sel = sel[sel.status == '顯著']
EXPERTS = []
for _, r in sel.iterrows():
    wl = pd.read_csv(r.list_file)
    col = r.proxy
    if r.rule == 'top30':
        t = wl[col].min(); f = (lambda t: lambda v: v >= t)(t); desc = f'≥ {t:.6g}'
    elif r.rule == 'bottom30':
        t = wl[col].max(); f = (lambda t: lambda v: v <= t)(t); desc = f'≤ {t:.6g}'
    else:
        f = lambda v: v > 0; desc = '> 0'
    EXPERTS.append(dict(psi=col, model=args.expert_fmt.format(short=col.split('_')[0]),
                        wset=set(wl.case.astype(str)), sel=f, rule=desc))
print('專家：' + '、'.join(f"{e['model']}（{e['psi']} {e['rule']}，清單 {len(e['wset'])} 張）" for e in EXPERTS))

S, C, TR, TE, TRS, TRC = [], [], [], [], [], []
for which in ('best', 'last'):
    ctl = load(args.control, which)
    if ctl is None:
        print(f'⚠ 缺少對照組 {args.control} 的 percase_{which}.csv，略過 {which}')
        continue
    # ---------- 雜訊基準：另一次 baseline vs 對照組 ----------
    nz = load(args.noise, which)
    if nz is not None and args.noise != args.control:
        for sp in ('train', 'val'):
            cs = ctl[ctl.split == sp].index.intersection(nz.index)
            dd = nz.loc[cs, 'dice'] - ctl.loc[cs, 'dice']
            S.append(dict(checkpoint=which, analysis='雜訊基準（另一次 baseline − 對照組）', split=sp, model=args.noise,
                          n=len(cs), before=ctl.loc[cs, 'dice'].mean(), after=nz.loc[cs, 'dice'].mean(),
                          delta=dd.mean(), sd=dd.std(), up=int((dd > 0).sum()), down=int((dd < 0).sum()), p=wil(dd)))
    for e in EXPERTS:
        ex = load(e['model'], which)
        if ex is None:
            print(f'⚠ 缺少 {e["model"]} 的 percase_{which}.csv，略過')
            continue
        # ---------- 1. 訓練集：被加權的影像 ----------
        cs = ctl[ctl.split == 'train'].index.intersection(ex.index)
        for c in cs:
            TR.append(dict(checkpoint=which, expert=e['model'], psi=e['psi'], case=c,
                           weighted='是' if c in e['wset'] else '否', before_control=ctl.loc[c, 'dice'],
                           after_expert=ex.loc[c, 'dice'], delta=ex.loc[c, 'dice'] - ctl.loc[c, 'dice']))
        t = pd.DataFrame([r for r in TR if r['checkpoint'] == which and r['expert'] == e['model']])
        W, U = t[t.weighted == '是'], t[t.weighted == '否']
        TRS.append(dict(checkpoint=which, expert=e['model'], psi=e['psi'], n_weighted=len(W),
                        before_control=W.before_control.mean(), after_expert=W.after_expert.mean(),
                        delta=W.delta.mean(), 提升=int((W.delta > 0).sum()), 下降=int((W.delta < 0).sum()),
                        wilcoxon_p=wil(W.delta)))
        TRC.append(dict(checkpoint=which, expert=e['model'], n_weighted=len(W), delta_weighted=W.delta.mean(),
                        n_unweighted=len(U), delta_unweighted=U.delta.mean(),
                        weighted_minus_unweighted=W.delta.mean() - U.delta.mean(), mannwhitney_p=mwu(W.delta, U.delta)))
        # ---------- 2. 測試集：特徵明顯的影像 ----------
        cs = ctl[ctl.split == 'val'].index.intersection(ex.index).intersection(PSI.index)
        for c in cs:
            v = PSI.loc[c, e['psi']]
            TE.append(dict(checkpoint=which, expert=e['model'], psi=e['psi'], rule=e['rule'], case=c, psi_value=v,
                           prominent='是' if (not pd.isna(v) and e['sel'](v)) else '否',
                           before_control=ctl.loc[c, 'dice'], after_expert=ex.loc[c, 'dice'],
                           delta=ex.loc[c, 'dice'] - ctl.loc[c, 'dice']))
        t = pd.DataFrame([r for r in TE if r['checkpoint'] == which and r['expert'] == e['model']])
        P_, N_ = t[t.prominent == '是'], t[t.prominent == '否']
        S.append(dict(checkpoint=which, analysis='測試集：特徵明顯組', split='val', model=e['model'], rule=e['rule'],
                      n=len(P_), before=P_.before_control.mean(), after=P_.after_expert.mean(), delta=P_.delta.mean(),
                      sd=P_.delta.std(), up=int((P_.delta > 0).sum()), down=int((P_.delta < 0).sum()), p=wil(P_.delta)))
        S.append(dict(checkpoint=which, analysis='測試集：整體', split='val', model=e['model'],
                      n=len(t), before=t.before_control.mean(), after=t.after_expert.mean(), delta=t.delta.mean(),
                      sd=t.delta.std(), up=int((t.delta > 0).sum()), down=int((t.delta < 0).sum()), p=wil(t.delta)))
        C.append(dict(checkpoint=which, expert=e['model'], n_prominent=len(P_), delta_prominent=P_.delta.mean(),
                      n_other=len(N_), delta_other=N_.delta.mean(),
                      prominent_minus_other=P_.delta.mean() - N_.delta.mean() if len(P_) else np.nan,
                      mannwhitney_p=mwu(P_.delta, N_.delta)))

for name, rows in [('weighted_cases_BUSI', TR), ('weighted_cases_BUSI_summary', TRS),
                   ('weighted_cases_BUSI_compare', TRC), ('test_prominent_BUSI', TE),
                   ('test_prominent_BUSI_summary', S), ('test_prominent_BUSI_compare', C)]:
    if rows:
        pd.DataFrame(rows).to_csv(f'{OUT}/{name}.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
pd.set_option('display.width', 250)
if TRS:
    print('\n【訓練集】被加權的影像：對照組 → 專家')
    print(pd.DataFrame(TRS).to_string(index=False, float_format='%.4f'))
    print('\n【訓練集】加權組 vs 未加權組（差距 = 加權本身的效果）')
    print(pd.DataFrame(TRC).to_string(index=False, float_format='%.4f'))
if S:
    print('\n【測試集】特徵明顯組、整體，以及雜訊基準')
    print(pd.DataFrame(S).to_string(index=False, float_format='%.4f'))
    print('\n【測試集】特徵明顯組 vs 其他影像')
    print(pd.DataFrame(C).to_string(index=False, float_format='%.4f'))
print(f'\n輸出目錄：{OUT}')