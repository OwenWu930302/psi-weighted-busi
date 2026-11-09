"""
BUSI（U-KAN）分析：對照組（權重 1）→ 專家（Ψ ×3）
  對照組 --control：不加權的 baseline（權重全為 1 時與原版 U-KAN 完全相同）
  Ψ 與「真正低分組」的來源 --psi_from：算 Ψ、選清單的那個 baseline（與對照組、專家都是不同的模型）
  雜訊基準 --noise：另一次同設定的 baseline

【訓練集】weighted_cases_BUSI*.csv：被加權的影像本身有沒有變好（同前版）

【測試集】（130 張，論文報告分數的那份）
  特徵明顯組：套用訓練清單的門檻（只用 Ψ，不看標註）
  真正低分組：--psi_from 的 baseline 在測試集上 Dice < 0.5（用標註；僅供分析，不參與任何選擇）
    BUSI 測試集的 Dice 是雙峰：大多數 > 0.8，另有一群 < 0.5（含整顆找錯位置的 Dice≈0）
    用另一個模型定義低分組，避免「以對照組自己的低分挑影像 → 下次自然回升」的回歸平均偏差
  覆蓋率 = 真正低分組中被 Ψ 挑到的比例；精確度 = 特徵明顯組中真的是低分的比例
  四格：明顯且低分（挑對的）、明顯但不低分（誤挑的好影像）、低分但未挑到（漏掉的）、其他
輸出（outputs/analysis_busi/）：
  test_BUSI_percase.csv      逐張：prominent（特徵明顯）、low_dice（真正低分）、對照組 → 專家
  test_BUSI_summary.csv      每個專家一列：覆蓋率、整體、特徵明顯組、真正低分組的前後變化
                             救回 = 對照組 < 0.5 → 專家 ≥ 0.5；新失敗 = 對照組 ≥ 0.5 → 專家 < 0.5
  test_BUSI_4cells.csv       每個專家一列：四格各自的張數與前後變化
  test_BUSI_noise.csv        雜訊基準：同設定重訓兩次的差（判斷其他數字是否超出雜訊）
用法（在 Seg_UKAN 資料夾）：python analyze_busi.py --lists <清單資料夾>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--psi_from', default='busi_UKAN_seed2981', help='算 Ψ、選清單、定義真正低分組的 baseline')
ap.add_argument('--control', default='busi_UKAN_base2', help='對照組（權重 1）')
ap.add_argument('--noise', default='busi_UKAN_seed2981', help='雜訊基準：另一次 baseline')
ap.add_argument('--lists', default='/mnt/data2t/BUSI/psi_lists')
ap.add_argument('--expert_fmt', default='busi_UKAN_{short}w3')
ap.add_argument('--low_t', default=0.5, type=float, help='真正低分組 = baseline Dice 低於此值')
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


def grp(prefix, g):
    """一組影像的前後變化"""
    return {f'{prefix}_張數': len(g), f'{prefix}_對照組': g.control.mean(), f'{prefix}_專家': g.expert.mean(),
            f'{prefix}_Δ': g.delta.mean(), f'{prefix}_升': int((g.delta > 0).sum()),
            f'{prefix}_降': int((g.delta < 0).sum()), f'{prefix}_p': wil(g.delta)}


PSI = load(args.psi_from, 'best')
PSI is not None or sys.exit(f'找不到 {O}/{args.psi_from}/percase_best.csv（先執行 predict_busi.py --psi）')
sel = pd.read_csv(f'{args.lists}/proxy_selection_busi.csv')
sel = sel[sel.status == '顯著']
EXPERTS = []
for _, r in sel.iterrows():
    lf = r.list_file if os.path.isabs(str(r.list_file)) else os.path.join(args.lists, os.path.basename(r.list_file))
    wl = pd.read_csv(lf)
    col = r.proxy
    if r.rule == 'top30':
        t = wl[col].min(); f = (lambda t: lambda v: v >= t)(t); desc = f'≥ {t:.6g}'
    elif r.rule == 'bottom30':
        t = wl[col].max(); f = (lambda t: lambda v: v <= t)(t); desc = f'≤ {t:.6g}'
    else:
        f = lambda v: v > 0; desc = '> 0'
    EXPERTS.append(dict(psi=col, model=args.expert_fmt.format(short=col.split('_')[0]),
                        wset=set(wl.case.astype(str)), sel=f, rule=desc))

# 真正低分組：psi_from 的 baseline 在測試集上 Dice < low_t（固定定義，與 checkpoint 無關）
VAL = PSI[PSI.split == 'val']
LOW_T = args.low_t
LOW = set(VAL.index[VAL.dice < LOW_T])
print(f'真正低分組：{args.psi_from} 測試集 Dice < {LOW_T}（{len(LOW)} / {len(VAL)} 張，'
      f'其中 Dice < 0.1 的完全失敗 {int((VAL.dice < 0.1).sum())} 張）')
print('專家：' + '、'.join(f"{e['model']}（{e['psi']} {e['rule']}，清單 {len(e['wset'])} 張）" for e in EXPERTS))

TR, TRS, TRC, TE, SUM, CELL, NOISE = [], [], [], [], [], [], []
for which in ('best', 'last'):
    ctl = load(args.control, which)
    if ctl is None:
        print(f'⚠ 缺少對照組 {args.control} 的 percase_{which}.csv，略過 {which}')
        continue
    nz = load(args.noise, which)
    if nz is not None and args.noise != args.control:
        for sp, nm in (('train', '訓練集'), ('val', '測試集')):
            cs = ctl[ctl.split == sp].index.intersection(nz.index)
            g = pd.DataFrame({'control': ctl.loc[cs, 'dice'], 'expert': nz.loc[cs, 'dice']})
            g['delta'] = g.expert - g.control
            NOISE.append({'檢查點': which, '資料': nm, '說明': f'{args.noise} − {args.control}（同設定重訓兩次）',
                          **grp('全部', g), '逐張Δ標準差': g.delta.std()})
    for e in EXPERTS:
        ex = load(e['model'], which)
        if ex is None:
            print(f'⚠ 缺少 {e["model"]} 的 percase_{which}.csv，略過')
            continue
        # ---------- 訓練集：被加權的影像 ----------
        cs = ctl[ctl.split == 'train'].index.intersection(ex.index)
        t = pd.DataFrame({'control': ctl.loc[cs, 'dice'], 'expert': ex.loc[cs, 'dice']})
        t['delta'] = t.expert - t.control
        t['weighted'] = ['是' if c in e['wset'] else '否' for c in t.index]
        for c, r in t.iterrows():
            TR.append(dict(checkpoint=which, expert=e['model'], psi=e['psi'], case=c, weighted=r.weighted,
                           before_control=r.control, after_expert=r.expert, delta=r.delta))
        W, U = t[t.weighted == '是'], t[t.weighted == '否']
        TRS.append(dict(checkpoint=which, expert=e['model'], psi=e['psi'], n_weighted=len(W),
                        before_control=W.control.mean(), after_expert=W.expert.mean(), delta=W.delta.mean(),
                        提升=int((W.delta > 0).sum()), 下降=int((W.delta < 0).sum()), wilcoxon_p=wil(W.delta)))
        TRC.append(dict(checkpoint=which, expert=e['model'], n_weighted=len(W), delta_weighted=W.delta.mean(),
                        n_unweighted=len(U), delta_unweighted=U.delta.mean(),
                        weighted_minus_unweighted=W.delta.mean() - U.delta.mean(), mannwhitney_p=mwu(W.delta, U.delta)))
        # ---------- 測試集 ----------
        cs = ctl[ctl.split == 'val'].index.intersection(ex.index).intersection(PSI.index)
        g = pd.DataFrame({'control': ctl.loc[cs, 'dice'], 'expert': ex.loc[cs, 'dice'],
                          'baseline': PSI.loc[cs, 'dice'], 'psi_value': PSI.loc[cs, e['psi']]})
        g['delta'] = g.expert - g.control
        g['prominent'] = [bool(not pd.isna(v) and e['sel'](v)) for v in g.psi_value]
        g['low'] = g.index.isin(LOW)
        for c, r in g.iterrows():
            TE.append(dict(checkpoint=which, expert=e['model'], psi=e['psi'], rule=e['rule'], case=c,
                           psi_value=r.psi_value, prominent='是' if r.prominent else '否',
                           low_dice='是' if r.low else '否', baseline_dice=r.baseline,
                           before_control=r.control, after_expert=r.expert, delta=r.delta))
        P, L = g[g.prominent], g[g.low]
        PL, PN, NL, NN = g[g.prominent & g.low], g[g.prominent & ~g.low], g[~g.prominent & g.low], g[~g.prominent & ~g.low]
        SUM.append({'檢查點': which, '專家': e['model'], 'Ψ': e['psi'], '門檻（訓練清單）': e['rule'],
                    '測試集張數': len(g), '真正低分組門檻': f'baseline Dice < {LOW_T}',
                    '覆蓋率（低分組被挑到）': len(PL) / max(len(L), 1),
                    '精確度（明顯組中真的低分）': len(PL) / max(len(P), 1),
                    **grp('整體', g),
                    '整體_救回': int(((g.control < 0.5) & (g.expert >= 0.5)).sum()),
                    '整體_新失敗': int(((g.control >= 0.5) & (g.expert < 0.5)).sum()),
                    **grp('特徵明顯組', P), **grp('真正低分組', L),
                    '明顯組比其他多': P.delta.mean() - g[~g.prominent].delta.mean() if len(P) else np.nan,
                    '明顯組比其他多_p': mwu(P.delta, g[~g.prominent].delta)})
        CELL.append({'檢查點': which, '專家': e['model'], 'Ψ': e['psi'],
                     **grp('明顯且低分（挑對）', PL), **grp('明顯但不低分（誤挑的好影像）', PN),
                     **grp('低分但未挑到（漏掉）', NL), **grp('其他', NN)})

for name, rows in [('weighted_cases_BUSI', TR), ('weighted_cases_BUSI_summary', TRS),
                   ('weighted_cases_BUSI_compare', TRC), ('test_BUSI_percase', TE), ('test_BUSI_summary', SUM),
                   ('test_BUSI_4cells', CELL), ('test_BUSI_noise', NOISE)]:
    if rows:
        pd.DataFrame(rows).to_csv(f'{OUT}/{name}.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)


def show(title, rows, cols):
    if rows:
        print(f'\n{title}')
        print(pd.DataFrame(rows)[cols].to_string(index=False, float_format='%.4f'))


show('【雜訊基準】同設定重訓兩次', NOISE, ['檢查點', '資料', '全部_張數', '全部_Δ', '逐張Δ標準差', '全部_升', '全部_降'])
show('【訓練集】加權組 vs 未加權組', TRC, ['checkpoint', 'expert', 'delta_weighted', 'delta_unweighted',
                                     'weighted_minus_unweighted', 'mannwhitney_p'])
for which in ('last', 'best'):
    rows = [r for r in SUM if r['檢查點'] == which]
    show(f'【測試集 {which}】覆蓋率', rows, ['專家', '特徵明顯組_張數', '真正低分組_張數', '覆蓋率（低分組被挑到）',
                                       '精確度（明顯組中真的低分）'])
    for grpn in ('整體', '特徵明顯組', '真正低分組'):
        show(f'【測試集 {which}】{grpn}：對照組 → 專家', rows,
             ['專家'] + [f'{grpn}_{k}' for k in ('張數', '對照組', '專家', 'Δ', '升', '降', 'p')]
             + (['整體_救回', '整體_新失敗'] if grpn == '整體' else []))
    rows = [r for r in CELL if r['檢查點'] == which]
    show(f'【測試集 {which}】四格的 Δ（張數）', [{'專家': r['專家'], **{k.split('_')[0]: f"{r[k]:+.4f}（{r[k.replace('_Δ', '_張數')]}）"
                                                       for k in r if k.endswith('_Δ')}} for r in rows],
         ['專家', '明顯且低分（挑對）', '明顯但不低分（誤挑的好影像）', '低分但未挑到（漏掉）', '其他'])
print(f'\n輸出目錄：{OUT}')
