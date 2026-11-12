#!/bin/bash
# BUSI 追加實驗（官方切分 517 / 130，不搬移任何影像；結果僅供參考）
#   1. base3：第三次不加權訓練 → 判斷對照組 base2 的 last 是否偏低（重訓雜訊）
#   2. ×10 專家 busi_UKAN_psi*w10n、3. ×20 專家 busi_UKAN_psi*w20n（loss ÷ n）
#   4. 分析：每個倍率 vs base2、vs base3 → outputs/analysis_busi_w{3,10,20}n*/，總表 outputs/analysis_busi_compare.csv
# 可中斷後重跑（已完成的訓練與預測會略過）
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_more.sh > run_busi_more.log 2>&1 &
cd "$(dirname "$0")"
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
FACTORS=${FACTORS:-"10 20"}
for f in train.py predict_busi.py train_busi_queue.sh analyze_busi.py; do
  [ -f $f ] || { echo "缺少 $f（請放在 Seg_UKAN 資料夾）"; exit 1; }
done
grep -q "wnorm" train.py || { echo "train.py 尚未修改（先執行 python patch_ukan_weighted.py）"; exit 1; }
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "1. base3（不加權）"
N=busi_UKAN_base3
if [ -f outputs/$N/model_last.pth ] && [ "$(tail -n +2 outputs/$N/log.csv 2>/dev/null | wc -l)" -ge 400 ]; then
  echo "已完成，略過訓練"
else
  # 與 train_busi_queue.sh 訓練 base2 的指令相同，只差名稱
  python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name $N --data_dir ./inputs --epochs 400 || exit 1
fi
for w in best last; do
  [ -f outputs/$N/percase_$w.csv ] && echo "已有 percase_$w.csv，略過預測" || python predict_busi.py --name $N --which $w || exit 1
done

for k in $FACTORS; do
  step "×$k 專家"
  WFACTOR=$k WNORM=n LISTS=$LISTS bash train_busi_queue.sh
done

step "分析"
for k in 3 $FACTORS; do
  python analyze_busi.py --lists $LISTS --expert_fmt "busi_UKAN_{short}w${k}n" --out analysis_busi_w${k}n \
    > outputs/analysis_busi_w${k}n.txt 2>&1 || { cat outputs/analysis_busi_w${k}n.txt; exit 1; }
  python analyze_busi.py --lists $LISTS --expert_fmt "busi_UKAN_{short}w${k}n" --out analysis_busi_w${k}n_vs_base3 \
    --control busi_UKAN_base3 --noise busi_UKAN_base2 > outputs/analysis_busi_w${k}n_vs_base3.txt 2>&1
done

FACTORS="3 $FACTORS" python - <<'PY'
import os
import pandas as pd
O = 'outputs'
pd.set_option('display.width', 220)
nz = pd.read_csv(f'{O}/analysis_busi_w3n/test_BUSI_noise.csv')
print('[重訓雜訊：其他 baseline − 對照組 base2]')
print(nz[['checkpoint', 'split', 'model_b', 'n', 'dice_a', 'dice_b', 'delta', 'sd_delta', 'n_up', 'n_down', 'wilcoxon_p']]
      .to_string(index=False, float_format='%.4f'))
rows = []
for k in os.environ['FACTORS'].split():
    d = f'{O}/analysis_busi_w{k}n'
    if not os.path.exists(f'{d}/test_BUSI_summary.csv'):
        continue
    tc = pd.read_csv(f'{d}/weighted_cases_BUSI_compare.csv').set_index('expert')
    ts = pd.read_csv(f'{d}/weighted_cases_BUSI_summary.csv').set_index('expert')
    s = pd.read_csv(f'{d}/test_BUSI_summary.csv')
    f3 = f'{d}_vs_base3/test_BUSI_summary.csv'
    s3 = pd.read_csv(f3) if os.path.exists(f3) else None
    for e in s.expert.unique():
        g = s[s.expert == e].set_index('group')
        r = dict(weight=k, expert=e, train_weighted_delta=ts.loc[e, 'delta'],
                 train_weighted_minus_unweighted=tc.loc[e, 'weighted_minus_unweighted'],
                 test_all=g.loc['all', 'after_expert'], all_delta=g.loc['all', 'delta'], all_p=g.loc['all', 'wilcoxon_p'],
                 prominent_delta=g.loc['prominent', 'delta'], low_dice_delta=g.loc['low_dice', 'delta'],
                 false_pick_delta=g.loc['false_pick', 'delta'])
        if s3 is not None:
            g3 = s3[s3.expert == e].set_index('group')
            r.update(all_delta_vs_base3=g3.loc['all', 'delta'], all_p_vs_base3=g3.loc['all', 'wilcoxon_p'])
        rows.append(r)
c = pd.DataFrame(rows)
c.to_csv(f'{O}/analysis_busi_compare.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
print('\n[權重倍率比較（last；Δ = 專家 − 對照組 base2）]')
print(c.to_string(index=False, float_format='%.4f'))
print(f'\n總表：{O}/analysis_busi_compare.csv；各倍率詳細：{O}/analysis_busi_w*n*/')
PY
step "全部完成"