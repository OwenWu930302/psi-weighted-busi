#!/bin/bash
# BUSI 全部流程，一次跑完（可中斷後重跑：已完成的訓練與預測會自動略過）
#   1. 修改 train.py（loss = Σ wᵢ·Lᵢ / n；--move_to_train）
#   2. 訓練 Ψ 專家 busi_UKAN_psi*w3n（對照組 base2 已完成，略過）→ 預測
#   3. 分析專家（outputs/analysis_busi/weighted_cases_BUSI*.csv、test_BUSI_*）
#   4. 搬移清單：測試集兩個 baseline 皆 Dice < 0.2 的影像
#   5. 訓練 busi_UKAN_moved（權重 1，上述影像移到訓練集）→ 預測
#   6. 分析搬移（outputs/analysis_busi/moved_BUSI_*.csv）
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_all.sh > run_busi_all.log 2>&1 &
cd "$(dirname "$0")"
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
MOVED=busi_UKAN_moved
MLIST=$LISTS/moved_busi_dice02.csv
for f in train.py patch_ukan_weighted.py predict_busi.py train_busi_queue.sh analyze_busi.py moved_busi.py; do
  [ -f $f ] || { echo "缺少 $f（請放在 Seg_UKAN 資料夾）"; exit 1; }
done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "1. 修改 train.py"
python patch_ukan_weighted.py || exit 1
grep -q "move_to_train" train.py && grep -q "default='n'" train.py || { echo "train.py 修改失敗"; exit 1; }

step "2. Ψ 專家（w3n）"
WNORM=n LISTS=$LISTS bash train_busi_queue.sh

step "3. 分析專家"
python analyze_busi.py --lists $LISTS --expert_fmt 'busi_UKAN_{short}w3n' || exit 1

step "4. 搬移清單"
python moved_busi.py make --list $MLIST || exit 1

step "5. 搬移後訓練 $MOVED"
if [ -f outputs/$MOVED/model_last.pth ] && [ "$(tail -n +2 outputs/$MOVED/log.csv 2>/dev/null | wc -l)" -ge 400 ]; then
  echo "已完成，略過訓練"
else
  python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name $MOVED --data_dir ./inputs \
    --move_to_train $MLIST || exit 1
fi
for w in best last; do
  [ -f outputs/$MOVED/percase_$w.csv ] && echo "已有 percase_$w.csv，略過預測" || python predict_busi.py --name $MOVED --which $w || exit 1
done

step "6. 分析搬移"
python moved_busi.py analyze --list $MLIST --models $MOVED
step "全部完成"