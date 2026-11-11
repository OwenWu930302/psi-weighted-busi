#!/bin/bash
# 依序訓練：baseline 第二次（對照組，含 model_last.pth；與第一次比較 = 重訓雜訊）→ 每個顯著 Ψ 的專家（×3）
# 全部與原作者設定相同（種子 2981、400 epoch、batch 8、Adam），只差在有沒有 --wcases
# 每個訓練完立即預測訓練集＋測試集（best、last 兩種 checkpoint）
# 可中斷重跑：已有 model_last.pth 且 log 跑滿 400 epoch 的會略過
# 用法（在 Seg_UKAN 資料夾、ukan 環境）：nohup bash train_busi_queue.sh > train_busi_queue.log 2>&1 &
# 專家 loss = Σ wᵢ·Lᵢ / n（busi_UKAN_psi*w3n，與原版的平均相同）；舊版 ÷Σw（w3）已不使用，只在重現時用 WNORM=sum
#   對照組 base2 已訓練完成會自動略過
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
[ -f train.py ] && grep -q "wcases" train.py || { echo "請在 Seg_UKAN 資料夾執行，且先執行 patch_ukan_weighted.py"; exit 1; }
[ -f $LISTS/proxy_selection_busi.csv ] || { echo "缺少 $LISTS/proxy_selection_busi.csv（先執行 screen_busi.py）"; exit 1; }

WNORM=${WNORM:-n}                                           # n：÷batch 張數（w3n）；sum：÷Σw（第一版，已不使用）
[ "$WNORM" = n ] && SUF=w3n || SUF=w3
EPOCHS=${EPOCHS:-400}                                       # 原作者設定 400；只在測試時改小
done_already() { [ -f outputs/$1/model_last.pth ] && [ "$(tail -n +2 outputs/$1/log.csv 2>/dev/null | wc -l)" -ge $EPOCHS ]; }

run() {   # run <名稱> [--wcases 清單]
  local n=$1; shift
  echo "===== $n　$(date '+%m/%d %H:%M')"
  if done_already $n; then echo "已完成，略過訓練"; else
    python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name $n --data_dir ./inputs --epochs $EPOCHS "$@" || return
  fi
  for w in best last; do
    [ -f outputs/$n/percase_$w.csv ] && echo "已有 percase_$w.csv，略過預測" || python predict_busi.py --name $n --which $w
  done
}

run busi_UKAN_base2
for f in $LISTS/wcases_busi_*.csv; do
  [ -f "$f" ] || { echo "沒有顯著的 Ψ，不訓練專家"; break; }
  psi=$(basename $f .csv | sed 's/wcases_busi_//')          # 例：psi1_convex
  run busi_UKAN_${psi%%_*}$SUF --wcases $f --wfactor 3.0 --wnorm $WNORM
done
echo "===== 全部完成　$(date '+%m/%d %H:%M')"