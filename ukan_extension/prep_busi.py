"""
把 Kaggle 原始 BUSI 整理成 U-KAN 需要的格式（取代作者已失效的 pre-processed 連結）
  來源：Kaggle「Breast Ultrasound Images Dataset」（U-KAN README 所附連結）
        解壓後為 Dataset_BUSI_with_GT/{benign,malignant,normal}/
  依論文：只用 benign + malignant（647 張），不含 normal
  一張影像有多個遮罩（_mask.png、_mask_1.png …）時取聯集（任一遮罩為病灶即為病灶）
輸出（U-KAN README 的結構）：
  <輸出>/images/<名稱>.png
  <輸出>/masks/0/<名稱>_mask.png     （0 / 255）
用法：python prep_busi.py <Dataset_BUSI_with_GT 資料夾> <輸出資料夾，例如 inputs/busi>
"""
import glob
import os
import shutil
import sys

import cv2
import numpy as np

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(f'{OUT}/images', exist_ok=True)
os.makedirs(f'{OUT}/masks/0', exist_ok=True)

stat = {}
for cls in ('benign', 'malignant'):
    d = os.path.join(SRC, cls)
    if not os.path.isdir(d):
        sys.exit(f'找不到 {d}，請確認第一個參數是 Dataset_BUSI_with_GT 資料夾')
    imgs = sorted(f for f in glob.glob(f'{d}/*.png') if '_mask' not in os.path.basename(f))
    n_multi = 0
    for f in imgs:
        name = os.path.basename(f)[:-4]                       # 例：benign (1)
        masks = sorted(glob.glob(os.path.join(d, glob.escape(name) + '_mask*.png')))
        if not masks:
            sys.exit(f'✗ {name} 沒有遮罩')
        img = cv2.imread(f)
        m = np.zeros(img.shape[:2], np.uint8)
        for mf in masks:
            a = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if a.shape != m.shape:
                sys.exit(f'✗ {mf} 大小 {a.shape} 與影像 {m.shape} 不同')
            m[a > 127] = 255
        n_multi += len(masks) > 1
        shutil.copy(f, f'{OUT}/images/{name}.png')
        cv2.imwrite(f'{OUT}/masks/0/{name}_mask.png', m)
    stat[cls] = (len(imgs), n_multi)

n = sum(v[0] for v in stat.values())
for cls, (k, mm) in stat.items():
    print(f'{cls:<10} {k:>4} 張（其中 {mm} 張有多個遮罩，已取聯集）')
print(f'合計 {n} 張 → {OUT}　（論文為 647 張：benign 437 + malignant 210）')
print(f'images：{len(os.listdir(OUT + "/images"))}　masks/0：{len(os.listdir(OUT + "/masks/0"))}')