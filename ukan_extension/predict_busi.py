"""
用訓練好的 U-KAN 預測 BUSI，存下每張影像的預測遮罩與逐張 Dice，並可計算十個 Ψ
  切分：與 train.py 完全相同（影像名稱排序 → train_test_split(test_size=0.2, random_state=dataseed)）
        train = 80%（517 張，模型訓練過）、val = 20%（130 張，論文報告分數用的那份，本研究當作測試集）
  前處理：與 train.py 的 val_transform 相同（Resize 256 → Normalize → /255），門檻 sigmoid ≥ 0.5
  逐張 Dice / IoU：在 256×256 上計算（與訓練時的評估相同解析度）
  Ψ（--psi）：沿用 psi-weighted-nnunet/psi/run_all_psi.py 的函式，只用預測遮罩，不用標註
    Ψ1 凸包比、Ψ6 連通元件、Ψ7 拓撲環：由預測遮罩計算
    Ψ8 旋轉一致性：影像旋轉 15° → 同一模型預測 → 轉回 → 1 − Dice(原預測, 轉回的預測)
    Ψ2 聲影一致性：未定義計算方式（同 nnU-Net 專案，標為 NaN）
    Ψ3/4/5/9/10：需要兩個以上的結構，BUSI 只有腫瘤一個結構，標為 NaN
輸出：outputs/<name>/percase_<best|last>.csv、outputs/<name>/pred_<best|last>/<split>/<影像>.png
用法（在 Seg_UKAN 資料夾）：
  python predict_busi.py --name busi_UKAN_seed2981 --which best --psi        # baseline：訓練集 + 測試集，含 Ψ
  python predict_busi.py --name busi_UKAN_psi1w3 --which last                  # 專家：只算 Dice
"""
import argparse
import importlib.util
import os
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations import Compose, Resize
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split

import archs

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--which', default='best', choices=['best', 'last'])
ap.add_argument('--splits', default='train,val')
ap.add_argument('--psi', action='store_true', help='計算十個 Ψ（含 Ψ8 旋轉推論）')
ap.add_argument('--psi_lib', default=os.path.expanduser('~/桌面/論文/psi-weighted-nnunet/psi/run_all_psi.py'))
ap.add_argument('--angle', default=15.0, type=float)
args = ap.parse_args()

D = f'{args.output_dir}/{args.name}'
config = yaml.load(open(f'{D}/config.yml'), Loader=yaml.FullLoader)
ck = f'{D}/model.pth' if args.which == 'best' else f'{D}/model_last.pth'
os.path.exists(ck) or sys.exit(f'找不到 {ck}' + ('（model_last.pth 只有用修改後的 train.py 訓練才會有）' if args.which == 'last' else ''))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'],
                                       embed_dims=config['input_list'], no_kan=config.get('no_kan', False))
missing = model.load_state_dict(torch.load(ck, map_location='cpu'), strict=True)   # 嚴格載入：參數對不上就報錯
model.to(dev).eval()
print(f'模型：{ck}（{config["arch"]}，dataseed {config["dataseed"]}）')

if args.psi:
    spec = importlib.util.spec_from_file_location('psi_lib', args.psi_lib)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

base = os.path.join(config['data_dir'], config['dataset'])
ids = sorted(glob(os.path.join(base, 'images', '*.png')))
ids = [os.path.splitext(os.path.basename(p))[0] for p in ids]
tr, va = train_test_split(ids, test_size=0.2, random_state=config['dataseed'])
SPLIT = {'train': tr, 'val': va}
tf = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])


def to_tensor(img):
    x = tf(image=img)['image'].astype('float32') / 255           # 與 dataset.py 相同
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


@torch.no_grad()
def predict(img):
    return (torch.sigmoid(model(to_tensor(img).to(dev)))[0, 0] >= 0.5).cpu().numpy()


def rotate(a, ang, border, interp):
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


rows = []
for sp in args.splits.split(','):
    od = f'{D}/pred_{args.which}/{sp}'
    os.makedirs(od, exist_ok=True)
    for k, i in enumerate(SPLIT[sp]):
        img = cv2.imread(os.path.join(base, 'images', i + '.png'))
        gt = cv2.imread(os.path.join(base, 'masks', '0', i + '_mask.png'), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(gt, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST) > 127
        p = predict(img)
        cv2.imwrite(f'{od}/{i}.png', p.astype(np.uint8) * 255)
        inter, uni = (p & g).sum(), (p | g).sum()
        r = dict(case=i, split=sp, dice=dice(p, g), iou=1.0 if uni == 0 else inter / uni,
                 pred_pixels=int(p.sum()), gt_pixels=int(g.sum()))
        if args.psi:
            # Ψ8：影像轉 15°（邊緣補影像最小值，同 nnU-Net 專案）→ 預測 → 轉回（最近鄰）→ 與原預測比較
            ir = rotate(img, args.angle, tuple(int(v) for v in img.reshape(-1, 3).min(0)), cv2.INTER_LINEAR)
            pr = predict(ir)
            pb = rotate(pr.astype(np.uint8), -args.angle, 0, cv2.INTER_NEAREST) > 0
            r.update(psi1_convex=P.psi1_convex(p), psi2_shadow=np.nan, psi3_inclusion=np.nan,
                     psi4_adjacency=np.nan, psi5_exclusion=np.nan, psi6_cc=P.psi6_cc(p),
                     psi7_genus=P.psi7_genus(p), psi8_rotation=1.0 - dice(p, pb),
                     psi9_nesting=np.nan, psi10_centroid=np.nan)
        rows.append(r)
        if (k + 1) % 50 == 0:
            print(f'  {sp} {k + 1}/{len(SPLIT[sp])}', flush=True)
d = pd.DataFrame(rows)
out = f'{D}/percase_{args.which}.csv'
if os.path.exists(out):                                           # 保留其他 split 已算好的列
    old = pd.read_csv(out)
    d = pd.concat([old[~old.split.isin(d.split.unique())], d], ignore_index=True)
d.to_csv(out, index=False)
for sp, g in d.groupby('split'):
    print(f'{sp:<5} {len(g):>4} 張　逐張 Dice 平均 {g.dice.mean():.4f}　IoU 平均 {g.iou.mean():.4f}')
print(f'→ {out}\n→ {D}/pred_{args.which}/')