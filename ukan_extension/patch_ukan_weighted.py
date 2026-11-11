"""
U-KAN 加入 Ψ 加權訓練（只改兩個檔，可重複執行；不給 --wcases 時訓練行為與原版完全相同）
  losses.py：BCEDiceLoss 新增 per_sample()，回傳每張影像的 0.5·BCE + (1 − Dice)
             （所有影像同為 256×256，故 per_sample 的平均 = 原本的 forward，數值完全相同）
  train.py ：
    --wcases <csv>    加權清單（第一欄 case = 影像名稱）
    --wfactor 3.0     清單內影像的權重（其餘為 1）
    --wnorm n|sum     loss 的分母（預設 n）
                        n  ：Σ wᵢ·Lᵢ / n，n = batch 張數（與原版的平均相同；專家 busi_UKAN_psi*w3n）
                        sum：Σ wᵢ·Lᵢ / Σ wᵢ（第一版，已不使用，只為重現舊模型保留）
                        權重全為 1 時兩者都等於原版的 loss
    --move_to_train <csv>  把清單中的測試影像（第一欄 case）移到訓練集（不給時切分與原版完全相同）
    每個 epoch 印出 [Ψ加權] 本輪加權樣本 x/y，供檢查
    每個 epoch 另存 model_last.pth（原版只存驗證集最佳的 model.pth）
用法（在 Seg_UKAN 資料夾）：python patch_ukan_weighted.py
  已用舊版修改過的 train.py 也可直接執行，只會補上缺少的部分
"""
import os
import shutil
import sys

for f in ('train.py', 'losses.py'):
    os.path.exists(f) or sys.exit('請在 U-KAN/Seg_UKAN 資料夾執行')

ORIG_LOSS = """            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
"""
V1_LOSS = """            if config.get('wcases'):                          # [Ψ加權] 加權平均 Σ wᵢ·Lᵢ / Σ wᵢ
                w = psi_weights(config, meta['img_id']).to(output.device)
                loss = (w * criterion.per_sample(output, target)).sum() / w.sum()
"""
V2_SUM = """            if config.get('wcases'):                          # [Ψ加權] Σ wᵢ·Lᵢ / Σ wᵢ（--wnorm sum）或 / n（--wnorm n）
                w = psi_weights(config, meta['img_id']).to(output.device)
                wsum = (w * criterion.per_sample(output, target)).sum()
                loss = wsum / (w.sum() if config.get('wnorm', 'sum') == 'sum' else len(w))
"""
V2_LOSS = """            if config.get('wcases'):                          # [Ψ加權] Σ wᵢ·Lᵢ / n（預設）；--wnorm sum 時 / Σ wᵢ（舊版）
                w = psi_weights(config, meta['img_id']).to(output.device)
                wsum = (w * criterion.per_sample(output, target)).sum()
                loss = wsum / (w.sum() if config.get('wnorm', 'n') == 'sum' else len(w))
"""
NEW_LOSS = """            output = model(input)
""" + V2_LOSS + """                psi_hit += int((w != 1).sum())
                psi_tot += len(w)
            else:
                loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
"""
WN_SUM = """    parser.add_argument('--wnorm', default='sum', choices=['sum', 'n'], help='loss 分母：Σw 或 batch 張數 n')
"""
WN_N = """    parser.add_argument('--wnorm', default='n', choices=['n', 'sum'], help='loss 分母：batch 張數 n（預設）或 Σw（舊版）')
"""
MV_ARG = """    parser.add_argument('--move_to_train', default=None, help='移到訓練集的測試影像清單 csv（第一欄 case）')
"""
PR_SUM = """              f"loss ÷ {'Σw' if config.get('wnorm', 'sum') == 'sum' else 'n'}")
"""
PR_N = """              f"loss ÷ {'Σw' if config.get('wnorm', 'n') == 'sum' else 'n'}")
"""
SPLIT = """    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
"""
MV_CODE = """    if config.get('move_to_train'):                       # [Ψ加權] 把清單中的測試影像移到訓練集
        mv = set(pd.read_csv(config['move_to_train'], encoding='utf-8-sig').iloc[:, 0].astype(str))
        train_img_ids = train_img_ids + [i for i in val_img_ids if i in mv]
        val_img_ids = [i for i in val_img_ids if i not in mv]
        print(f"[Ψ加權] 移到訓練集 {len(mv)} 張 → 訓練 {len(train_img_ids)}、測試 {len(val_img_ids)}")
"""
WF = """    parser.add_argument('--wfactor', default=3.0, type=float, help='清單內影像的權重，其餘為 1')
"""

# (原文, 新文, 可略過)：依序套用；新文已存在就略過；「可略過」的項目找不到原文時也略過
E = {
 'losses.py': [(
"""        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
""",
"""        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

    def per_sample(self, input, target):
        \"\"\"[Ψ加權] 每張影像各自的 loss：0.5·BCEᵢ + (1 − Diceᵢ)；對張數平均即等於 forward()\"\"\"
        num = target.size(0)
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none').view(num, -1).mean(1)
        smooth = 1e-5
        p = torch.sigmoid(input).view(num, -1)
        t = target.view(num, -1)
        dice = (2. * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)
        return 0.5 * bce + (1 - dice)
""", False)],
 'train.py': [
  ("""    parser.add_argument('--no_kan', action='store_true')
""",
   """    parser.add_argument('--no_kan', action='store_true')
    # [Ψ加權] 不給 --wcases 時與原版完全相同
    parser.add_argument('--wcases', default=None, help='Ψ 加權清單 csv（第一欄 case）')
""" + WF, False),
  (WN_SUM, WN_N, True),                                   # 舊版已修改過的 train.py：預設改為 n
  (WF, WF + WN_N, False),
  (WN_N, WN_N + MV_ARG, False),
  ("""        print(f"[Ψ加權] 清單 {config['wcases']}：{len(_PSI_CASES)} 張，權重 ×{config['wfactor']}")
""", """        print(f"[Ψ加權] 清單 {config['wcases']}：{len(_PSI_CASES)} 張，權重 ×{config['wfactor']}，"
              f"loss ÷ {'Σw' if config.get('wnorm', 'n') == 'sum' else 'n'}")
""", True),
  (PR_SUM, PR_N, True),                                   # 舊版已修改過的 train.py：預設改為 n
  ("""def train(config, train_loader, model, criterion, optimizer):
""",
   """_PSI_CASES = None


def psi_weights(config, ids):
    \"\"\"[Ψ加權] 清單內的影像權重 = wfactor，其餘 = 1\"\"\"
    global _PSI_CASES
    if _PSI_CASES is None:
        _PSI_CASES = set(pd.read_csv(config['wcases'], encoding='utf-8-sig').iloc[:, 0].astype(str))
        print(f"[Ψ加權] 清單 {config['wcases']}：{len(_PSI_CASES)} 張，權重 ×{config['wfactor']}，"
              f"loss ÷ {'Σw' if config.get('wnorm', 'n') == 'sum' else 'n'}")
    return torch.tensor([config['wfactor'] if str(i) in _PSI_CASES else 1.0 for i in ids])


def train(config, train_loader, model, criterion, optimizer):
""", False),
  ("""    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
""",
   """    pbar = tqdm(total=len(train_loader))
    psi_hit = psi_tot = 0
    if config.get('wcases') and config['deep_supervision']:
        raise ValueError('[Ψ加權] 未支援 deep_supervision')
    for input, target, meta in train_loader:
""", False),
  (V1_LOSS, V2_LOSS, True),                               # 舊版已修改過的 train.py
  (V2_SUM, V2_LOSS, True),
  (ORIG_LOSS, NEW_LOSS, False),
  ("""    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
""",
   """    pbar.close()
    if psi_tot:
        print(f'[Ψ加權] 本輪加權樣本 {psi_hit}/{psi_tot} ({psi_hit / psi_tot:.1%})')

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
""", False),
  (SPLIT, SPLIT + MV_CODE, False),
  ("""        trigger += 1
""",
   """        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_last.pth')   # [Ψ加權] 另存最後一個 epoch
        trigger += 1
""", False),
 ],
}

texts, n = {}, 0
for f, reps in E.items():
    s = open(f, encoding='utf-8').read()
    for old, new, optional in reps:
        if new in s:
            continue
        if s.count(old) != 1:
            if optional:
                continue
            sys.exit(f'✗ {f} 找不到要修改的位置：{old.strip()[:50]}　未修改任何檔案')
        s = s.replace(old, new)
        n += 1
    texts[f] = s
for f, s in texts.items():
    if s == open(f, encoding='utf-8').read():
        continue
    compile(s, f, 'exec')
    if not os.path.exists(f + '.orig'):
        shutil.copy(f, f + '.orig')                       # 原檔備份
    open(f, 'w', encoding='utf-8').write(s)
    print('已修改', f, '（原檔：' + f + '.orig）')
print(f'完成（修改 {n} 處）' if n else '已修改過，略過')