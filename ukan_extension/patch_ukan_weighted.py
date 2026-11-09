"""
U-KAN 加入 Ψ 加權訓練（只改兩個檔，可重複執行；不給 --wcases 時訓練行為與原版完全相同）
  losses.py：BCEDiceLoss 新增 per_sample()，回傳每張影像的 0.5·BCE + (1 − Dice)
             （所有影像同為 256×256，故 per_sample 的平均 = 原本的 forward，數值完全相同）
  train.py ：
    --wcases <csv>    加權清單（第一欄 case = 影像名稱）
    --wfactor 3.0     清單內影像的權重（其餘為 1）
    loss = Σ wᵢ·Lᵢ / Σ wᵢ（加權平均，與 nnU-Net 舊版 ÷Σw 相同；batch 只有 8，不用 ÷n）
    每個 epoch 印出 [Ψ加權] 本輪加權樣本 x/y，供檢查
    每個 epoch 另存 model_last.pth（原版只存驗證集最佳的 model.pth）
用法（在 Seg_UKAN 資料夾）：python patch_ukan_weighted.py
"""
import os
import shutil
import sys
import time

for f in ('train.py', 'losses.py'):
    os.path.exists(f) or sys.exit('請在 U-KAN/Seg_UKAN 資料夾執行')

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
""")],
 'train.py': [
  ("""    parser.add_argument('--no_kan', action='store_true')
""",
   """    parser.add_argument('--no_kan', action='store_true')
    # [Ψ加權] 不給 --wcases 時與原版完全相同
    parser.add_argument('--wcases', default=None, help='Ψ 加權清單 csv（第一欄 case）')
    parser.add_argument('--wfactor', default=3.0, type=float, help='清單內影像的權重，其餘為 1')
"""),
  ("""def train(config, train_loader, model, criterion, optimizer):
""",
   """_PSI_CASES = None


def psi_weights(config, ids):
    \"\"\"[Ψ加權] 清單內的影像權重 = wfactor，其餘 = 1\"\"\"
    global _PSI_CASES
    if _PSI_CASES is None:
        _PSI_CASES = set(pd.read_csv(config['wcases'], encoding='utf-8-sig').iloc[:, 0].astype(str))
        print(f"[Ψ加權] 清單 {config['wcases']}：{len(_PSI_CASES)} 張，權重 ×{config['wfactor']}")
    return torch.tensor([config['wfactor'] if str(i) in _PSI_CASES else 1.0 for i in ids])


def train(config, train_loader, model, criterion, optimizer):
"""),
  ("""    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
""",
   """    pbar = tqdm(total=len(train_loader))
    psi_hit = psi_tot = 0
    if config.get('wcases') and config['deep_supervision']:
        raise ValueError('[Ψ加權] 未支援 deep_supervision')
    for input, target, meta in train_loader:
"""),
  ("""            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
""",
   """            output = model(input)
            if config.get('wcases'):                          # [Ψ加權] 加權平均 Σ wᵢ·Lᵢ / Σ wᵢ
                w = psi_weights(config, meta['img_id']).to(output.device)
                loss = (w * criterion.per_sample(output, target)).sum() / w.sum()
                psi_hit += int((w != 1).sum())
                psi_tot += len(w)
            else:
                loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
"""),
  ("""    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
""",
   """    pbar.close()
    if psi_tot:
        print(f'[Ψ加權] 本輪加權樣本 {psi_hit}/{psi_tot} ({psi_hit / psi_tot:.1%})')

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
"""),
  ("""        trigger += 1
""",
   """        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_last.pth')   # [Ψ加權] 另存最後一個 epoch
        trigger += 1
"""),
 ],
}

texts, todo = {}, []
for f, reps in E.items():
    s = texts[f] = open(f, encoding='utf-8').read()
    for old, new in reps:
        if new in s:
            continue
        if s.count(old) != 1:
            sys.exit(f'✗ {f} 找不到要修改的位置：{old.strip()[:50]}　未修改任何檔案')
        todo.append((f, old, new))
for f, old, new in todo:
    texts[f] = texts[f].replace(old, new)
for f in {f for f, _, _ in todo}:
    compile(texts[f], f, 'exec')
    if not os.path.exists(f + '.orig'):
        shutil.copy(f, f + '.orig')                       # 原檔備份
    open(f, 'w', encoding='utf-8').write(texts[f])
    print('已修改', f, '（原檔：' + f + '.orig）')
print(f'完成（修改 {len(todo)} 處）' if todo else '已修改過，略過')