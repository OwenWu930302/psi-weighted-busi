# Ψ-Weighted Expert Models on BUSI（U-KAN）

**把 [psi-weighted-nnunet](https://github.com/OwenWu930302/psi-weighted-nnunet) 的 Ψ 加權專家流程，套用到 BUSI 乳房超音波腫瘤分割與 U-KAN（AAAI 2025）模型。**

Ψ 的定義、proxy 篩選、加權清單規則、加權 loss 公式，全部沿用 psi-weighted-nnunet；
差別在分割模型（U-KAN，而非 nnU-Net）與資料集（BUSI 乳房超音波）。
BTCV、ACDC 的結果請見原倉庫，REFUGE 請見 [psi-weighted-refuge](https://github.com/OwenWu930302/psi-weighted-refuge)。

---

## 流程

```
① baseline（官方 U-KAN，原作者設定：80 / 20 隨機切分、種子 2981、400 epoch）
        │
        ▼
② baseline 預測訓練集 517 張 → 逐張 Dice、Ψ1–Ψ10 → proxy 篩選（低分組 = Dice 最低 30%；不使用測試集）
        │   選出 Ψ1 凸包比、Ψ6 連通元件、Ψ8 旋轉一致性
        ▼
③ 依 Ψ 取最明顯的影像 → 加權清單（Ψ1、Ψ8 各 155 張，Ψ6 33 張）
        │
        ▼
④ 專家模型：清單影像 loss × 3，其餘不變          ⑤ 對照組：同設定、不加權，重新訓練一次
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
⑥ 訓練集：被加權影像在「專家 vs 對照組」下的 Dice 變化
⑦ 測試集（20%，130 張）：套用訓練清單的門檻挑出「特徵明顯」的影像，比較專家 vs 對照組
```

$$L = \frac{\sum_i w_i \cdot L_i}{\sum_i w_i}, \qquad L_i = 0.5\cdot\mathrm{BCE}_i + (1-\mathrm{Dice}_i), \qquad w_i \in \{1,\ 3\}$$

U-KAN 官方的 loss（`BCEDiceLoss`）本來就逐張計算 Dice，所有影像同為 256×256，所以權重全為 1 時與官方 loss 完全相同
（實測差異 1.2×10⁻⁷，為浮點誤差；修改後的程式不加權時，前 3 個 epoch 的 loss 與原版逐位相同）。
因此**對照組就是不加權的 baseline**，不需要 nnU-Net 版的 1.0001 倍率。batch 只有 8 張，採 ÷Σw（加權平均）；
BTCV 的經驗顯示小 batch 用 ÷n 會讓 loss 尺度大幅跳動。

---

## 與 psi-weighted-nnunet 的差異

| | psi-weighted-nnunet（BTCV / ACDC） | 本倉庫（BUSI） |
|---|---|---|
| 分割模型 | nnU-Net v2.8.1 | [U-KAN](https://github.com/CUHK-AIM-Group/U-KAN)（AAAI 2025）：卷積 + Tokenized KAN 的 U 形網路，約 6.35M 參數 |
| 資料 | 3D 影像的 2D 切片 | 2D 超音波影像，縮放至 256×256 |
| 切分 | BTCV 公開切分、ACDC 官方切分 | 原作者做法：影像名稱排序後 `train_test_split(test_size=0.2, random_state=2981)`，訓練 517 / 測試 130 |
| loss | CE + 整批 Dice | 0.5·BCE + **逐張** Dice（官方實作） |
| 對照組 | 倍率 1.0001（讓 loss 也逐張計算） | 不給清單（權重全為 1），與官方完全相同 |
| proxy 篩選 | 5 折 OOF／測試集（ACDC Ψ1/6/7 †） | baseline 對**訓練集**的預測（同 ACDC 的 all 模式），完全不看測試集 |
| 選用的權重 | 最後一個 epoch | **兩種都報告**：last（最後一個 epoch，主要依據）、best（原作者做法：在 20% 集合上挑最好的 epoch） |
| Ψ | 依資料集 | BUSI 只有腫瘤一個結構：可計算 Ψ1、Ψ6、Ψ7、Ψ8；Ψ3/4/5/9/10 需要兩個以上的結構，Ψ2 未定義 |

---

## 主要結果

### 1. baseline 重現

| | IoU | Dice（= 論文的 F1） |
|---|---|---|
| **本研究：best（原作者做法）** | **0.6883** | **0.8056** |
| 本研究：last（最後一個 epoch） | 0.6511 | — |
| 作者 GitHub 單次結果（種子 2981） | 0.6526 | 0.7875 |
| 論文 3 次平均 | 0.6338 ± 0.0283 | 0.7640 ± 0.0290 |

（訓練過程印出的 IoU / Dice 為每個 batch 合併計算後平均，與論文相同；以下分析一律用**逐張** Dice。）

### 2. proxy 篩選（baseline 對訓練集 517 張的預測；低分組 = Dice ≤ 0.9118，155 張）

| Ψ | AUC | p | 規則 | 加權清單 |
|---|---|---|---|---|
| **Ψ1 凸包比** | **0.780** | 6.6×10⁻²⁴ | 前 30%（≥ 1.05127） | 155 張 |
| **Ψ8 旋轉一致性** | **0.705** | 1.3×10⁻¹³ | 前 30%（≥ 0.331525） | 155 張 |
| **Ψ6 連通元件** | 0.537 | 0.0014 | > 0（預測碎成兩塊以上） | 33 張 |
| Ψ7 拓撲環 | 0.497 | 0.13 | 不顯著（實心腫瘤幾乎都等於 1） | — |

### 3. 同設定重訓的雜訊（另一次 baseline − 對照組，best）

| | Δ | 逐張標準差 | 升 / 降 |
|---|---|---|---|
| 訓練集 | −0.0263 | 0.062 | 81 / 436 |
| 測試集 | −0.0093 | **0.099** | 64 / 63 |

設定完全相同，只因 best 挑到的 epoch 不同，訓練集就有 436 張下降；**best 的比較主要反映挑到哪個 epoch**，因此以下以 **last** 為主要依據。

### 4. 訓練集（last）：被加權的影像本身

| 專家 | 清單 | 對照組 → 專家 | Δ | 升 / 降 | 加權組比未加權組多升 | p |
|---|---|---|---|---|---|---|
| Ψ1 | 155 | 0.9625 → 0.9681 | +0.0057 | 132 / 23 | **+0.0130** | < 0.001 |
| Ψ6 | 33 | 0.9494 → 0.9723 | **+0.0229** | 31 / 2 | **+0.0265** | < 0.001 |
| Ψ8 | 155 | 0.9655 → 0.9670 | +0.0015 | 91 / 64 | **+0.0113** | < 0.001 |

與 ACDC、BTCV 相同：**×3 加權讓清單上的影像本身擬合得更好**。

### 5. 測試集（last）：Ψ 挑到的影像 vs 真正低分的影像

測試集 130 張的 Dice 是雙峰：大多數 > 0.8，另有 15 張 < 0.5，其中 **9 張 Dice < 0.1（整顆腫瘤找錯位置）**。
因此分成兩種組別來看：

- **特徵明顯組**：Ψ 超過訓練清單的門檻（只用 baseline 的預測，不看標註）
- **真正低分組**：baseline（`busi_UKAN_seed2981`）Dice < 0.5，15 張（用標註，僅供分析）。
  用另一個模型而非對照組來定義，避免「用對照組自己的低分挑影像，重訓後自然回升」的偏差。

**(a) Ψ 有沒有挑到真正低分的影像**

| 專家 | 特徵明顯組 | 覆蓋率（15 張低分被挑到） | 精確度（明顯組中真的低分） |
|---|---|---|---|
| Ψ1 | 48 | 80%（12） | 25% |
| Ψ6 | 21 | 53%（8） | 38% |
| Ψ8 | 38 | **93%（14）** | 37% |

**(b) 對照組 → 專家**

| 專家 | 整個測試集 Δ | 救回 / 新失敗 | 特徵明顯組 Δ（升 / 降） | 真正低分組 Δ（升 / 降） |
|---|---|---|---|---|
| Ψ1 | −0.0002 | 3 / 1 | −0.0088（17 / 27） | +0.0001（3 / 6） |
| Ψ6 | +0.0050 | 2 / 0 | +0.0163（11 / 6） | +0.0168（4 / 4） |
| Ψ8 | +0.0048 | 5 / 2 | +0.0324（18 / 15） | +0.0420（6 / 4） |
| 雜訊（兩次 baseline，best） | −0.0093 | 3 / 1 | | |

救回 = 對照組 < 0.5 → 專家 ≥ 0.5；新失敗 = 對照組 ≥ 0.5 → 專家 < 0.5。所有 p > 0.1。

**(c) 四格（Δ，張數）**

| 專家 | 明顯且低分（挑對） | 明顯但不低分（誤挑的好影像） | 低分但未挑到（漏掉） | 其他 |
|---|---|---|---|---|
| Ψ1 | +0.0007（12） | **−0.0119（36）** | −0.0021（3） | +0.0050（79） |
| Ψ6 | +0.0242（8） | +0.0115（13） | +0.0084（7） | +0.0025（102） |
| Ψ8 | +0.0460（14） | +0.0245（24） | −0.0138（1） | −0.0065（91） |

- Ψ 能找到低分影像（Ψ8 覆蓋 93%、Ψ1 80%），但明顯組裡 6–7 成是原本就不差的影像。
- Ψ1 的下降集中在「誤挑的好影像」（36 張，−0.012）：形狀不規則但原本分割得好的腫瘤，加權後反而變差。
- 9 張 Dice < 0.1 的影像，對照組與三個專家**全部仍 < 0.03**：這類錯誤是位置找錯，不是邊界形狀，加權形狀類 Ψ 無法修正。
- Ψ8 在挑對的影像上 +0.046、救回 5 張，方向正確，但與兩次 baseline 的差異（救回 3 / 新失敗 1）相比仍在雜訊範圍內。

<details>
<summary>補充：best（原作者做法）的結果</summary>

| 專家 | 訓練集被加權影像 Δ | 加權組多升 | 整個測試集 Δ（p） | 特徵明顯組 Δ（升 / 降） | 真正低分組 Δ（升 / 降） |
|---|---|---|---|---|---|
| Ψ1 | −0.0312 | +0.0082 | −0.0160（0.039） | −0.0415（17 / 29） | −0.0297（3 / 8） |
| Ψ6 | −0.0080 | +0.0230 | −0.0033（0.191） | +0.0276（9 / 10） | −0.0393（3 / 8） |
| Ψ8 | −0.0020 | +0.0160 | −0.0138（0.132） | −0.0259（15 / 19） | −0.0497（2 / 9） |

best 各自挑到不同的 epoch，與 last 的正負號經常相反（第 3 節），僅供參照。
</details>

### 結論

> Ψ 能在沒有標註的情況下挑出困難的影像（訓練集篩選 AUC 0.78 / 0.71；測試集真正低分影像的覆蓋率 Ψ8 93%、Ψ1 80%），
> ×3 加權也確實讓訓練集上被加權的影像變好（三個專家皆顯著），
> 但效果沒有轉移到測試集：專家在低分組的變化都在重訓雜訊範圍內，Ψ1 甚至讓誤挑的好影像變差。
> 主要原因有三：
> 1. 加權清單來自 baseline 對**訓練集**的預測，訓練集 Dice 已達 0.92，清單反映的是擬合後殘留的形狀，而不是模型真正不會的影像；
> 2. BUSI 最困難的影像是位置找錯（Dice ≈ 0），形狀類的加權無法修正；
> 3. 重訓雜訊大（逐張標準差約 0.10），15 張低分組的平均差異需大於約 ±0.04 才可能分辨。
>
> 另外，原作者「在 20% 集合上挑最好 epoch 並報告同一集合」的做法會引入很大的雜訊，比較加權效果時應以固定的 epoch 為準。

---

## 重現

需要 NVIDIA GPU（本研究使用 RTX 4080）。第 1 步之後全部在 `U-KAN/Seg_UKAN` 資料夾執行。`setup_ukan.sh` 預設把資料夾放在 `/mnt/data2t/BUSI`、桌面 `~/桌面/BUSI` 為捷徑，可用 `DATA=… LINK=… bash setup_ukan.sh` 改位置。

```bash
# 1. 環境與官方程式（建立 conda 環境 ukan、下載 U-KAN、修正官方程式無法執行的兩處、改寫 requirements）
bash ukan_extension/setup_ukan.sh

# 2. 資料：Kaggle「Breast Ultrasound Images Dataset」（U-KAN README 所附連結）
#    官方 pre-processed 連結已失效，改由原始資料整理：只用 benign + malignant（647 張），多個遮罩取聯集
python prep_busi.py <Dataset_BUSI_with_GT> inputs/busi

# 3. baseline（原作者指令）
python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_UKAN_seed2981 --data_dir ./inputs

# 4. 加入 Ψ 加權（不給 --wcases 時與原版完全相同）
python patch_ukan_weighted.py

# 5. baseline 預測 + Ψ、篩選並產生清單
python predict_busi.py --name busi_UKAN_seed2981 --which best --psi
python screen_busi.py outputs/busi_UKAN_seed2981/percase_best.csv <清單資料夾>

# 6. 對照組（第二次 baseline）與專家，依序訓練並預測
LISTS=<清單資料夾> bash train_busi_queue.sh

# 7. 分析
python analyze_busi.py --lists <清單資料夾>
```

`ukan_extension/` 的檔案複製到 `U-KAN/Seg_UKAN` 後使用（`setup_ukan.sh` 除外）。`predict_busi.py` 的 Ψ 計算沿用
psi-weighted-nnunet 的 `psi/run_all_psi.py`（以 `--psi_lib` 指定路徑）。

### 官方程式的修正（`setup_ukan.sh`）

| 問題 | 現象 | 修正 |
|---|---|---|
| `train.py` 讀取 `archs.__all__`，`archs.py` 未定義 | 一執行就 `AttributeError` | `archs.py` 第一行加 `__all__ = ['UKAN']` |
| `metrics.py` 需要 `medpy`，requirements 未列 | `ImportError` | 加入 |
| `scipy==1.5.4`、`scikit-image==0.17.2` | 作者指定的 Python 3.10 沒有對應版本，無法安裝 | 移除固定版本（程式未直接使用） |
| `timm==0.3.2` | 使用 `torch._six`，與作者指定的 PyTorch 1.13 不相容 | `timm==0.4.12` |
| numpy / opencv / albumentations 未固定版本 | 會裝到與 PyTorch 1.13 或程式不相容的新版 | 固定版本（見 `ukan_extension/requirements_ukan.txt`） |

---

## 檔案

| 位置 | 內容 |
|---|---|
| `ukan_extension/patch_ukan_weighted.py` | 在官方 `train.py`、`losses.py` 加入 Ψ 加權（`--wcases`、`--wfactor`）與每個 epoch 存 `model_last.pth`；修改處皆標 `[Ψ加權]` |
| `ukan_extension/ukan_patch.diff` | 修改後與官方原檔的完整差異 |
| `ukan_extension/predict_busi.py` | 預測訓練集與測試集、逐張 Dice、Ψ1–Ψ10（Ψ8 含旋轉推論）；切分與官方 `train.py` 相同 |
| `ukan_extension/screen_busi.py` | proxy 篩選與加權清單（規則同 psi-weighted-nnunet） |
| `ukan_extension/train_busi_queue.sh` | 依序訓練對照組與專家，訓練完立即預測（best、last） |
| `ukan_extension/analyze_busi.py` | 訓練集加權前後；測試集的覆蓋率、特徵明顯組與真正低分組的前後變化、四格、重訓雜訊 |
| `ukan_extension/prep_busi.py`、`setup_ukan.sh`、`requirements_ukan.txt` | 資料整理、環境建置、改寫後的套件版本 |
| `weight_lists/` | 三個加權清單（影像名稱、Ψ 值、baseline Dice） |
| `results/psi_selection/proxy_selection_busi.csv` | proxy 篩選結果 |
| `results/percase/` | 每個模型逐張的 Dice（`<模型>_<best或last>.csv`）；`busi_UKAN_seed2981_best.csv` 含十個 Ψ |
| `results/analysis/` | `weighted_cases_BUSI*`（訓練集）；`test_BUSI_summary`（覆蓋率與各組前後）、`test_BUSI_4cells`（四格）、`test_BUSI_percase`（逐張）、`test_BUSI_noise`（雜訊基準） |
| `results/training_logs/` | 每個模型每個 epoch 的 loss 與 IoU |

資料、影像與模型權重不包含在倉庫中。U-KAN 的程式不包含在本倉庫，見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 參考

- Li et al., *U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation*, AAAI 2025. [論文](https://arxiv.org/abs/2406.02918)／[程式](https://github.com/CUHK-AIM-Group/U-KAN)
- Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.（BUSI）