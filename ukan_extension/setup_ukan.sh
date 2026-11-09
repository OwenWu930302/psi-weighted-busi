#!/bin/bash
# U-KAN（AAAI 2025）BUSI 實驗環境建置：可重複執行，已完成的步驟會略過
#   1. 資料夾放在 /mnt/data2t/BUSI（容量最大的硬碟），桌面 ~/桌面/BUSI 為捷徑
#   2. 下載官方程式 U-KAN
#   3. 修正官方程式無法執行的地方（只加一行）＋改寫 requirements.txt（原檔備份為 requirements_original.txt）
#   4. 建立新的 conda 環境 ukan（Python 3.10、PyTorch 1.13.0 + CUDA 11.6，皆為作者指定）
#   5. 檢查 GPU 與所有套件是否能正常載入
# 用法：bash setup_ukan.sh
set -e
DATA=${DATA:-/mnt/data2t/BUSI}
LINK=${LINK:-$HOME/桌面/BUSI}
REPO=$DATA/U-KAN
SEG=$REPO/Seg_UKAN

echo "===== 1. 資料夾"
mkdir -p $DATA
if [ -d "$LINK" ] && [ ! -L "$LINK" ]; then            # 桌面已有實體資料夾 → 內容搬到 data2t 再換成捷徑
  echo "桌面已有 BUSI 資料夾，內容搬到 $DATA"
  shopt -s dotglob nullglob
  for f in "$LINK"/*; do mv "$f" $DATA/; done
  rmdir "$LINK"
fi
[ -L "$LINK" ] || ln -s $DATA "$LINK"
ls -ld "$LINK"

echo "===== 2. 下載官方程式"
[ -d $REPO/.git ] || git clone https://github.com/CUHK-AIM-Group/U-KAN.git $REPO
echo "版本：$(cd $REPO && git log -1 --format='%h %ad' --date=short)"

echo "===== 3. 修正"
cd $SEG
# (a) train.py 第 42 行讀 archs.__all__，但官方 archs.py 沒有定義 → 一執行就 AttributeError
grep -q "^__all__" archs.py || sed -i "1i __all__ = ['UKAN']   # 修正：官方 train.py 需要 archs.__all__，原檔未定義" archs.py
# (b) requirements.txt：原檔在 Python 3.10 無法安裝（scipy 1.5.4、scikit-image 0.17.2 沒有 3.10 版）、
#     timm 0.3.2 與 PyTorch 1.13 不相容（使用已移除的 torch._six）、缺少 medpy、
#     未固定版本的 numpy / opencv / albumentations 會裝到不相容的新版
[ -f requirements_original.txt ] || cp requirements.txt requirements_original.txt
cat > requirements.txt <<'REQ'
# 改寫自 requirements_original.txt，使其能在作者指定的 Python 3.10 + PyTorch 1.13 安裝與執行
# PyTorch 另外安裝（同原檔）：
#   pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
numpy<2                          # PyTorch 1.13 以 NumPy 1.x 編譯
opencv-python==4.10.0.84         # 更新版要求 NumPy 2
opencv-python-headless==4.10.0.84
albumentations==1.3.1            # 程式使用 geometric.transforms.Flip，新版已移除
scikit-image<0.25
timm==0.4.12                     # 原為 0.3.2：會 import torch._six（PyTorch 1.9 起已移除）
medpy                            # metrics.py 需要，原檔漏列
pandas
pyyaml
tqdm
tensorboardX
scikit-learn
scipy
matplotlib
pillow
REQ
echo "requirements.txt 已改寫（原檔：requirements_original.txt）"

[ -n "$SKIP_ENV" ] && { echo "（測試模式：略過環境建置）"; exit 0; }
echo "===== 4. conda 環境 ukan"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^ukan " || conda create -n ukan python=3.10 -y
conda activate ukan
python -c "import torch" 2>/dev/null || pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

echo "===== 5. 檢查"
python - <<'PY'
import torch, numpy, cv2, albumentations, timm, medpy, tensorboardX
print('torch', torch.__version__, '| numpy', numpy.__version__, '| albumentations', albumentations.__version__, '| timm', timm.__version__)
print('CUDA 可用：', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU：', torch.cuda.get_device_name(0), '| 編譯支援', torch.cuda.get_arch_list())
    x = torch.randn(2, 3, 64, 64, device='cuda')
    torch.nn.Conv2d(3, 8, 3).cuda()(x).sum().item()
    print('GPU 運算測試：OK')
import archs, losses, dataset, metrics
m = archs.UKAN(1, 3, False, embed_dims=[128, 160, 256])
if torch.cuda.is_available():
    m = m.cuda()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 256, 256, device='cuda'))
    print('U-KAN 前向測試：OK，輸出', tuple(y.shape))
print('全部檢查完成')
PY
echo
echo "完成。程式位置：$LINK/U-KAN/Seg_UKAN（實際在 $SEG）"
echo "之後使用：conda activate ukan && cd $LINK/U-KAN/Seg_UKAN"