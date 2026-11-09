# Third-party notices

## U-KAN

- Repository: https://github.com/CUHK-AIM-Group/U-KAN (commit `b20bf63`, 2025-06-24)
- Paper: Li et al., *U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation*, AAAI 2025.
- The U-KAN repository does not include a license file, so **no U-KAN source file is redistributed here**.
  Users obtain U-KAN from its official repository (`ukan_extension/setup_ukan.sh` clones it).
- `ukan_extension/patch_ukan_weighted.py` modifies the user's own copy of U-KAN's `train.py` and `losses.py`
  in place. To locate the edit points it contains short excerpts (a few lines each) of those files.
  All inserted code is marked `[Ψ加權]`; the original files are kept as `*.orig`.
- `ukan_extension/ukan_patch.diff` shows the resulting changes with a few lines of surrounding context.
- `ukan_extension/predict_busi.py` imports U-KAN's `archs` module at run time and reproduces the data split
  and preprocessing of U-KAN's `train.py` / `val.py`; it contains no copied U-KAN code.
- `ukan_extension/requirements_ukan.txt` is a rewritten version of U-KAN's `requirements.txt`
  (see the README for the reasons for each change).

## BUSI dataset

- Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images*. Data in Brief, 28:104863, 2020.
- Images and masks are not included in this repository. `results/` contains only per-image metrics
  and Ψ values identified by the original file names.

## psi-weighted-nnunet

- `predict_busi.py` uses the Ψ functions in `psi/run_all_psi.py` of
  https://github.com/OwenWu930302/psi-weighted-nnunet (same author, Apache-2.0).