# QTI_ML Repository Onboarding Guide

This guide explains how this repository is structured, how the training/evaluation pipelines fit together, and how to run it in a clean environment from VS Code.

---

## 1) What this repository is for

This repo contains two closely related PyTorch workflows for diffusion MRI / QTI modeling:

1. **Supervised regression pipeline (`QTI_MLP`)**
   - Learns to predict scalar invariants (e.g., `FA`, `uFA`, etc.) from diffusion-weighted signal volumes.
   - Uses paired input/output training data (`.nii.gz` + `.mat` target invariants).

2. **Physics-informed self-supervised pipeline (`QTIp_MLP`)**
   - Predicts tensor-parameter representation first.
   - Reconstructs diffusion signal with physics/tensor math (`QTIp_tensor_math.py`) and trains on signal reconstruction losses.

There is also a full **performance evaluation stack** (`Eval_metrics.py`, figure scripts, notebooks) for computing and plotting metrics such as nRMSE, pSNR, and SSIM.

---

## 2) High-level project layout

### Core model / data code
- `QTI_MLP.py`
  - Main supervised dataset class (`QTI_Dataset`), model definitions (`QTI_MLP`, `QTIb_MLP`), utility funcs, train/test loops.
- `QTIp_MLP.py`
  - Physics-informed dataset class (`QTIp_Dataset`), model class (`QTIp_MLP`), train/test loops.
- `QTIp_tensor_math.py`
  - Physics/tensor helper math used by QTIp training and inference.
- `dw_signal_l1_loss.py`
  - Optional custom loss module around DW signal operations.

### Training entry points
- `QTI_MLP_train.py`
  - Script-style training entry for supervised model.
- `QTIp_MLP_train.py`
  - Script-style training entry for physics-informed model.

### Prediction / inference entry points
- `QTI_MLP_predict.py`
  - Loads trained supervised model and writes predictions.
- `QTIp_MLP_predict.py`
  - Loads trained QTIp model, reconstructs signal/params, writes outputs.

### Evaluation and plotting
- `Eval_metrics.py`
  - `QTI_Fit_Dataset` and metric computations (nRMSE, pSNR, SSIM, diff maps) + plotting utilities.
- `run_performance_eval.py`
  - Script wrapper around `Eval_metrics.py`.
- `nRMSE_figure.py`, `pSNR_figure.py`, `SSIM_figure.py`, `loss_curve_plt.py`, `ReLU_plot.py`
  - Focused plotting scripts.
- Multiple notebooks (`*.ipynb`)
  - Experimentation / analysis / paper-figure generation.

### Results / experiment logs
- `runs/`
  - TensorBoard logs and run folders for many experiments.

---

## 3) Data conventions used in code

The code assumes neuroimaging and MATLAB artifacts in specific shapes/formats:

- **Diffusion signal input**: `.nii.gz`
- **Targets / metadata**: `.mat` (e.g., dps/xps structures)
- **Mask**: `.nii.gz` binary mask

Common flow in dataset classes:
1. Read NIfTI (`nibabel`) and convert to tensors.
2. Reorder dimensions (`torch.permute`) to repository conventions.
3. Apply mask and optional slice exclusion.
4. Normalize / threshold / mask invalid values.
5. Flatten to voxel-wise batches for MLP training.

The scripts are mostly hardcoded with user-local paths from earlier runs, so first practical task is replacing those paths with your local dataset paths.

---

## 4) Pipeline mental model

### A) Supervised path (`QTI_MLP`)
1. Build `QTI_Dataset` from signal + scalar invariants.
2. Apply preprocessing (masking, normalization, flattening).
3. Train `QTI_MLP`/`QTIb_MLP` with train/test loops.
4. Save model checkpoints.
5. Run `QTI_MLP_predict.py` for inference.
6. Evaluate predictions with `Eval_metrics.py` / notebooks.

### B) Physics-informed path (`QTIp_MLP`)
1. Build `QTIp_Dataset` from signal + xps (+ mask).
2. Preprocess signal and b-tensors.
3. Train `QTIp_MLP` to produce latent/tensor params.
4. Use `QTIp_tensor_math` to reconstruct DW signal and compute loss.
5. Save model; predict via `QTIp_MLP_predict.py`.
6. Export `.nii` / `.mat` outputs for downstream analysis.

---

## 5) Why VS Code stopped recognizing packages

Most likely cause: **interpreter mismatch**.

Even if you previously used:
- `C:/miniforge3/envs/QTI_Project/python.exe`

VS Code may currently be pointed to another interpreter (global Python, base conda env, or a different env), so imports like `matplotlib` fail there.

### Verify quickly in VS Code terminal
Run:
```powershell
python -c "import sys; print(sys.executable)"
```
If this is not `.../envs/QTI_Project/python.exe`, VS Code is using the wrong interpreter.

---

## 6) Clean setup in this repo

Created files:
- `environment.yml` (conda-first setup)
- `requirements.txt` (pip fallback)
- `setup_env.ps1` (automated setup commands)
- `verify_environment.py` (import smoke-test)

### Recommended setup path
1. Open terminal in repo root (`C:\QTI_ML`).
2. Run:
   ```powershell
   .\setup_env.ps1
   ```
3. In VS Code, choose interpreter **Python (QTI_Project)**.
4. Run:
   ```powershell
   python .\verify_environment.py
   ```

If you already have the env and just want package sync:
```powershell
conda activate QTI_Project
python -m pip install -r .\requirements.txt
python .\verify_environment.py
```

---

## 7) Package inventory used by this repo

External Python deps detected from source and notebooks:
- `torch`
- `torchmetrics`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `nibabel`
- `cmasher`
- `tensorboard` (via `torch.utils.tensorboard`)
- Notebook stack: `jupyter`, `ipykernel`

Optional ecosystem notes:
- MATLAB appears in historical workflow (`.m` file and notebook comments), but it is not a required Python package for core scripts to import.

---

## 8) First productive tasks when onboarding

1. **Path normalization pass**
   - Replace hard-coded macOS paths with project config or relative paths.
2. **Choose one canonical entry script**
   - e.g., keep one training script per pipeline with arg parsing.
3. **Create minimal reproducible run**
   - Single subject / tiny subset for fast iteration.
4. **Lock versions once stable**
   - Freeze exact versions after first successful end-to-end run.

---

## 9) Known characteristics (important when debugging)

- Script-centric style with many historical path blocks and commented alternatives.
- Heavy reliance on tensor shape manipulations and masking order.
- Different dtypes across pipelines (`QTIp` uses float64 defaults).
- Many analysis notebooks and experiment folders under `runs/`.
- Assumes domain-specific file structures and naming conventions.

---

## 10) Suggested daily workflow in VS Code

1. Select `QTI_Project` interpreter.
2. Run `verify_environment.py` once per machine/session reset.
3. Start from the pipeline script you need (`QTI_MLP_*` or `QTIp_MLP_*`).
4. Keep one local dataset config at top of script while onboarding.
5. Use TensorBoard on `runs/` for training checks.
6. Use `Eval_metrics.py`/notebooks for post-hoc analysis.

---

If you want, next step is to convert the hardcoded path blocks into a small config file + CLI arguments so the repo becomes plug-and-play on your Windows machine.
