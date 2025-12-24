# 🚀 How to Run Step 2 - Complete Guide

## 📋 Prerequisites

1. ✅ Virtual environment activated
2. ✅ Features downloaded (384 files in `data/video/omnivore/`)
3. ✅ Models trained (or use existing checkpoints)

---

## 🔧 Step 1: Activate Environment

```powershell
cd "D:\polito\aml project\aml-2025-mistake-detection"
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

---

## 📊 Step 2: Run Error Type Analysis (Part 2a)

### For MLP Model:
```powershell
python -m core.evaluate_error_types `
    --variant MLP `
    --backbone omnivore `
    --split recordings `
    --ckpt "checkpoints/error_recognition/MLP/omnivore/error_recognition_recordings_omnivore_MLP_video_epoch_4.pt" `
    --threshold 0.4 `
    --save_csv
```

### For Transformer Model:
```powershell
python -m core.evaluate_error_types `
    --variant Transformer `
    --backbone omnivore `
    --split recordings `
    --ckpt "checkpoints/error_recognition/Transformer/omnivore/error_recognition_recordings_omnivore_Transformer_video_epoch_1.pt" `
    --threshold 0.4 `
    --save_csv
```

### For LSTM Model:
```powershell
python -m core.evaluate_error_types `
    --variant LSTM `
    --backbone omnivore `
    --split recordings `
    --ckpt "checkpoints/error_recognition/LSTM/omnivore/error_recognition_recordings_omnivore_LSTM_video_epoch_8.pt" `
    --threshold 0.4 `
    --save_csv
```

**Note:** Replace epoch numbers with your best/latest checkpoints!

---

## 🔄 Step 3: Compare All Baselines (Part 2b)

```powershell
python compare_baselines.py `
    --split recordings `
    --backbone omnivore `
    --mlp_ckpt "checkpoints/error_recognition/MLP/omnivore/error_recognition_recordings_omnivore_MLP_video_epoch_4.pt" `
    --transformer_ckpt "checkpoints/error_recognition/Transformer/omnivore/error_recognition_recordings_omnivore_Transformer_video_epoch_1.pt" `
    --lstm_ckpt "checkpoints/error_recognition/LSTM/omnivore/error_recognition_recordings_omnivore_LSTM_video_epoch_8.pt" `
    --save_csv
```

---

## 📈 Step 4: Run on Step Split (Optional)

If you want to evaluate on `step` split (threshold 0.6):

```powershell
# Error type analysis for step split
python -m core.evaluate_error_types --variant MLP --backbone omnivore --split step --ckpt "YOUR_CHECKPOINT.pt" --threshold 0.6 --save_csv

# Baseline comparison for step split
python compare_baselines.py --split step --backbone omnivore --mlp_ckpt "..." --transformer_ckpt "..." --lstm_ckpt "..." --save_csv
```

---

## 📁 Output Locations

After running, you'll find results in:

- **Error Type Analysis**: `results/error_type_analysis/`
  - Files: `{variant}_{backbone}_{split}_error_type_analysis.csv`

- **Baseline Comparison**: `results/baseline_comparison.csv`

---

## 🎯 Quick Test (Using Current Checkpoints)

Since you have some checkpoints already, let's test with LSTM:

```powershell
# Activate environment
cd "D:\polito\aml project\aml-2025-mistake-detection"
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1

# Test LSTM error type analysis
python -m core.evaluate_error_types --variant LSTM --backbone omnivore --split recordings --ckpt "checkpoints/error_recognition/LSTM/omnivore/error_recognition_recordings_omnivore_LSTM_video_epoch_8.pt" --threshold 0.4 --save_csv
```

---

## ⚠️ Troubleshooting

### If you get "CUDA out of memory":
- Add `--device cpu` to use CPU instead

### If checkpoint path is wrong:
- Find your checkpoints: `Get-ChildItem -Path "checkpoints" -Recurse -Filter "*.pt"`

### If you want to wait for training to complete:
- Check training progress in terminal
- Best models are saved as `*_best.pt` or highest epoch number

---

## 📝 Expected Output

### Error Type Analysis Output:
```
================================================================================
ERROR TYPE ANALYSIS - LSTM (omnivore) - recordings split
================================================================================

📊 OVERALL METRICS:
----------------------------------------
  Accuracy:  XX.XX%
  Precision: XX.XX%
  Recall:    XX.XX%
  F1 Score:  XX.XX%
  AUC:       XX.XX%

📋 PER ERROR TYPE METRICS:
--------------------------------------------------------------------------------
Error Type                   Count      Acc     Prec   Recall       F1      AUC
--------------------------------------------------------------------------------
Technique Error                 XX   XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%
Preparation Error               XX   XX.XX%   XX.XX%   XX.XX%   XX.XX%   XX.XX%
...
```

### Baseline Comparison Output:
```
+-------------+----------+----------+-----------+--------+-------+-------+
| Model       | Backbone | Accuracy | Precision | Recall |    F1 |   AUC |
+-------------+----------+----------+-----------+--------+-------+-------+
| MLP         | omnivore |    XX.XX |     XX.XX |  XX.XX | XX.XX | XX.XX |
| Transformer | omnivore |    XX.XX |     XX.XX |  XX.XX | XX.XX | XX.XX |
| LSTM        | omnivore |    XX.XX |     XX.XX |  XX.XX | XX.XX | XX.XX |
+-------------+----------+----------+-----------+--------+-------+-------+
```

