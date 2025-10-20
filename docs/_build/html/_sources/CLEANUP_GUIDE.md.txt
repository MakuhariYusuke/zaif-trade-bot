# Project Cleanup and Optimization Guide

**Date:** 2025-10-08  
**Version:** 3.6.1  
**Purpose:** Clean workspace and prepare for production training

---

## 📋 Overview

このガイドでは、プロジェクトの整理整頓と設定ファイルの最適化を実施します。

### Goals

1. ✅ 古いモデルとチェックポイントのアーカイブ
2. ✅ ルート直下のPythonファイルの整理
3. ✅ 設定ファイルの最適化
4. ✅ クリーンな環境での学習準備

---

## 🚀 Quick Start

### Option 1: Automated Cleanup (Recommended)

```powershell
# Run the automated cleanup script
.\scripts\cleanup_project.ps1
```

### Option 2: Manual Cleanup

以下のセクションに従って手動で整理します。

---

## 📁 Phase 1: Archive Old Models

### 1.1 Create Archive Directories

```powershell
New-Item -ItemType Directory -Force -Path "models\archived"
New-Item -ItemType Directory -Force -Path "checkpoints\archived"
```

### 1.2 Archive Hyperparameter Search Results

```powershell
# Move all test models to archive
Move-Item "models\*_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\aggressive_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\curriculum_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\scalping_*.zip" "models\archived\" -ErrorAction SilentlyContinue
```

### 1.3 Archive Old Checkpoints

```powershell
# Move old training sessions
Move-Item "checkpoints\scalping*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\iterative_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\test-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\smoke_test_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
```

**Result:** Clean models/ and checkpoints/ directories with only active/recent files.

---

## 📝 Phase 2: Reorganize Root Python Files

### 2.1 Create Script Subdirectories

```powershell
New-Item -ItemType Directory -Force -Path "scripts\analysis"
New-Item -ItemType Directory -Force -Path "scripts\data"
New-Item -ItemType Directory -Force -Path "scripts\validation"
New-Item -ItemType Directory -Force -Path "scripts\debug"
New-Item -ItemType Directory -Force -Path "tests\manual"
```

### 2.2 Move Analysis Scripts

```powershell
# Analysis and benchmarking
Move-Item "ablation_study.py" "scripts\analysis\"
Move-Item "analyze_*.py" "scripts\analysis\"
Move-Item "benchmark_*.py" "scripts\analysis\"
Move-Item "compare_*.py" "scripts\analysis\"
Move-Item "feature_*.py" "scripts\analysis\"
Move-Item "*_analysis.py" "scripts\analysis\"
```

### 2.3 Move Data Generation Scripts

```powershell
# Data generation
Move-Item "create_*.py" "scripts\data\"
Move-Item "fetch_*.py" "scripts\data\"
Move-Item "generate_*.py" "scripts\data\"
```

### 2.4 Move Validation Scripts

```powershell
# Validation
Move-Item "validate_*.py" "scripts\validation\"
Move-Item "check_*.py" "scripts\validation\"
Move-Item "stress_test.py" "scripts\validation\"
```

### 2.5 Move Debug Scripts

```powershell
# Debug utilities
Move-Item "debug_*.py" "scripts\debug\"
Move-Item "monitor_*.py" "scripts\debug\"
Move-Item "demo_*.py" "scripts\debug\"
Move-Item "quick_*.py" "scripts\debug\"
```

### 2.6 Move Test Files

```powershell
# Manual test files
New-Item -ItemType Directory -Force -Path "tests\manual"
Move-Item "test_*.py" "tests\manual\" -ErrorAction SilentlyContinue
Move-Item "simple_backtest.py" "scripts\analysis\"
```

### 2.7 Remove Temporary Files

```powershell
# Clean up temporary/obsolete files
Remove-Item "temp*.py" -ErrorAction SilentlyContinue
Remove-Item "fix_*.py" -ErrorAction SilentlyContinue
Remove-Item "clean_*.py" -ErrorAction SilentlyContinue
Remove-Item "remove_*.py" -ErrorAction SilentlyContinue
```

**Result:** Clean root directory with only essential entry points:
- `run_training.py`
- `run_training_memory_optimized.py`
- `live_trade.py`
- `backtest_model.py`

---

## ⚙️ Phase 3: Optimize Configuration Files

### 3.1 Backup Original Configuration

```powershell
# Backup the original
Copy-Item "configs\training\ppo_balanced_mem_optimized.json" `
          "configs\training\ppo_balanced_mem_optimized_backup_20251008.json"
```

### 3.2 Apply Optimized Configuration

```powershell
# Replace with optimized version
Copy-Item "configs\training\ppo_balanced_mem_optimized_v3.6.1.json" `
          "configs\training\ppo_balanced_mem_optimized.json"
```

### 3.3 Validate Configuration

```powershell
# Check configuration consistency
python scripts\check_config_consistency.py `
  configs\training\ppo_balanced_mem_optimized.json
```

**Expected Output:**
```
✅ Configuration validation passed
✅ No inconsistencies found
✅ env_config matches top-level settings
✅ All required fields present
```

---

## 🧪 Phase 4: Verify Clean Environment

### 4.1 Run Import Check

```powershell
# Verify all imports work
python -c "from ztb.trading.constants import ACTION_HOLD, ACTION_BUY, ACTION_SELL; print('✅ Constants OK')"
python -c "import ztb.training.unified_trainer; print('✅ Trainer OK')"
python -c "import ztb.trading.environment; print('✅ Environment OK')"
```

### 4.2 Run Test Suite

```powershell
# Run unit tests
python -m pytest tests\unit\trading\live\test_live_trade.py -v
```

**Expected:** 7/7 PASS

### 4.3 Check Disk Space

```powershell
# Check archived space savings
$archivedModels = (Get-ChildItem "models\archived" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
$archivedCheckpoints = (Get-ChildItem "checkpoints\archived" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host "Archived Models: $([math]::Round($archivedModels, 2)) MB"
Write-Host "Archived Checkpoints: $([math]::Round($archivedCheckpoints, 2)) MB"
Write-Host "Total Archived: $([math]::Round($archivedModels + $archivedCheckpoints, 2)) MB"
```

---

## 📊 Summary of Changes

### Directory Structure (Before → After)

```
/
├── models/                    ├── models/
│   ├── (100+ .zip files)  →   │   ├── archived/ (old files)
│   └── ...                    │   ├── ppo_100k_optimized.zip
│                               │   └── ensemble_config.json
│
├── checkpoints/               ├── checkpoints/
│   ├── (50+ session dirs) →   │   ├── archived/ (old sessions)
│   └── ...                    │   └── ppo_balanced_mem_optimized/
│
├── (50+ .py files)        →   ├── run_training.py
│                               ├── live_trade.py
│                               ├── backtest_model.py
│                               │
│                               ├── scripts/
│                               │   ├── analysis/ (20+ files)
│                               │   ├── data/ (10+ files)
│                               │   ├── validation/ (10+ files)
│                               │   └── debug/ (10+ files)
│                               │
│                               └── tests/
│                                   └── manual/ (test files)
```

### Configuration Files

- ✅ **ppo_balanced_mem_optimized.json**: Optimized with v3.6.1 improvements
- ✅ **Bug #45 fixed**: env_config consistency
- ✅ **Full documentation**: All settings explained
- ✅ **Metadata added**: Version tracking, bug fixes recorded

---

## ✅ Verification Checklist

- [ ] Old models archived to `models/archived/`
- [ ] Old checkpoints archived to `checkpoints/archived/`
- [ ] Root Python files reorganized to `scripts/*/`
- [ ] Temporary files removed
- [ ] Configuration backed up
- [ ] Optimized configuration applied
- [ ] Configuration validated
- [ ] All imports working
- [ ] Tests passing (7/7)
- [ ] Disk space checked

---

## 🎯 Ready for Training

Once all verification steps are complete, you're ready to start production training:

```powershell
# Start training with optimized configuration
python run_training.py --config configs\training\ppo_balanced_mem_optimized.json
```

---

## 📝 Rollback (If Needed)

If you need to revert changes:

```powershell
# Restore original configuration
Copy-Item "configs\training\ppo_balanced_mem_optimized_backup_20251008.json" `
          "configs\training\ppo_balanced_mem_optimized.json"

# Restore files from archive (example)
Move-Item "models\archived\*" "models\" -ErrorAction SilentlyContinue
```

---

**Last Updated:** 2025-10-08  
**Status:** Ready for execution  
**Estimated Time:** 10-15 minutes
