# Bug #48 水平展開調査レポート

**調査日時**: 2025-10-08 12:20
**調査範囲**: ztb/training/**/*.py 内の全環境構築箇所

---

## 🔍 調査結果サマリー

### **バグ発見箇所: 2ファイル**
1. ✅ **修正済み**: `ztb/training/sell_mitigation_ppo_trainer.py`
2. ✅ **修正済み**: `ztb/training/trainers/sell_mitigation_trainer.py`

### **問題なし: 8ファイル**
以下のファイルは正しく設定を渡しています:
- `ztb/training/ppo_trainer.py` → `config`を直接渡す
- `ztb/training/ppo_trainer_old.py` → `config`を直接渡す
- `ztb/training/binary_search/base_optimizer.py` → `env_config`に`reward_settings`含む
- `ztb/training/curriculum_transition.py` → `config`を直接構築
- `ztb/training/simple_reward.py` → `get_trading_env_config()`使用
- `ztb/training/train_simple_reward.py` → `get_trading_env_config()`使用
- `ztb/training/training_utils.py` → `config`を引数で受け取り直接渡す
- `ztb/training/curriculum_learning.py` → `UnifiedTrainer`経由(間接)

---

## 📋 詳細な修正内容

### **1. ztb/training/sell_mitigation_ppo_trainer.py (Line 281-289)**

#### 修正前:
```python
env_config = {
    "curriculum_stage": self.config.get("curriculum_stage", "full"),
    "allow_reverse": self.allow_reverse,
    "transaction_cost": self.config.get("transaction_cost", 0.001),
    "max_position_size": self.config.get("max_position_size", 1.0),
    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
    "reward_scaling": self.config.get("reward_scaling", 1.0),
    # ★ reward_settings が欠落
}
```

#### 修正後:
```python
env_config = {
    "curriculum_stage": self.config.get("curriculum_stage", "full"),
    "allow_reverse": self.allow_reverse,
    "transaction_cost": self.config.get("transaction_cost", 0.001),
    "max_position_size": self.config.get("max_position_size", 1.0),
    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
    "reward_scaling": self.config.get("reward_scaling", 1.0),
    # ★ BUG FIX #48: Pass reward_settings from config to environment
    "reward_settings": self.config.get("reward_settings", {}),
}
```

---

### **2. ztb/training/trainers/sell_mitigation_trainer.py (Line 190-198)**

#### 修正前:
```python
env_config = {
    "curriculum_stage": self.config.get("curriculum_stage", "full"),
    "allow_reverse": self.allow_reverse,
    "transaction_cost": self.config.get("transaction_cost", 0.001),
    "max_position_size": self.config.get("max_position_size", 1.0),
    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
    "reward_scaling": self.config.get("reward_scaling", 1.0),
    # ★ reward_settings が欠落
}
```

#### 修正後:
```python
env_config = {
    "curriculum_stage": self.config.get("curriculum_stage", "full"),
    "allow_reverse": self.allow_reverse,
    "transaction_cost": self.config.get("transaction_cost", 0.001),
    "max_position_size": self.config.get("max_position_size", 1.0),
    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
    "reward_scaling": self.config.get("reward_scaling", 1.0),
    # ★ BUG FIX #48: Pass reward_settings from config to environment
    "reward_settings": self.config.get("reward_settings", {}),
}
```

---

## 🎯 影響分析

### **修正した2ファイルの使用状況**

1. **sell_mitigation_ppo_trainer.py**
   - 用途: SELL回避緊急修正で使用中 (**重要度: 最高**)
   - 影響: v3.6.1, v3.6.3の検証で実際に使用
   - 結果: reward_settings未反映によりSELL率改善失敗

2. **trainers/sell_mitigation_trainer.py**
   - 用途: 旧バージョンのSELL緩和トレーナー
   - 影響: 現在未使用だが、将来の再利用時に問題になる可能性
   - 結果: 予防的修正

### **問題なかったファイルの共通パターン**

安全な実装パターン:
1. **configを直接渡す**: `HeavyTradingEnv(df=df, config=self.config)`
2. **完全なenv_config構築**: `reward_settings`を含めて構築
3. **ヘルパー関数使用**: `get_trading_env_config()`経由

危険な実装パターン:
1. **部分的なenv_config構築**: 必要な項目のみ列挙 → 漏れが発生

---

## 📝 推奨事項

### **1. コーディングガイドライン追加**
```python
# ❌ 避けるべき: 部分的な env_config 構築
env_config = {
    "curriculum_stage": ...,
    "transaction_cost": ...,
    # reward_settings が漏れる可能性
}

# ✅ 推奨: config を直接渡す
env = HeavyTradingEnv(df=df, config=self.config)

# ✅ または: ヘルパー関数を使う
env_config = get_trading_env_config({...})
env = HeavyTradingEnv(df=df, config=env_config)
```

### **2. 単体テスト追加**
```python
def test_reward_settings_propagation():
    """reward_settingsが環境に正しく渡されることを確認"""
    config = {
        "reward_settings": {
            "profit_bonus_multipliers": [1.0, 2.0, 3.0]
        }
    }
    trainer = SELLBiasMitigationPPOTrainer(...)
    # trainer内のenv.reward_settingsを確認
    assert trainer.env.reward_settings["profit_bonus_multipliers"] == [1.0, 2.0, 3.0]
```

### **3. 設定伝播の可視化**
トレーニング開始時にreward_settingsをログ出力:
```python
logger.info(f"reward_settings: {env.reward_settings}")
```

---

## ✅ 水平展開完了チェックリスト

- [x] sell_mitigation_ppo_trainer.py - **修正済み**
- [x] trainers/sell_mitigation_trainer.py - **修正済み**
- [x] ppo_trainer.py - 問題なし(config直接渡し)
- [x] ppo_trainer_old.py - 問題なし(config直接渡し)
- [x] binary_search/base_optimizer.py - 問題なし(reward_settings含む)
- [x] curriculum_transition.py - 問題なし(config直接構築)
- [x] simple_reward.py - 問題なし(ヘルパー使用)
- [x] train_simple_reward.py - 問題なし(ヘルパー使用)
- [x] training_utils.py - 問題なし(config引数渡し)
- [x] curriculum_learning.py - 問題なし(UnifiedTrainer経由)

---

## 📊 まとめ

### **修正ファイル数**: 2/10
### **問題検出率**: 20%
### **重要度**: **CRITICAL** (本番使用中のコードに影響)

**結論**:
- SELL回避修正で使用中の`sell_mitigation_ppo_trainer.py`の重大バグを修正
- 同様のパターンが1ファイルに存在し、予防的に修正
- 他の8ファイルは安全な実装パターンを使用

**次のアクション**:
1. ✅ 修正完了 → 再検証実行中
2. 📝 コーディングガイドライン更新
3. 🧪 単体テスト追加(将来の課題)

---

**調査完了日時**: 2025-10-08 12:25
**ステータス**: ✅ 水平展開完了・全修正適用済み
