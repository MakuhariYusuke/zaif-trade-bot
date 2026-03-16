# Phase 3 Action Space エラー詳細分析

**作成日:** 2026-01-25  
**問題:** SAC学習時に "Discrete(3) が提供されているが Box が必要" エラー  
**影響範囲:** 全48実験がブロック中

---

## 1. エラーの症状

### エラーメッセージ
```
AssertionError: The algorithm only supports Box as action spaces but Discrete(3) was provided
```

### 発生箇所
- **ファイル:** `stable_baselines3/common/base_class.py` (line 181)
- **タイミング:** SAC model 初期化時（trainer.train() 実行直後）
- **試行実験:** 単一実験・最小構成でも同じエラー

---

## 2. 根本原因の特定

### 2.1 環境生成の二重構造

**問題のある環境クラスが2つ存在:**

1. **`ztb/training/environments/heavy_trading_env.py`**  
   - UnifiedTrainer が使用する軽量版
   - **常に continuous action space** (Box) をハードコード
   ```python
   # Line 86-88
   self.action_space = spaces.Box(
       low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
   )
   ```

2. **`ztb/trading/environment/heavy_env/core.py`**  
   - 本番環境用の完全版 HeavyTradingEnv
   - action_type 設定を**正しく読み取る**
   ```python
   # mixins/initialization.py Line 489-503
   if use_continuous_actions:
       self.action_space = spaces.Box(...)
   else:
       self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)
   ```

### 2.2 UnifiedTrainer の環境選択ロジック

**`ztb/training/unified_trainer/trainer.py` (Line 2093-2140):**

```python
def _create_v433_training_environment(self) -> Optional[Any]:
    try:
        mod = importlib.import_module(
            "ztb.training.environments.heavy_trading_env"  # ← 軽量版を使用！
        )
        HeavyTradingEnv = getattr(mod, "HeavyTradingEnv", None)
    except Exception:
        HeavyTradingEnv = None
```

**現状:** 常に `ztb/training/environments/` (軽量版) をロード  
**問題:** この軽量版は **action_type 設定を無視**して常に Box を返す

### 2.3 なぜ Discrete(3) が返るのか？

**推測される流れ:**

1. UnifiedTrainer が軽量版 HeavyTradingEnv をインポート
2. 軽量版は Box を返す（正しい）
3. **しかし別の箇所で環境が再生成される可能性**
   - Walk-Forward の window 切り替え時
   - Feature Engineering 後の環境再構築時
   - その際に action_space が Discrete にリセットされる

**または:** 軽量版の初期化コードに条件分岐が存在し、特定条件で Discrete に切り替わる

---

## 3. Walk-Forward との関連性

### 3.1 多窓学習の仕組み

```python
# configs/walk_forward.yaml の想定動作
walk_forward:
  enabled: True
  n_splits: 4  # 4つの時間窓で学習

# 各窓ごとに:
# 1. データを分割
# 2. 特徴量エンジニアリング
# 3. 環境を再生成  ← ここで action_space が変わる可能性
# 4. モデル学習
```

### 3.2 環境再生成のリスク

**問題点:**
- Walk-Forward の各 window で環境が**破棄→再生成**される
- 再生成時に config が正しく引き継がれない
- 初回生成は Box、2回目以降は Discrete になる可能性

**メモリリークとの関連:**
- 774MB → 3.5GB の増加は、環境が**完全に破棄されずメモリに残留**
- 4窓 × 複数特徴量セット = 大量のオブジェクト残存
- これ以上詰めると不具合誘発のリスク（ユーザー指摘の通り）

---

## 4. 修正方針（慎重なアプローチ）

### 4.1 最小限の修正（推奨）

**Option A: 軽量版環境の使用を停止**

```python
# trainer.py Line 2093 を修正
mod = importlib.import_module(
    "ztb.trading.environment.heavy_env.core"  # 完全版を使用
)
```

**利点:**
- 1行の変更で解決
- action_type 設定が正しく読み取られる
- Walk-Forward でも config が引き継がれる

**リスク:**
- 完全版は重い（メモリ増加の可能性）
- 既存の軽量版依存コードが影響を受ける

---

**Option B: Walk-Forward を一時無効化**

```python
# test_single_experiment.py の config
config = {
    "training": {
        "walk_forward": {
            "enabled": False  # 単一窓で学習
        }
    }
}
```

**利点:**
- 環境再生成の問題を回避
- シンプルな学習フローで問題切り分け
- メモリリーク軽減

**リスク:**
- 48実験の設計変更が必要（4窓 → 1窓 = 12実験に減少）
- Walk-Forward の効果を検証できない

---

### 4.2 Walk-Forward を詰めすぎない理由

**ユーザー指摘:**
> "一本で学習していたのを複数本で学習するようにするための機能なのでこれ以上詰めるのも難しい。詰めすぎると逆に不具合を誘発しかねない"

**技術的背景:**
1. **データ分割の複雑性:**
   - 時系列の順序保持
   - オーバーラップの制御
   - 各窓の独立性確保

2. **環境のライフサイクル:**
   - 窓ごとに環境破棄→再生成
   - 状態の完全リセット
   - メモリ管理の複雑化

3. **Feature Engineering との相互作用:**
   - 各窓で異なる特徴量セット
   - 正規化パラメータの窓間独立性
   - 品質フィルタリングの一貫性

**結論:** Walk-Forward は既に複雑な実装。これ以上の深掘りは避けるべき。

---

## 5. 推奨アクション

### Step 1: 環境クラスの切り替え（最優先）

```python
# ztb/training/unified_trainer/trainer.py
# Line 2093-2098 を修正

def _create_v433_training_environment(self) -> Optional[Any]:
    try:
        # 完全版を使用（action_type設定を尊重）
        mod = importlib.import_module(
            "ztb.trading.environment.heavy_env.core"
        )
        HeavyTradingEnv = getattr(mod, "HeavyTradingEnv", None)
```

### Step 2: 単一実験でテスト

```powershell
# Walk-Forward 無効化した minimal config で検証
python scripts/v459/test_single_experiment.py
```

**期待結果:**
- Box action space が使用される
- SAC 初期化が成功
- 5000 timesteps 完走

### Step 3: Walk-Forward 1窓でテスト

```python
# walk_forward を再有効化（n_splits: 1）
config["training"]["walk_forward"] = {
    "enabled": True,
    "n_splits": 1  # 単一窓で動作確認
}
```

**期待結果:**
- 環境再生成後も Box を維持
- メモリリークの発生状況確認
- 完走可能性の評価

### Step 4: 実験計画の調整

**Case A: Walk-Forward が安定動作**
- 元の計画通り 4窓 × 12実験 = 48サンプル

**Case B: Walk-Forward が不安定**
- 1窓 × 12実験 = 12サンプル
- 統計的検定力は低下するが実行可能
- Mann-Whitney U test は n=12 でも有効

---

## 6. メモリリーク問題との分離

### 現状のメモリ使用量
```
774MB (初期) → 1142MB (学習開始) → 3512MB (失敗時)
増加: 2738MB (354% 増加)
閾値: 800MB (342% 超過)
```

### 原因の切り分け

**Action Space 問題:**
- 環境初期化で失敗→学習開始前にエラー
- メモリリークとは**別問題**

**メモリリーク問題:**
- 学習が進むにつれて増加
- Walk-Forward + Feature Engineering が主因
- gc.collect() では不十分

**結論:** 両者は独立した問題。Action Space を先に解決すべき。

---

## 7. リスク評価

### 高リスク（避けるべき）
❌ Walk-Forward の内部実装を変更  
❌ Feature Engineering のキャッシュ機構を改造  
❌ 環境の初期化順序を大幅に変更  

### 中リスク（慎重に実施）
⚠️ 完全版 HeavyTradingEnv への切り替え（メモリ増加の可能性）  
⚠️ Walk-Forward の窓数削減（実験設計への影響）  

### 低リスク（推奨）
✅ 環境インポート先の1行変更  
✅ Walk-Forward 無効化テスト  
✅ 単一実験での動作確認  

---

## 8. 次のステップ

1. **即時対応:** trainer.py の環境インポート先を変更
2. **検証:** test_single_experiment.py で動作確認
3. **判断:** Walk-Forward の安定性を評価
4. **実行:** 実験計画の最終決定（48 or 12サンプル）

**重要:** ユーザーの指摘通り、Walk-Forward を過度に詰めずシンプルな解決策を優先する。
