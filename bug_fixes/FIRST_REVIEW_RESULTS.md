# 潜在的バグレビュー - 徹底調査

**日付:** 2025年10月8日  
**目的:** 根本的なバグが複数見つかったため、さらなる潜在的バグを洗い出す

---

## 🔍 調査対象領域

### 1. ⚠️ **Critical: MaskablePPO使用箇所の総点検**

#### 発見された問題箇所

**問題1: simple_backtest.py (Line 62)**
```python
# ❌ 問題: predict_with_masksを使用していない
action, _states = model.predict(obs, deterministic=True)
```

**深刻度:** Critical  
**影響:** MaskablePPOモデル使用時にaction_masksが渡されず、不正確な予測

**問題2: debug_model_predictions.py (Line 68)**
```python
# ❌ 問題: predict_with_masksを使用していない
action, _ = model.predict(obs, deterministic=False)
```

**深刻度:** Critical  
**影響:** デバッグ時の動作確認が不正確

**問題3: regime_evaluation.py (Line 148)**
```python
# ❌ 問題: MaskablePPOをロードしているがaction_masksなし
model = MaskablePPO.load(model_path)
...
action, _ = model.predict(obs, deterministic=False)
```

**深刻度:** Critical  
**影響:** レジーム評価が完全に不正確

**問題4: test_paper_trading.py (Line 167)**
```python
# ❌ 問題: MaskablePPOをロードしているがaction_masksなし
model = MaskablePPO.load(args.model_path)
...
action, _ = model.predict(obs_reshaped, deterministic=True)
```

**深刻度:** Critical  
**影響:** ペーパートレーディングテストが不正確

---

### 2. 🔍 環境とPositionManagerの同期問題

#### 調査: _last_trade_stepの二重管理

**ファイル:** 
- `ztb/trading/environment/environment.py`
- `ztb/trading/environment/components/position_manager.py`

**懸念:**
- 環境に`self._last_trade_step`がある
- PositionManagerに`self._last_trade_step`がある
- 二重管理で不整合が発生する可能性

**要確認:**
```python
# environment.py
self._last_trade_step = ...

# position_manager.py  
self._last_trade_step = ...
```

これらが同期されているか?

---

### 3. 🔍 get_legal_actionsとaction_masksの一貫性

#### 調査ポイント

**ファイル:** `ztb/trading/environment/environment.py`

```python
def get_legal_actions(self) -> NDArray[np.int32]:
    # Returns legal actions based on various constraints
    
def action_mask(self) -> NDArray[np.bool_]:
    # Returns action masks for MaskablePPO
    
def get_action_masks(self) -> NDArray[np.bool_]:
    # Alias for action_mask()
```

**懸念:**
- これらのメソッドが同じロジックを使っているか?
- 不整合があると、環境側の制限とMaskablePPOの制限が矛盾

---

### 4. 🔍 報酬計算の正確性

#### 調査: 未実現損益と実現損益の混同

**ファイル:** `ztb/trading/environment/components/reward_calculator.py`

**懸念ポイント:**
1. ポジション保有中の報酬計算は未実現損益を使うべき
2. ポジションクローズ時の報酬は実現損益を使うべき
3. これらが混同されていないか?

**要確認:**
```python
def calculate_reward(...):
    # 未実現PnLと実現PnLが正しく使い分けられているか?
```

---

### 5. 🔍 トレード記録の正確性

#### 調査: trades_historyの記録タイミング

**ファイル:** 
- `ztb/trading/environment/components/position_manager.py`
- `ztb/trading/environment/environment.py`

**懸念:**
- ポジション反転時にトレードが2回記録されているか?(close + open)
- PositionManagerのtrades_countと実際のtrades_history長が一致するか?

---

### 6. 🔍 環境リセット処理の完全性

#### 調査: reset()で状態が完全にリセットされるか

**ファイル:** `ztb/trading/environment/environment.py`

**要確認項目:**
- `self._last_trade_step` がリセットされるか?
- PositionManagerの状態が完全にリセットされるか?
- RewardCalculatorの累積値がリセットされるか?
- `self._consecutive_trade_steps` がリセットされるか?

---

## 📋 次のアクション

### 優先度: Critical (即座に修正)
1. ✅ simple_backtest.py → predict_with_masks適用
2. ✅ debug_model_predictions.py → predict_with_masks適用  
3. ✅ regime_evaluation.py → predict_with_masks適用
4. ✅ test_paper_trading.py → predict_with_masks適用

### 優先度: High (詳細調査必要)
5. ⬜ _last_trade_stepの二重管理問題を調査
6. ⬜ get_legal_actionsとaction_masksの一貫性を確認
7. ⬜ 報酬計算の未実現/実現PnL使い分けを確認

### 優先度: Medium (検証)
8. ⬜ トレード記録の正確性を検証
9. ⬜ 環境リセット処理の完全性を検証

---

## 🎯 見つかった問題の深刻性

### なぜこれほど多くのバグが見落とされたか?

1. **MaskablePPO移行時の不完全な対応**
   - 既存のPPOコードをMaskablePPOに移行した際、評価スクリプトが更新されなかった
   - `predict_with_masks`ユーティリティを作ったが、既存コードに適用しなかった

2. **統合テストの不足**
   - バックテスト/評価スクリプトの自動テストがない
   - MaskablePPO使用時の動作を検証するテストがない

3. **コードレビューの不足**
   - action_masksが必須であることの認識が不十分
   - predict()呼び出し全箇所の総点検が行われなかった

---

## 💡 再発防止策

### 1. Lintルール追加
```python
# MaskablePPO.load()の後にpredict()がある場合、警告
# → predict_with_masks使用を強制
```

### 2. 統合テストの追加
```python
# test_all_scripts_use_masks.py
# すべての評価スクリプトでMaskablePPO使用時にaction_masksが渡されることを確認
```

### 3. ドキュメント明記
```markdown
# CRITICAL: MaskablePPO使用時の必須事項
1. 必ず predict_with_masks() を使用すること
2. model.predict() の直接呼び出しは禁止
3. action_masks がない予測は不正確
```

---

**調査継続中...**
