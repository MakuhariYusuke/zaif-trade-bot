# Doc14レビュー対応方針 (15)

**Date**: 2026-01-23  
**対象**: Doc14 Phase 2仕様書レビュー指摘事項  
**Status**: 📋 対応方針策定

---

## 1. Executive Summary

Doc14レビューで**Critical 2件、Major 6件、Minor 2件**の指摘を受けました。検証の結果、以下の事実が判明しました：

### 1.1 検証結果サマリー

| 指摘 | 検証結果 | 対応方針 |
|------|----------|----------|
| **C1: Phase 1完了根拠** | ✅ `should_block`は既に`should_enter`に修正済み | Doc説明の明確化 |
| **C2: P1-1定義ズレ** | ❌ Doc00="close明示"、Doc12="理由付与" | **仕様修正必要** |
| **M1: TradeType基準不一致** | ❌ 実装=long_open/close、Doc12=long_entry_win/loss | **仕様修正必要** |
| **M2: reason生成層不明** | ⚠️ 生成経路の設計が不足 | 設計追加必要 |
| **M3: AB統計仕様不整合** | ⚠️ Doc00=4seed×4split、Doc12=2seed | Phase 2は簡易版で進行 |
| **M4: AB条件比較不足** | ⚠️ A/B条件の定義が不足 | 設計追加必要 |
| **M5: Reporter統合仕様不足** | ⚠️ Training側API移植計画が不明確 | 詳細設計追加 |
| **M6: MTF/Scaler後ろ倒し** | ⚠️ Phase 3延期は妥当だが文書化不足 | 明文化 |
| **m1: テスト数不整合** | ⚠️ 103 vs 196 vs 123の表記揺れ | 統一 |
| **m2: 基本8種vs12種** | ⚠️ 表現の誤差 | 修正 |

### 1.2 対応優先度

**即時対応必要（Phase 2実装前）**:
1. ✅ **C1対応**: Phase 1完了根拠の明確化（検証済み、説明追加のみ）
2. ⚠️ **C2対応**: P1-1仕様の修正（Doc00準拠に戻す）
3. ⚠️ **M1対応**: TradeType基準の統一（実装に合わせる）

**Phase 2実装時に対応**:
4. ⚠️ **M2対応**: reason生成層の設計追加
5. ⚠️ **M4対応**: AB条件比較の設計追加
6. ⚠️ **M5対応**: Reporter統合詳細設計

**Phase 3で対応（Phase 2では明文化）**:
7. ✅ **M3対応**: AB統計仕様（Phase 2は簡易版と明記）
8. ✅ **M6対応**: MTF/Scaler延期の明文化

---

## 2. 各指摘の検証結果と対応

### 2.1 [Critical] C1: Phase 1完了根拠の揺れ

**指摘内容**:
> Entry Gate I/O不整合（`should_block`参照）とhold変換の挙動が残っており、P0完了の根拠が揺れています。

**検証結果**: ✅ **指摘は誤り（実装は正しい）**

**実装確認**:
```python
# ztb/trading/environment/fast_intraday_env_v456.py:572
if not gate_result["should_enter"]:
    # エントリーブロック → HOLDに変換
    action = self._convert_to_hold_action()
```

**事実**:
- ✅ `should_enter`キーを使用（`should_block`は使用していない）
- ✅ Phase 1テストで検証済み（test_p01_p02_completion.py: 10/10パス）
- ✅ hold変換挙動も正常（`_convert_to_hold_action()`実装済み）

**問題の原因**:
- Doc12の説明が不十分だった
- Phase 1完了報告でshould_enter使用の根拠が明確に記載されていなかった

**対応方針**: ✅ **説明追加のみ（実装変更不要）**
- Doc12に実装確認の根拠を追加
- Phase 1完了報告にコード引用を追加

**優先度**: Low（説明の問題のみ）

---

### 2.2 [Critical] C2: P1-1定義ズレ

**指摘内容**:
> P1-1の内容がDoc00のP1定義（"close"の明示処理）とズレています。Phase 2の計画では"理由付与"に軸足があり、**本来のP1バグが未解決のまま**になる可能性が高い。

**検証結果**: ❌ **指摘は正しい（仕様修正必要）**

**Doc00の定義**:
```markdown
# Doc00: Phase 2 (P1)
| Issue | 修正内容 | ファイル | テスト |
|-------|----------|----------|--------|
| Trade Type分類 | "close"の明示処理 | evaluator.py, reporter.py | 統計検証 |
```

**Doc12の記述**:
```markdown
# Doc12: P1-1
- Exit Type詳細分類（TP/SL/Reversal/Manual）
- Entry Reason記録（Signal/Reentry）
- Hold期間中のReason記録
```

**問題点**:
- ❌ Doc00の「"close"の明示処理」が欠落
- ❌ "理由付与"が主軸になり、本来のP1バグ（close処理の不足）が未解決のまま

**"close"の明示処理とは**:
- 現在の実装では`long_close`/`short_close`はあるが、これが「意図的なclose」か「反転による決済」か不明
- P1-1の本質は「closeアクションを明示的に記録すること」

**対応方針**: ⚠️ **P1-1仕様の修正必要**

**修正内容**:
1. P1-1の主目的を「"close"の明示処理」に戻す
2. 理由付与は**副次的な拡張**として位置づける
3. close明示処理の実装方針を明確化:
   - `long_close`/`short_close`に`close_reason`フィールド追加
   - `close_reason`: `"tp"` (利確), `"sl"` (損切), `"reversal"` (反転決済), `"manual"` (手動)

**優先度**: High（Phase 2実装前に修正必要）

---

### 2.3 [Major] M1: TradeType基準不一致

**指摘内容**:
> 既存TradeTypeの前提がDoc04と一致していません。Doc12は`long_entry_win/loss`等を既存分類として扱っていますが、Doc04は`long_open/long_close/long_add/...`の設計です。

**検証結果**: ❌ **指摘は正しい（仕様修正必要）**

**実装の確認**:
```python
# ztb/evaluation/walk_forward/reporter.py:29
# 実際の実装
Trade Type: "long_open", "long_close", "long_add", "long_reduce",
            "short_open", "short_close", "short_add", "short_reduce",
            "reverse", "hold"
```

**Doc12の誤記**:
```python
# Doc12で記載していた内容（誤り）
TradeType: Literal[
    "long_entry_win", "long_entry_loss",
    "short_entry_win", "short_entry_loss",
    ...
]
```

**問題点**:
- ❌ Doc12が実装と異なる分類体系を記載
- ❌ Phase 0.2aで実装済みの分類（Doc04準拠）を無視
- ❌ "win/loss"ベースの分類は実装されていない

**対応方針**: ⚠️ **Doc12の記述修正**

**修正内容**:
1. Doc12のTradeType分類をDoc04準拠（実装準拠）に修正
2. P1-1で追加するのは既存分類への「理由フィールド追加」
3. "win/loss"分類は導入しない（統計計算で対応）

**優先度**: High（Phase 2実装前に修正必要）

---

### 2.4 [Major] M2: reason生成層不明

**指摘内容**:
> `exit_reason/entry_reason/hold_reason`は追加されるものの、どの層（env/evaluator）で生成するかが記載されておらず、**実質的に全てNoneのまま**になる懸念があります。

**検証結果**: ⚠️ **指摘は正しい（設計不足）**

**問題点**:
- ❌ reasonフィールドの生成経路が不明
- ❌ envとevaluatorのどちらで生成するか未定義
- ❌ 生成ロジックが未設計

**対応方針**: ⚠️ **生成経路の設計追加**

**設計方針**:
1. **exit_reason**: envで生成
   - TP判定: `net_pnl > 0`
   - SL判定: `net_pnl < 0`
   - Reversal判定: 反転検出時
   - Manual: 上記以外のclose

2. **entry_reason**: envで生成は困難 → **Phase 2では見送り**
   - Signal判定にはRL信号の履歴が必要（envは持たない）
   - Reentry判定にも履歴が必要
   - → Phase 3でevaluator層での記録を検討

3. **hold_reason**: envで生成は困難 → **Phase 2では見送り**
   - Waiting/Avoiding判定にはゲート情報が必要
   - → Phase 3でゲートログとの統合を検討

**Phase 2での実装**:
- ✅ `exit_reason`のみ実装（env層で生成）
- ⚠️ `entry_reason`/`hold_reason`はPhase 3に延期

**優先度**: Medium（Phase 2実装時に対応）

---

### 2.5 [Major] M3: AB統計仕様不整合

**指摘内容**:
> AB Testingの統計仕様がDoc00と不整合です。Doc12は2-seed成功・two-sided検定・多重比較補正なしで進めていますが、Doc00は4seed×4split・Holm-Bonferroni・効果量を前提としています。

**検証結果**: ⚠️ **指摘は正しい（意図的な簡略化）**

**Doc00の仕様**:
```markdown
| **サンプル数** | 各条件n ≥ 16（4seed × 4split） |
| **多重比較補正** | Holm-Bonferroni法（3比較） |
| **効果量** | Cliff's Delta（|d| > 0.33で中程度） |
```

**Doc12の計画**:
```markdown
- 2 seed比較成功
- Mann-Whitney U検定（two-sided）
- 多重比較補正なし
```

**差異の理由**:
- Phase 2の目的はAB Testing**基盤構築**
- 完全な統計検定はPhase 3で実施予定
- Phase 2では「複数seed結果の統合・比較機能」の実装に集中

**対応方針**: ✅ **Phase 2は簡易版と明記**

**修正内容**:
1. Doc12に「Phase 2は簡易版」と明記
2. Phase 3で本格的な統計検定実装を計画
3. Phase 2の完了条件を「2 seed比較動作確認」に限定

**理由**:
- Phase 2の工数は4日（統計検定実装含めると7日超）
- AB Testing基盤が整えば、Phase 3での拡張は容易
- 段階的実装が現実的

**優先度**: Low（意図的な設計判断、明文化のみ）

---

### 2.6 [Major] M4: AB条件比較不足

**指摘内容**:
> AB Testingは単一条件のseed統合に留まり、**条件A/Bの定義・保存・比較導線**が不足しています。Entry Gate ON/OFFなどの比較が設計上成立していません。

**検証結果**: ⚠️ **指摘は正しい（設計不足）**

**問題点**:
- ❌ 条件A（例: Entry Gate ON）と条件B（例: Entry Gate OFF）の定義方法が不明
- ❌ 各条件の結果を別ディレクトリに保存する設計がない
- ❌ 2条件の比較機能（`compare_two_conditions()`）があるが、使用方法が不明

**対応方針**: ⚠️ **条件定義・保存設計の追加**

**設計方針**:
1. **条件の定義方法**:
   - Config YAMLで条件を定義
   - 各条件に名前を付与（例: "gate_on", "gate_off"）

2. **結果の保存構造**:
```
results/
├── condition_a_gate_on/
│   ├── seed_0/
│   │   ├── val_seed0.csv
│   │   └── test_seed0.csv
│   ├── seed_1/
│   └── ...
└── condition_b_gate_off/
    ├── seed_0/
    └── ...
```

3. **比較スクリプト**:
```python
# scripts/compare_conditions.py（新規）
comparator_a = ABTestingComparator("results/condition_a_gate_on")
comparator_b = ABTestingComparator("results/condition_b_gate_off")

comparison = comparator_a.compare_with_condition(
    other=comparator_b,
    metric="net_roi",
    alpha=0.05
)
```

**Phase 2での実装**:
- ✅ 条件定義・保存構造の設計
- ✅ 比較スクリプトの実装
- ⚠️ 自動化はPhase 3（Phase 2は手動実行）

**優先度**: Medium（Phase 2実装時に対応）

---

### 2.7 [Major] M5: Reporter統合仕様不足

**指摘内容**:
> Reporter統合はTrainingReporter削除が前提ですが、Training側が必要とするメトリクス/APIをBacktestReporterにどう移植するかの仕様が不足しています（破壊的変更リスク）。

**検証結果**: ⚠️ **指摘は正しい（詳細設計不足）**

**問題点**:
- ❌ TrainingReporterが提供するAPIの洗い出しが不足
- ❌ BacktestReporterへの移植計画が不明確
- ❌ 互換性テストの計画がない

**対応方針**: ⚠️ **詳細設計の追加**

**設計方針**:
1. **TrainingReporter APIの洗い出し**:
   - 使用箇所をgrep検索で特定
   - 必要なメソッド・プロパティをリスト化

2. **BacktestReporterへの移植**:
   - `training_mode`フラグで動作切り替え
   - Training専用メソッドの追加（エピソード統計等）

3. **互換性テスト**:
   - 既存Training Scriptでの動作確認
   - メトリクス値の一致確認

**Phase 2での実装**:
- ✅ API洗い出し（実装前）
- ✅ 移植計画の詳細化
- ✅ 互換性テストの実施

**優先度**: High（Phase 2実装前に設計完了）

---

### 2.8 [Major] M6: MTF/Scaler後ろ倒し

**指摘内容**:
> MTF因果性検証とScaler境界の厳密化がPhase 3に後ろ倒しされています。前回の積み残しがPhase 2計画に反映されておらず、評価基盤の前提が弱いままです。

**検証結果**: ⚠️ **指摘は正しい（明文化不足）**

**事実**:
- MTF因果性検証はPhase 0で仕様策定のみ（実装未完了）
- Scaler境界の厳密化も同様
- Phase 2では対応しない方針だが、明文化が不足

**対応方針**: ✅ **Phase 3延期の明文化**

**修正内容**:
1. Doc12に「MTF/Scaler厳密化はPhase 3対応」と明記
2. Phase 2で対応しない理由を記載:
   - Phase 2の焦点はP1バグ修正（Trade Type、Entry Price、Reporter、AB Testing）
   - MTF/ScalerはP2バグ（Phase 3対応）
   - Phase 2工数を4日に抑えるための判断

3. Phase 3での対応を明記:
   - MTF因果性検証実装
   - Scaler fit境界の警告→エラー化

**優先度**: Low（明文化のみ）

---

### 2.9 [Minor] m1: テスト数不整合

**指摘内容**:
> テスト総数の整合が取れていません（Phase 1=103/103 vs 196/196、または123/123）。報告の信頼性が下がります。

**検証結果**: ⚠️ **指摘は正しい（表記揺れ）**

**事実**:
- Phase 0統合: 9件
- Phase 1単体: 94件
- Phase 1合計: 103件（9+94）

**Doc12の表記揺れ**:
- 箇所A: 103/103（正しい）
- 箇所B: 196/196（Phase 2含む予測値、誤解を招く）
- 箇所C: 123/123（計算ミス）

**対応方針**: ✅ **表記統一**

**修正内容**:
- Phase 1完了時点: 103/103
- Phase 2完了予定: 196/196（103 + Phase 2の93件）
- 各箇所で文脈を明確化

**優先度**: Low（表記の問題のみ）

---

### 2.10 [Minor] m2: 基本8種vs12種

**指摘内容**:
> 「基本8種」と書きながら列挙は12種です。表現の誤差が仕様理解を混乱させます。

**検証結果**: ⚠️ **指摘は正しい（表現ミス）**

**事実**:
- 基本8種: long/short × open/close/add/reduce = 8種
- +2種: reverse, hold
- 合計10種（Doc12では12種と誤記）

**対応方針**: ✅ **表現修正**

**修正内容**:
- "基本8種" → "基本8種 + reverse/hold = 10種"
- 列挙内容も正確に記載

**優先度**: Low（表現の問題のみ）

---

## 3. 対応計画

### 3.1 即時対応（Phase 2実装前）

| 項目 | 作業 | 工数 | 担当 |
|------|------|------|------|
| C1 | Phase 1完了根拠の明確化 | 0.1日 | Doc |
| **C2** | **P1-1仕様の修正（close明示に戻す）** | **0.3日** | **仕様** |
| **M1** | **TradeType基準の統一（実装準拠）** | **0.2日** | **仕様** |
| M5 | Reporter統合詳細設計 | 0.2日 | 設計 |
| m1/m2 | 表記揺れの修正 | 0.1日 | Doc |

**小計**: 0.9日（1日以内）

### 3.2 Phase 2実装時に対応

| 項目 | 作業 | 工数 | タイミング |
|------|------|------|-----------|
| M2 | reason生成層の実装（exit_reasonのみ） | 0.3日 | P1-1実装時 |
| M4 | AB条件比較設計の実装 | 0.5日 | P1-4実装時 |

**小計**: 0.8日（Phase 2工数に含まれる）

### 3.3 Phase 3で対応（Phase 2で明文化）

| 項目 | 作業 | 工数 | Phase |
|------|------|------|-------|
| M3 | AB統計検定の本格実装 | 2日 | Phase 3 |
| M6 | MTF/Scaler厳密化 | 2日 | Phase 3 |

**小計**: 4日（Phase 3工数）

### 3.4 修正版Phase 2工数見積もり

| フェーズ | 元見積もり | 事前対応 | 実装時対応 | 合計 |
|----------|------------|----------|-----------|------|
| 事前準備 | 0日 | +0.9日 | - | 0.9日 |
| Phase 2実装 | 4日 | - | +0.8日 | 4.8日 |
| **合計** | **4日** | - | - | **5.7日** |

**推奨**: Phase 2全体で**6日確保**（バッファ含む）

---

## 4. Doc12修正方針

### 4.1 Critical修正（即時）

**Section 2.2.1: P1-1定義の修正**

**Before（誤り）**:
```markdown
#### P1-1: Trade Type Classification（詳細分類拡張）

**Phase 2での拡張**: Exit Type詳細分類（TP/SL/Reversal/Manual）
```

**After（修正）**:
```markdown
#### P1-1: Trade Type Classification（close明示処理）

**Doc00定義**: "close"の明示処理

**Phase 0.2aの実装**:
- 基本8種分類実装済み（long/short × open/close/add/reduce）
- reverse, hold実装済み
- 合計10種のTrade Type分類

**Phase 2での対応**:
1. **主目的**: closeアクションの明示的記録
   - `long_close`/`short_close`に終了理由を記録
   - `close_reason`: "tp" (利確), "sl" (損切), "reversal" (反転決済), "manual" (手動)

2. **副次的拡張**（Phase 3検討）:
   - entry_reason: Signal/Reentry判定（履歴情報必要）
   - hold_reason: Waiting/Avoiding判定（ゲート情報必要）
```

**Section 3.1: 既存実装分析の修正**

**Before（誤り）**:
```python
# Phase 0.2aで実装済み（誤り）
TradeType: Literal[
    "long_entry_win", "long_entry_loss",
    ...
]
```

**After（修正）**:
```python
# Phase 0.2aで実装済み（Doc04準拠）
TradeType: Literal[
    "long_open", "long_close", "long_add", "long_reduce",
    "short_open", "short_close", "short_add", "short_reduce",
    "reverse", "hold"
]
# 合計10種（基本8種 + reverse + hold）
```

### 4.2 Major修正（実装前）

**Section 4.2.2: reason生成層の明確化**

**追加内容**:
```markdown
#### タスク2: P1-1 Trade Type拡張（0.5日）

**実装方針**:

1. **close_reason生成**（env層で実装）:
```python
# fast_intraday_env_v456.py
if is_closing_position:
    if net_pnl > 0:
        close_reason = "tp"  # Take Profit
    elif net_pnl < 0:
        close_reason = "sl"  # Stop Loss
    elif is_reversal:
        close_reason = "reversal"
    else:
        close_reason = "manual"
    
    info["close_reason"] = close_reason
```

2. **entry_reason/hold_reason**:
   - Phase 2では実装見送り（履歴情報が必要）
   - Phase 3でevaluator層での記録を検討
```

**Section 4.2.4: AB条件比較の設計追加**

**追加内容**:
```markdown
#### タスク4: P1-4 AB Testing実装（1.0日）

**実装内容**:

3. **条件定義・保存構造**:
```yaml
# config/v459/experiments/ab_test_gate.yaml
conditions:
  - name: "gate_on"
    entry_gate:
      enabled: true
  - name: "gate_off"
    entry_gate:
      enabled: false

ab_testing:
  seeds: [0, 1, 2, 3]
  output_base: "results/ab_test_gate"
```

4. **結果保存構造**:
```
results/ab_test_gate/
├── gate_on/
│   ├── seed_0/
│   └── ...
└── gate_off/
    ├── seed_0/
    └── ...
```

5. **比較スクリプト**:
```python
# scripts/v459/compare_ab_conditions.py（新規）
def compare_conditions(
    condition_a_dir: Path,
    condition_b_dir: Path,
    metric: str = "net_roi"
) -> Dict[str, Any]:
    # 2条件の統計的比較
    ...
```
```

### 4.3 Documentation修正（明文化）

**Section 6: リスク評価**

**追加内容**:
```markdown
### 6.4 Phase 3延期項目の明確化

**Phase 2では対応しない項目**:
1. MTF因果性検証（P2バグ → Phase 3対応）
2. Scaler fit境界の厳密化（警告→エラー化 → Phase 3対応）
3. AB Testing本格統計検定（4seed×4split、多重比較補正 → Phase 3対応）
4. entry_reason/hold_reason実装（履歴情報必要 → Phase 3対応）

**延期理由**:
- Phase 2の焦点: P1バグ修正（Trade Type、Entry Price、Reporter、AB Testing基盤）
- Phase 2工数: 4-6日に抑制
- Phase 3で拡張が容易な設計を採用
```

**Section 1.2: Phase 2完了条件の修正**

**Before**:
```markdown
- [ ] AB Testing動作確認（2 seed比較成功）
```

**After**:
```markdown
- [ ] AB Testing基盤構築完了（2条件 × 2 seed比較成功）
- 注: 本格的な統計検定（4seed×4split、多重比較補正）はPhase 3で実装
```

---

## 5. まとめ

### 5.1 対応サマリー

| カテゴリ | 件数 | 即時対応 | 実装時対応 | Phase 3対応 |
|----------|------|----------|-----------|-------------|
| Critical | 2 | 2 | 0 | 0 |
| Major | 6 | 2 | 2 | 2 |
| Minor | 2 | 2 | 0 | 0 |
| **合計** | **10** | **6** | **2** | **2** |

### 5.2 修正後のPhase 2計画

**Phase 2スコープ（修正版）**:
- P1-1: **close明示処理**（close_reason記録）
- P1-2: Entry Price更新
- P1-3: Reporter統合
- P1-4: AB Testing基盤（2条件 × 2 seed比較）

**Phase 2完了条件（修正版）**:
- ✅ P1バグ全修正（4/4件）
- ✅ close_reason記録動作（env→reporter）
- ✅ Entry Price反転時更新
- ✅ Reporter統一（3→1実装）
- ✅ AB Testing基盤動作（2条件 × 2 seed比較）
- ✅ 全テスト合格（Phase 0/1/2統合）

**Phase 3延期項目（明文化）**:
- MTF因果性検証
- Scaler境界厳密化
- AB統計検定本格実装
- entry_reason/hold_reason実装

### 5.3 工数見積もり（修正版）

| 作業 | 工数 |
|------|------|
| Doc12修正（事前対応） | 0.9日 |
| Phase 2実装 | 4.8日 |
| **合計** | **5.7日** |

**推奨確保日数**: 6日（バッファ含む）

---

## 6. 次のステップ

1. ✅ Doc15（本文書）をユーザーに提示
2. ⚠️ ユーザー承認後、Doc12修正実施
3. ⚠️ 修正版Doc12でセルフレビュー再実施
4. ✅ Phase 2実装開始

**Doc15完成、ユーザーレビュー待ち** ✅

---

**End of Response Plan**
