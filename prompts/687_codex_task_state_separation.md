# Codex Task: 638# State 分離 — last_executed_side vs last_attempted_side (687#)

## 目的
`SideSelector` の状態管理で、**実約定**と**試行のみ（NFQ/preflight 失敗）** を混同している問題を修正する。
これにより `balance_freeze_cycles` の誤発動を防ぎ、資本効率を改善する。

## 背景

### 638# で特定された問題（P0: "true root cure"）
- 現状: `last_executed_side` は fill 成功・NFQ ともに更新される
- 問題: `preflight_insufficient` (資本不足でブロック) 後でも side が更新され、`balance_freeze` が不要に発動
- 結果: 次サイクルで「前回と同じ側」を避けようとして、本来可能な取引を skip
- 影響: 4/1 データで preflight_insufficient=35%（NFQ 最大要因）。side 誤更新が freeze 誤発動を誘発

### 638# 提案の三要素
1. `last_executed_side`（実約定のみ更新）と `last_attempted_side`（試行で更新）を分離
2. `preflight_insufficient` 失敗時は `last_executed_side` を更新しない
3. `balance_freeze_cycles` は `last_executed_side` のみを参照

## タスク

### Task 1: SideSelector の状態分離

**主要対象**: `scripts/v460/lib/` 配下の SideSelector 実装（`side_selector.py` あるいは `fill_cycle_executor.py` 内）

1. 現在の `last_executed_side` フィールド特定:
   - grep で `last_executed_side` または `_last_side` の定義箇所を特定
   - `balance_freeze` ロジックとの結合箇所を確認

2. 状態フィールドの分離:
   ```python
   # Before: 単一状態
   self._last_side: str | None = None  # fill/NFQ 両方で更新
   
   # After: 分離
   self._last_executed_side: str | None = None  # fill 成功時のみ更新
   self._last_attempted_side: str | None = None  # 全試行で更新（従来動作）
   ```

3. 更新箇所の修正:
   - **fill 成功パス**: `_last_executed_side` と `_last_attempted_side` 両方を更新
   - **NFQ / preflight 失敗パス**: `_last_attempted_side` のみ更新
   - **balance_freeze**: `_last_executed_side` を参照するように変更

4. ログ改善:
   - side 更新時に `executed=buy attempted=sell` のようなログ出力
   - freeze 判定時に参照した side を明記

### Task 2: FillRecord への記録

**対象**: `fill_record_builder.py`, `fill_quality.py`

1. `last_executed_side: str | None` フィールドを FillRecord に追加
2. `last_attempted_side: str | None` フィールドを FillRecord に追加
3. 事後分析で freeze 誤発動を検出できるようにする

### Task 3: テスト

**対象**: `tests/unit/v460/` 配下

1. fill 成功後: `last_executed_side` と `last_attempted_side` が両方更新
2. NFQ 後: `last_attempted_side` のみ更新、`last_executed_side` 不変
3. preflight_insufficient 後: `last_attempted_side` のみ更新
4. freeze 判定: `last_executed_side` を参照（`last_attempted_side` に影響されない）
5. 初回（None 状態）: 両方 None で freeze は発動しない

### Task 4: 既存テスト互換

1. `last_executed_side` の名称変更に追随するテスト修正
2. `python -m pytest tests/ -x --tb=short` で全テスト pass 確認

## 受け入れ基準

- [ ] `last_executed_side` は fill 成功時のみ更新される
- [ ] `last_attempted_side` は全試行で更新される
- [ ] `balance_freeze` は `last_executed_side` のみ参照
- [ ] FillRecord に両フィールドが記録される
- [ ] 新規テスト 5 件以上、全テスト pass
- [ ] 既存の side_selector / freeze 関連テストが全て pass

## リスク評価

- **中リスク**: SideSelector は取引サイクルの中心。分岐変更は慎重に
- **ロールバック**: `last_attempted_side` を `balance_freeze` 参照に戻すだけで旧動作に復帰
- **検証**: 新 FillRecord フィールドで freeze 誤発動率を追跡可能
