# Codex Task: SkipGate bypass モード実装 (686# Phase SG-1)

## 目的
予測力ゼロと実証された SkipGate を、fill ブロックを停止しつつスコア記録は継続する
bypass モードに移行する。4日間で 48 件/日（NFQ の 14%）の不要ブロックを解放し、
即時の収益機会を回復する。

## 背景

### 672# 情報理論分析 + 686# 追加検証

| 検証 | 結果 |
|------|------|
| SkipGate MI (672#) | ≈ 0（AS と相互情報量ゼロ） |
| SG score×AS 四分位 (686#) | Q1=29%, Q2=26%, Q3=25%, Q4=25%（フラット） |
| 全 pre-fill 特徴量の AS 判別力 (686#) | 最大 \|r\|=0.155 (spread_offset_ratio)、実質ゼロ |
| skip_gate NFQ 件数 (4/1) | 48 件（全 NFQ 344 件の 14%） |

**結論**: SkipGate は AS を予測できず、fill を無駄にブロックしている。
ただし完全削除はリスクがあるため、**bypass モード** でスコア記録を維持しつつ
ブロックを停止する。将来の特徴量改善時に再有効化可能な設計とする。

### 既存実装の確認ポイント
- SkipGate の実装: `scripts/v460/lib/` 配下（skip_gate 関連ファイル）
- SkipGate 評価: `skip_gate_evaluator.py`
- FillRecord への記録: `skip_gate_fill_record.py`, `skip_gate_result_fields.py`
- YAML 設定: `configs/v460/fill_test.yaml` の `skip_gate` セクション

## タスク

### Task 1: bypass モードの追加

**設計方針**: 既存の SkipGate 評価ロジックは**一切変更しない**。判定後のブロック行動のみを制御する。

1. YAML 設定に `bypass_mode` を追加:
   ```yaml
   skip_gate:
     enabled: true        # 既存（スコア計算の有無）
     bypass_mode: true    # 新規: true=スコア計算するがブロックしない
     # ... 既存の閾値等はそのまま
   ```

2. SkipGate 設定クラスに `bypass_mode: bool = False` を追加

3. fill 判定ロジック（skip_gate でブロックする箇所）を修正:
   ```python
   # 変更前
   if sg_result.should_skip:
       return NFQ(reason="skip_gate", ...)
   
   # 変更後
   if sg_result.should_skip:
       if self.config.skip_gate.bypass_mode:
           # スコアは記録するがブロックしない
           logger.info("SkipGate bypass: score=%.2f (would_skip=True)", sg_result.score)
           # fill_record に bypass フラグを記録
       else:
           return NFQ(reason="skip_gate", ...)
   ```

4. FillRecord に `skip_gate_bypassed: bool` フィールドを追加
   - bypass_mode=True かつ should_skip=True の場合に True を記録
   - これにより、後から「bypass しなければブロックされていた fill」の PnL を分析可能

### Task 2: テスト

1. **ユニットテスト** (`tests/unit/v460/` 配下):
   - `bypass_mode=False`: 従来どおり skip_gate ブロックが発動
   - `bypass_mode=True`: should_skip=True でもブロックされず、`skip_gate_bypassed=True` が記録
   - `bypass_mode=True`: should_skip=False では `skip_gate_bypassed=False`
   - 既存の SkipGate テストが全て pass することを確認

2. **YAML 設定テスト**:
   - `bypass_mode` が YAML → config に正しくパースされることを確認
   - デフォルト値 (`False`) のテスト

### Task 3: YAML 適用

**対象ファイル**: `configs/v460/fill_test.yaml`

```yaml
skip_gate:
  bypass_mode: true  # 686# SG-1: MI≈0, 予測力ゼロ実証。ブロック停止、スコア記録継続
```

## 受け入れ基準

- [ ] `bypass_mode: true` で SkipGate スコアが計算・記録されるがブロックしない
- [ ] FillRecord に `skip_gate_bypassed` フラグが記録される
- [ ] `bypass_mode: false` (デフォルト) で既存動作が完全に維持される
- [ ] 全既存テストが pass
- [ ] 新規ユニットテスト 3 件以上追加

## リスク評価

- **低リスク**: bypass_mode は既存ロジックを変更せず、ブランチ分岐を 1 箇所追加するのみ
- **ロールバック**: `bypass_mode: false` に戻すだけで即座に元の動作に復帰
- **検証**: bypass された fill の PnL を `skip_gate_bypassed=True` で追跡可能
