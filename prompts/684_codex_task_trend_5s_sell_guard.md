# Codex Task: mid_price_trend_5s Sell Veto Layer (684# Phase M1)

## 目的
sell 側の最大損失要因である「上方短期トレンド中の sell」を、既存レイヤー（velocity_skip, toxic_sell_veto）とは独立した新 guard で防御する。

## 背景

### 4/1 データ検証結果 (684#)
| 条件 | n | avg PnL30 (bps) |
|------|---|:---:|
| trend_5s > 0 sell | 14 | **-3.25** |
| trend_5s ≤ 0 sell | 25 | -0.80 |
| trend_5s > 0 + obi > 0.1 sell | 6 | **-6.01** |
| trend_5s > 0 + fast fill sell | 4 | **-5.16** |

### 既存防御層との差分
- **velocity_skip**: `price_velocity_bps`（60s EMA）ベース。閾値 4.0 bps。5s の急変を捉えない
- **toxic_sell_veto**: spread + OBI + VPIN の複合条件。trend_5s は未使用
- **提案**: 5s の短期方向性を直接参照する独立レイヤー

### 重要な注意（684# 盲点 ③ より）
`price_velocity_bps` ≥ 2.0 の sell は PnL=-0.16 で**良好**。velocity が高い sell は「勢いに乗った約定」で mean reversion 利益あり。したがって velocity_skip の閾値引下げではなく、**trend_5s という別指標での防御が正しいアプローチ**。

## タスク

### Task 1: trend_5s Sell Guard の実装

**主要対象ファイル**: `scripts/v460/lib/fill_test_executor.py` または offset pipeline の該当箇所

1. 新 guard の位置を確認:
   - offset pipeline 内の既存 guard（velocity_skip, toxic_sell_veto）の実装箇所を特定
   - 同一ファイル内に `_apply_trend_5s_sell_guard()` メソッドを追加
   - 評価順序: toxic_sell_veto の**後**、最終 offset 確定の**前**

2. ロジック:
   ```python
   def _apply_trend_5s_sell_guard(
       self, side: str, trend_5s: float, current_offset: float
   ) -> tuple[float, str]:
       """trend_5s ベースの sell 防御。
       
       Returns:
           (adjusted_offset, action) where action is "boost"/"veto"/"none"
       """
       if side != "sell":
           return current_offset, "none"
       
       cfg = self.config.trend_5s_sell_guard
       if not cfg.enabled:
           return current_offset, "none"
       
       if trend_5s > cfg.hard_veto_threshold_bps:
           return current_offset, "veto"  # caller が skip 処理
       
       if trend_5s > cfg.threshold_bps:
           boosted = current_offset * cfg.offset_boost_factor
           return boosted, "boost"
       
       return current_offset, "none"
   ```

3. 設定クラス追加 (`fill_config.py` 内):
   ```python
   @dataclass
   class Trend5sSellGuardConfig:
       enabled: bool = True
       threshold_bps: float = 0.5
       hard_veto_threshold_bps: float = 3.0
       offset_boost_factor: float = 1.5
   ```

### Task 2: YAML 配線

**対象ファイル**: `configs/v460/fill_test.yaml`

```yaml
# trend_5s_sell_guard セクションを追加
trend_5s_sell_guard:
  enabled: true
  threshold_bps: 0.5        # soft guard 発動閾値 (bps)
  hard_veto_threshold_bps: 3.0  # hard veto 閾値 (bps)
  offset_boost_factor: 1.5  # soft mode 時の offset 乗数
```

ConfigReloader が hot-reload できることを確認:
- `fill_config.py` の YAML → dataclass マッピングに `trend_5s_sell_guard` を追加
- hot-reload テスト（YAML 変更→即時反映）

### Task 3: FillRecord への記録

**対象ファイル**: fill_records を構築するコード

FillRecord の出力に以下を追加:
- `trend_5s_guard_triggered: bool` — guard が反応したか（boost or veto）
- `trend_5s_guard_action: str` — "boost" / "veto" / "none"
- `trend_5s_at_order: float` — 判定に使った trend_5s 値

これにより post-hoc 分析で guard の効果を評価可能にする。

### Task 4: テスト

**新規作成**: `tests/unit/v460/test_trend_5s_sell_guard.py`

```python
class TestTrend5sSellGuard:
    def test_sell_soft_boost(self):
        """sell + trend_5s=1.0 (> threshold 0.5) → offset が boost される"""
    
    def test_sell_hard_veto(self):
        """sell + trend_5s=4.0 (> hard_veto 3.0) → veto 返却"""
    
    def test_sell_below_threshold(self):
        """sell + trend_5s=0.3 (< threshold 0.5) → 変更なし"""
    
    def test_buy_not_affected(self):
        """buy + trend_5s=5.0 → 変更なし（sell のみ対象）"""
    
    def test_disabled(self):
        """enabled=false → 任意の trend_5s でも変更なし"""
    
    def test_negative_trend_5s(self):
        """sell + trend_5s=-2.0 (下落中) → 変更なし（sell に有利）"""
    
    def test_boost_factor_applied(self):
        """offset=0.4, boost_factor=1.5 → 0.6 に変化"""
    
    def test_fill_record_has_guard_fields(self):
        """FillRecord に trend_5s_guard_triggered/action/value が含まれる"""

class TestTrend5sConfig:
    def test_yaml_loading(self):
        """fill_test.yaml の trend_5s_sell_guard セクションが正しくロードされる"""
    
    def test_default_values(self):
        """Trend5sSellGuardConfig のデフォルト値が仕様通り"""
```

**既存テスト実行**:
```bash
python -m pytest tests/ -x --tb=short
```

## 制約
- `git commit --no-verify -m "684# trend_5s sell guard layer"` でコミット
- `git add .` 禁止。対象ファイルを個別指定
- Any 型禁止、mypy 準拠
- 既存 velocity_skip / toxic_sell_veto を変更しない（独立レイヤーとして追加）
- hot-reload 対応必須（ConfigReloader 経由）
- buy 側に一切影響を与えないこと

## 成果物
1. `_apply_trend_5s_sell_guard()` 実装
2. `Trend5sSellGuardConfig` dataclass
3. `fill_test.yaml` に設定セクション追加
4. FillRecord に guard 記録フィールド追加
5. テストファイル `test_trend_5s_sell_guard.py` (全 pass)
6. 全既存テスト pass 確認
