# 151# P3-03 設計: dynamic_position_sizer (AS 確率連動ロットサイジング)

**日時**: 2026-02-23  
**種別**: design (設計)  
**Phase**: P3 (効率改善)  
**前提**: 149# §2 P3-03, 143# R-1b (regime_lot), 084# P(AS) 可観測性

---

## §1 背景と目的

### 1.1 問題

149# §8.2 の Oracle レポートで以下が判明:

| セグメント | n | 実績 PnL30s (bps) | Oracle PnL30s (bps) |
|-----------|---|-------------------|---------------------|
| all | 1182 | **-0.24** | +2.64 |
| buy | 591 | -0.00 | +2.82 |
| sell | 591 | **-0.47** | +2.44 |

Oracle 上限との乖離が大きく、特に sell 側で逆選別 (AS) による損失が顕著。

**現状のロット制御**:
- `_current_lot` は BalanceChecker 管理の固定値 (0.001-0.005 BTC)
- `_regime_adjusted_lot()` で regime 別倍率を乗算 (143# R-1b)
- `_try_auto_lot_size()` で N サイクルごとの fill_rate/AS_ratio/PnL ベースの段階的増減 (033# 方策 B, 現在 **disabled**)
- **SkipGate の AS 確率がサイクル単位で算出されているのにロットに反映されていない**

つまり「AS 確率が高い = 逆選別リスクが高い注文」でもフルロットで出しているため、1 件あたりの損失が不要に大きい。

### 1.2 目的

SkipGate が算出した **P(AS) をサイクル単位のロット決定に連動** させ、AS リスクに応じてエクスポージャーを動的に縮小する。

### 1.3 期待効果

AS 確率が高い (閾値ギリギリで PASS した) 注文のロットを縮小し:
- **sell 側 PnL の改善**: ロット縮小により 1 件あたり AS 損失を低減
- **Oracle gap 縮小**: フルロット × 低 AS 注文に集中 → 効率的な資本配分
- **リスク調整後リターン (Sharpe) 向上**

### 1.4 プロジェクト大義との整合

本プロジェクトの大義は**短期間での高収益性システム**。P(AS) は既に算出済みのデータであり、追加の推論コスト・API コストゼロでロットを最適化できるため、**最小工数で最大の P&L 改善が見込める施策**。

---

## §2 設計選択肢

### 2.1 案 A: 線形スケーリング (推奨)

AS 確率に対して線形にロットを縮小する。

```
lot_factor = max(floor, 1.0 - scale * as_prob)
effective_lot = regime_adjusted_lot × lot_factor
```

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `scale` | `1.0` | AS 確率 → ロット縮小の傾斜 |
| `floor` | `0.3` | 最小ロット倍率 (これ以下には縮小しない) |

例 (scale=1.0, floor=0.3):

| P(AS) | lot_factor | ロット (base=0.003) |
|-------|-----------|---------------------|
| 0.10 | 0.90 | 0.0027 |
| 0.40 | 0.60 | 0.0018 |
| 0.60 | 0.40 | 0.0012 |
| 0.80 | 0.30 (floor) | 0.0009 |

**メリット**: 直感的、パラメータ少、解釈容易  
**デメリット**: 非線形のリスク構造を近似しきれない場合あり

### 2.2 案 B: 段階的バケット

AS 確率をバケットに分割し、各バケットに固定倍率を割り当て。

```yaml
confidence_lot_buckets:
  - {as_prob_max: 0.30, factor: 1.00}  # 高信頼: フルロット
  - {as_prob_max: 0.50, factor: 0.70}  # 中信頼
  - {as_prob_max: 0.70, factor: 0.40}  # 低信頼
  - {as_prob_max: 1.00, factor: 0.20}  # 超低信頼
```

**メリット**: 明示的な閾値制御、bucket 別の分析が容易  
**デメリット**: パラメータ多、bucket 境界の連続性が不自然

### 2.3 案 C: Kelly 基準応用

期待値 × 確率に基づく最適ベット比率。

```python
# 簡易 Kelly: lot_factor = (E[win] × P(win) - E[loss] × P(loss)) / E[win]
# ただし AS が二値分類のため P(win)=1-as_prob, P(loss)=as_prob
```

**メリット**: 理論的に最適  
**デメリット**: E[win]/E[loss] の推定が必要、fill_test のコスト構造 (スプレッド、AS severity) の正確なモデル化が前提

### 2.4 選定

| 観点 | 案 A | 案 B | 案 C |
|------|------|------|------|
| 実装コスト | ◎ 低 | ○ 中 | △ 高 |
| 解釈容易性 | ◎ | ○ | △ |
| 最適性 | ○ | ○ | ◎ |
| 既存整合 | ◎ | ○ | △ |
| パラメータ少 | ◎ | △ | △ |

**結論: 案 A (線形スケーリング) を推奨**。

1. SkipGate の AS 確率は 0-1 連続値なので線形近似で十分実用的
2. `floor` で最小ロットを保証し、極端な縮小を防止
3. Phase C データで `scale`/`floor` を事後チューニング可能
4. 将来的に案 C の知見 (最適 scale の理論値導出) に進化させる余地あり

---

## §3 案 A 詳細設計

### 3.1 ロット算出フロー

```
base_lot (_current_lot)          [BalanceChecker 管理]
  ↓
regime_adjusted_lot              [143# R-1b: regime_lot_multiplier]
  ↓ 
confidence_adjusted_lot  ← NEW  [151# P3-03: AS 確率連動]
  ↓
final_lot = max(min_order_btc, confidence_adjusted_lot)
```

**合成ルール**: `regime_factor × confidence_factor` の二段階乗算。

注: `_try_auto_lot_size()` (方策 B) との関係:
- 方策 B は **N サイクル単位** で `_current_lot` 自体を上下させる長期適応
- P3-03 は **1 サイクル単位** で regime/confidence に基づく短期調整
- 互いに直交する (方策 B が base_lot を変え、P3-03 がサイクル単位でスケーリング)

### 3.2 FillTestConfig 拡張

```python
@dataclass
class FillTestConfig:
    # --- 151# P3-03: AS 確率連動ロットサイジング ---
    enable_confidence_lot: bool = False
    confidence_lot_scale: float = 1.0       # AS 確率 → lot 縮小の傾斜
    confidence_lot_floor: float = 0.3       # lot 倍率の下限
    confidence_lot_mode: str = "as"         # "as" (AS 確率) / "pnl" (予測 PnL bps)
    confidence_lot_pnl_zero: float = 0.0    # mode=pnl: PnL=0 の lot_factor
    confidence_lot_pnl_max: float = 5.0     # mode=pnl: lot_factor=1.0 になる PnL (bps)
```

### 3.3 YAML 設定

```yaml
# configs/v460/fill_test.yaml
confidence_lot:
  enabled: false                # 有効化
  scale: 1.0                   # AS prob → lot 縮小傾斜
  floor: 0.3                   # 最小 lot 倍率
  mode: "as"                   # "as" or "pnl"
  # mode=pnl 用 (将来)
  pnl_zero_factor: 0.5         # PnL=0 bps 時の lot 倍率
  pnl_max_bps: 5.0             # lot_factor=1.0 になる PnL (bps)
```

### 3.4 `_confidence_lot_factor()` メソッド

```python
def _confidence_lot_factor(
    self,
    as_prob: float | None,
    pred_pnl: float | None,
) -> float:
    """151# P3-03: AS 確率/予測 PnL に基づくロット倍率.

    Returns:
        1.0 (無効時 / 確率なし) or [floor, 1.0] の倍率.
    """
    if not self.config.enable_confidence_lot:
        return 1.0

    mode = self.config.confidence_lot_mode
    if mode == "as" and as_prob is not None:
        import math
        if not math.isfinite(as_prob):
            return 1.0
        scale = self.config.confidence_lot_scale
        floor = self.config.confidence_lot_floor
        raw = 1.0 - scale * as_prob
        return max(floor, min(1.0, raw))

    if mode == "pnl" and pred_pnl is not None:
        import math
        if not math.isfinite(pred_pnl):
            return 1.0
        pnl_max = self.config.confidence_lot_pnl_max
        if pnl_max <= 0:
            return 1.0
        floor = self.config.confidence_lot_floor
        # PnL が高い → 1.0, PnL が 0 以下 → floor
        raw = pred_pnl / pnl_max
        return max(floor, min(1.0, raw))

    return 1.0
```

### 3.5 ロット算出統合: `_effective_order_lot()` メソッド (新規)

```python
def _effective_order_lot(
    self,
    as_prob: float | None = None,
    pred_pnl: float | None = None,
) -> float:
    """151# 統合ロット算出: base × regime × confidence.

    Args:
        as_prob: SkipGate の AS 確率 (mode=as 用).
        pred_pnl: SkipGate の予測 PnL bps (mode=pnl 用).

    Returns:
        最終注文ロット (min_order_btc 以上保証).
    """
    regime_lot = self._regime_adjusted_lot()
    conf_factor = self._confidence_lot_factor(as_prob, pred_pnl)
    lot = regime_lot * conf_factor

    min_lot = self.config.min_order_btc
    lot = max(lot, min_lot)
    if self.config.max_lot > 0:
        lot = min(lot, self.config.max_lot)

    if conf_factor < 1.0:
        logger.debug(
            f"[confidence_lot] as_prob={as_prob}, factor={conf_factor:.2f} "
            f"→ lot={lot:.4f} (regime={regime_lot:.4f})"
        )
    return lot
```

### 3.6 `run_single_cycle()` への統合

現在の呼び出しフロー:

```python
# 現状 (L953-975)
sg = await self._evaluate_skip_gate(
    ..., order_lot=self._regime_adjusted_lot(),
)
...
_order_lot = self._regime_adjusted_lot()
```

変更後:

```python
# 1. SkipGate 判定 (regime_lot で skip/pass 判定)
sg = await self._evaluate_skip_gate(
    ..., order_lot=self._regime_adjusted_lot(),
)
skip_gate_as_prob = sg.as_prob
skip_gate_score = sg.score  # pred_pnl (mode=pnl 用)
...

# 2. 発注ロット決定: regime × confidence
_order_lot = self._effective_order_lot(
    as_prob=skip_gate_as_prob,
    pred_pnl=skip_gate_score,
)
```

**重要**: SkipGate 判定は `_regime_adjusted_lot()` (confidence 未適用) で行う。
confidence_lot はあくまで「PASS した注文のエクスポージャー調整」であり、
skip/pass の判定基準に lot を混入させない (因果逆転の回避)。

### 3.7 FillRecord 記録フィールド

```python
@dataclass
class FillRecord:
    # 151# P3-03: confidence lot 情報
    confidence_lot_factor: float | None = None  # 適用された倍率
```

### 3.8 side 別パラメータ (オプション, Phase 2)

sell 側の AS リスクが高い場合、side 別に `scale`/`floor` を分離可能:

```yaml
confidence_lot:
  enabled: true
  scale: 1.0
  floor: 0.3
  scale_sell: 1.2    # sell はより積極的に縮小
  floor_sell: 0.2    # sell の最小倍率を下げる
```

初期実装では side 共通とし、Phase C データの side 別分析結果で要否を判断する。

---

## §4 既存機構との整合

### 4.1 regime_lot_multiplier (143# R-1b)

| | regime_lot | confidence_lot |
|---|---|---|
| 粒度 | regime 単位 (3種) | サイクル単位 (連続値) |
| 適用対象 | 全サイクル | skip_gate PASS のサイクルのみ |
| base_lot 変更 | 一時的乗算 | 一時的乗算 |
| 合成 | `base × regime` | `base × regime × confidence` |

### 4.2 方策 B: _try_auto_lot_size (033#)

- 方策 B は N サイクルの **集計統計** に基づいて `_current_lot` = base_lot を変更
- P3-03 は **1 サイクル** の AS 確率に基づいて一時倍率を適用
- 両者は独立動作。方策 B と P3-03 を同時有効にしても二重縮小の心配は不要
  (base_lot を方策 B が下げた上で、さらに P3-03 が confidence で調整する)

### 4.3 balance_shrink (121#)

- 残高不足時のロット半減 (`balance_shrink_divisor`)
- これは `_current_lot` 自体を永続的に変更する
- P3-03 は `_current_lot` を変更しない → 競合なし

### 4.4 SkipGate skip/pass 判定

- SkipGate は AS 確率 ≥ 閾値でスキップ (注文しない)
- P3-03 は AS 確率 < 閾値 (PASS) だが閾値ギリギリの場合にロットを縮小
- 補完関係: SkipGate がバイナリ判定のところに連続的なリスク調整を追加

```
P(AS) = 0.2  →  SkipGate: PASS,  confidence_lot_factor = 0.80  (ほぼフル)
P(AS) = 0.5  →  SkipGate: PASS,  confidence_lot_factor = 0.50  (半減)
P(AS) = 0.65 →  SkipGate: SKIP   (注文しない)
```

### 4.5 144# CRITICAL: preflight-lot 整合性

149# §5 で指摘された「preflight 前に regime lot が反映されていない」問題は P3-03 にも波及する。
confidence_lot は regime_lot のさらに後段で適用されるため、preflight が過大ロット基準で判定するリスクは同じ。
ただし confidence_lot は常に **縮小方向** (factor ≤ 1.0) なので safety 上は許容可能。
本格対応は 144# CRITICAL 修正後。

---

## §5 テスト計画

| # | テストケース | 入力 | 期待結果 |
|---|-------------|------|---------|
| T1 | 無効時は 1.0 | `enable_confidence_lot=False` | factor = 1.0 |
| T2 | AS 確率 0.0 → factor = 1.0 | `as_prob=0.0, scale=1.0` | 1.0 |
| T3 | AS 確率 0.5 → factor = 0.5 | `as_prob=0.5, scale=1.0` | 0.5 |
| T4 | AS 確率 1.0 → floor で制限 | `as_prob=1.0, floor=0.3` | 0.3 |
| T5 | NaN/inf → 1.0 | `as_prob=float('nan')` | 1.0 |
| T6 | None → 1.0 | `as_prob=None` | 1.0 |
| T7 | regime × confidence 合成 | `regime=0.7, conf=0.5` | lot = base × 0.7 × 0.5 |
| T8 | min_order_btc 保証 | 縮小結果が min 以下 | lot = min_order_btc |
| T9 | max_lot 上限 | 拡大なし確認 | lot ≤ max_lot |
| T10 | mode=pnl | `pred_pnl=2.5, pnl_max=5.0` | factor = 0.5 |
| T11 | side 別 (Phase 2) | sell_scale=1.2 | factor 差異確認 |
| T12 | FillRecord 記録 | PASS 注文 | `confidence_lot_factor` が記録される |

---

## §6 リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| ロット縮小過多で約定機会の逸失 (false positive) | 収益減 | `floor=0.3` で最小保証。Phase C 後にデータ分析でチューニング |
| AS 確率の校正バイアス | 不適切な縮小比 | 138# P1-03 の score calibration (isotonic regression) で校正済み |
| _regime_adjusted_lot × confidence の二重縮小 | 過小ロット | min_order_btc で絶対下限保証 |
| mode=pnl での予測誤差 | ロット不整合 | 初期は mode=as のみ使用。mode=pnl は Phase C データ検証後 |
| preflight との整合 (144# CRITICAL) | 過大ロット基準 | confidence_lot は縮小方向のみなので safety 上は許容。根本対応は別途 |

---

## §7 実装計画

### 7.1 工数見積

| 作業 | 工数 |
|------|------|
| FillTestConfig 拡張 + YAML | 0.1日 |
| `_confidence_lot_factor()` + `_effective_order_lot()` | 0.2日 |
| `run_single_cycle()` 統合 | 0.1日 |
| FillRecord フィールド追加 | 0.05日 |
| テスト (T1-T10) | 0.2日 |
| **合計** | **~0.65日** |

### 7.2 実装順序

1. FillTestConfig に設定追加 (default disabled)
2. `_confidence_lot_factor()` メソッド追加 + 単体テスト
3. `_effective_order_lot()` メソッド追加 + 統合テスト
4. `run_single_cycle()` の `_order_lot` 算出を置き換え
5. FillRecord に `confidence_lot_factor` フィールド追加
6. YAML config に `confidence_lot` セクション追加
7. 回帰テスト (既存テスト全 PASS 確認)

### 7.3 初期パラメータ決定基準

Phase C データ (fill_records) から:

```python
# 分析スクリプト (Phase C 完了後に実行)
import pandas as pd
df = pd.read_json("fill_records_*.jsonl", lines=True)

# AS 確率 vs 実績 PnL の相関
corr = df[df.filled].groupby(
    pd.cut(df["skip_gate_as_prob"], bins=10)
).agg({"post_fill_30s_pnl": ["mean", "count"]})

# 最適 scale: 逆選別率が高い bucket の寄与損失を最小化する scale を決定
# 最適 floor: 全 bucket でロットがある程度確保される下限
```

---

## §8 将来拡張

| 項目 | 条件 | 内容 |
|------|------|------|
| side 別パラメータ | Phase C sell 分析後 | `scale_sell`, `floor_sell` を独立化 |
| Kelly 基準移行 | AS severity 推定が安定後 | 案 C の理論的最適比に移行 |
| time-of-day 連動 | 時間帯分析後 | 深夜帯のリスク増に応じた追加縮小 |
| 方策 B 統合 | 方策 B 再有効化時 | confidence_lot の効果を方策 B の判定に反映 |

---

## §9 Codex レビュー依頼事項

### 9.1 設計レビュー

- 案 A (線形スケーリング) は第一歩として妥当か。初期段階から案 B (バケット) にすべきか
- `floor=0.3` (70% 縮小上限) は保守的すぎるか、あるいは攻めすぎか
- AS 確率の校正品質 (138# P1-03 isotonic) が不十分な場合のフォールバック

### 9.2 統合レビュー

- §3.6 の因果逆転回避設計 (SkipGate 判定は regime_lot、発注は confidence_lot) は適切か
- `_effective_order_lot()` の配置は FillTestRunner のメソッドで良いか、別モジュール化すべきか
- 144# CRITICAL との相互作用リスクの評価

### 9.3 コードベース確認依頼

```
scripts/v460/run_fill_test.py:
  L448-466: _regime_adjusted_lot() — P3-03 で呼び出す基盤
  L953-975: run_single_cycle SkipGate → 発注フロー — P3-03 統合箇所
  L1186-1191: FillRecord skip_gate_* フィールド — confidence_lot_factor 追加先

scripts/v460/lib/fill_config.py:
  L63-64: enable_dynamic_lot / max_lot — 方策 B との関係
  L68-70: regime_lot_multipliers — regime_lot との合成

scripts/v460/ml/skip_gate.py:
  L130-142: SkipDecision — as_probability / threshold_used フィールド
  L184-310: evaluate() — AS 確率算出ロジック

configs/v460/fill_test.yaml:
  L56-70: lot_sizing セクション — confidence_lot 追加先
```
