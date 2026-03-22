# 549# ログ分析深堀り：EWMA汚染修正・サイドカー死因・AS構造パターン

- **日付**: 2026-03-23
- **親ドキュメント**: 547# (Codexレビュー), 548# (本質論)
- **目的**: post-restart fill_test ログ (`run_id: 1774174930_d186a612`) の深堀り分析結果と、実施した修正の記録

---

## §1. 概況: post-restart ログ分析結果

| 指標 | 値 |
|------|-----|
| 総サイクル | 161 |
| フィル数 | 30 (18.6%) |
| 合計PnL | -27.5 bps |
| sell_dynamic_kill 比率 | 26% (pre-restart: 3%) |
| sidecar_offset_bps | 全30件 = 0.0 |

5つの改善優先項目を特定し、個別に深堀りを実施した。

---

## §2. P0: sell_dynamic_kill スパイラル — 根本原因と修正

### 2.1 根本原因

sell_kill_state の pnl_history は4件のみ:

```
[-13.536, 1.12, -6.755, 4.231]
```

EWMA α=0.05 で再構築すると:
- seed = -13.536 (最初の値)
- EWMA = α × v + (1-α) × EWMA を逐次適用
- 最終 EWMA ≈ **-11.66 bps**

ranging 閾値 = -0.7 bps であるため、-11.66 << -0.7 → **全サイクルで kill 発動**。

時間減衰 τ=600s での回復時間:
```
t_recover = -ln(0.7 / 11.664) × 600 ≈ 1688s ≈ 28分
```

**単一の極端な AS イベント (-13.54 bps) が EWMA を約28分間汚染し、26%のsellサイクルがブロックされた。**

これは systematic risk ではなく idiosyncratic shock であり、EWMA が本来追跡すべき傾向ではない。

### 2.2 修正: EWMA 入力クランプ (Winsorization)

`DynamicKillConfig` に `ewma_input_clamp_bps` フィールドを追加。`track()` 時に入力 PnL を `[-clamp, +clamp]` に制限する。

```python
# ztb/risk/sell_dynamic_kill.py track() 内
clamped = pnl_bps
clamp = self._config.ewma_input_clamp_bps
if clamp > 0:
    clamped = max(-clamp, min(clamp, pnl_bps))
```

**設定値**: `ewma_input_clamp_bps: 5.0` (sell/buy 両方)

クランプ適用後の効果:
- seed = clamp(-13.536) = **-5.0** (13.54 → 5.0 に制限)
- 最終 EWMA ≈ -4.26 (vs 無クランプ -11.66)
- 回復時間: -ln(0.7/5.0) × 600 ≈ **1181s ≈ 20分** (vs 28分)
- `_rebuild_ewma_from_history()` にも同一クランプを適用 (整合性確保)

### 2.3 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ztb/risk/sell_dynamic_kill.py` | `ewma_input_clamp_bps` フィールド追加, `track()` / `_rebuild_ewma_from_history()` にクランプ適用 |
| `configs/v460/fill_test.yaml` | sell/buy 両方に `ewma_input_clamp_bps: 5.0` 追加 |
| `scripts/v460/lib/fill_config.py` | `sell/buy_dynamic_kill_ewma_input_clamp_bps` フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | YAML パース追加 |
| `scripts/v460/run_fill_test.py` | `DynamicKillConfig` 構築に新フィールド追加 |
| `tests/unit/v460/test_549_ewma_input_clamp.py` | 新規テスト 11件 (全PASSED) |

### 2.4 理論的根拠

Winsorization (Wilcox 2010): 外れ値の影響を制限するロバスト統計手法。EWMA (RiskMetrics 1996) と組み合わせ、systematic trend のみを追跡し idiosyncratic outlier による contamination を防止する。

**pnl_history には生値を保持** — クランプは EWMA 計算時のみ適用。これにより診断・デバッグ時に実際の PnL 値が参照可能。

---

## §3. P0: サイドカー完全死亡 — 真の原因

### 3.1 547#/548# の誤診を訂正

547# は「compound conservatism (quadratic + toxicity)」を、548# は「pipeline の因果関係」をサイドカー0の原因として分析したが、**いずれも誤診**。

### 3.2 真の原因: SAC モデルの neutral fallback

`cache/sidecar_signal.json` の内容:

```json
{
  "model_version": "neutral",
  "directional_bias": 0.0,
  "confidence": 0.0,
  "timestamp": "..."
}
```

SAC retrain scheduler が **neutral fallback** を出力している:
- OOS gate failure (バリデーション不合格)
- training exception
- model load failure

のいずれかが原因で、実際の方向性予測モデルが稼働していない。

`compute_sidecar_offset_bps_v2()` のロジック:
```python
if abs(bias) < dead_zone:  # dead_zone=0.10
    return 0.0  # bias=0.0 → 即座に 0 返却
```

### 3.3 対応方針

コードバグではなくML pipeline の問題。以下が必要:
- SAC モデルの学習・評価ログ確認
- OOS gate の閾値見直し
- 学習データの品質確認

本セッションではコード修正対象外。

---

## §4. P1: ceiling 0.30 SHA 比較

post-restart run 内のSHA遷移を分析:

| SHA | fills | avg_eff | avg_pnl | AS率 |
|-----|-------|---------|---------|------|
| d79e (ceiling 0.30) | 22 | 0.267 | -1.00 | 22.7% |
| 8a63 (ceiling 0.25) | 8 | 0.250 | -0.70 | 25.0% |

**サンプルサイズ不足で統計的結論は出せない**が、ceiling 0.30 → 0.25 の変更は avg_pnl を 30% 改善した可能性がある。ceiling 0.30 のフィルは AS 被害直撃時のダメージが大きい。

---

## §5. P1: Velocity + OBI と AS パターン

### 5.1 発見

全7件の AS fill を分析:
- **全件 VG (Velocity Guard) = True** — VG は検知しているがブロックしていない
- **skip_gate_as_prob = 空** — SkipGate モデルがこれらをスコアリングしていない
- **Sell AS パターン**: velocity > 0 (価格上昇中) + OBI < 0 (ask-heavy / bid-thin)

### 5.2 構造的課題

VG は velocity に比例した offset を加算するが、取引自体はブロックしない。つまり:
- 価格が急上昇中 (velocity > 0) に sell を出す
- order book は ask 側に厚い (OBI < 0 = 不利)
- VG offset が不十分なまま fill → 逆選択

### 5.3 今後の検討

1. VG に velocity + OBI の複合条件でのブロック閾値を導入
2. SkipGate モデルの AS パターン学習データ拡充
3. AS fill の velocity / OBI 分布をモニタリングに追加

本セッションではコード修正対象外 (設計検討として記録)。

---

## §6. テスト結果

```
test_549_ewma_input_clamp.py: 11 passed
test_349_ewma_fixes.py: 25 passed (リグレッションなし)
```

---

## §7. まとめと次のアクション

| 項目 | 状態 | アクション |
|------|------|-----------|
| sell_dynamic_kill EWMA スパイラル | **修正済** | ewma_input_clamp_bps=5.0 適用、テスト通過 |
| サイドカー死亡 | **原因特定** | ML pipeline 調査が必要 (コード問題ではない) |
| ceiling 0.30 効果 | **観測記録** | データ蓄積待ち |
| VG + AS パターン | **構造記録** | 設計検討として保留 |
| preflight_insufficient | **547# P0参照** | pricing pipeline の根本改善が必要 |

**最優先**: fill_test を現行 SHA (6a7f49c73 → 549# 修正後) で再起動し、EWMA クランプの実効果を検証する。
