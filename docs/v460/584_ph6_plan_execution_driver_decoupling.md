# 584# Phase 6 計画: Execution Driver Decoupling & Validation

**Date**: 2026-03-24
**Phase**: ph6 (Phase 5.5 完了後の次期フェーズ)
**Status**: Draft
**Prerequisites**: Phase 5.5 完了 (582# A/B dispatcher, 579# tanh inventory skew, 555# CalibrationMap)
**Gate**: G3-pnl / G3.1-stress の前段として Execution 層の品質向上を図る

---

## §0 大義との接続

> 000# §0: 「短期間での高収益性システム」の実現。ただし短期的成果と長期的な健全性のバランスを取る。

Phase 5 運用データ (n=3,869, 2026-03-12〜03-22) の実績:
- **Fill Rate**: 25.2% (G1.2-full F1 閾値 70% を大幅に下回る)
- **Buy 側 PnL**: -0.28 bps (損失)
- **Sell 側 PnL**: +0.21 bps (利益だが p10=-8.26 bps のテールリスク)
- **preflight_insufficient**: 34.7%

**Phase 6 の役割**: Execution 層の構造的問題を解消し、G1.2-full 再通過への道筋を作る。

---

## §1 Phase 5.5 成果の棚卸し

| # | 成果 | 状態 | 次ステップ |
|---|------|------|-----------|
| 582# | RMS Additive Pipeline (A/B dispatcher) | ✅ 実装済・無効 | A/B テスト実施 |
| 581# | True Additive 設計 + RMS 数式 | ✅ 仕様確定 | 582# に包含 |
| 580# | FillRecord schema 修正 + eDRC hard_cap 順序修正 | ✅ 修正済 | — |
| 579# | Inventory Skew tanh 平滑化 | ✅ 実装済 | 効果測定 |
| 575# | eDRC パラメータ推定 (α=0.020, β=0.40) | ✅ 推定済・無効 | 有効化判断 |
| 555# | CalibrationMap Entry Gate 統合 | ✅ 実装済・無効 | 有効化判断 |
| 554# | Raw data gap fill (22K bars) | ✅ 完了 | — |
| 553# | OHLCV auto-update pipeline | ✅ 稼働中 | — |

**共通パターン**: 基盤は実装済みだが **全て disabled**。Phase 6 は検証・有効化フェーズ。

---

## §2 Phase 6 タスク定義

### §2.1 A/B Validation: Additive Pipeline (P1 — CRITICAL)

**目的**: 582# RMS additive pipeline の multiplicative chain に対する優位性を実測で検証。

**方法**:
1. `experimental_additive_pipeline: true` を YAML で有効化
2. 72h (G1.1-quick 準拠) のデータ収集
3. `analyze_fill_logs.py` の `section_execution_quality_comparison` で A/B 比較
4. `section_buffer_decomposition` で Toxicity/Liquidity バッファ分離の効果確認

**成功基準**:

| # | 指標 | 条件 | 根拠 |
|---|------|------|------|
| V1 | spread_capture_bps (additive) | ≥ multiplicative の median | 収益性維持 |
| V2 | adverse_selection_cost_bps | additive ≤ multiplicative | リスク削減確認 |
| V3 | offset_ceiling clamp 発火率 | additive < multiplicative 10% 以上 | 乗算爆発抑止の実証 |
| V4 | tox_buffer / liq_buffer 分離度 | 相関 < 0.5 | 独立バッファとして機能 |

**FAIL 時**: additive を disabled に戻す。RMS 以外の加法合成 (max, weighted sum) を検討。

**依存**: 現行 fill_test の `experimental_additive_pipeline` フラグ切替のみ。コード変更不要。

---

### §2.2 Smart Preflight Integration (P2 — HIGH)

**目的**: preflight_insufficient (34.7%) の構造的削減。

**現状の問題**:
- preflight 拒否は「残高不足」の事後検知 → API コール浪費
- inventory_skew (tanh 平滑化済み) と preflight_pause が独立動作
- 在庫偏り → 片側 preflight 失敗 → pause → 機会損失の悪循環

**設計**:

```python
def _should_skip_preflight(self, side: str) -> bool:
    """579# Phase 6: 在庫スコアから preflight 失敗を予測し API コールを節約."""
    inv_score = self._maker_price.get_inventory_skew_score(side)
    if abs(inv_score) > self.config.preflight_skip_inv_threshold:
        return True  # 在庫偏りが大きすぎる → preflight 不要
    return False
```

**新規 Config パラメータ**:

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `smart_preflight_enabled` | false | Smart Preflight 有効化 |
| `preflight_skip_inv_threshold` | 0.7 | inv_skew_score がこの閾値超過で skip |

**成功基準**:
- preflight_insufficient 率 34.7% → 20% 以下
- fill_rate への影響 ±3% 以内

**実装場所**: `orchestrator_balance.py` (`_resolve_balance_and_preflight`)

---

### §2.3 Buy 側品質改善 (P3 — HIGH)

**目的**: Buy 側 avg_pnl -0.28 bps → 0 bps 以上

**現状分析** (579# / 576#):
- Buy 側は EV offset / velocity guard / toxicity budget 全てが「保守的方向」に作用
- 結果としてスプレッドを過剰に取り、約定率低下 + 逆選択コスト増加
- Cross-venue lead-lag guard が buy 側で過剰適用 (503# 指摘)

**アプローチ**:

| # | 施策 | 実装量 | 期待効果 |
|---|------|--------|---------|
| B1 | Buy 側 offset ceiling 緩和 | Config 変更のみ | スプレッド縮小 → 約定率向上 |
| B2 | Buy 側 EV offset sensitivity 引き下げ | Config 変更のみ | EV guard 過剰反応の抑制 |
| B3 | Cross-venue guard buy 側閾値の独立化 | 小規模コード変更 | guard 過剰適用の是正 |

**検証**: 各施策を段階的に投入し、72h データで比較。

---

### §2.4 Sell 側テールリスク抑制 (P4 — MEDIUM)

**目的**: Sell 側 p10=-8.26 bps → -5.0 bps 以内

**eDRC 活用**:
- 575# で推定済み: α=0.020, β=0.40, hard_cap=1.0
- `edrc.enabled: true` で有効化
- 高ボラティリティ + 逆方向 OFI 時に ceiling を自動引き下げ

**検証基準**:

| # | 指標 | 条件 |
|---|------|------|
| T1 | sell_p10_bps | ≥ -5.0 bps |
| T2 | sell_avg_pnl_bps | ≥ +0.15 bps (現行比 -30% 以内) |
| T3 | sell_fill_rate | ≥ 現行比 -5% |

---

### §2.5 Entry Gate 有効化判断 (P5 — MEDIUM)

**目的**: CalibrationMap (555#) による EV negative サイクルの事前ブロック。

**前提**: 554# の calibration_batch.py で 15,531 件の fill records から学習済み。

**有効化手順**:
1. `entry_gate.enabled: true`
2. 72h データ収集
3. ブロック率とブロック対象のPnL分布を検証
4. ブロックされたサイクルの事後 PnL が負であることを確認

**成功基準**: ブロック対象の avg_pnl30 < -0.5 bps (正しく損失サイクルを予測)

---

### §2.6 Retrain Scheduler 再起動 (P6 — HIGH)

**目的**: SAC Sidecar の neutral fallback 状態を解消し、方向バイアス信号を復活。

**現状**: `cache/sidecar_signal.json` が `model_version: "neutral"` のまま。
retrain_scheduler は実装済みだが未起動。

**手順**:
1. OHLCV データ鮮度確認 (553# auto-update が稼働中のはず)
2. `retrain_scheduler.py --once` でテスト実行
3. 出力モデルの OOS ROI 確認 (val_ratio=0.10, 000# §3.4 準拠)
4. 正常なら daemon 起動

**リスク**: 426# で確認済みのレジーム汎化限界。retrain_scheduler の stale data guard (48h) は実装済み。

---

## §3 実行順序とスケジュール

```
Phase 6.0: Validation & Activation
├─ P6 Retrain Scheduler 再起動       ← 即日 (前提条件の回復)
├─ P1 Additive A/B test (72h)        ← データ収集開始
├─ P2 Smart Preflight                 ← P1 と並行実装可能
├─ P3 Buy 側 Config 調整              ← P1 結果待ち (additive の影響把握後)
├─ P4 eDRC 有効化 (72h)              ← P3 完了後
└─ P5 Entry Gate 有効化               ← P4 と並行可能

Phase 6.1: Assessment
├─ G1.2-full 再判定 (168h)
├─ G3-pnl 暫定評価
└─ Phase 7 判断 (SAC retrain + Sidecar v2 本格運用)
```

**クリティカルパス**: P6 → P1 → P3 → G1.2-full 再判定

---

## §4 Gate 影響

| Gate | Phase 6 での変化 |
|------|----------------|
| G1.2-full | 再判定対象: P1-P5 完了後に F1-F8 を再計測 |
| G2-train | 影響なし (SAC 学習パイプラインは変更しない) |
| G3-pnl | P1/P3/P4 の結果次第で初回判定可能 |
| G3.1-stress | P4 (eDRC) の有効化で stress 耐性改善を期待 |
| G4-live | Phase 6 完了後に投入判断 |

---

## §5 リスク

| 重要度 | リスク | 緩和策 |
|--------|--------|--------|
| ⭐⭐⭐ | Additive pipeline で収益性悪化 | A/B フラグで即戻し可能 |
| ⭐⭐⭐ | eDRC 有効化でスプレッドが過剰収縮 → 約定率崩壊 | hard_cap=1.0 で上限制約済み。段階的 α/β 調整 |
| ⭐⭐ | Smart Preflight で本来約定可能だったサイクルを skip | inv_threshold を保守的に開始 (0.7) |
| ⭐⭐ | Retrain Scheduler crash (570# 調査で crash vector 確認済み) | stale data guard + 例外ハンドリング強化済み |
| ⭐ | Entry Gate の false positive | ブロック率モニタリング + CalibrationMap のオンライン更新 |

---

## §6 成果物

| # | 成果物 | 形式 |
|---|--------|------|
| 1 | A/B テスト結果レポート | docs/v460/585# rpt |
| 2 | Smart Preflight 実装 + テスト | scripts/v460/lib/ + tests/ |
| 3 | Buy 側 Config 最適化結果 | docs/v460/586# rpt |
| 4 | eDRC 有効化結果 | docs/v460/587# rpt |
| 5 | G1.2-full 再判定レポート | docs/v460/ rpt |

---

## Appendix A: 改訂履歴

| 日付 | 変更 | 理由 |
|------|------|------|
| 2026-03-24 | 初版作成 | Phase 5.5 完了 (582#)、Phase 6 開始準備 |
