# 664# 662/663レビュー検証 + Deadlock Escape 実装

## 概要
662# (多角的バリデーション) と 663# (理論レビュー＋利益提案) の外部レビュー文書を
コードベースに対して逐一検証し、盲点を拾い上げた上で、最優先課題である
inventory deadlock の escape 機構を実装した。

---

## 1. 662# 検証結果

### 検証済み項目

| 662# の主張 | 検証結果 | 根拠 |
|---|---|---|
| inventory_deadlock が #1 優先 | ✅ 確認 | `orchestrator_balance.py:240-280`: 検出のみで脱出機構なし。最大92連続サイクル(~3h)のロックダウンをログで確認 |
| toxic_sell_veto_as_offset は発火している | ✅ 確認 (661#の0カウント主張を修正) | 観測ウィンドウの問題。実際には発火している |
| 660# の mixed-SHA 分析は過学習リスク | ✅ 妥当な指摘 | 異なるSHAの結果を混合比較するのは統計的に問題 |
| non-reloadable field の警告 | ✅ 確認 | restart 前は反映されていなかった |

---

## 2. 663# 検証結果

### Proposal A: Deadlock Jitter / Anti-Windup
- **検証**: ✅ 脱出機構は確かに不在
- **コード根拠**: `orchestrator_balance.py:240-280` — 検出 + WARNING ログ + guard_fire のみ。カウンタは fill 成功時のみリセット (`orchestrator_post_cycle.py:58`)
- **対応**: **664# で Deadlock Escape として実装** (後述)

### Proposal B: Stale Sidecar Penalty
- **検証**: ✅ stale 時に offset = 0 になる動作を確認
- **コード根拠**: `cycle_gate_aggregator.py:435-448` — `if sidecar_signal is not None:` ゲートにより stale → offset 0
- **⚠ 663# の影響度過大評価**: `sidecar_max_boost_bps = 0.2` (YAML) → 影響は最大 0.2 bps ≈ ~22 JPY。663#の "+500 JPY penalty" 提案は不均衡。P2 として保留。

### Proposal C: Reservation Price Time Decay
- **検証**: 未実装を確認。`loss_boost` 指数減衰 (`maker_price.py:1135-1160`) が部分的代替
- **コード根拠**: `257_phg_rpt_codebase_sweep.md` で P1 future work として記載済み
- **対応**: 将来実装。現時点では loss_boost が部分的に A-S モデルの役割を担う

### Proposal D: eDRC Activation
- **検証**: ✅ 二重ゲートで完全オフを確認
  - Gate 1: `experimental_additive_pipeline = False` → `fill_config.py:378-417` の eDRC パス全体をスキップ
  - Gate 2: `edrc_alpha = 0.0, edrc_beta = 0.0` → exp(0) = 1 → 動的調整なし
- **対応**: 独立した検証セッションが必要。664# では対象外

---

## 3. 盲点 (662#/663# に含まれない発見)

### Blind Spot 1: Kyle λ / Amihud ILLIQ 死コード
- `kyle_lambda_enabled = True` + `amihud_illiq_enabled = True` だが `imbalance_enabled = False`
- → depth cache が更新されない → Kyle λ / Amihud ILLIQ は常に skip
- FillTestConfig ロード時に UserWarning が発火するが、実行に影響なし
- **影響**: 計算リソースの無駄遣い (軽微)。要 config 整合修正

### Blind Spot 2: Dual Price Control Competition
- `micro_timeout_enabled = True` + `stale_order_enabled = True`
- micro-timeout の 15s requote と stale_order の repricing が競合する可能性
- 現状問題なしだが、両方の timeout 設定が近い場合にレース条件のリスク

### Blind Spot 3: Deadlock Feedback Loop
- 高ボラ → σ 上昇 → ATR floor 膨張 → 全クオート infeasible → fill なし → σ 更新不可 → 膠着
- 648# の σ-refresh-before-guards が部分的に緩和するが完全解決ではない
- **664# の Deadlock Escape がこのフィードバックループの最終防衛線**

---

## 4. 実装: 664# Deadlock Escape

### 設計
648# で検出のみだった inventory deadlock に対し、長期化時にスプレッドガードを
一時緩和する escape 機構を追加。

```
通常時:
  effective_min = max(S_abs, BPS_floor, ATR_floor)  ← 3-tier (625#)

Escape 発動時 (counter >= deadlock_escape_threshold):
  effective_min = max(S_abs, BPS_floor, ATR_floor) × deadlock_escape_spread_mult
  例: 0.5 → min_spread 半減 → 狭いスプレッドでもクオート可能に
```

### フロー
```
orchestrator_balance._check_inventory_deadlock()
  └─ counter >= escape_threshold かつ opposite_nfq >= 2
     └─ maker_price.set_deadlock_escape(True) ← 🆕
        └─ _enforce_spread_guards() で effective_min *= spread_mult

fill 成功 → orchestrator_post_cycle
  └─ deadlock_escape_active → set_deadlock_escape(False) ← 🆕
```

### Config パラメータ
| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `deadlock_escape_threshold` | int | 0 (無効) | escape 発動サイクル数。0=無効 |
| `deadlock_escape_spread_mult` | float | 0.5 | escape 中の effective_min 乗数 |

### 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/fill_config.py` | config パラメータ追加 |
| `scripts/v460/lib/maker_price.py` | `_deadlock_escape_active` フラグ + `set_deadlock_escape()` + `_enforce_spread_guards` 緩和 |
| `scripts/v460/lib/orchestrator_balance.py` | `_check_inventory_deadlock` 内で escape 有効化 |
| `scripts/v460/lib/orchestrator_post_cycle.py` | fill 成功時に escape 解除 |
| `tests/unit/v460/test_664_deadlock_escape.py` | 16 テスト (config / flag / 機能 / ソースコントラクト) |

### 有効化方法
YAML に以下を追加:
```yaml
deadlock_escape_threshold: 20    # 20サイクル ≈ 40分で発動
deadlock_escape_spread_mult: 0.5 # min_spread を半減
```

---

## 5. テスト結果
- 664# 新規テスト: 16/16 passed
- 648# 既存テスト: 22/22 passed (回帰なし)
- 239# 既存テスト: 29/29 passed (回帰なし)
- 全 v460 テスト: 4245 passed, 5 skipped, 0 failed

---

## 6. TODO (次セッション以降)

| 優先度 | 項目 | 概要 |
|---|---|---|
| P1 | YAML 有効化 | `deadlock_escape_threshold: 20` を本番 YAML に投入 + bot restart |
| P1 | Kyle/Amihud config 整合 | `imbalance_enabled: true` にするか kyle/amihud を `false` に |
| P2 | Stale sidecar offset boost | 影響 0.2 bps と小さいが防御としては有効 |
| P2 | eDRC 独立検証 | alpha/beta のキャリブレーション必要 |
| P3 | Reservation price | A-S モデル full 実装 (257# で P1 future work) |
