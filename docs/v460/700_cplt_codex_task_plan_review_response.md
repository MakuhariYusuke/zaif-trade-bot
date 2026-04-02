# 700# Codex タスク計画: 697#/698# レビュー対応

## 概要

699# クロスバリデーションで確認されたバグ修正と構造的問題への対応として、Codex に投入する 4 タスクを策定。
697#/698# レビュー指摘 + 699# 独自発見の盲点から、**即効性の高い修正**を優先する。

## 優先度選定根拠

699# で特定した行動方針から、以下の基準で選定:
1. **P0 かつ Codex 向き** (明確な仕様、テスト可能、局所的変更)
2. **収益直結度** (trending_down -63.54 bps が最大損失源)
3. **リスク** (enabled: false のバグ修正は低リスク、パラメータ変更は中リスク)

---

## タスク一覧

### 優先度マトリクス

| 優先度 | タスク | 根拠 | 期待効果 | 工数 |
|--------|--------|------|----------|------|
| **P0-1** | Protocol 688 NFQ フィルタ修正 | 697# 指摘, 699# 盲点F | 分析基盤正確性 | 2-3h |
| **P0-2** | spread_as_guard 閾値修正 + 有効化準備 | 697# 指摘 | AS 防御 +0.2-0.4 bps | 3-4h |
| **P0-3** | インベントリスキューイング強化 | 698# 指摘, 699# 盲点A | ドリフト抑制 | 4-5h |
| **P1-1** | trending_down regime exit 戦略 | 699# 盲点A,C,D | RT損失 -63→-30 bps | 5-6h |

---

### Task 1: Protocol 688 NFQ フィルタ修正 (P0-1)

**問題**: `protocol_688.py:349` で `_cancel_payload(records)` が全キャンセルを返すセマンティックバグ。
過去の「NFQ 分析」結果はすべて全キャンセル分析だった。

**修正箇所**:
- `scripts/v460/analysis/protocols/protocol_688.py`
- NFQ ペイロード生成に `cancel_reason == "no_feasible_quote"` フィルタ追加
- 既存 `_cancel_payload` は汎用キャンセルとして残す (他セクションで利用)
- 新関数 `_nfq_payload` を追加

**テスト**:
- `tests/unit/v460/test_700_protocol_688_nfq_fix.py`
- NFQ のみのフィルタリング検証
- 全キャンセル vs NFQ キャンセルの分離確認
- 空データ・NFQ ゼロケースの処理

**成果物**: protocol 688 の NFQ セクションが正確な NFQ データのみを表示

---

### Task 2: spread_as_guard 閾値修正 + 有効化準備 (P0-2)

**問題**: `fill_test.yaml:1289` の `threshold: 1500.0` は bps 単位で 15% 相当。意図は JPY spread か、実際のスプレッド bps かが曖昧。

**修正箇所**:
- `scripts/v460/lib/entry_gate_adjustments.py`: 閾値ロジックの単位明確化
- `configs/v460/fill_test.yaml`: `threshold: 15.0` (bps) に修正 + コメント追記
- `spread_as_guard.enabled: false` は維持 (有効化は手動)
- spread 分布の p50/p75/p90 を分析して閾値の妥当性を検証するユーティリティ追加

**テスト**:
- `tests/unit/v460/test_700_spread_as_guard_fix.py`
- 閾値が bps 単位で正しく動作する検証
- 境界値テスト (threshold ± 0.1 bps 周辺)
- enabled=false 時のバイパス確認
- ev_penalty 適用の数値精度

**成果物**: spread_as_guard が正しい単位で安全に有効化可能な状態

---

### Task 3: インベントリスキューイング強化 (P0-3)

**問題**: `deque(maxlen=100)` は直近 100 fill のみ追跡。時間減衰 τ=3600s はあるが、数百 fill にわたる漸進的ドリフトは検出できない。698# はwindow=1000 を提案するが、τ=3600s との併用で効果は逓減的。

**修正箇所**:
- `scripts/v460/lib/maker_price.py`: `InventorySkewing` クラス
  - `maxlen` を config 可能に (デフォルト 300, config: `inventory_skewing_window`)
  - `max_factor` の段階的スケーリング: 通常 0.4, ドリフト検出時 0.6
  - ドリフト検出: 直近 window の buy/sell 比率が `neutral_band` 外に一定時間滞在
- `configs/v460/fill_test.yaml`: 新パラメータ追加
  - `inventory_skewing_window: 300`
  - `max_factor_drift: 0.6`
  - `drift_detection_threshold: 0.15` (neutral_band * 3)
  - `drift_detection_window_sec: 1800`

**テスト**:
- `tests/unit/v460/test_700_inventory_skewing.py`
- window 拡大が長期ドリフト検出に寄与する検証
- 段階的 max_factor スケーリングの動作確認
- ドリフト検出 → max_factor 引き上げの遷移テスト
- 時間減衰との相互作用テスト
- 既存動作への非破壊性 (window=100 同等動作の後方互換)

**成果物**: 長期在庫ドリフトへの耐性を持つインベントリスキューイング

---

### Task 4: trending_down regime exit 戦略 (P1-1)

**問題**: 699# 盲点A — trending_down ラウンドトリップが -63.54 bps (全 RT 損失の 349%)。
trending_down 時に buy fill が滞留し、regime 遷移後に損失拡大する。

**修正箇所**:
- `scripts/v460/lib/maker_price.py` or 新ファイル `regime_exit_strategy.py`
  - trending_down 検出時の buy ポジション exit 指標追加
  - `max_trending_down_exposure` config: trending_down 時の buy 累積上限
  - 上限超過時に aggressive skewing (max_factor 引き上げ or NFQ 発動)
- `scripts/v460/lib/fill_record_builder.py`: regime exit 関連メトリクス追加

**分析フェーズ** (実装前):
- trending_down 時の buy fill → sell fill までの hold 時間分布
- regime 遷移 (trending_down → ranging/up) のタイミングと RT クローズのタイミング
- sell_hour_boost と trending_down の interaction 分析

**テスト**:
- `tests/unit/v460/test_700_regime_exit.py`
- trending_down exposure 上限の動作確認
- regime 遷移時の max_factor 切替
- 通常 regime での非介入確認
- MCB との interaction テスト

**成果物**: trending_down 時の損失抑制メカニズム

---

## Codex 投入順序

```
Phase 1 (同時投入可):
  Task 1 (Protocol 688 NFQ fix)   ← 独立、局所的
  Task 2 (spread_as_guard fix)    ← 独立、局所的

Phase 2 (Phase 1 テスト後):
  Task 3 (Inventory skewing)      ← maker_price.py 変更

Phase 3 (Phase 2 検証後):
  Task 4 (Regime exit strategy)   ← Task 3 の inventory 機構に依存
```

---

## 見送りタスク (次回以降)

| タスク | 理由 |
|--------|------|
| as_trailing_gate 有効化 | config 変更のみ。Codex 不要、手動対応 |
| AS フィールドドキュメント | ドキュメントタスク。Codex 不向き |
| 分析ツール統合 (697# Option D) | 大規模リファクタ。安定期に実施 |
| Sidecar 改善 | 100% stale だが根本原因がインフラ側 |
| NFQ 0% 閾値見直し | 分析タスク。Task 4 の結果を待って判断 |

---

*生成: 2026-04-02 by cplt (700#)*
*入力: 697#, 698#, 699#*

---

## 2026-04-03 follow-up: prompt-to-runtime review

今回の実装前点検で、prompt 記述をそのまま当てるより current runtime に寄せる方が安全な点を確認した。

- Task 1:
  - `protocol_688` の `nfq` は本当に cancel 全体を見ていたため、prompt どおり修正で正しかった
- Task 2:
  - 本体はすでに `spread_threshold_bps` / `ev_penalty_bps` 契約へ移行済みだった
  - 問題は命名より `1500.0bps` という実質無効な閾値で、修正点は threshold 値 + validation + backward-compat parser に絞るのが正解だった
- Task 3:
  - inventory skewing は新しい並列実装ではなく、既存 `maker_price` の drift/state telemetry を伸ばす方が live path に自然だった
- Task 4:
  - regime exit は単独 tracker で終わらせず、NFQ / fill telemetry / pricing veto まで通す必要があった

このレビュー結果に基づき、700 batch は prompt の意図を保持しつつ、current code path に沿う最小実装へ寄せた。

## 実装ステータス

| タスク | 状態 | 補足 |
|--------|------|------|
| Protocol 688 NFQ fix | 完了 | `nfq` と `cancels` を分離し、過去の cancel 全体集計バグを解消 |
| spread_as_guard fix | 完了 | 命名変更ではなく、`15.0bps` 系 threshold + validation + backward-compat parser に寄せた |
| inventory skewing | 完了 | 既存 `maker_price` の drift telemetry / max_factor escalation を拡張 |
| regime exit strategy | 完了 | 単独 tracker で終わらせず、NFQ / pricing veto / fill telemetry まで接続 |

## 隠れタスクとして回収したもの

- hot-reload / validation / YAML drift allowlist の追随
- fill observability への telemetry 流し込み
- protocol / analysis での spread 分布補助セクション追加
- broad regression を通すための current threshold 系テスト追随

## 横展開できたもの

- `FillRecord` telemetry は runtime だけでなく analysis / fill_quality / tests まで接続
- `spread_as_guard` の current 契約は hot-reload / validation / protocol 側へ横展開
- inventory / regime-exit は maker path に閉じず NFQ / cancel taxonomy / observability まで反映
