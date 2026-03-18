# 475# 472提案評価 + 474セルフレビュー + 残課題改修

## 概要
472#の3つのアーキテクチャ提案を評価し、474#のセルフレビューを実施し、470-472の残課題を改修。

---

## §1 472# 三提案の評価

### ① mmap IPC (Memory-Mapped State) — ❌ REJECT

**提案**: プロセス間通信を mmap / SharedMemory + C構造体で 0-copy 化し、GIL 遅延を突破。

**評価**:
- **現状の IPC**: atomic JSON ファイル (sidecar_signal_io.py) + multiprocessing.Manager (cache_coordination.py)
- **現状のタイミング粒度**: 120秒サイクル、5秒ポーリング間隔、15秒マイクロタイムアウト
- **mmap の利得**: IPC レイテンシ ~1ms → ~1μs (×1000 改善)
- **問題**: 120秒サイクルで 1ms→1μs の改善は誤差以下。ボトルネックは API RTT (~200ms) とポーリング間隔 (5s)

**却下理由**:
1. GIL はボトルネックではない — メインループは asyncio (I/O bound)、CPU 集約処理は ProcessPoolExecutor
2. 現タイミング粒度 (秒単位) に対してμs最適化は premature optimization
3. C構造体定義やメモリ安全性の複雑さがリターンに見合わない
4. **再検討条件**: サブ秒判断サイクルに移行する場合のみ

### ② XGBoost Micro-prediction — ⚠️ DEFER

**提案**: 板の厚み偏り (Orderbook Imbalance) + Tick Velocity の極小 XGBoost/LightGBM で、
1ms 以下の推論で 1 秒以内の mid 方向変動を予測する「反射神経ドメイン」。

**評価**:
- **既存機能**: SkipGate (LightGBM classifier) は pnl30/pnl120 基準で skip/go 判定済み
- **既存特徴量**: depth_imbalance, trade_flow_imbalance, VPIN, velocity — 既にパイプラインに存在
- **問題**:
  1. ポーリング間隔 5 秒 → 1ms 推論の利点が活きない
  2. P0 修正 (ceiling=0.20) で mid+718 JPY バッファを確保済み → 936 JPY 逆行の主問題に対処済み
  3. **order_monitor.py に adverse_drift cancellation が既に実装** (drift_bps_buy=4.0, drift_bps_sell=5.0)
    → 逆行時のキャンセル機構は稼働中
  4. XGBoost はモデルリスク (過学習、レジーム変化) を追加

**代替**: 既存の adverse drift detection (order_monitor.py) が同等の保護を提供。
追加 ML モデルの複雑さなしで、閾値ベースの反射的キャンセルで十分。

### ③ Shadow Orders (仮想約定追跡) — ⚠️ DEFER (最有望)

**提案**: 本番稼働中に「別パラメータならいくらで約定したか」をメモリ上で追跡。
Counterfactual 差分を off-policy 学習のフィードバックに活用。

**評価**:
- **既存基盤**: run_observation.py (板観測モード)、archived/counterfactual.py (過去実装)
- **利点**:
  1. 本番環境での安全な A/B テスト
  2. 機会損失 (Opportunity Cost) の定量化
  3. SAC への off-policy 教師データ
- **現段階で不適切な理由**:
  1. P0 修正後のデータがまだ蓄積されていない (24h 未満)
  2. 利益が出ていないシステムの A/B テストは意味が薄い
  3. まず P0 効果の検証が先決

**実装予定時期**: P0 修正後 3-7 日のデータ蓄積・分析後。
run_observation.py を拡張する形で最小構成の Shadow Order Engine を構築。

---

## §2 474# セルフレビュー

### P0: Sell パラメータ修正 — ✅ 正常

| 項目 | 検証結果 |
|------|----------|
| ceiling_sell=0.20 | `resolve_offset_ceiling("sell")` → 0.20 を返す。最終段クランプが正しく機能 |
| floor=0.05 | パイプラインが 0.05-0.20 の範囲で自然な offset 分布を生成可能 |
| `_effective_max_ratio` | max(0.30, 0.20)=0.30 を中間段上限に返す。最終段 0.20 クランプとの組み合わせで正しい出力 |

**`_effective_max_ratio` の max() は min() にすべきか？**

→ **いいえ**。設計意図は中間段の探索幅を確保すること。
最終段の `resolve_offset_ceiling(0.20)` が最終出力を必ず保証するため、
中間段が 0.30 まで探索しても最終結果に影響しない。
ただし docstring が旧値 (0.30-0.50) を参照していたため更新した。

### P1: _recalc 差分公式 — ✅ 正確

```python
delta = spread_at_order * (old_ratio - new_ratio)
sell: price + delta  # ratio↓ → delta>0 → mid から離れる (正しい)
buy:  price - delta  # ratio↓ → delta>0 → mid から離れる (正しい)
```

- spread=0 or None → `order_price` そのまま返却 ✅
- old_ratio == new_ratio → delta=0、変化なし ✅
- half-spread エラーなし、mid 推定を介さない直接差分 ✅

### P2: Micro-timeout 公式 — ✅ 正確 (軽微な注意点あり)

```python
sell: order_price = round(_rq_mid + _rq_spread * (0.5 - effective_offset_ratio))
buy:  order_price = round(_rq_mid + _rq_spread * (effective_offset_ratio - 0.5))
```

- 公式はベース数式 `sell = mid + spread*(0.5 - ratio)` と一貫 ✅
- **注意**: `spread_at_order=None` 時 → `_rq_spread=0.0` → `order_price=mid`
  - micro_timeout は `enabled: false` なので現時点では非活性
  - 将来的に有効化する際は spread 取得の堅牢化が必要

### P3: retrain_scheduler lockfile — ✅ 機能的 (軽微な TOCTTOU)

- O_CREAT|O_EXCL は Windows 互換 ✅
- PID 生存チェックによる stale lock 回収 ✅
- **TOCTTOU**: unlink → re-acquire 間に別プロセスが割り込む可能性
  - retrain_scheduler の起動頻度は低く (手動 or hot_swap のみ)、実質リスク極小
  - 改善案: portalocker による OS レベル排他 (lock_manager.py と同等)

### lock_manager parent PID — ✅ 正確

- parent_pid 取得失敗時は None → 自 PID のみ除外 (安全フォールバック) ✅
- Start-Process ランチャーの false positive を正しく回避 ✅

---

## §3 470-472 残課題の棚卸し

### 済み (474# で対処)

| 項目 | ステータス |
|------|-----------|
| P0: Sell ceiling/floor | ✅ 474# |
| P1: _recalc 半spread | ✅ 474# |
| P2: micro-timeout 公式 | ✅ 474# |
| P3: orphan cleanup | ✅ 474# |

### 済み (既存実装で対処)

| 項目 | 発見 |
|------|------|
| 470# P3: Mid-movement cancellation | ✅ order_monitor.py に adverse_drift detection 実装済み (drift_bps_buy=4.0, drift_bps_sell=5.0) |
| 470# P4: Post-fill wait 延長 | ✅ post_fill_wait_sec_sell=90.0 (168# で対応済み) |

### 本 475# で対処

| 項目 | 内容 |
|------|------|
| 470# P2: Sell-side filtering | `skip_ranging_sell_low_vol` + soft mode 実装 → Gate 2b として追加 |
| `_effective_max_ratio` docstring | 旧値 (0.30-0.50) 参照を現状 (ceiling < base) に更新 |

### 保留 (データ蓄積・監視後に判断)

| 項目 | 理由 |
|------|------|
| 470# P1: EV sell-side fix | P0 修正後のデータで EV 分布変化を観察する必要あり。EV が依然 -0.78 固定なら sell 側 EV weight を 0 に設定 |
| 471# P3: Offset semantics unification | 中期リファクタリング。現状 3 モジュール (maker_price, offset_pipeline, micro-timeout) で ratio の定義が異なるが、474# P1-P2 で直接的な数式バグは修正済み |
| 472# Shadow Orders | P0 効果検証後に実装 |

---

## §4 実装変更サマリ

### 4.1 `skip_ranging_sell_low_vol` (Gate 2b)

**変更ファイル**:
- `scripts/v460/lib/fill_config.py` — `skip_ranging_sell_low_vol`, `ranging_sell_low_vol_as_offset` パラメータ追加
- `scripts/v460/lib/cycle_gate_aggregator.py` — `_check_ranging_sell_low_vol` メソッド + Gate 2b チェーン
- `configs/v460/fill_test.yaml` — 両パラメータを `true` (soft mode で開始)
- `tests/unit/v460/test_475_ranging_sell_low_vol.py` — 9 テスト全通過

**設計**: buy 側 `_check_ranging_buy_low_vol` (169# B1') と完全対称。
ranging + sell + vol_ratio < low_vol_threshold 時にゲート発火。
soft mode (as_offset=true) では hard skip せず、maker_price の offset boost に委譲。

### 4.2 `_effective_max_ratio` docstring 更新

`maker_price.py` — 旧コメント「0.30-0.50 の範囲で有効に機能」を
「ceiling < base の場合は base を返し、最終段 ceiling clamp が上限を保証」に修正。

---

## §5 次のアクション

1. **P0 効果観察 (3-7日)**: sell fill の PnL 分布、逆行距離、offset 分布をモニタリング
2. **EV sell-side 判断**: 蓄積データで EV スコアの sell 分布を分析。定数なら weight=0 に
3. **Shadow Order MVP**: P0 効果確認後、run_observation.py ベースで仮想約定追跡を実装
4. **Offset semantics unification**: 中期リファクタリングとして ratio → absolute JPY 移行を計画

---

*作成日: 2026-03-19*
*対象: 472# §4 提案評価 + 474# セルフレビュー + 470# P2 実装*
