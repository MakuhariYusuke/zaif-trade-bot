# 370# ph2 方向転換: 368#/369# レビュー受容 + 緊急対応

| 項目 | 値 |
|---|---|
| 文書番号 | 370# |
| フェーズ | ph2 G1.1-exec |
| 前提文書 | 367#, 368# (Codex), 369# (Gemini) |
| 作業日 | 2026-03-10 |
| 方針 | 収益最優先 / 心肺蘇生 / 方向転換 |

---

## §1 レビュー結果の独自検証

368# (Codex) と 369# (Gemini) の指摘を実データ・コードで全件検証した。

### 1.1 検証結果サマリ

| Finding | 重大度 | 検証結果 | 根拠 |
|:-------:|:------:|:--------:|------|
| F1 | CRITICAL | ✅ **確認** | `orchestrator_mid_cycle.py` L135 の `evaluate()` 呼出しに `sidecar_signal=` 引数なし。`read_sidecar_signal()` は live path から未参照。Gate 側は受容コード実装済み (`cycle_gate_aggregator.py` L189, L350-352) |
| F2 | CRITICAL | ✅ **確認** | `sac_retrain_scheduler.py` L648 で `obs, _ = env.reset()` → 訓練ウィンドウ先頭の obs。コメント `# 最新 obs で推論` は実態と不一致 |
| F3 | HIGH | ✅ **確認** | scheduler の deploy gate は `gross_roi > 0` のみ。G2 相当の seed stability / worst-window 検証なし |
| F4 | HIGH | ✅ **確認: 367# の FIX-0 は誤り** | `post_fill_30s_pnl` は全日 100% coverage。367# は `post_fill_30s_pnl_bps`（存在しないフィールド名）を参照していたバグ |
| F5 | HIGH | ✅ **確認** | BDK DEADLOCK 2回 (08:35, 09:49)、DD HALT 05:30→cooldown release 07:31→**re-arm 11:31→以降 halt cycle #30 まで完全停止** |
| F6 | HIGH | ✅ **確認** | 直近10件の MTM ログ全てで `spreadPnL` < 0、`btcMTM` >> 0。spread 損失は -717～-851 JPY、MTM は +26,000～+79,000 JPY。利益源は明確に inventory |
| F7-F9 | MEDIUM | ✅ 確認 | train-time wrapper と live hard guard の分離必要性、stale artifact 混在、ranging sell 逆選択 |

### 1.2 367# の誤り訂正

367# §2.4 「post_fill PnL 計測の完全欠落 (0%)」は **フィールド名の参照ミス**。

```python
# 367# で参照したフィールド (存在しない):
post_fill_30s_pnl_bps  → 0/27 (0%)

# 正しいフィールド:
post_fill_30s_pnl      → 27/27 (100%) ✅
post_fill_60s_pnl      → 16/27 (59%)
post_fill_120s_pnl     → 16/27 (59%)
```

367# の FIX-0 は **降格: 再発監視** とする。

### 1.3 F5 時系列 (03-10 の実態)

```
00:00  Day reset (prev_day +78.11bps ← BTCの上げ相場)
03:18  State restored: pnl=-12.46bps, halted=False
04:23  SOFT: -37.56bps <= -30.0 → sell lot 0.5 ← sell が損失主因
05:30  HALT: -50.68bps <= -50.0 after 59 fills ← daily DD hard halt 発動
07:31  Cooldown release (2h経過): lot_scale=0.3 ← 慎重に再開
08:35  BDK DEADLOCK WARNING (10 consecutive gate blocks) ← buy が動けない
09:49  BDK DEADLOCK WARNING 2回目 ← buy の EWMA が threshold 付近
11:31  DD RE-ARM: post-release PnL -31.39bps <= -10.0bps ← lot30%でも-31bps
       → 再 halt (no further release this day)
11:33~ Halt cycle #0, #10, #20, #30... → 当日終了まで完全停止
```

**03-10 の活動時間**: ~00:00-05:30 (5.5h) + 07:31-11:31 (4h) = **約9.5h / 24h = 39.6% duty cycle**
残り60%の時間は**完全に停止**。稼いでいないどころか、機会損失が甚大。

### 1.4 F6 利益構造の核心

```
直近 10 件の MTM ログ:
  spreadPnL:  min=-851, max=-717 JPY (常にマイナス)
  btcMTM:     min=+26,340, max=+79,647 JPY (常にプラス)
  比率:       spread損失 / MTM利益 ≈ 1-3%

結論: Makerとしては赤字。BTCの価格上昇に乗っているだけ。
```

369# の「単なる運の良いガチホトレーダー」は正確な表現。

---

## §2 方向転換の決定

### 2.1 367# からの優先順位変更

```
367# (旧):                          370# (新):
  FIX-0 post_fill PnL 修復 ←最優先    TUNE-4R BDK ranging緩和 ←最優先 ✅ 実施済
  TUNE-4R BDK ranging 閾値            DD re-arm budget緩和 ←最優先 ✅ 実施済
  SG-1 skip gate calibration          DD cooldown短縮 ←最優先 ✅ 実施済
                                      OPS-5 Task Scheduler
                                      SG-1 skip gate calibration  
                                      FIX-0 → 再発監視に降格
```

### 2.2 本ドキュメントで実施した YAML 変更

| 変更 | ファイル | 旧値 | 新値 | 根拠 |
|------|---------|:----:|:----:|------|
| **TUNE-4R** | `fill_test.yaml` L631 | (なし) | `ranging: -1.0` | SDK TUNE-3 成功の横展開。03-10 BDK 21件中 ranging=14件の解消狙い |
| **BDK duration** | `fill_test.yaml` L626 | `1800` (30min) | `900` (15min) | DEADLOCK WARNING 2回/日 → duty cycle 回復 |
| **DD cooldown** | `fill_test.yaml` L699 | `7200` (2h) | `3600` (1h) | halt 時間短縮 (停止=0利益) |
| **DD re-arm** | `fill_test.yaml` L702 | `-10.0` | `-25.0` | lot 30% での -10bps は 4h で再 halt 確定 (03-10 実績: -31bps) |

### 2.3 テスト結果

```
4493 passed, 0 failed, 13 warnings (56.38s)
```

`KNOWN_YAML_OVERRIDES` に `dd_cooldown_rearm_budget_bps`, `buy_dynamic_kill_max_duration_sec` を追加。

---

## §3 368#/369# 受容事項の整理

### 3.1 完全に受容

| # | 指摘 | 対応 |
|:--:|------|------|
| F1 | SAC sidecar 未接続 | **受容**。配線は P0 だが、先に ph2 止血が必要。接続時は `evaluate()` に `sidecar_signal=` を渡す + `run_single_cycle()` に offset 伝搬 |
| F2 | signal が env.reset() で過去を見ている | **受容**。Feature Registry から最新 row で推論するよう変更必要 |
| F4 | 367# FIX-0 は stale | **受容**。フィールド名参照ミスを確認。降格 |
| F5 | BDK + DD が本丸 | **受容**。YAML 変更で即時対応実施済み |
| F6 | spread 赤字 / MTM 依存 | **受容**。SAC を offset 微調整ではなく inventory bias に寄せるべき |
| F7 | train wrapper と live guard の分離 | **受容**。整理表は 368# §4 を採用 |

### 3.2 条件付き受容

| # | 指摘 | 判断 |
|:--:|------|------|
| F3 | scheduler deploy gate 弱すぎ | **受容だが後回し**。まず live 配線 (F1/F2) が先。deploy gate 強化は SAC が実際に signal を出せてから |
| F9 | ranging での SAC は neutral 優勢 | **受容**。ただし SAC 自身が ranging で neutral を学習するのが理想。外部制約で強制するのは最初だけ |

### 3.3 368#/369# の共通見解 (合意事項)

1. **SAC は sidecar として正しい** — direct quote policy は情報粒度不足
2. **今のまま ph3 に進んでも儲からない** — ph2 の BDK/DD が生死の問題
3. **live hard guard は外してはならない** — dynamic_kill, DD, toxicity veto, post_only 等
4. **SAC 出力は inventory bias に寄せるべき** — spreadPnL 赤字 / MTM 黒字の構造から自然

---

## §4 SAC sidecar 接続の修正計画 (F1/F2)

### 4.1 F2 修正: signal 生成の現在市場化

```
現状:
  obs, _ = env.reset()  # → 訓練ウィンドウ先頭
  action = model.predict(obs)

修正案:
  latest_features = feature_registry.get_latest_row()
  obs = env.observation_builder.build_from_features(latest_features)
  action = model.predict(obs)
```

**対象**: `scripts/v460/ml/sac_retrain_scheduler.py` L648-656
**工数**: 1-2h (observation builder の API 確認含む)

### 4.2 F1 修正: live 配線

```python
# orchestrator_mid_cycle.py L135 付近:
sidecar_sig = self._read_sidecar_signal()  # 追加

_gate_result = self._cycle_gate.evaluate(
    ...,
    sidecar_signal=sidecar_sig,  # 追加
)

# orchestrator_mid_cycle.py L383 付近:
record = await self.run_single_cycle(
    ...,
    sidecar_offset_bps=_gate_result.sidecar_offset_bps,  # 追加
)
```

**対象**: `orchestrator_mid_cycle.py`, `fill_cycle_executor.py`
**工数**: 1h (既存 gate ロジック完備のため配線のみ)
**前提**: F2 修正完了後 (signal 品質確保が先)

---

## §5 次のアクション (Profit-First 順)

```
✅ 完了: TUNE-4R (BDK ranging -1.0)
✅ 完了: BDK max_kill_duration 1800→900
✅ 完了: DD cooldown 7200→3600
✅ 完了: DD rearm budget -10→-25

即時:   OPS-5 (Task Scheduler IgnoreNew) — 5min
        fill test 再起動 (新 YAML 適用)

Week 1: F1/F2 SAC sidecar 配線 + signal 現在化
        SG-1 skip gate sell 側 calibration 調査
        BDK staged response 設計 (hard kill → participation縮小先行)

Week 2: SAC inventory bias mode 設計
        K1/K2 再計測
        GATE-1 要否判定
```

---

## §6 コミット履歴

| コミット | 内容 |
|---------|------|
| (本コミット) | 370# TUNE-4R + DD 緩和 + 方向転換 |

---

## §7 367# 訂正事項

367# の以下の記述は**誤り**として訂正する:

1. **§2.4 「post_fill PnL 計測の完全欠落」**: フィールド名参照ミス (`_bps` suffix)。実際は 30s=100%, 60s/120s=59% で coverage あり
2. **§4.1 FIX-0 最優先**: 降格。本丸は BDK deadlock + DD re-arm
3. **§3 「DEGRADED 判定自体の信頼性が疑わしい」**: PnL coverage は正常に機能。DEGRADED は pass_mean_pnl=-1.090bps で実態を反映している
