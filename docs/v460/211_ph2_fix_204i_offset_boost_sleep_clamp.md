# 211# 204# I offset boost + sleep clamp + halt 可視化

> **日付**: 2026-03-02  
> **前提**: 210# (203#/204# 残課題解消) 完了後、204#/205# 全項目監査で検出した残 2 件 + 運用観測で発覚した sleep バグを修正  
> **コミット**: `4edc35679` (offset boost + link 修正), `ab912a1cf` (sleep clamp + halt log), `b56ba1eea` (halt persist interval fix)

---

## 1. 背景

204#/205# ドキュメントの全項目を監査した結果、以下 2 件が未実装/未修正であった:

1. **204# I: Per-fill loss cap — offset boost** (interval 延長 + toxic veto は実装済みだが、3 層目の offset 拡大が未実装)
2. **205# §4.5: 198# broken link** (index.md のリンクが旧ファイル名のまま)

加えて、再起動後の初動監視で以下の運用バグを発見:

3. **`_effective_sleep()` 30 分スリープ**: halt(×5) + soft_drawdown(×3) の乗算で `120 × 5 × 3 = 1800s` となり、209# M4 の `max_cycle_sleep_sec=600` クランプが通常パスのみに適用されていた
4. **halt 中のログ無音**: halt パスに `logger.*` 呼び出しがなく、数時間ログファイルが無更新となる可視性問題

---

## 2. 修正内容

### 204# I: Per-fill loss offset boost

| 項目 | 内容 |
|---|---|
| 問題 | 大損 fill 後の防御が interval 延長 (202# A) + toxic veto (207# §4) の 2 層のみ。offset を広げて次回 fill の利益率を改善する第 3 層が欠如 |
| 設計 | `loss_boost_offset_mult=1.5` — loss_cooldown 発動時に 1 サイクル限定で offset を 1.5 倍に拡大。`_scale_offset_ratio()` 経由で `max_offset_ratio` クランプ下で適用 |
| 修正 | (1) `FillTestConfig.loss_boost_offset_mult: float = 1.5` 追加、(2) `MakerPriceCalculator._loss_boost_mult` slot + `set_loss_boost()` setter、(3) `compute()` に one-shot boost 適用ブロック (FFD boost 前)、(4) orchestrator の loss_cooldown 発動時に `set_loss_boost()` を呼び出し |
| one-shot 設計 | `compute()` で `_loss_boost_mult` 読み取り後に `1.0` にリセット。1 回の offset 計算で消費され、次サイクル以降は通常 offset に復帰 |
| ファイル | `fill_config.py`, `maker_price.py`, `fill_loop_orchestrator.py` |

### 198# broken link 修正

| 項目 | 内容 |
|---|---|
| 問題 | `index.md` の 198# リンクが `198_postmortem_20260301_drawdown_analysis.md` だが、実ファイル名は `198_ph2_rpt_drawdown_postmortem_20260301.md` |
| 修正 | リンク先を実ファイル名に修正 |
| ファイル | `docs/v460/index.md` |

### `_effective_sleep()` max_cycle_sleep_sec clamp

| 項目 | 内容 |
|---|---|
| 問題 | 209# M4 で追加した `max_cycle_sleep_sec=600` は通常サイクル完了パスのみに適用。`_effective_sleep()` は halt(×5) + soft_drawdown(×3) 乗算で `120 × 5 × 3 = 1800s` (30 分) のスリープとなり、bot が長時間無応答 |
| 影響 | halt + soft_drawdown 併発時に 30 分間一切のログ/state 更新/heartbeat 停止 → lock stale 誤判定リスク |
| 修正 | `_effective_sleep()` 内で `min(_raw, max_cycle_sleep_sec)` クランプを適用。`max_cycle_sleep_sec=0` (無効) のケースも考慮 |
| ファイル | `fill_loop_orchestrator.py` |

### halt サイクル可視化ログ

| 項目 | 内容 |
|---|---|
| 問題 | halt パスに `logger.*` 呼び出しがなく、fill record/state 保存も `progress_log_interval=50` 毎のみ。halt 中はログファイルが数時間無更新となり、bot 死活判定が困難 |
| 修正 | halt entering + 10 iter 毎に `logger.info("[daily_drawdown] Halt cycle #N")` を出力 |
| ファイル | `fill_loop_orchestrator.py` |

### halt 中 state/record 保存間隔の適正化

| 項目 | 内容 |
|---|---|
| 問題 | halt 中の state 保存・fill record 記録が通常サイクル用の `progress_log_interval=50` を流用しており、600s × 50 = 8.3 時間に 1 回しか保存されない。halt 中に再起動すると数時間分の halt iteration カウントが巻き戻る |
| 修正 | halt 専用の `_HALT_PERSIST_INTERVAL = 10` を導入し、state 保存・fill record・ログ出力の 3 つを統一。600s × 10 = 約 100 分間隔で保存 |
| 検証 | 04:07:39 に `Halt cycle #10` ログ出力を確認。#0 (02:27:39) から 100 分で正確に動作 |
| ファイル | `fill_loop_orchestrator.py` |

---

## 3. 三層防御の完成

204# I の実装により、大損 fill 後の防御が 3 層完成:

| 層 | 施策 | 効果 | 実装 |
|---|---|---|---|
| 1 | Interval 延長 | `loss_cooldown_interval_mult=2.0` でサイクル間隔を倍増 → 市場安定を待つ | 202# A |
| 2 | Toxic veto | `toxic_fill_veto_threshold_bps=-5.0` 以下の大損 fill 後 3 サイクル同側注文を封鎖 | 207# §4 |
| 3 | **Offset boost** | `loss_boost_offset_mult=1.5` で次回 fill の指値を有利方向に拡大 (1 サイクル限定) | **211#** |

---

## 4. 変更ファイル一覧

| ファイル | 変更量 | 変更内容 |
|---|---|---|
| `scripts/v460/lib/fill_config.py` | +2 | `loss_boost_offset_mult` フィールド |
| `scripts/v460/lib/maker_price.py` | +32 | `_loss_boost_mult` slot/init/setter + compute 適用 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +32/−7 | loss boost 呼び出し + sleep clamp + halt log + halt persist interval |
| `docs/v460/index.md` | +2/−2 | 198# リンク修正 + 211# エントリ追加 |
| `docs/v460/211_ph2_fix_204i_offset_boost_sleep_clamp.md` | +120 | 本ドキュメント |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +35 | 3 新規テスト |

合計: **+223/−9** (6 files), コミット 4 件 (`4edc35679`, `ab912a1cf`, `21d5703eb`, `b56ba1eea`)

---

## 5. テスト

### 新規テスト (3件)

| クラス | テスト数 | 内容 |
|---|---|---|
| `TestLossBoostOffset211` | 3 | 初期 noop (`_loss_boost_mult == 1.0`)、setter 動作、config デフォルト (`1.5`) |

### 結果

- **211# テスト**: 3 passed
- **210# + 211# 結合**: 14 passed
- **v460 全体**: 81 passed (test_168 file), 2542 passed (full suite)

---

## 6. 運用観測で発見した問題と対応

| 問題 | 発見方法 | 原因 | 対応 |
|---|---|---|---|
| 再起動後 11 分間ログ無出力 | `Get-Content -Tail` + state file LastWriteTime 監視 | `_effective_sleep(5.0)` × `soft_dd_mult=3.0` = 1800s sleep。209# M4 clamp が通常パスのみ | `_effective_sleep()` に clamp 追加 (`ab912a1cf`) |
| halt 中のログ完全無音 | 上記調査の過程で発見 | halt パスに `logger.*` 呼び出しなし | entering + 10 iter 毎にログ出力 (`ab912a1cf`) |
| state file が 8.3h 更新されない | halt cycle #10 確認時に state LastWriteTime が entering 時のまま | `progress_log_interval=50` は 600s halt 周期で 8.3h になる | halt 専用 `_HALT_PERSIST_INTERVAL=10` (100 分間隔) に変更 (`b56ba1eea`) |

---

## 7. 残課題

| ID | 重要度 | 内容 | 備考 |
|---|---|---|---|
| **P0-A** | **HIGH** | **Operator Alert Flag (手動リスクフラグ)** | §8 参照。ニュース等で事前把握したリスクを即座に bot に伝達する仕組み。ファイルタッチ型で実装量極小 |
| P1-B | MEDIUM | Micro Circuit Breaker (複数時間軸の価格急変検知) | 5 分/15 分/1h 窓で自動 halt/offset boost |
| P1-C | MEDIUM | Spread Anomaly Detector (spread 急拡大→自動 alert) | 流動性枯渇の最速市場内シグナル |
| 204# K–Q | P2 | σ-linked offset, OFI/PIN, Friday filter 等 | 長期施策 (205# §7) |
| H4 | HIGH | SellDynamicKillManager rolling PnL window 非永続化 | 設計要 (210# §6) |
| spread staleness 60s | LOW | ハードコード → Config 外部化 | 優先度低 |

---

## 8. 地政学イベント対応提案 (P0-A: Operator Alert Flag)

### 背景

2026-02-28 の米・イスラエルによるイラン攻撃 (Operation Epic Fury) で以下の事実が判明:

- **BTC は 攻撃直後に $67K → $63K 急落** (2/28)、ハメネイ師死亡確認後 $68K 反発 (3/1)、現在 $66K 付近
- **ホルムズ海峡封鎖警告**: 世界の石油・ガス輸送の 20% が通過する要衝
- 土曜攻撃→**月曜の伝統市場オープンが真のプライシング**。CME 先物/S&P500 連鎖でBTCに波及

### 問題

現在の bot は「価格が動いた後」にしか反応できない。ニュースで事前にリスクを把握していても、bot に即座に伝える手段がない。

### 提案: ファイルタッチ型 Alert Mode

```
results/v460/fill_test/alert_mode.json の存在をサイクル先頭でチェック
```

**発動例:**
```powershell
# 即座に halt
echo '{"halt": true}' > results/v460/fill_test/alert_mode.json

# 縮小運転 (offset 2倍 + lot 半減 + interval 3倍)
echo '{"offset_mult": 2.0, "lot_mult": 0.5, "interval_mult": 3.0}' > results/v460/fill_test/alert_mode.json

# 解除
del results/v460/fill_test/alert_mode.json
```

**パラメータ:**

| キー | 型 | デフォルト | 効果 |
|---|---|---|---|
| `halt` | bool | false | true で完全停止 (fill record に "operator_halt" 記録) |
| `offset_mult` | float | 1.0 | offset に乗算 (>1.0 でワイドに) |
| `lot_mult` | float | 1.0 | lot に乗算 (<1.0 で縮小) |
| `interval_mult` | float | 1.0 | サイクル間隔に乗算 (>1.0 で低頻度) |
| `reason` | str | "" | ログに記録する理由テキスト |

**設計原則:**
- hot-reload YAML とは独立。YAML 変更は構造的、alert は一時的
- ファイル削除で即復帰 (状態を持たない)
- サイクル先頭の 1 ファイル存在チェック → パフォーマンス影響なし
- 地政学に限らず全種のイベント (取引所メンテ、大口移動等) に汎用利用可能
