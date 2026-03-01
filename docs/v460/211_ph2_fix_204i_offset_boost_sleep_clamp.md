# 211# 204# I offset boost + sleep clamp + halt 可視化

> **日付**: 2026-03-02  
> **前提**: 210# (203#/204# 残課題解消) 完了後、204#/205# 全項目監査で検出した残 2 件 + 運用観測で発覚した sleep バグを修正  
> **コミット**: `4edc35679` (offset boost + link 修正), `ab912a1cf` (sleep clamp + halt log)

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
| `scripts/v460/lib/fill_loop_orchestrator.py` | +22/−1 | loss boost 呼び出し + sleep clamp + halt log |
| `docs/v460/index.md` | +1/−1 | 198# リンク修正 |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +35 | 3 新規テスト |

合計: **+92/−2** (5 files), コミット 2 件

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
| 再起動後 11 分間ログ無出力 | `Get-Content -Tail` + state file LastWriteTime 監視 | `_effective_sleep(5.0)` × `soft_dd_mult=3.0` = 1800s sleep。209# M4 clamp が通常パスのみ | `_effective_sleep()` に clamp 追加 |
| halt 中のログ完全無音 | 上記調査の過程で発見 | halt パスに `logger.*` 呼び出しなし | entering + 10 iter 毎にログ出力 |

---

## 7. 残課題

| ID | 重要度 | 内容 | 備考 |
|---|---|---|---|
| 204# K–Q | P2 | σ-linked offset, OFI/PIN, Friday filter 等 | 長期施策 (205# §7) |
| H4 | HIGH | SellDynamicKillManager rolling PnL window 非永続化 | 設計要 (210# §6) |
| spread staleness 60s | LOW | ハードコード → Config 外部化 | 優先度低 |
