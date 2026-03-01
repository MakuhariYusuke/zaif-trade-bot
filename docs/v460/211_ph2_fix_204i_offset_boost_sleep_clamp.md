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
| **P0-A** | **HIGH** | **Operator Alert Flag (手動リスクフラグ)** | §8 参照。ファイルタッチ型で実装量極小 |
| P1-B | MEDIUM | Micro Circuit Breaker (複数時間軸の価格急変検知) | §9 参照。5 分/15 分/1h 窓で自動 halt/offset boost |
| P1-C | MEDIUM | Spread Anomaly Detector (spread 急拡大→自動 alert) | §10 参照。流動性枯渇の最速市場内シグナル |
| P2-D | LOW | 外部シグナルフィード (F&G, Funding Rate, RSS/News) | §11 参照。P0-A の alert_mode.json 自動ライター |
| P2-E | LOW | ニュース特徴量としてのモデル統合 | §12 参照。外部データを observation space に追加 |
| 204# K–Q | P2 | σ-linked offset, OFI/PIN, Friday filter 等 | 長期施策 (205# §7) |
| H4 | HIGH | SellDynamicKillManager rolling PnL window 非永続化 | 設計要 (210# §6) |
| spread staleness 60s | LOW | ハードコード → Config 外部化 | 優先度低 |

---

## 8. 地政学イベント対応提案 (P0-A: Operator Alert Flag)

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

### 背景事実 (本提案の動機)

> 216# F: 214# §4.3 に従い事実叙述を仕様から分離。

2026-02-28 の中東軍事衝突で以下の市場事実が観測された:

- BTC は衝突直後に $67K → $63K 急落 (2/28)、翌日 $68K 反発 (3/1)、以後 $66K 付近
- ホルムズ海峡封鎖警告 — 世界の石油・ガス輸送の 20% が通過する要衝
- 土曜衝突 → 月曜の伝統市場オープンが真のプライシング。CME 先物/S&P500 連鎖で BTC に波及

これらの事実は、bot が「ニュースレベルのリスクを事前に bot に伝える手段がない」
という問題の根拠であり、本提案の直接的な動機となった。

---

## 9. 自動防御提案 (P1-B: Micro Circuit Breaker)

### 概要

複数時間軸の **価格変動率** を監視し、異常急変を検知したら自動で offset boost / halt を発動する。
P0-A が「人間がトリガー」であるのに対し、P1-B は **市場データから自動検知** する防御層。

### 検知ロジック

```
各サイクルで直近の価格変動率 (%) を複数窓で計算:
  - 5 分窓 (短期ショック)
  - 15 分窓 (中期急変)
  - 1 時間窓 (ドリフト加速)

閾値超過で段階的に防御発動
```

**段階:**

| レベル | 条件 | アクション |
|---|---|---|
| **CAUTION** | いずれかの窓で変動率 > 1σ (直近 24h 基準) | ログ警告のみ |
| **WARNING** | 2 窓以上で > 1.5σ、または 1 窓で > 2σ | offset ×1.5 + interval ×2.0 |
| **HALT** | 2 窓以上で > 2σ、または 1 窓で > 3σ | 自動 halt (5 分間クールダウン後に再評価) |

**σ 計算:**
- 直近 24h の各窓変動率の標準偏差を rolling で維持
- `deque(maxlen=窓数分の24hサンプル)` で O(1) 更新
- 初期 warmup 中 (サンプル不足) はデフォルト閾値 (5分: 0.5%, 15分: 1.0%, 1h: 2.0%) を使用

### 既存資産との関係

| 既存 | 差異 |
|---|---|
| `regime_detector.py` | 20 観測 (~40 分) の hysteresis → 急変検知に遅すぎる |
| `circuit_breaker.py` | API 失敗 (CLOSED/OPEN/HALF_OPEN) が対象 → 価格変動は未対象 |
| Volatility Guard (VG) | 4h ATR 基準 → 長期指標。数分のフラッシュクラッシュを検知不能 |

**P1-B は「VG と regime_detector の間を埋める 5分〜1時間の急変検知レイヤー」** として位置づけ。

### 実装見積

| 項目 | 量 |
|---|---|
| 新規ファイル | `scripts/v460/lib/micro_circuit_breaker.py` (~150行) |
| orchestrator 変更 | サイクル先頭で `check()` 呼び出し追加 (~10行) |
| config 追加 | `mcb_enabled`, `mcb_caution_sigma`, `mcb_warning_sigma`, `mcb_halt_sigma`, `mcb_cooldown_sec` |
| テスト | ~80行 (閾値超過/復帰/warmup のユニットテスト) |

---

## 10. 流動性枯渇検知提案 (P1-C: Spread Anomaly Detector)

### 概要

**bid-ask spread の急拡大** は、地政学イベント・取引所障害・大口操作のいずれでも最初に現れる市場内シグナル。
価格が動く前に spread が拡大するケースが多く、**最速の自動検知手段** となる。

### 検知ロジック

```
各サイクルで取得済みの ticker.spread を蓄積:
  - 直近 1h の spread 中央値を基準値とする
  - 現在 spread / 基準値 = spread_ratio

spread_ratio が閾値を超えたらアクション発動
```

**段階:**

| レベル | 条件 (spread_ratio) | アクション |
|---|---|---|
| **NORMAL** | < 2.0 | 通常運転 |
| **WIDE** | 2.0 〜 4.0 | offset を `spread_ratio × 0.5` 倍に拡大 (spread に追従) |
| **DRY** | 4.0 〜 8.0 | offset ×3.0 + interval ×2.0 + lot ×0.5 |
| **FROZEN** | > 8.0 | 自動 halt (30 秒ごとに再評価) |

### 既存資産との関係

| 既存 | 差異 |
|---|---|
| `spread_staleness` (60s) | 古い spread の検知 → spread の「大きさ」は見ていない |
| `maker_price.py` の spread 適応 | offset 計算時に spread を考慮済みだが、**異常値に対する防御** (halt/lot縮小) 機能はない |
| Volatility Guard | ATR ベース → spread 独立 |

**P1-C は「spread が異常に開いた = 流動性が枯渇した」状態を検知し、約定リスクを自動回避する専用レイヤー。**

### 実装見積

| 項目 | 量 |
|---|---|
| 新規ファイル | `scripts/v460/lib/spread_anomaly_detector.py` (~100行) |
| orchestrator 変更 | spread 取得直後に `check(spread)` 呼び出し (~5行) |
| config 追加 | `sad_enabled`, `sad_wide_ratio`, `sad_dry_ratio`, `sad_frozen_ratio`, `sad_baseline_window_sec` |
| テスト | ~60行 |
| 所要データ | 既存 ticker.spread のみ (追加 API 不要) |

### P1-B / P1-C の連携

```
P1-B (価格急変) と P1-C (spread 急拡大) は独立に検知するが、
両方が同時に WARNING 以上 → 即 HALT (AND 条件での escalation)
```

これにより単独での false positive を抑制しつつ、真の急変イベントでは素早く反応できる。

---

## 11. 外部シグナルフィード提案 (P2-D)

### 概要

市場外部のデータソースを定期的に取得し、bot の防御判断に組み込む。
P0-A (手動) / P1-B,C (市場データ自動) の上位レイヤーとして、**より早期の予兆検知** を狙う。

### 候補データソース

| ソース | 取得方法 | 更新頻度 | 検知可能な事象 |
|---|---|---|---|
| **Fear & Greed Index** | [alternative.me API](https://api.alternative.me/fng/) (無料) | 24h | 市場センチメント極端化 (F&G < 20: Extreme Fear) |
| **Funding Rate** | coincheck / binance API | 8h | レバレッジ偏り → 清算カスケードの予兆 |
| **Open Interest** | 取引所 API | 1h | OI 急増 → ボラ拡大の前兆 |
| **Google Trends** | pytrends | 4h | "bitcoin crash" 等の検索急増 |
| **X (Twitter) Sentiment** | 自前クローラー or 有料 API | 15min | インフルエンサーの panic 発信 |
| **RSS/News** | feedparser | 5min | "Iran", "war", "hack", "exchange down" 等のキーワード |

### 設計方針

```python
# 独立プロセス (bot 本体とは別)
class ExternalSignalFeed:
    def poll(self) -> SignalLevel:
        """各ソースを巡回し、最も深刻なレベルを返す"""
        ...
    
    def write_alert(self, level: SignalLevel, reason: str):
        """P0-A の alert_mode.json に書き出す"""
        ...
```

- **P0-A のファイルタッチ方式を再利用**: 外部シグナルフィードが `alert_mode.json` を書き出す → bot 本体は P0-A ロジックのみで対応
- bot 本体とは疎結合 (別プロセス / cron)
- 有料 API に依存しない階層 (F&G + Funding Rate は無料で取得可能)

### 優先度

- **P2 (中期)**: P0-A + P1-B,C が先に完成していれば、外部シグナルは「alert_mode.json の自動ライター」として追加するだけ
- F&G Index のみの最小実装: ~50行、cron 5分ごと

---

## 12. ニュース特徴量としてのモデル統合提案 (P2-E)

### 概要

P2-D の外部シグナルを **ML モデルの特徴量** として取り込み、エージェントの行動決定に直接反映させる。
P2-D が「ルールベースの防御」であるのに対し、P2-E は **学習ベースの適応** を目指す。

### アーキテクチャ

```
            ┌──────────────────────────────┐
            │       Observation Space       │
            │  ┌─────────┐ ┌────────────┐  │
            │  │ 市場特徴 │ │ 外部特徴量 │  │
            │  │ (既存)   │ │ (新規追加) │  │
            │  └─────────┘ └────────────┘  │
            │         ↓         ↓          │
            │  ┌──────────────────────┐    │
            │  │   SAC / PPO Agent    │    │
            │  └──────────────────────┘    │
            │         ↓                    │
            │  action (buy/sell/hold)       │
            └──────────────────────────────┘
```

### 追加特徴量候補

| 特徴量 | 次元 | 正規化 | 根拠 |
|---|---|---|---|
| `fear_greed_normalized` | 1 | [0, 1] (0=Extreme Fear, 1=Extreme Greed) | 低 F&G 時は sell/hold バイアス学習 |
| `funding_rate_zscore` | 1 | z-score (直近 7d 基準) | 異常な funding → 清算リスク |
| `oi_change_pct` | 1 | 直近 4h 変化率 % | OI 急増 = ボラ予兆 |
| `news_sentiment_score` | 1 | [-1, +1] (NLP スコア) | ネガティブニュース → 防御行動 |
| `news_volume_zscore` | 1 | z-score (直近 24h 記事数) | ニュース急増 = イベント発生 |
| `geopolitical_risk_flag` | 1 | {0, 1} | 戦争/制裁等のキーワード検出時 |

### 実装上の課題

| 課題 | 対策案 |
|---|---|
| **特徴量の遅延** | F&G は 24h 更新 → stale な特徴量がノイズに。timestamp embedding で鮮度を学習させる |
| **学習データの希少性** | 地政学イベントは年に数回 → synthetic data augmentation (既存価格データに人工的なショックを注入) |
| **次元の呪い** | 6 次元追加 → 既存 ~60 特徴量の 1 割。影響は限定的だが、ablation study で効果検証必須 |
| **NLP モデルの運用コスト** | 軽量モデル (DistilBERT / TF-IDF + ロジスティック回帰) で news_sentiment を前処理 |
| **再学習頻度** | retrain_scheduler に組み込み。外部特徴量追加時は warm-start (既存重みを保持) |

### ロードマップ

| フェーズ | 内容 | 依存 |
|---|---|---|
| **E-1** | F&G Index のみを observation に追加、ablation study | P2-D の F&G ポーリング実装 |
| **E-2** | Funding Rate + OI を追加 | 取引所 API 拡張 |
| **E-3** | NLP ベースの news_sentiment 導入 | ニュースクローラー + 軽量 NLP モデル |
| **E-4** | geopolitical_risk_flag (keyword matching → ML 分類器) | E-3 のデータパイプライン |

### 優先度

- **P2 (長期)**: P0-A / P1-B,C / P2-D が全て機能してからの最終段階
- モデル再学習を伴うため、**十分な学習データ蓄積** (最低 3ヶ月のイベントデータ) が前提
- E-1 (F&G のみ) は比較的低コストで実験可能 → P2-D 完成後に着手推奨
