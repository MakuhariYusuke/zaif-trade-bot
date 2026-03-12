# 376# Gemini 3.1 Pro: 374#–375# 深層レビューと v456/v454 資産復活による真の Phase 3 統合

| 項目 | 値 |
|---|---|
| 文書番号 | 376# |
| フェーズ | ph3 (Live Injection & Env Parity) |
| 対象 | 370#–375#, 及び埋もれた v454/v456 シリーズ |
| 作成 | Gemini 3.1 Pro |
| 観点 | 収益最優先 / 歴史的遺産の再評価 / Offline-Online 歪みの是正 |

---

## §1 結論と 374# / 375# への最終判決

ユーザーからの要請を受け、**374# の設計案、Codex の 375# レビュー、および現在のコードベースに横たわる「膨大な見落とし」** を精査した。
Codex 氏の 375# における指摘（「3.0bpsはスプレッドよりでかい自殺行為」「LiteTradingEnvは現状不一致」「今はDaily Drawdown起因の停止が最大のブロッカーである」）は**完全に正しい**。374# をそのまま実装すればBotは一瞬で口座を吹き飛ばして死ぬ。

しかし、Codex 氏の 375# もまた、手元の「現在（v460）のコード」に縛られすぎており、**旧バージョン（v454, v456）にすでに完成していた「Phase 3 の理想郷」が放置されているという最大の盲点**を見落としている。

本稿では、AIの幻覚（374#）をバッサリ切り捨てつつ、過去の偉大な遺産（v456）をサルベージし、最短で Live 収益を叩き出すための「本物のアーキテクチャ統合」を提示する。

---

## §2 判明した「絶望的な見落とし（Blind Spots）」

### 1. 【CRITICAL】 M1-M5 特徴量の致命的な Offline/Online 不一致
374# は「M2〜M5 を LiteTradingEnv に追加しよう (Phase 3.2)」と語っているが、実態はもっと悲惨である。
- **Online (Live) の現実**: `configs/v460/fill_test.yaml` では `bayesian_regime: true`, `glft_dynamic_k: true`, `vpin_vol_sync: true` が見事に稼働しており、`run_fill_test.py` は高度なベイズ推定とVPINで動いている。
- **Offline (SAC Train) の現実**: `scripts/v460/build_features.py` には M2-M5 を計算する処理が **1行も存在しない**。`btc_jpy_1m_full_registry_features.parquet` は単なる12次元の 1分足 OHLCV である。
- **結論**: 今の Phase 3 モデルは、Live エンジンが見ている「高度な市場理論」を一切見ずに、1分足の価格だけを見て学習している。これでは「Sidecar」として的外れな指示しか出せない。Phase 3.2 の前に、**「Parquet に M2-M5 を事前計算する ETL の実装」** が絶対の前提となる。

### 2. 【CRITICAL】 存在しない「LiteTradingEnv」と、埋もれた真の神器「`FastIntradayEnvV456`」
374# §4 で `LiteTradingEnv` の拡張が提唱されているが、現在の codebase にそのようなファイルは存在しない。`sac_train.py` は依然として巨大で遅い `HeavyTradingEnv` を使用している。
**ここで最大の盲点**: ワークスペースの `ztb/trading/environment/fast_intraday_env_v456.py` を見よ。
ここには、以下を完全に備えた **超高速・88次元対応の完璧な環境** がすでに完成して放置されている。
- `GroupedFeatureScaler` (次元ごとの最適な正規化)
- `Cyclical Time Features`
- **Lost Alpha Restoration: Ichimoku Calculation for Trend Guidance** (M2/M3の代用となるトレンド追従)
- Numpy vectorized Tracking

**結論**: 374# で想像上のLite環境を作るのではなく、`sac_train.py` のインポートを `FastIntradayEnvV456` に切り替えるだけで、Phase 3 は劇的な速度とスケーリング能力を取り戻すことができる。

### 3. 【HIGH】 v454 の `MarketRegimeClassifier` の忘却
「M2 Bayesian Regime を Parquet にどうやって計算するか？」という問いの答えは、すでに `scripts/v454/verify_regime_distribution.py` や `MarketRegimeClassifier` の中に存在する。
v460 では Live ロジック (`FillTestRegimeDetector`) だけが先行してしまったため、v454 のオフラインレジーム分類資産との連携が完全に断絶している。

---

## §3 374# における「3.0bps 比例ブースト」の本当の恐ろしさ

Codex 375# も指摘したが、 `max_boost_bps = 3.0` は単なる「値が大きすぎる」以上の致命的な副作用を引き起こす。
1. 現在の中央値スプレッドは約 2,282 JPY。
2. 3.0bps のオフセット（約 3,274 JPY）が乗ると、実質的に **クロスした（指値がスプレッドの反対側を突き抜ける）注文** になる。
3. 結果として、`post_only` ガードに直撃して Live サイクル上の `postonly_crossing_skip` が多発し、注文が全く出なくなる（Haltの連発）。
4. 374# 提案の Linear な Proportional Boost 自体は非常に優秀である。ただし、**絶対的な上限を `0.1 bps` または `0.2 bps` にロック** し、影響を「待ち行列上の僅かな優位性（Queue Position Edge）」に留めなければならない。

---

## §4 真の「一番儲かる」Phase 3 統合ロードマップ

過去の vXXX 資産を復活させ、375# の堅実な警告を受け入れた上で、以下のステップで実装を強制する。これ以上の「新規アイディア（Closed-Loop Reward 等）」はゴミ箱へ捨てること。

### Step 1: データの血脈を繋ぐ (Data Parity Restoration)
- `scripts/v460/build_features.py` に、現在 Live で稼働している「M2 Bayesian Regime」「M5 VPIN VolSync」と同等の計算ロジック（あるいは v454 の `MarketRegimeClassifier`）を追加する。
- 12次元でなく、M2-M5 を含んだ Parquet データセットを再生成する。

### Step 2: `FastIntradayEnvV456` の完全復活と配線
- `scripts/v460/lib/tasks/sac_train.py` の 106行目付近、`HeavyTradingEnv` のインポートを破棄し、`FastIntradayEnvV456` に置換する。
- v456 の `GroupedFeatureScaler` と一目均衡表（Lost Alpha）を有効化し、SAC の観測空間（Observation Space）を一気に強靭化する。

### Step 3: Phase 3.1 Proportional Boost の縮小実行 (Live Injection)
- 374# の `compute_sidecar_offset_bps_v2` (Proportional Boost) を `sidecar_types.py` に導入するが、デフォルト値を以下に厳格に固定する。
  ```python
  max_boost_bps = 0.15  # 絶対に 1.0 を超えない
  shaping = "linear"
  dead_zone = 0.10
  ```
- 368# で指摘された「`orchestrator_mid_cycle.py` で `sidecar` が Gate に全く渡されていない」という配線漏れを修正し、この 0.15bps のブーストが実際に指値に反映されることを `fill_test.log` で確認する。

### 統括
現状の Phase 3 (v460) は、過去のバージョン（v454/v456）ですでに解決していた環境構築問題をリセットしてしまい、さらに Live と Offline の特徴量が完全に乖離しているという「基礎的なバグ」を抱えている。
**「AI（SAC）を賢くする前に、AIに過去（v456）の視力を取り戻させ、今現場（Live）で見ているもの（M2-M5）と同じものを見せよ」**。それが出来て初めて、Proportional Boost が利益を生み出す。これこそが、アーキテクチャの真の回復である。