# 635# CPLT: 収益性改善に向けたアルファ保全と赤字要因の直接排除 (from 634#)

## 背景と課題認識
634# の分析にて、システムが期待通りの収益を上げていない（儲かっていない）原因が定量的に特定されました。
特に、以下の2点が深刻な「アルファの漏出・毀損」をもたらしています。

1. **Ranging相場でのSell側の慢性的な赤字**:
   - `sell/ranging` は取引機会が多いものの、期待値がマイナス（WR低下・平均PnL悪化）に陥っている。
   - 逆に `buy/ranging` はプラスの期待値を持っており、ここでSellにサイクルを浪費することが機会損失となっている。
2. **`no_feasible_quote` や `preflight_insufficient` による機会損失と連打**:
   - 発注不可能な状態（板の枯渇や残高不足など）にも関わらず、同じサイドばかりを連続して選択し続け、アービトラージの機会を逃している。

## 介入内容
上記の事実から、遠回りな全体巻き戻し（Rollback）ではなく、問題箇所を**直接切除・誘導**する外科的改修を実施しました。

### 1. `no_feasible_quote` での Side 凍結 (Spam防止)
- **対象**: `orchestrator_post_cycle.py` (`_process_cancel`)
- **内容**: キャンセル理由が `no_feasible_quote` だった場合、対象のSideを**2サイクル凍結 (freeze)** させるロジックを追加しました。
- **効果**: 「発注できないのに連続してSellを選び続ける」といった無駄なサイクル消費を強制終了させ、反対売買や次の一手に計算リソースと機会を譲るようになります。

### 2. Ranging相場でのBuy優先アジェンダ (Positive Alphaへの誘導)
- **対象**: `side_selector.py`
- **内容**: レジームが `ranging` の場合、通常の計算で `sell` が選ばれても、**強制的に `buy` へ切り替える**(`ranging_buy_priority`)ロジックを注入しました。
- **効果**: 儲かっていない `sell` を避け、プラスアルファが確認されている `buy` にのみ参加することで、Rangingブレイク時の利益を最大化します。極端な偏りを防ぐため、3回連続で発動した後は通常のSide Selectionに戻ります。

### 3. Ranging×SellのSkipGateペナルティ (Negative Alphaの隔離)
- **対象**: `skip_gate_evaluator.py`, `fill_config.py`, `fill_config_parser.py`, `configs/v460/fill_test.yaml`
- **内容**: 予測ベースのSkipGateにおいて、`sell` 注文かつ `ranging` レジームの場合に限り、Skip閾値のオフセットに対して **`+0.5`のペナルティ**（`sell_ranging_offset`）を課すように改修しました。
- **効果**: Side Selector側でSellが選ばれてしまった場合でも、予測ゲート側で通常より高い厳格さ（勝率見込み）を要求し、微悪な見込みの注文を確実にDrop（スキップ）させます。

### 4. ログ集計スクリプトの拡張
- **対象**: `analyze_fill_logs.py`
- **内容**: キャンセル理由（`cancel_reason`）の集計において、単なる回数だけでなく `Side / Regime` 別の内訳（トップ3のキャンセル理由に対する詳細）を出力するように改修。
- **効果**: 今後、`no_feasible_quote` や特定の不発理由がどのレジーム・サイドに偏っているかを可視化できるようになり、次なる異常の早期発見をサポートします。

## 総括・推敲
今回の改修は、「なぜ儲からないのか」に対する極めて直接的かつ合理的な対処です。
無理なポジションを取ろうとし続けることで発生する間接的ロス（機会損失・スプレッド負け）を、**物理的なSide凍結とアジェンダ上書き**という2段構えで遮断しました。

今後はこの設定で運用を行い、以下の指標を注視します。
1. **Ranging相場でのBuy/Sell比率の改善** (Buyが増加し、Sellが減少しているか)
2. **`no_feasible_quote` の連続発生数の減少**
3. **全体のPnLの底上げ** (特に微小取引によるスプレッド負けの減少)

---

## レビュー (636#)

### 発見・修正したバグ (5件)

| # | 問題 | 深刻度 | 修正内容 |
|---|------|:------:|---------|
| 1 | `skip_gate_evaluator.py` で `getattr(self._config, ...)` 使用 → 255# テスト違反 | **High** | `self._config.skip_gate_sell_ranging_offset` に変更 |
| 2 | `orchestrator_post_cycle.py` で `getattr(self, "_side_selector")` 使用 | Medium | `self._side_selector` に直接アクセス |
| 3 | `side_selector.py` で `ranging_buy_priority_consecutive` が `FillTestConfig` に未定義 → `getattr` fallback | Medium | `ranging_buy_priority_max_consecutive` として FillTestConfig / YAML / parser に正式追加 |
| 4 | `offset_ceil=0.5` で `sell_ranging_offset=0.5` が事実上無効化 (常に clamp 上限到達) | **High** | `offset_ceil` を `0.5→0.8` に引き上げ。penalty が hour_offset と加算されて効果を発揮 |
| 5 | `side_selector.py` の dead code: `self._last_side != "buy"` は `base_side == "sell"` のとき常に False | Low | 条件を `self._consecutive_same_side < max_consecutive` に簡素化 |

### 追加の修正

| # | 問題 | 修正内容 |
|---|------|---------|
| 6 | 630# テスト更新漏れ: `test_fill_quality.py` が VG `velocity_threshold_bps=12.0` を期待 (実際は 630# で 6.0) | テスト値を `6.0` に修正 |
| 7 | `sell_ranging_offset` が YAML に未記載 (code default 依存) | `configs/v460/fill_test.yaml` の skip_gate セクションに追加 |
| 8 | `ranging_buy_priority_max_consecutive` が YAML に未記載 | `smart_side` セクションに追加 |

### 設計レビュー所見

1. **Ranging buy priority のパターン**: buy→(forced buy×3)→sell→buy→... の周期で 5:1 の buy:sell 比率。634# の sell/ranging=-1.93bps 根拠から初期値として妥当だが、市場構造変化時に sell 機会を逃すリスクあり。`ranging_buy_priority_max_consecutive=3` は YAML hot-reload 可能なので運用中調整可
2. **skip_gate penalty + ranging priority + freeze の 3 重防御**: sell/ranging は (1) 80% side selection で回避、(2) 選ばれても +0.5 offset で skip 率大幅増、(3) no_feasible 時は 2 cycle freeze。過剰抑制のリスクあるが sell/ranging の net=-1.93bps からして初期値として合理的
3. **no_feasible_quote freeze の全レジーム適用**: trending_up での sell も freeze される。634# P1 の趣旨 (spam 防止) に基づけば全レジーム適用は妥当。freeze=2 cycles は短くリスク低
4. **`offset_ceil` 引き上げ (0.5→0.8)**: sell_ranging_penalty の有効性確保に必須。非 sell/ranging 時にも hour_offset の厳格化範囲が拡大する副作用あり。問題が出れば sell_ranging_offset 値を下げて対応可

### テスト

- **新規**: `test_634_sell_ranging_suppression.py` (11 tests)
  - Ranging buy priority: 4 tests (基本動作, 連続上限, non-ranging, frozen override)
  - Skip gate penalty: 3 tests (config field, source check, offset_ceil 整合)
  - No feasible freeze: 2 tests (source check, freeze 動作)
  - Config integration: 2 tests (default, YAML round-trip)
- **既存**: 259 passed (test_255, test_336, test_fill_quality, test_254, test_250 含む)