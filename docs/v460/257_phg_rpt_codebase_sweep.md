# 257# Codebase Sweep — ドキュメント整合・市場理論・再利用・技術的負債

> 256# セルフレビュー完了後の包括的 Sweep  
> 日付: 2026-03-03

---

## DOCUMENT NAME ISSUES

### A. ディスク上に存在するが index.md に未参照 (10件)

| ファイル | 状態 | 対応 |
|---|---|---|
| `002_ph0_rev_001.md` | 欠番宣言済み (セッションノート) | index 欠番一覧に明記 or 削除 |
| `004_ph0_rev_fix.md` | 同上 | 同上 |
| `007_ph1_resp_006.md` | 同上 | 同上 |
| `008_ph1_ver_007.md` | 同上 + naming violation (`ver`) | 同上 |
| `012_ph2_resp_011.md` | 同上 | 同上 |
| `060_ph2_impl_summary.md` | `060_ph2_ml_improvement.md` と番号重複 | 旧ファイル削除 or 統合 |
| `139_ph2_review_fixes_137_138.md` | `139_ph2_fix_review_137_138.md` と重複 | 旧名削除 |
| `151_ph3_plan_dynamic_position_sizer.md` | ph3 → `151_ph2_plan_dynamic_position_sizer.md` (ph2) と重複 | ph3 版削除 |
| `255_codebase_sweep_report.md` | index に `—` (ファイルなし) で記載 | リンク追加 |
| `256_self_review_and_sweep.md` | 同上 | リンク追加 |

### B. 命名規則違反 (主要 37件)

パターン: `NNN_phX_TYPE_description.md`

| ファイル | 違反内容 | 修正案 |
|---|---|---|
| `059_impl_summary.md` | フェーズタグ欠落 | `059_ph2_impl_summary.md` |
| `096_ph2_impl.md` | TYPE 後の description 欠落 | `096_ph2_impl_095_review_response.md` |
| `119_fill_test_161h_analysis.md` | フェーズタグ欠落 | `119_ph2_rpt_fill_test_161h_analysis.md` |
| `189_alt_horizon_macro_integration.md` | フェーズタグ欠落 | `189_ph2_impl_alt_horizon_macro_integration.md` |
| `255_codebase_sweep_report.md` | フェーズ/TYPE 欠落 | `255_phg_rpt_codebase_sweep.md` |
| `256_self_review_and_sweep.md` | 同上 | `256_phg_rpt_self_review_and_sweep.md` |
| (他 31件) | TYPE が規約外 (`codex_review_package`, `dryrun_10h_analysis`, etc.) | P2 — 一括リネーム |

### C. 254#–256# index.md 未リンク

index.md L300-302 に `| 254 | impl | — |` の形式で記載されているが、ファイルリンクなし。
255#, 256# も同様。→ **257# でリンク追加**

---

## MARKET THEORY P1 (implement in 257#)

### MT-1: Avellaneda-Stoikov 予約価格導入

- **ファイル**: `scripts/v460/lib/maker_price.py` L828 `compute()`
- **現状**: `spread * offset_ratio` ヒューリスティック。offset_ratio にレジーム・在庫・VG・ibalance 等の層を重ねている
- **理論**: AS モデルの予約価格: $r(s,q,t) = s - q \cdot \gamma \cdot \sigma^2 \cdot (T-t)$
  - $s$: mid price, $q$: inventory (inv_net_imbalance), $\gamma$: risk aversion, $\sigma$: volatility (RegimeDetector), $T-t$: 残セッション時間
- **適用方法**: `_apply_as_reservation_price()` 新ステージとしてパイプライン挿入。
  既存の inv_skew + regime boosts を1つの理論的フレームワークに統合可能。
  しかし **既存の層が複雑なため、まずは offset 修正値として加算するのが安全**。
- **優先度**: P1 — 収益性直結。offset_ratio の根拠を理論化し、パラメータ自動調整への道を拓く。
- **リスク**: 既存の inv_skew/regime boosts との二重補正。AS 予約価格モードを flag で切替可能にする。

### MT-2: Kyle's Lambda (価格インパクト係数) → offset 動的調整

- **ファイル**: `scripts/v460/lib/maker_price.py`
- **現状**: offset_ratio は static config + レジーム倍率。自己注文の市場インパクトは未考慮。
- **理論**: Kyle (1985): $\Delta P = \lambda \cdot Q$ — 注文サイズに比例する価格インパクト
  - $\lambda = \sigma_V / \sigma_u$ (情報トレーダーVolatility / 非情報Volatility)
- **適用方法**: 直近のOB深度 (`_last_bid_depth`, `_last_ask_depth`) と自己注文サイズから
  簡易 $\lambda_{est} = spread / (2 \cdot depth\_volume)$ を推定し、offset を調整。
- **優先度**: P2 — BTC/JPY maker で 0.001 BTC の注文ではインパクトは軽微。Coincheck の板厚次第。

### MT-3: VPIN 連続リスクモジュレーター化

- **ファイル**: `scripts/v460/lib/maker_price.py` L717 `_apply_volatility_guard()`
- **現状**: VPIN は閾値超過時のみ binary トリガー (boost or not)
- **理論**: Easley, López de Prado, O'Hara (2012): VPIN は情報非対称性の連続指標
- **適用方法**: `vpin_offset_mult = 1.0 + α * (vpin - baseline)` として連続的に offset を調整。
  閾値型のまま残すオプションも YAML config で選択可能に。
- **優先度**: P1 — 既存 VPIN 計算を活用でき、実装量が少ない。

### MT-4: Amihud 非流動性比率 → spread_adaptive 補強

- **ファイル**: `scripts/v460/lib/maker_price.py` L666 `_apply_spread_adaptive()`
- **現状**: `narrow_spread_bps` / `wide_spread_bps` 固定閾値で2段階判定
- **理論**: Amihud (2002): $ILLIQ = \frac{1}{D} \sum \frac{|R_d|}{V_d}$ — 日次リターン/出来高比率
  - 高 ILLIQ = 低流動性 → より保守的な offset が必要
- **適用方法**: SkipGate の `trade_vel_60s` データソースから簡易 Amihud 比率を計算し、
  spread_adaptive の閾値を動的に調整。
- **優先度**: P2 — 計算は軽いが、約定データの蓄積が必要。

### MT-5: Kelly Criterion → lot_sizer 連携

- **ファイル**: `scripts/v460/lib/lot_sizer.py`, `lot_manager.py`
- **現状**: ロットサイズは config の `lot_size` + `confidence_lot_multiplier` で静的/半動的
- **理論**: Kelly (1956): $f^* = \frac{p \cdot b - q}{b}$ — 勝率 $p$、ペイオフ比 $b$
  - _recent_records の win_rate / avg_win:avg_loss から動的 Kelly fraction 計算可能
- **適用方法**: `lot_sizer.kelly_fraction()` メソッドを追加。`confidence_lot_multiplier` の
  upper bound として使用 (half-Kelly が一般的)。
- **優先度**: P2 — _recent_records deque が既に利用可能。ただし短期 win_rate は不安定。

---

## REUSE OPPORTUNITIES

### R-1: `MakerPriceCalculator._scale_offset_ratio()` → 共有ユーティリティ化

- **現在地**: `maker_price.py` L463 (staticmethod)
- **利用者**: maker_price.py 内 7箇所
- **候補利用者**: `cycle_gate_aggregator.py` での offset 判定、`skip_gate_evaluator._ev_weighted_as_offset()`
- **実装**: `scripts/v460/lib/offset_utils.py` に移動、`_scale_offset_ratio()` + `_resolve_trending_boost()` をセットで

### R-2: `velocity_math.compute_instant_velocity_bps()` → skip_gate_evaluator

- **現在地**: `velocity_math.py` L45
- **現利用者**: `maker_price.py`
- **候補利用者**: `skip_gate_evaluator.py` L991–1030 — 現在 adapter 経由で独自に velocity 計算
  trade_vel_60s は別信号だが、instant velocity も gate features に追加可能
- **判定**: P2 — instant と trade_vel_60s は目的が異なるため、安易な統合は不適切

### R-3: `ob_utils.best_bid_ask()` → maker_price.compute() 内の直接アクセス排除

- **現在地**: `ob_utils.py` L46
- **問題**: `maker_price.py` L859-862 は `ob.bids[0][0]`/`ob.asks[0][0]` と直接アクセス。
  `ob_utils.extract_price()` を使うべき。
- **影響**: API レスポンス形式変更時の breakage リスク
- **優先度**: P1 — 統一すれば dual-format 問題がここにも波及しなくなる

### R-4: `FillTestConfig` の `_effective_sell_offset_floor()` → config_access に移動

- **現在地**: `maker_price.py` L353
- **問題**: config 依存の計算だが MakerPriceCalculator のメソッド
- **候補**: `config_access.py` の pure function として抽出し、他モジュール (cycle_gate_aggregator) でも参照可能に
- **優先度**: P3

---

## FUNCTIONALITY P1 (implement in 257#)

### F-1: `run_continuous()` God Method 分割 — **CRITICAL**

- **ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py` L636–L2330
- **現状**: **~1700 行**。`_heartbeat_loop` 定義後の while ループ本体が全て1メソッド内。
- **提案**: 以下の 4 サブメソッドに分割:
  1. `_run_startup()` → L636–L830 (ロック取得・resume・warmup = 195行)
  2. `_run_main_loop()` → L850–L2295 (while ループ本体)
  3. `_run_one_cycle()` → ループ内の1サイクル分 (side決定→注文→post-fill処理)
  4. `_run_teardown()` → L2295–L2330 (heartbeat cancel・結果返却)
- **God method 行数上限 150 行 (maker_price.py 基準)** に合わせるべき

### F-2: `order_monitor._resolve_regime_name()` — Protocol 型安全化

- **ファイル**: `scripts/v460/lib/order_monitor.py` L123–130
- **現状**: `regime_detector: object | None` + `getattr` × 2 + `hasattr` × 1
- **修正**: `RegimeDetectorLike` Protocol を定義し、型を `RegimeDetectorLike | None` に変更
- ```python
  class RegimeDetectorLike(Protocol):
      @property
      def current_regime(self) -> FillTestRegime: ...
  ```

### F-3: `skip_gate_evaluator` adapter Protocol 定義

- **ファイル**: `scripts/v460/lib/skip_gate_evaluator.py` L991, L1015
- **現状**: `getattr(adapter, "get_recent_trades", None)` / `getattr(adapter, "get_orderbook", None)`
- **修正**: evaluate() の `adapter` 引数に `GateAdapterProtocol` を定義:
  ```python
  class GateAdapterProtocol(Protocol):
      async def get_recent_trades(self, symbol: str, ...) -> ...: ...
      async def get_orderbook(self, symbol: str, depth: int = ...) -> ...: ...
  ```

### F-4: `_recent_records` パイプライン — 正常動作確認済み ✅

- `deque(maxlen=200)` (L97) → `append(record)` (L2064) → `_check_regime_stop_conditions` (L552) で消費
- パイプライン: append → check → deque 自動ローテーション
- **256# の修正は正しく機能している**

### F-5: `_apply_regime_boosts()` 153行 → 分割

- **ファイル**: `scripts/v460/lib/maker_price.py` L513
- **提案**: 5つの if ブロックを個別メソッドに:
  1. `_apply_trending_boost()` — L530–L562
  2. `_apply_high_vol_boost()` — L564–L582
  3. `_apply_ranging_discount()` — L584–L627
  4. `_apply_low_vol_boost()` — L629–L656
  5. `_apply_unknown_buy_guard()` — L658–L676
- maker_price.py の GOD OBJECT 警告ボックスに「`_apply_regime_boosts()` 行数上限: 50 行」を追記

---

## P2 (defer)

### P2-1: index.md ファイル名一括リネーム (37件の命名規則違反)

- 多くは歴史的事情 (初期命名 + セッション間断絶) によるもの
- git mv + index.md 更新の一括バッチ処理で対応可能
- リスク: 他ドキュメント内の相互参照リンクが壊れる → grep で検出・修正
- 作業量: 中。ドキュメント品質には寄与するが収益性には無関係。

### P2-2: `evaluate()` in skip_gate_evaluator.py 346行 → 分割

- feature 構築、モデル推論、ev_weighted 判定、OB fetch の 4 フェーズに分割可能
- 現状 `_try_ev_weighted_decision()` は既に分割済みだが、残りの特徴量構築が長い

### P2-3: `evaluate()` in cycle_gate_aggregator.py 211行 → 分割

- Gate 1–9 を個別メソッドに抽出する設計は既に検討済み (192#)
- 各 Gate の判定ロジックが相互依存しているため、単純分割は困難

### P2-4: `update_pnl()` in daily_drawdown_guard.py 104行

- soft/hard/per-side の 3 段階 DD 処理を個別メソッドに分割

### P2-5: `ob_utils.py` getattr — MarketDataAccessor 内部

- L50-51, L122, L133: `getattr(ob, "bids/asks", None)` は設計上必要 (unknown OB type 対応)
- Protocol 化するなら `OrderBookSnapshot` Protocol を adapter パッケージで定義し、
  api adapter が返す型を OrderBookSnapshot に準拠させる

### P2-6: `lock_manager.py` pass in except — 意図的

- L96, 98, 104: stale lock 検出の psutil/OS エラーは意図的 ignore (ドキュメント済み)
- ただし L155 `except Exception:` + debug log — こちらは改善済み (255#)

### P2-7: Kelly Criterion + AS 予約価格 — 理論統合

- MT-1 (AS) と MT-5 (Kelly) を組み合わせた**最適ポジション管理フレームワーク**:
  - AS予約価格 → offset 決定
  - Kelly → lot size 決定
  - 両者は γ (リスク回避パラメータ) を共有可能
- 学術的には Guéant-Lehalle-Fernandez-Tapia (2013) が統合フレームワークを提供

---

## SUMMARY

| カテゴリ | P1 | P2 | 合計 |
|---|---|---|---|
| Document Issues | 3 (リンク追加・重複ファイル削除) | 37 (命名規則修正) | 40 |
| Market Theory | 2 (MT-1 AS予約価格, MT-3 VPIN連続化) | 3 (MT-2/4/5) | 5 |
| Reuse | 1 (R-3 ob_utils 統一) | 3 (R-1/2/4) | 4 |
| Functionality | 3 (F-1 god method, F-2/F-3 Protocol) | 4 (P2-2/3/4/5) | 7 |
| **合計** | **9** | **47** | **56** |

### 最重要 P1 (257# で実装)

1. **F-1**: `run_continuous()` 1700行 → 4分割 (最大の構造問題)
2. **MT-1**: AS予約価格パイプラインステージ追加 (収益性直結)
3. **MT-3**: VPIN連続リスクモジュレーター化 (実装量小)
4. **F-2/F-3**: order_monitor + skip_gate_evaluator Protocol型安全化
5. **F-5**: `_apply_regime_boosts()` 153行 → 5分割
6. **R-3**: maker_price.compute() 内の直接 OB アクセス → ob_utils 統一
7. **Doc**: 254#-256# index.md リンク追加 + 重複ファイル削除
