# 343# P2 改善: パラメータ有効化 / inv_bypass gradual 化 / EWMA mode

> **種別**: impl  
> **日付**: 2026-03-08  
> **前提**: 343# P1 (`6195f1515`) + 342# 設計・市場理論深掘り調査 (Finding B/D)  
> **テスト**: v460 unit 4227 passed, +21 new test cases  

---

## §1 概要

343# P1 の §5 残課題から P2 タスク 5 件を一括実施。
休眠パラメータ 3 件の有効化、inv_bypass のステップ関数廃止と gradual 化、
DynamicKillManager の EWMA mode 追加を行った。

| # | 施策 | 参照 | 概要 |
|---|------|------|------|
| A | velocity_ema_alpha 有効化 | 343#P1 §5 発見 | `1.0 → 0.3`: bid-ask bounce 抑制 |
| B | ranging_obi_asymmetry_factor 有効化 | 343#P1 §5 発見 | `0.0 → 0.3`: OBI 方向シグナル活用 |
| C | inv_decay_tau_sec 有効化 | 343#P1 §5 発見 | `0.0 → 1800.0`: 古い fill 履歴の時間減衰 |
| D | inv_bypass gradual 化 | 342#B | ステップ関数廃止 + inv_relaxation 拡大 |
| E | DynamicKillManager EWMA mode | 342#D | count-based rolling mean → EWMA 選択可 |

---

## §2 施策詳細

### §2.1 (A) velocity_ema_alpha 有効化

**問題**: `velocity_ema_alpha=1.0` は EMA のフル通過 (平滑化なし) と同義。
Zaif の BTC/JPY 板は薄いため、bid-ask bounce で velocity に短期スパイクが生じ、
velocity_skip が不要な誤発火を起こす。

**変更**: `1.0 → 0.3`

EMA の effective window は $\frac{2}{\alpha} - 1 = \frac{2}{0.3} - 1 \approx 5.7$ tick。
数 tick のバウンスでスパイクが平滑化され、真の方向性 velocity のみが残る。

**変更ファイル**: `fill_test.yaml`, `fill_config.py`

---

### §2.2 (B) ranging_obi_asymmetry_factor 有効化

**問題**: `ranging_obi_asymmetry_factor=0.0` で OBI の方向情報が ranging 市場で
完全に無視されていた。ranging で売り板に偏りがあるにも関わらず、
offset がシンメトリックに計算される。

**変更**: `0.0 → 0.3`

OBI (Order Book Imbalance) が ranging 判定時に方向性を持つ場合、
offset に `obi_direction * 0.3` の非対称バイアスを加算。
マーケットメイカーが板の片側に寄りすぎないようにする。

**変更ファイル**: `fill_test.yaml`, `fill_config.py`

---

### §2.3 (C) inv_decay_tau_sec 有効化

**問題**: `inv_decay_tau_sec=0.0` で在庫ネット不均衡の時間減衰が無効。
1 時間前の 0.01 BTC fill と 1 秒前の fill が同一重みで在庫偏りに寄与し、
stale な在庫履歴が skew に影響を与え続ける。

**変更**: `0.0 → 1800.0` (30 分 τ)

指数減衰: $\text{weight}(t) = e^{-\Delta t / \tau}$

- 30 分経過で重み ≈ $e^{-1} = 0.37$ (63% 減衰)
- 60 分経過で重み ≈ $e^{-2} = 0.14$ (86% 減衰)
- 値は `_decayed_imbalance()` で既実装。`τ=0` では減衰なし (`weight=1.0` 固定)。

**変更ファイル**: `fill_test.yaml`, `fill_config.py`

---

### §2.4 (D) inv_bypass gradual 化 (342#B)

**問題**: `sell_guard_inv_bypass_threshold=0.3` のステップ関数が、
在庫偏り (imbalance) が 0.3 を跨ぐたびに **0↔full bypass** の
不連続ジャンプを起こし、cycle_gate_aggregator のゲート判定がチャタリングする。

$$
\text{bypass}(x) = \begin{cases} 1 & x \geq 0.3 \\ 0 & x < 0.3 \end{cases}
$$

この不連続性はノイズの多い在庫偏り信号で頻繁にフリップし、
MM のポジション管理を不安定にする。

**解決**:

1. **ステップ関数廃止**: `sell_guard_inv_bypass_threshold: 0.3 → 0.0` (bypass 無効化)
2. **gradual 補完**: `sell_dynamic_kill_inv_relaxation_max_bps: 0.3 → 0.5`

inv_relaxation は在庫偏りに **比例** して kill 閾値を緩和する gradual mechanism:

$$
\text{relaxation}(x) = \text{scale} \cdot |x| \cdot \text{max\_bps}
$$

bypass 廃止で失われる保護を、max_bps の拡大 (0.3→0.5) で補填。
buy 側は `buy_guard_inv_bypass_threshold` が存在しないため変更なし。

**Glosten-Milgrom 非対称**: sell 側のみ bypass 廃止→relaxation 強化のため、
sell `max_bps` (0.5) > buy `max_bps` (0.3) の非対称が生じる。
sell はもともと squeeze リスクが高く、in-crypto MM では sell 側の flow quality が
統計的に低い (informed sell > informed buy) ため、この非対称は合理的。

**変更ファイル**: `fill_test.yaml`, `fill_config.py`, `config_hot_reload.py`

---

### §2.5 (E) DynamicKillManager EWMA mode (342#D)

**問題**: `DynamicKillManager` の `is_kill_active()`, `check_kill()`, `assess_toxicity()` は
PnL 履歴の **count-based rolling mean** (直近 N 件の算術平均) で判定する。
この方法はすべての fill に同一重みを与え、regime 遷移時の追従が遅い。
10 連続損失が発生しても、window 内に古い利益 fill があると kill 発動が遅延する。

**解決**: **EWMA (Exponentially Weighted Moving Average)** mode の追加。
J.P.Morgan RiskMetrics (1996) のアプローチに基づく。

$$
\text{EWMA}_t = \alpha \cdot \text{pnl}_t + (1 - \alpha) \cdot \text{EWMA}_{t-1}
$$

**パラメータ**: `ewma_alpha = 0.05` (effective window ≈ $\frac{2}{\alpha} - 1 = 39$ fills)

**実装構造**:

```python
# DynamicKillConfig に追加
ewma_alpha: float = 0.0  # 0.0 = count-based (従来), >0 = EWMA

# DynamicKillManager に追加
_ewma_value: float | None  # None = 未初期化

# track() での更新
if self.config.ewma_alpha > 0:
    if self._ewma_value is None:
        self._ewma_value = pnl  # 初回: seed
    else:
        a = self.config.ewma_alpha
        self._ewma_value = a * pnl + (1 - a) * self._ewma_value

# _get_rolling_mean() ヘルパー
def _get_rolling_mean(self) -> float | None:
    if self.config.ewma_alpha > 0:
        return self._ewma_value  # EWMA
    # fallback: count-based (従来ロジック)
    ...
```

**3 箇所の呼び出し元を統一**:
- `is_kill_active()`: `_get_rolling_mean()` で kill 状態判定
- `check_kill()`: `_get_rolling_mean()` で kill 発動判定
- `assess_toxicity()`: `_get_rolling_mean()` で toxicity 評価

**count-based との比較**:

| 特性 | count-based | EWMA |
|------|------------|------|
| 重み分布 | 均一 (1/N) | 指数減衰 |
| regime 遷移追従 | 遅い (N 件必要) | 速い (α で制御) |
| stale データ影響 | 大 (window 端まで同重み) | 小 (自然減衰) |
| メモリ | O(N) deque | O(1) スカラー |
| hot-reload | window 変更で不連続 | α 変更で自然遷移 |

**FillTestConfig 側**:
- `sell_dynamic_kill_ewma_alpha: float = 0.05`
- `buy_dynamic_kill_ewma_alpha: float = 0.05`
- YAML, parser, hot-reload すべて対応。

**変更ファイル**: `sell_dynamic_kill.py`, `fill_config.py`, `fill_config_parser.py`,
`run_fill_test.py`, `config_hot_reload.py`, `fill_test.yaml`

---

## §3 テスト

### 新規テスト (21 cases)

| クラス | テスト数 | 内容 |
|--------|---------|------|
| `TestVelocityEmaAlphaDefault` | 1 | code default == 0.3 |
| `TestRangingObiAsymmetryDefault` | 1 | code default == 0.3 |
| `TestInvDecayTauDefault` | 1 | code default == 1800.0 |
| `TestInvBypassGradual` | 5 | bypass=0, relaxation=0.5, gate 挙動, DynamicKill no bypass |
| `TestEwmaMode` | 11 | default α, validation, count-based fallback, EWMA 計算, seed, kill 検出, false kill 回避, regime 遷移追従, is_kill_active, assess_toxicity, FillConfig defaults |
| `TestYamlParsing` | 1 | 全 P2 パラメータの YAML→Config 統合テスト |

### 既存テスト修正 (3 files)

| ファイル | 修正内容 | 理由 |
|---------|---------|------|
| `test_169_c1_c3_c4_config.py` | `inv_bypass_threshold: 0.3 → 0.0` | bypass 廃止に追従 |
| `test_229_cleanup_counter_rename.py` | `inv_decay_tau: 0.0 → 1800.0` | default 変更に追従 |
| `test_337_sell_side_countermeasures.py` | `max_bps: 0.3 → 0.5`, `sell ≤ buy → sell ≥ buy` | bypass 廃止→relaxation 拡大に追従 |

### 回帰テスト

- v460 全体: **4227 passed**, 0 failed
- ドリフト防止テスト: **4 passed** (全 P2 変更で code↔YAML 同期済)

---

## §4 変更ファイル一覧

| ファイル | 変更 | 施策 |
|---------|------|------|
| `configs/v460/fill_test.yaml` | A–E | パラメータ有効化 + EWMA α + inv_bypass |
| `scripts/v460/lib/fill_config.py` | A–E | code default 同期 + ewma_alpha fields |
| `scripts/v460/lib/fill_config_parser.py` | E | ewma_alpha パーサー追加 |
| `scripts/v460/run_fill_test.py` | E | DynamicKillConfig 構築に ewma_alpha |
| `scripts/v460/lib/config_hot_reload.py` | D, E | hot-reload 対象追加 |
| `ztb/risk/sell_dynamic_kill.py` | E | EWMA core: config, state, track(), _get_rolling_mean() |
| `tests/unit/v460/test_343_p2_improvements.py` | new | 21 test cases |
| `tests/unit/v460/test_169_c1_c3_c4_config.py` | fix | inv_bypass assertion 修正 |
| `tests/unit/v460/test_229_cleanup_counter_rename.py` | fix | inv_decay_tau assertion 修正 |
| `tests/unit/v460/test_337_sell_side_countermeasures.py` | fix | relaxation max_bps assertion 修正 |

---

## §5 残課題

| 342# | 内容 | 状態 | 備考 |
|------|------|------|------|
| E | sell post_fill_wait_sec 非対称化 | 未着手 | YAML パラメータ追加のみで対応可 |
| F | velocity ルールの AS-aware 化 | 未着手 | skip_gate 制御の複雑さ増加リスク |
| G | preflight 差分対称性 | 未着手 | 低優先度 (情報的) |
| — | skip_gate_score_calibration | 未着手 | Isotonic regression による score 校正 |
