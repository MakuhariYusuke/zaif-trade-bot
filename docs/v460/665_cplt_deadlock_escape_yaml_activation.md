# 665# Deadlock Escape YAML 有効化 + Stale Warning 修正

## 概要
664# で実装した Deadlock Escape をプロダクション YAML で有効化し、
YAML パーサーの対応を追加。併せて 330# B4 の古い UserWarning を修正。

コミット: `588197da8`

---

## 1. 変更内容

### 1.1 YAML パーサー対応
**`fill_config_parser.py`**: `tuning_map` に以下を追加:
- `deadlock_escape_threshold` (`int`, default 0)
- `deadlock_escape_spread_mult` (`float`, default 0.5)

### 1.2 プロダクション YAML 有効化
**`configs/v460/fill_test.yaml`** tuning セクション:
```yaml
deadlock_escape_threshold: 20        # ≈40 min で発動 (2 min/cycle × 20)
deadlock_escape_spread_mult: 0.5     # min_spread を半減
```

### 1.3 Stale UserWarning 削除 (330# B4 誤報修正)
**`fill_config_validation.py`**: Kyle λ / Amihud ILLIQ の生死に関する警告を削除。

- **誤**: `imbalance_enabled=False` → depth cache 未更新 → Kyle/Amihud は死コード
- **正**: `_run_pre_order_phase` が `imbalance_enabled` に無関係に
  `_compute_orderbook_imbalance()` を毎サイクル呼出し → depth cache は常に更新
- 警告は 330# B4 時点の誤解に基づく古いもの → 削除

### 1.4 テスト更新
| ファイル | 変更 |
|---|---|
| `test_336_yaml_code_drift_prevention.py` | `KNOWN_YAML_OVERRIDES` に `deadlock_escape_threshold` 追加 |
| `test_346_fill_config_validation.py` | `test_kyle_without_imbalance_warns` → `test_kyle_without_imbalance_no_longer_warns` |

---

## 2. 664# TODO 消化状況

| 項目 | 状態 |
|---|---|
| P1: YAML 有効化 + parser | ✅ 完了 |
| P1: Kyle/Amihud config 整合 | ✅ 訂正 (正常稼働中、stale warning 削除) |
| P2: Stale sidecar offset boost | 未着手 |
| P2: eDRC 独立検証 | 未着手 |
| P3: Reservation price (A-S full) | 未着手 |

---

## 3. 検証結果 (666# 初動分析)

Bot 再起動 (PID 38276, SHA `588197da8`, 11:08 JST) 後 ~7.5h の運用データ:

### 3.1 Deadlock Escape 実績

**5 回発火、5/5 全て fill 成功で解除** (100% 完走率):

| # | Activated | Deactivated | Duration | Counter | Fill Side | 30s PnL |
|---|-----------|-------------|----------|---------|-----------|---------|
| 1 | 12:13:13 | 12:16:46 | 3.5 min | 20 | sell | +2.65 bps |
| 2 | 13:27:05 | 13:37:38 | 10.6 min | 20 | buy | +2.10 bps |
| 3 | 14:04:46 | 14:15:34 | 10.8 min | 20 | sell | +2.56 bps |
| 4 | 15:23:06 | 15:31:02 | 7.9 min | 22 | sell | +2.16 bps |
| 5 | 16:44:52 | 16:58:16 | 13.4 min | 21 | sell | +3.21 bps |

- **Escape fills 合計: +12.68 bps** (avg +2.54 bps/fill)
- 通常 fills (11 件): -8.29 bps → **Escape なしでは赤字セッション**
- Escape 頻度: 約 1 回/時間、平均 9.2 分間持続

### 3.2 全体パフォーマンス

| 指標 | 値 |
|---|---|
| Fill rate | 7.4% (16/201) |
| 30s PnL | +4.39 total, +0.27 avg, 50% WR |
| 60s PnL | +19.16 total, +2.13 avg, 67% WR |
| Side balance | sell 9 / buy 7 (661# 修正効果確認) |

### 3.3 Cancel Reason 分布

| Reason | Count | % |
|--------|-------|---|
| `preflight_insufficient` | 102 | 45% |
| `no_feasible_quote` | 51 | 23% |
| `spread_too_narrow` | 34 | 15% |
| `skip_gate` | 12 | 5% |
| `mcb_halt` | 9 | 4% |

---

## 4. 変更ファイル一覧

| ファイル | 変更種別 |
|---|---|
| `scripts/v460/lib/fill_config_parser.py` | tuning_map 追加 |
| `configs/v460/fill_test.yaml` | deadlock_escape パラメータ追加 |
| `scripts/v460/lib/fill_config_validation.py` | 330# B4 stale warning 削除 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | KNOWN_YAML_OVERRIDES 追加 |
| `tests/unit/v460/test_346_fill_config_validation.py` | テスト名変更 + assert 修正 |
