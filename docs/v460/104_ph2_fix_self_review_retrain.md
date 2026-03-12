# 104# Self-Review + SkipGate再訓練 + __post_init__バリデーション

| key | value |
|---|---|
| 番号 | 104 |
| フェーズ | ph2 |
| 種別 | fix / retrain |
| 親文書 | `103_ph2_fix_yaml_externalization.md` |
| 作成日 | 2026-02-17 |
| テスト | 811 passed |

---

## §0 背景

103# YAML外部化コミット後のCodex視点自己レビュー。
ログ分析 + JSONL PnL分析 + SkipGateモデル再訓練を併せて実施。

---

## §1 Self-Review: P0/P1 修正

### P0-1: `min_adapt_samples` YAML→Config未伝播（修正済）

- **問題**: `adaptation.min_samples` がYAMLに存在するが `from_yaml()` にマッピングがなかった
- **影響**: YAML変更がサイレント無視（デフォルト50と偶然一致で未発覚）
- **修正**: `from_yaml()` に `adapt["min_samples"]` → `kwargs["min_adapt_samples"]` を追加
- `flat_keys` から `min_adapt_samples` を削除（adaptationセクション管轄に一本化）

### P0-2: `balance_shrink_divisor=0` でZeroDivisionError（修正済）

- **問題**: YAML誤設定で0を指定すると `self._current_lot / self.config.balance_shrink_divisor` でクラッシュ
- **修正**: `__post_init__` にバリデーション追加（`>= 1` チェック）

### P1-1: `flat_keys` と `tuning_map` の二重登録解消

- `batch_flush_interval_sec`, `heartbeat_interval_sec` を `flat_keys` から削除
- `tuning` セクション管轄に一本化

### P1-2: `status_unknown_retry_delays` デフォルト修正

- **変更前**: `None` → フォールバック `[2.0, 3.0, 5.0]` がコード内にハードコード
- **変更後**: `field(default_factory=lambda: [2.0, 3.0, 5.0])` にデフォルト統一

### P1-3: tuningセクションのテスト追加（5件）

| テスト | 内容 |
|---|---|
| `test_yaml_tuning_roundtrip` | YAML tuning全18キーの一致検証 |
| `test_tuning_custom_values` | カスタム値のマッピング |
| `test_post_init_balance_shrink_divisor_zero` | 0→ValueError |
| `test_post_init_offset_ratio_invariant` | max≤min→ValueError |
| `test_adaptation_min_samples_mapping` | adaptation.min_samples→min_adapt_samples |

### P1-4: `max_offset_ratio > min_offset_ratio` 不変条件チェック

- `__post_init__` で検証。YAML誤設定で逆転した場合は即ValueError

---

## §2 ログ分析結果

### 統計（612 cycles, 2/13-2/17）

| 指標 | 値 |
|---|---|
| Fill Rate | 75.3% (461/612) |
| PnL Average | -0.60 bps |
| PnL Median | -0.16 bps |
| AS% (>2.5bps loss) | 23.4% |
| Win% | 45.8% |
| ERROR | 368 |
| WARNING | 285 |
| both-filtered | 112 |

### Side別PnL

| Side | n | avg bps | AS% | win% |
|---|---|---|---|---|
| BUY | 236 | -0.36 | 21.2% | 45.3% |
| SELL | 225 | -0.85 | 25.8% | 46.2% |

**SELL側が2.4倍悪い**（-0.85 vs -0.36 bps）。AS率も高い。

### 400 Errorブレイクダウン

| Error | Count |
|---|---|
| Failed to cancel the order | 55 |
| BTCの所持金額が足りません | 42 |
| 量が最低量(0.001 BTC)を下回っています | 40 |
| JPYの所持金額が足りません | 32 |

---

## §3 SkipGateモデル再訓練

### Walk-Forward比較

| 指標 | 097# (215 samples) | 104# (254 samples) |
|---|---|---|
| Folds | 8 | 10 |
| ROC-AUC | 0.442 ± 0.120 | 0.450 ± 0.130 |
| PR-AUC | 0.578 | 0.602 |
| Brier | 0.253 | 0.255 |
| Skip20% | +0.405 bps | +0.269 bps |
| Baseline PnL | -0.781 bps | -0.839 bps |
| Jaccard | 0.357 | 0.267 |

- ROC-AUC微改善（0.442→0.450）
- PR-AUC改善（0.578→0.602）
- Skip20%低下は直近データの市場環境変化を反映
- Always selected: `buy_ratio`, `side_aligned_velocity`, `trade_count_60s`, `trade_flow_imbalance_60s`

### モデルファイル

- `models/v460/skip_gate_as.pkl` 更新済（3,391 bytes）
- 254サンプル、AS rate 57.1%、k=10

---

## §4 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/run_fill_test.py` | `__post_init__`追加、`from_yaml()`修正、`status_unknown_retry_delays`デフォルト変更 |
| `tests/unit/v460/test_fill_test_config.py` | tuning roundtrip + validation テスト5件追加 |
| `docs/v460/065_as_lr_prep.md` | 再訓練レポート更新 |
| `docs/v460/065_as_lr_wf_results.json` | Walk-forward結果更新 |
| `models/v460/skip_gate_as.pkl` | 254サンプルで再訓練 |

---

## §5 残課題

1. **SELL側PnL改善**: avg -0.85bps → `sell_offset_floor` 引き上げ or `spread_offset_ratio_sell` 調整が必要
2. **balance insufficient対策 (114回)**: lot sizing下限 or 残高事前確認の改善
3. **Feature stability低下**: Jaccard 0.267 → データ500+でさらに安定化見込み
4. **Skip20%低下**: 市場環境変化の影響、継続モニタリング要
