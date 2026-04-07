# CX4: skip_gate bypass_mode 段階的停止 & threshold 最適化

## 背景

`skip_gate.bypass_mode: true` (686#) で SkipGate はスコア計算・テレメトリのみ行い、ブロックしない observe モード。
CX1 分析で threshold=0.6 が最良 (pass avg=+0.032 vs block avg=-0.172) と判明。
bypass_mode を安全に停止し、段階的にブロッキングを復活させるフレームワークが必要。

## 現在の構成

```yaml
skip_gate:
  bypass_mode: true
  pnl_threshold: -0.5
  max_skip_rate: 0.3
  adaptive_threshold: true
  target_skip_rate_buy: 0.15
  target_skip_rate_sell: 0.20
```

## 要件

### 1. ドライラン CLI

`scripts/v460/analysis/skip_gate_bypass_dryrun.py` を作成:

```bash
.venv/Scripts/python.exe -m scripts.v460.analysis.skip_gate_bypass_dryrun \
  --results-dir results/v460/fill_test \
  --date-from 2026-04-04 --date-to 2026-04-06 \
  --threshold-range 0.1,0.2,0.4,0.6,0.8 \
  --json
```

出力:
- threshold 別の block_count, block_rate, blocked_avg_pnl, passed_avg_pnl, net_pnl_impact
- side 別 (buy/sell) の同上
- regime 別 (ranging/trending_up/trending_down) の同上
- fill rate impact estimation: `1.0 - block_rate * (1 - max_skip_rate)`

### 2. side 別有効化 YAML パス

```yaml
skip_gate:
  bypass_mode_buy: true    # 新規: buy 側は慎重に
  bypass_mode_sell: false   # 新規: sell 側から先行
```

`scripts/v460/lib/skip_gate_evaluator.py` の bypass_mode チェックを side-aware に拡張:
- `bypass_mode: true` で従来互換（両方 bypass）
- `bypass_mode_buy` / `bypass_mode_sell` が設定されていればそちらを優先
- FillTestConfig, parser, validation, hot-reload に追随

### 3. テスト

- `tests/unit/v460/test_710_skip_gate_bypass_dryrun.py`:
  - CLI 引数パース
  - threshold 別集計ロジック
  - side-aware bypass の分岐テスト
- 既存テスト(test_enricher_skip_gate.py) の regression 確認

### 4. 安全制約

- `max_skip_rate` を超える threshold は自動で警告ログ出力
- ドライランで block_rate > 0.5 の threshold は WARN 表示
- adaptive_threshold がアクティブなら threshold は動的上書きされることを CLI に明示

## 影響範囲

| ファイル | 変更 |
|---------|------|
| `scripts/v460/analysis/skip_gate_bypass_dryrun.py` | 新規 |
| `scripts/v460/lib/skip_gate_evaluator.py` | bypass_mode side-aware 拡張 |
| `scripts/v460/lib/fill_config.py` | bypass_mode_buy/sell フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | parser 追随 |
| `scripts/v460/lib/fill_config_validation.py` | validation 追随 |
| `scripts/v460/lib/config_hot_reload.py` | hot-reload 追随 |
| `configs/v460/fill_test.yaml` | bypass_mode_buy/sel 追加（初期値 true） |
| `tests/unit/v460/test_710_skip_gate_bypass_dryrun.py` | 新規テスト |

## 成功基準

1. ドライラン CLI が正常動作し、threshold 別レポートを JSON 出力
2. side-aware bypass が既存テストを破壊しない
3. `bypass_mode: true` のまま `bypass_mode_sell: false` に設定可能
