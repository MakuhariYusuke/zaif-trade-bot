# 709# 708 Codex 実装レビュー: skip_gate quality / entry_gate / SAG

## 概要

708# で委譲した `CX1/CX2/CX3` を current runtime に照らして再検証し、
「prompt の指摘はどこまで正しいか」「そのまま実装すると危険な箇所はどこか」を整理した。
本稿は実装レビューと再現用メモを兼ねる。

## 実装サマリ

### CX1: skip_gate score 品質分析

追加:
- `scripts/v460/analysis/skip_gate_quality_analysis.py`
- `scripts/v460/analysis/protocols/protocol_708_skip_gate_quality.py`
- `scripts/v460/analysis/analyze_708_skip_gate_quality.py`
- `tests/unit/v460/test_708_skip_gate_quality_protocol.py`

再現コマンド:

```bash
.venv/Scripts/python.exe -m scripts.v460.analysis.analyze_708_skip_gate_quality \
  --results-dir results/v460/fill_test \
  --date-from 2026-04-01 \
  --date-to 2026-04-06 \
  --json
```

主結果:
- pre count: `822`
- post count: `962`
- post mean: `-0.0283`
- post bimodality coefficient: `0.5595`
- forced_pass 主因: `skip_rate_limit(35%>30%)`
- threshold 候補: `0.4-0.6` が `0.1` より良い

判断:
- prompt の「二峰性検証」は正しかった
- ただし結論は「selector 崩壊」単独ではなく、`forced_pass` と feature shift の複合だった

### CX2: entry_gate redesign

更新:
- `scripts/v460/lib/entry_gate_guard.py`
- `tests/unit/v460/test_690_entry_gate_guard.py`
- `tests/unit/v460/test_704_entry_gate_side_aware.py`

判断:
- prompt の dead code 指摘は正しかった
- ただし stale guard を弱める変更は避けるべきだった
- 実装では
  1. stale check を維持
  2. buy mild-negative suppress を auto-disable より前へ
  3. sell は entry-gate 本判定に委譲
  の順に整理した

### CX3: SAG penalty redesign

更新:
- `scripts/v460/lib/fill_config.py`
- `scripts/v460/lib/fill_config_parser.py`
- `scripts/v460/lib/fill_config_validation.py`
- `scripts/v460/lib/config_hot_reload.py`
- `scripts/v460/lib/entry_gate_adjustments.py`
- `configs/v460/fill_test.yaml`
- `tests/unit/v460/test_695_spread_as_guard.py`
- `tests/unit/v460/test_169_config_hot_reload.py`
- `tests/unit/v460/test_346_fill_config_validation.py`

判断:
- prompt の「定数税」診断は正しい
- ただし offset 直加算へ一気に切り替えるより、まず現行 EV path で opt-in 実装する方が安全
- 実装したもの:
  - `redesign_enabled`
  - `active_threshold_bps`
  - `inverse_penalty_reference_bps`
  - `inverse_penalty_floor_bps`
  - `inverse_penalty_cap_bps`

## hidden task

1. parser / validation / hot-reload の同時追随
- runtime 変更だけでは drift する

2. analysis protocol 登録
- 単発 script だけだと再利用性が低いので `protocols/__init__.py` へ登録

3. heavy test への横展開
- `test_enricher_skip_gate.py` の実データ sample 窓を縮小
- 708 作業と合わせて heavy subset の固定費を下げた

## 現時点の推奨

1. `skip_gate_threshold`
- runtime 変更より先に `0.4 / 0.6` の counterfactual 比較継続

2. `entry_gate_buy_suppress_ev_threshold`
- `-0.5` は据え置きにせず候補比較が必要
- ただし runtime デフォルトは即変更しない

3. `SAG redesign`
- opt-in のまま AB 実験へ進める
- `spread_threshold_bps` をいきなり大きく動かすより、まず redesign flag を切って比較する

## 回帰確認

- focused subset:
  - `140 passed in 2.46s`
- heavy subset:
  - `360 passed, 1 skipped, 5 warnings`
- `py_compile`:
  - 対象ファイル通過

## 所見

708# の prompt は問題提起としては有効だったが、
実装は current runtime 契約に沿って変形する必要があった。
特に `entry_gate` と `SAG` は、「正しい指摘」と「そのまま入れると危ない変更」が混ざっていた。
今回の対応は、その差を吸収した安全側の実装といえる。
