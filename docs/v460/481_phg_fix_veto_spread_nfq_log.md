# 481# fix: veto_threshold 6→8bps + min_spread 1000→700 + NFQ log reason

> **種別**: fix  
> **日付**: 2026-03-18  
> **コミット**: b5e4fa8d2324  
> **前提**: 480# 検証結果に基づく P0 改修

---

## §1 概要

480# で 478#/479# レビューを検証した結果、fill rate 低迷の真因が特定された。
本セッションではその P0 アクション 3 件を実施する。

---

## §2 変更内容

### §2.1 veto_threshold_bps: 6.0 → 8.0

**ファイル**: [configs/v460/fill_test.yaml](../../configs/v460/fill_test.yaml)

```yaml
# Before
veto_threshold_bps: 6.0

# After
veto_threshold_bps: 8.0  # 481# 6.0→8.0: 49件中41件(84%)が6-8帯,本当にtoxicな8bps超のみvetoする
```

**根拠** (480# §4, §6):
- cross_venue_veto が Buy 側 58 件中 49 件 (83%) を占める最大の抑制要因
- 49 件の `spread_bps` 分布: range 6.00-9.46, **median=7.07bps**, 84% (41件) が 6-8bps 帯
- 6bps 閾値は過剰防御 → 8bps に緩和し、本当に toxic な大乖離のみ veto

**期待効果**: Buy 側 fill rate の改善（49件中41件＝84%が解放対象）

### §2.2 min_spread_jpy: 1000 → 700

**ファイル**: [configs/v460/fill_test.yaml](../../configs/v460/fill_test.yaml)

```yaml
# Before
min_spread_jpy: 1000  # 190# C: 1200→1000

# After
min_spread_jpy: 700   # 481# 1000→700: Phase1段階緩和 (spread<1000が55件/17.1%,700以下は約30件回収見込)
```

**根拠** (480# §1, §6):
- spread<1000 JPY で弾かれているレコードが 55 件 (17.1%)
- 479# は 100 への急進的引き下げを提案したが、リスク管理上 Phase 1 として 700 を選択
- Phase 2 として 500 への追加引き下げを予定

### §2.3 NFQ エスカレーションログに last_reason 追加

**ファイル**: [scripts/v460/lib/fill_cycle_executor.py](../../scripts/v460/lib/fill_cycle_executor.py) L681-686

```python
# Before
f"consecutive infeasible quotes ({side}) — constraint set collapse "
f"(min_spread={self.config.min_spread_jpy}, "
f"sell_max_spread={self.config.sell_max_spread_jpy})"

# After
f"consecutive infeasible quotes ({side}) — "
f"last_reason={e.reason}, "
f"min_spread={self.config.min_spread_jpy}, "
f"sell_max_spread={self.config.sell_max_spread_jpy}"
```

**根拠** (480# §1):
- NFQ 59 件の内訳が 49件=cross_venue_veto / 10件=spread_too_narrow と判明
- 旧ログは `constraint set collapse` としか出力せず真因判別不可
- `last_reason=` タグにより NFQ の背景にある InfeasibleQuoteError の reason が即座に判明

---

## §3 テスト結果

```
57 tests passed (481# 関連)
964 total tests (1 failure: test_stale_lock_reclaimed — Bot PID=68232 起動中のため。無関係)
```

---

## §4 今後の課題

| 優先度 | 内容 | 状態 |
|--------|------|------|
| P1 | 24h 経過後の fill_records で効果検証 | ⏳ 待機 |
| P2 | Phase 2: min_spread_jpy 500 への更なる緩和 | 未実施 |
| P2 | NFQ skip record に cross_venue_lead_lag_spread_bps を付加 | 未実施 |
| P3 | PnL 品質改善: AS率 32.3% の根本対策 | 未実施 |
