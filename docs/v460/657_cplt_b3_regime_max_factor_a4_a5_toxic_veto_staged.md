# 657# B-3 regime別max_factor + A-4/A-5 toxic_sell_veto段階化

## 概要

656# 深掘り分析の推奨事項を実装。在庫管理とtoxic flow防御の両面で段階的応答を導入。

| 項目 | 理論根拠 | 変更内容 |
|------|----------|----------|
| **B-3** | Ho-Stoll / Cartea-Jaimungal | trending時inv_skew完全停止→低減max_factorで継続 |
| **A-4** | Glosten-Milgrom staged response | toxic_sell_veto hard skip→offset boost |
| **A-5** | 指数減衰 sticky防止 | 連続veto α^n 減衰 |

## B-3: regime別max_factor

### 問題 (656# §2)
249# regime_gate_enabled=True は trending時にinv_skewを**完全停止**していた。
これにより BTC偏重が解消されず、JPY枯渇→preflight_insufficient→sell_kill の
負のフィードバックループが発生。

### 解決策
trending時は `inv_skew_max_factor_trending=0.15` (ranging時 0.4 の 37.5%) で在庫管理を
継続。完全停止ではなく低減された強度で方向α保全と在庫管理を両立。

### 変更ファイル
- `fill_config.py`: `inv_skew_max_factor_trending: float = 0.15` 追加
- `fill_config_parser.py`: YAML `max_factor_trending` → config フィールドマッピング
- `maker_price.py`: `_apply_inventory_skew()` — binary gate → regime別max_factor選択
  - `inv_skew_regime_gate_enabled=True`: 後方互換で従来の完全停止
  - `inv_skew_regime_gate_enabled=False` (新デフォルト): trending時に低減max_factor使用
- `fill_test.yaml`:
  - `regime_gate_enabled: true → false` (binary gateを廃止)
  - `max_factor_trending: 0.15` 追加

### 期待効果
- trending時もBTC偏重の漸進的解消 (+inventory管理継続)
- JPY枯渇→preflight_insufficientの負ループ軽減
- 方向αの37.5%以上を保全 (low max_factorでtanh曲線が緩やか)

## A-4: toxic_sell_veto ソフト化

### 問題 (656# §1)
654# toxic_sell_veto はall-or-nothing hard veto。条件充足で即スキップし、
連続発火時にsell機会を完全喪失→JPY補充不能。

### 解決策
velocity_skip_as_offset パターンを踏襲。全条件充足時もoffset boost で保守的発注。

- `toxic_sell_veto_as_offset_enabled: true` → ソフトモード
- `toxic_sell_veto_offset_boost_factor: 1.8` → 80% offset boost
- offset pipeline (additive/multiplicative) に `sg_toxic_veto_offset_mult` として配線

### 変更ファイル
- `fill_config.py`: `toxic_sell_veto_as_offset_enabled`, `toxic_sell_veto_offset_boost_factor` 追加
- `fill_config_parser.py`: YAML マッピング追加
- `fill_config_results.py`: `SkipGateResult.toxic_veto_offset_mult` フィールド追加
- `skip_gate_evaluator.py`: toxic_sell_veto ブロック段階化
- `fill_cycle_executor.py`: `sg_toxic_veto_offset_mult` パラメータ配線
- `offset_pipeline.py`: additive pipeline に toxic_veto offset 消費追加
- `multiplicative_pipeline.py`: multiplicative pipeline に toxic_veto offset 消費追加

## A-5: 連続veto時間減衰

### 問題
toxic_sell_veto が連続発火するとsell機会がstickyに喪失。

### 解決策
`_toxic_veto_consecutive_count` カウンタ + α^(n-1) 指数減衰。

- `toxic_sell_veto_decay_alpha: 0.7` → 1回目100%, 2回目70%, 3回目49%...
- `decay < 0.5` の場合: hard modeでもソフトにフォールバック
- `boost_effective = 1.0 + (boost - 1.0) * decay` でboost強度も減衰
- 条件不充足時にカウンタリセット

### YAML設定変更
```yaml
# 657# A-4: toxic_sell_veto ソフト化
toxic_sell_veto_as_offset_enabled: true
toxic_sell_veto_offset_boost_factor: 1.8
# 657# A-5: 連続 veto 時間減衰
toxic_sell_veto_decay_alpha: 0.7
```

## テスト

`tests/unit/v460/test_657_regime_max_factor_and_toxic_veto_offset.py` — 13テスト

| クラス | テスト数 | 内容 |
|--------|----------|------|
| `TestB3RegimeMaxFactor` | 8 | ranging/trending max_factor差分、後方互換、降下互換 |
| `TestA4ToxicVetoAsOffset` | 2 | config/result フィールド存在確認 |
| `TestA5ToxicVetoDecay` | 3 | α^n減衰数式、boost_effective計算、フォールバック |

## 656# 推奨事項のステータス

| 項目 | ステータス | 備考 |
|------|-----------|------|
| B-3 regime別max_factor | ✅ 実装済 | このPR |
| A-4 toxic_sell_veto段階化 | ✅ 実装済 | このPR |
| A-5 連続veto時間減衰 | ✅ 実装済 | このPR |
| C-4 sell_dynamic_kill ARL最適化 | ⏳ 保留 | 50+ RT蓄積後にARL計測が必要 |
| D-2/D-6 preflight balance改善 | ⏳ 保留 | B-3効果の計測を先行 |

## セルフレビュー (658#)

657# コミット後の自己レビューで以下を検出・修正。

### 修正済

| ID | 種別 | 内容 | 対応 |
|----|------|------|------|
| R1 | dead code | `_conditions_met = sum([...])` 未使用変数 (skip_gate_evaluator.py) | 削除 |
| R2 | comment | fill_config.py A-4 コメントが存在しない `soft_max_conditions` を参照 | 修正 |
| R3 | log level | `[inv_skew]` がINFOで毎サイクル出力 | debug化 + 60秒毎INFOサマリ (time throttle) |
| R4 | observability | inv_skew ログに regime 別 max_factor 情報なし | `max_f=` フィールド追加 |
| R5 | _exec_stages | multiplicative_pipeline の stage 記録に toxic_veto 欠落 | 追加 |
| R6 | readability | toxic_sell_veto OR 分岐の可読性 | コメント補強 + `_soft_mode` 変数導入 |

### 要注意事項

- **regime 遷移時の offset 不連続**: ranging→trending 遷移で max_factor が 0.4→0.15 に step change。
  tanh で滑らかだが regime detector の遅延分ジャンプが発生。fill records で計測推奨。
