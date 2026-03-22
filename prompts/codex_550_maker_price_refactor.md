# Codex Task: MakerPrice Stage Orchestration 明確化 + 周辺リファクタリング

> 対象: 521# (master architecture carry-forward) の `maker_price.py` 設計メモに基づく段階的改善
> 前提: AGENTS.md の規約に従うこと。`git add .` 禁止、個別指定。コミットは `--no-verify`。

---

## 背景

### 現状

`scripts/v460/lib/maker_price.py` は **1,231行 / 48メソッド / 44インスタンス変数** の God Object に近い状態。
既に以下の pure helper は `ztb/trading/pricing/` に抽出済み:

| 抽出済モジュール | 内容 |
|---|---|
| `inventory_math.py` | 在庫減衰・インバランス計算 |
| `offset_math.py` | offset ratio 算術 |
| `spread_adaptive.py` | spread adaptive ステージ |
| `price_finalization.py` | spread guard / finalization |
| `offset_ceiling.py` | ceiling clamp |
| `stage_tracking.py` | offset stage 記録 |
| `boost_math.py` | boost 計算 |
| `ofi_lite.py` | OFI-Lite 計算 |
| `contracts.py` | 型契約 |
| `offset_amount.py` | offset amount 計算 |

3つの Mixin にも責務分割が進んでいる:
- `maker_risk_guards.py` (574行) — VG, Cross-Venue, AS guard
- `maker_microstructure.py` (364行) — Amihud, Kyle λ, OBI
- `maker_regime_boost.py` (338行) — レジーム boost、sell hour boost

### 521# の設計方針 (遵守すること)

> - state object 化を急がず、stage orchestration を明示化する
> - pure helper は引き続き `ztb.trading.pricing.*` へ寄せる
> - `compute()` の public/inspection 契約は壊さない

### 547# §5.3 の提案 (参考)

> `MakerPrice` から microstructure state/telemetry を分離する設計メモを先に作る
> - pricing core
> - microstructure state
> - telemetry recorder
> の 3 分割くらいのメモは先に作っておくと後が楽

---

## Task 1: MakerPrice の State 分類とロードマップ設計メモ

### やること

`maker_price.py` の 44 インスタンス変数を以下の3カテゴリに **分類** する設計メモを `docs/v460/` に作成する (ドキュメント番号は `docs/v460/` 内の最大番号 + 1 を採番):

1. **Pricing Core State** — 価格計算に直接必要な状態 (config, base_offset, regime_detector 等)
2. **Microstructure Cache** — 市場データキャッシュ (OB snapshot, OFI, spread, mid_price, VPIN, σ 等)
3. **Telemetry / Diagnostic** — 観測・ログ用の最終値 (last_vg_*, last_offset_stages, last_sigma 等)

各変数をカテゴリに振り分け、**将来の分割が可能な境界線** を明示する。

### 制約
- **実装分割はしない** (設計メモのみ)
- 521# の「state object 化を急がず」の方針は維持
- `compute()` の public 契約は変更しない
- 現在の Mixin 構造 (RiskGuardsMixin, MicrostructureMixin, RegimeBoostMixin) との整合性を検証する

---

## Task 2: `compute()` の stage 実行シーケンス文書化

### やること

`compute()` メソッド (L868~) の **stage 実行順序** を明示的に文書化する:

1. 各 stage の名前、呼び出し順、入力・出力
2. stage 間の依存関係 (どの stage が前の stage の結果に依存するか)
3. 「独立」な stage と「依存」な stage の区別

これを Task 1 の設計メモに含めるか、既存の 521# を更新する形で記録する。

---

## Task 3: ToxicityAssessment / ToxicityLevel の型切り出し

### やること

`ztb/risk/sell_dynamic_kill.py` 内の以下2型を **shared module** に移動:

```python
class ToxicityLevel(enum.Enum): ...    # L56-66
class ToxicityAssessment: ...          # L73-88
```

移動先候補: `ztb/risk/toxicity_types.py` (新規)

### 理由

547# F5:
> `ToxicityAssessment` / `ToxicityLevel` 型は `sell_dynamic_kill` に依存しており、完全独立ではない

これらの型は `ztb/risk/toxicity_budget.py` からもインポートされており、`sell_dynamic_kill` が型定義の SSOT になるのは不自然。

### 制約
- 既存の全 import を更新する (`from ztb.risk.sell_dynamic_kill import ToxicityAssessment` → `from ztb.risk.toxicity_types import ...`)
- `sell_dynamic_kill.py` には re-export を残す (互換性維持):
  ```python
  from ztb.risk.toxicity_types import ToxicityAssessment, ToxicityLevel  # noqa: F401
  ```
- 全テスト通過を確認: `python -m pytest tests/ -x --tb=short`

### 影響範囲の確認コマンド
```powershell
Select-String -Path (Get-ChildItem -Recurse -Filter "*.py" | Select-Object -ExpandProperty FullName) -Pattern "from ztb\.risk\.sell_dynamic_kill import.*Toxicity"
```

---

## Task 4: telemetry schema version の導入

### やること

547# §5.2 の提案:
> `offset_stages_schema_version` 的なものを入れる

`MakerPriceCalculator.compute()` が返す `offset_stages` JSON に、schema version フィールドを追加する。

```python
# stage_tracking.py 等の適切な場所に
OFFSET_STAGES_SCHEMA_VERSION = "549"  # 最後に offset_stages 構造を変更したドキュメント#
```

これにより mixed-SHA 集計時に「キーがある run と無い run」を区別可能にする。

### 制約
- `MakerPriceResult` や `last_offset_stages` の既存フィールドを壊さない
- version は文字列でよい（数値よりフレキシブル）

---

## 実施順序

1. Task 3 (ToxicityTypes 切り出し) — 最もスコープが小さく、テスト可能
2. Task 4 (schema version) — 追加のみ、既存破壊なし
3. Task 1+2 (設計メモ) — ドキュメント作業

各 Task は個別にコミットすること。

---

## テスト

- 各 Task 後にテストを実行: `python -m pytest tests/ -x --tb=short`
- 既存テスト 131+ を壊さないこと
- Task 3 では `ToxicityAssessment` / `ToxicityLevel` のインポート元変更のリグレッションに注意

---

## 参照ドキュメント

- `docs/v460/521_phg_master_deferred_and_architecture_carryforward.md` — master architecture
- `docs/v460/547_phg_rev_540_546_spec_dilemmas_and_multifaceted_recommendations.md` — §5.3, F5, F8
- `docs/v460/549_log_analysis_ewma_clamp_sidecar_as_pattern.md` — 直前の分析結果
- `AGENTS.md` — コーディング規約、git 運用ルール
