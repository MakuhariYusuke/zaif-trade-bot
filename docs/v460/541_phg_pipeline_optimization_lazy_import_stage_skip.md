# 541# Pipeline 最適化: Lazy Import 引き上げ + Pre-order Disabled Stage スキップ

- **日付**: 2026-03-22
- **前提**: 540# §2 の pipeline 分析（14 段中 11 段 identity、spread_adapt 主犯）
- **コミット**: `d376e0511`

---

## §1 Lazy import のモジュールレベル引き上げ

**対象**:
- `scripts/v460/lib/maker_risk_guards.py`: `time`, `datetime`, `timezone`, `pathlib.Path`, `ztb.io.jsonl.append_jsonl` を `_emit_vg_event()` 内の毎回 import からモジュールトップに移動
- `scripts/v460/lib/offset_pipeline.py`: `compute_ev_offset_multiplier`, `MacroTrend`, `current_utc_hour` を各メソッド内の inline import からモジュールトップに移動

**効果**: サイクルあたり 8 回の `importlib` ルックアップを排除

---

## §2 Pre-order disabled stage のスキップ最適化

**対象**: `scripts/v460/lib/maker_price.py` の `compute()` パイプライン

以下 5 段に config gate チェックを追加し、disabled 時は関数呼び出しをスキップし `_record_offset_stage` のみ実行:

| Stage | Config Flag | Default |
|-------|------------|---------|
| kyle | `kyle_lambda_enabled` | False |
| amihud | `amihud_illiq_enabled` | False |
| imb_risk | `imbalance_enabled` | False |
| buy_as_guard | `buy_as_guard_enabled` | False |
| ffd | `fast_fill_defense_enabled` | False |

**効果**: サイクルあたり 5 回の不要な関数呼び出し + 引数セットアップを排除（ステージ記録は保持）

---

## §3 陳腐化テストの修正

- `test_093_side_params.py`: `sa_boost` 変数名テスト → `_apply_spread_adaptive` メソッド存在テストに更新
- `test_168_low_vol_offset_boost.py`: FFD テストに `fast_fill_defense_enabled=True` を追加（スキップ最適化に対応）

---

## §4 spread_adapt 挙動の解明

540# で「主犯」と特定した spread_adapt の実装を確認:

- `narrow_spread_bps: 2.5` — スプレッド < 2.5bps で boost ×2.0（buy）/ ×2.5（sell）
- `wide_spread_bps: 4.5` — スプレッド > 4.5bps で ×0.5（縮小）
- Coincheck の典型スプレッドは 1-2.5bps → **ほぼ全サイクルで narrow boost が発火**
- offset 0.15 × 2.0 = 0.30 → ceiling 0.25 でクランプ → **clamp 飽和の直接原因**
- これは意図的設計（狭スプレッド時の AS リスク軽減）だが、ceiling との組み合わせで情報が失われる

**対策**: 542# で ceiling 0.25→0.30 に引き上げ実施
