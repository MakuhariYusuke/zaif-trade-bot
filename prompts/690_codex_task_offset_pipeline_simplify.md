# Codex Task: 690# Offset Pipeline 簡素化 (672# P2)

## 目的
672# §6 で「offset pipeline controls nothing」と判明した 9 段乗算 pipeline を
整理・簡素化する。具体的には、実効性が検証されたステージを残し、
ML MI≈0 由来の冗長ステージを統合・簡略化する。

## 背景

### 672# の分析結果
- **§6**: offset [0.09, 0.27] vs [0.27, 0.61] で fill prob 80% → 83%, PnL -0.23 → -0.24
  → offset 値の変化が PnL に有意な影響を与えていない
- **§2**: SkipGate MI = 0.0517 nats (H(PnL) の 2.2%) → ML features の予測力≈0
- **§4**: min_spread_atr が inverted logic → 686# cap_bps で対応済
- **§5**: break-even half-spread = 2.11 bps vs quoted avg 1.11 bps → spread が支配的

### 現行 9 段 pipeline (multiplicative_pipeline.py)
| # | Stage | Source | 実効性 |
|---|---|---|---|
| 1 | EV offset | sg_ev_score | ❌ MI≈0 由来 |
| 2 | Velocity Guard | sg_velocity_bps | ⚠️ 条件付き有効 |
| 3 | Toxic Veto (657#) | sg_toxic_veto | ⚠️ 新規、観察中 |
| 4 | Trend 5s Guard (684#) | sg_trend_5s_guard | ⚠️ 新規、観察中 |
| 5 | Macro Trend | trending_offset_mult | ✅ レジーム連動 |
| 6 | Toxicity Model | toxicity_offset_mult | ⚠️ 条件付き |
| 7 | VG Supplement | velocity_bps fallback | ❌ VG 未発動時のみ |
| 8 | Alert Mode | alert_offset_mult | ✅ 運用必須 |
| 9 | Final Clamp | ceiling | ✅ 安全装置 |

### 簡素化方針 (672# §10 P2)
- **Stage 1 (EV offset)** → disable フラグ追加 (デフォルト=%disabled)。MI≈0 なので offset への寄与なし
- **Stage 7 (VG Supplement)** → Stage 2 (Velocity) に統合
- **Stage 6 (Toxicity)** → 有効性未検証。disable フラグ追加
- **既存ステージの構造は維持** (段階的ロールアウト、A/B テスト可能)
- **_exec_stages JSON ログは全ステージ維持** (分析互換性)

## タスク

### Task 1: ステージ disable フラグ追加

**対象**: `scripts/v460/lib/fill_config.py`, `fill_config_parser.py`, `fill_config_validation.py`

```python
# 新規フィールド
offset_ev_stage_enabled: bool = False       # Stage 1: EV offset (default=off, MI≈0)
offset_toxicity_stage_enabled: bool = True  # Stage 6: Toxicity model
offset_vg_supplement_enabled: bool = False  # Stage 7: VG supplement (統合予定)
```

### Task 2: YAML 設定

**対象**: `configs/v460/fill_test.yaml`

```yaml
# Offset Pipeline Stage Controls (690# / 672# P2)
offset_ev_stage_enabled: false              # MI≈0, offset 寄与なし
offset_toxicity_stage_enabled: true         # 観察継続
offset_vg_supplement_enabled: false         # velocity stage に統合
```

### Task 3: multiplicative_pipeline.py 修正

**対象**: `scripts/v460/lib/multiplicative_pipeline.py`

1. Stage 1 (EV): `offset_ev_stage_enabled=false` 時は `_apply_offset_multiplier` を skip
2. Stage 6 (Toxicity): `offset_toxicity_stage_enabled=false` 時は skip
3. Stage 7 (VG Supplement): `offset_vg_supplement_enabled=false` 時は skip
4. **重要**: `_exec_stages` JSON には disabled ステージも `null` で記録 (分析互換性)
5. disabled ステージの logger.info は出力しない (ログ量削減)

```python
# Stage 1: EV offset
_ev_offset_applied = False
_ev_offset_mult_applied: float | None = None
_ev_score_pretrade: float | None = sg_ev_score
if (
    self.config.offset_ev_stage_enabled          # ← 690# 追加
    and sg_ev_score is not None
    and self.config.skip_gate_ev_as_offset_enabled
    and spread_at_order is not None
    and spread_at_order > 0
    and order_price > 0
):
    # ... 既存ロジック
```

### Task 4: VG Supplement → Velocity 統合

**対象**: `scripts/v460/lib/multiplicative_pipeline.py`

Stage 7 (VG Supplement, L194-214) の条件を Stage 2 (Velocity) に統合:

```python
# Stage 2: Velocity Guard (+ VG supplement 統合)
if self.config.offset_vg_supplement_enabled:
    # 旧 Stage 7 の条件をここに移動 (フォールバックルート)
    pass
# else: legacy VG supplement は Stage 7 で処理 (互換性)
```

**注意**: 実際の統合は `offset_vg_supplement_enabled: false` で
旧 Stage 7 を disable するだけ。コード削除はしない。

### Task 5: Pipeline 統計ログ追加

**対象**: `scripts/v460/lib/multiplicative_pipeline.py`

サイクル N 回ごとに pipeline 統計をサマリーログ出力:

```python
# 690# Pipeline Stage Stats (every 100 cycles)
if cycle_count % 100 == 0:
    logger.info(
        "[690# pipeline_stats] stages: ev=%d/%d vel=%d/%d toxic=%d/%d "
        "trend5s=%d/%d macro=%d/%d tox_model=%d/%d vg_supp=%d/%d alert=%d/%d clamp=%d/%d",
        *stage_counters,
    )
```

これにより「どのステージが実際に offset を変更しているか」を定量観測できる。

### Task 6: テスト

**新規作成**: `tests/unit/v460/test_690_offset_pipeline.py`

1. `offset_ev_stage_enabled=false` → EV ステージが skip される
2. `offset_ev_stage_enabled=true` → 従来動作
3. `offset_toxicity_stage_enabled=false` → toxicity ステージが skip される
4. `offset_vg_supplement_enabled=false` → VG supplement が skip される
5. `_exec_stages` JSON に disabled ステージが null で記録される
6. 全ステージ enabled → 従来の 9 段動作と同一出力
7. pipeline_stats カウンタの正確性
8. hot-reload でステージ enable/disable が即反映
9. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 動作仕様

1. 各ステージの `_enabled` フラグが false → そのステージの `_apply_offset_multiplier` を skip
2. `_exec_stages` JSON には全ステージを含む (disabled は null)
3. ログ出力は enabled ステージのみ
4. pipeline_stats は 100 サイクルごとにサマリー出力
5. 全ステージ enabled → 完全に従来動作 (regression なし)
6. デフォルト: EV=off, VG Supplement=off, Toxicity=on, 他=常時 on

## 受け入れ基準

- [ ] 3 ステージの disable フラグが機能する
- [ ] `_exec_stages` JSON の後方互換性維持
- [ ] disabled ステージの offset 計算を完全に skip
- [ ] pipeline_stats ログが 100 サイクルごとに出力
- [ ] 全ステージ enabled で従来動作と同一
- [ ] 新規テスト 9 件以上、全テスト pass
- [ ] hot-reload 対応

## リスク評価

- **低リスク**: デフォルトは EV/VG off のみ (MI≈0 の裏付けあり)
- **ロールバック**: 各フラグを true にすれば即時従来動作復帰
- **672# 理論根拠**: offset が PnL に有意な影響を与えていない (§6)
- **_exec_stages 互換**: JSON フィールド名・構造は不変
