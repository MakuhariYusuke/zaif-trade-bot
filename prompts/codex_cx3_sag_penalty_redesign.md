# Codex Prompt CX3: SAG penalty 比例化設計

## 背景
`spread_as_guard` (SAG) は現在:
- `spread_threshold_bps: 15.0` → Coincheck の実際の spread median ≈ 2 bps に対して **100% 発動**
- `ev_penalty_bps: 0.5` → **一律 0.5 bps の定数税** (spread 幅に関係なく)
- 総計: 474 fills × 0.5 bps = **237 bps の税収** (Post Apr 4-6)

706# は「定数税」と指摘。707# は「精緻化」を提案。
708# 分析: penalty は spread に反比例すべき（spread が狭い＝AS リスク高い → penalty 大、spread が広い → penalty 小）。

## タスク

### Phase 1: 現行 SAG 実装の理解

1. `spread_as_guard` の実装コードを特定し全体フローを把握:
   - penalty がどこで EV に加算されるか
   - regime_guard_overrides の `spread_as_guard_penalty_multiplier` との相互作用
   - penalty が offset pipeline のどの段階で適用されるか

2. spread 分布の定量分析 (Post Apr 4-6):
   - `spread_bps` フィールドの分布 (mean, median, P10, P90)
   - SAG penalty 後の `entry_gate_ev` 分布

### Phase 2: 比例化設計

3. 設計案 A: threshold 現実化
   - `spread_threshold_bps` を 15→4 に下げ、spread > 4bps では SAG 不発動
   - fill records で spread > 4bps の割合と PnL を検証

4. 設計案 B: spread 反比例 penalty
   - `penalty = base_penalty * (threshold / max(spread, 0.1))`
   - spread が狭いほど penalty が大きくなる
   - spread=1bps なら penalty=2.0bps, spread=2bps なら penalty=1.0bps, spread=4bps なら penalty=0.5bps

5. 設計案 C: penalty を offset 加算に変換
   - EV penalty ではなく offset_ratio への直接加算
   - spread が狭い → offset を上げる → 約定価格が保守的になる
   - 706# の思想（guard → pricing）に沿う

### Phase 3: 実装

6. 選択した設計を実装:
   - `FillTestConfig` に新パラメータ追加
   - SAG 適用コードを修正
   - YAML に新パラメータを追加（デフォルトは現行互換）
   - テストを追加

7. 後方互換性テスト:
   - 既存テストが全パスすることを確認
   - 新パラメータのバリデーション追加

## 成果物
- 設計分析: `docs/v460/708_sag_penalty_redesign.md`
- コード修正 + テスト
- YAML 変更（enabled: false でデフォルト、AB テスト用）
- commit message: `708# CX3: SAG penalty proportional redesign`

## 制約
- `git commit --no-verify` を使用
- テストは `python -m pytest tests/unit/v460/ -x --tb=short` で確認
- 現行動作を壊さない（新パラメータは opt-in）
