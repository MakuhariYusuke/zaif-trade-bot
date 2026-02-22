# 99# 98#レビュー妥当性評価と実行計画

**Date**: 2026-02-08  
**対象**: 98# Phase B 批判的再分析と収益化再設計ロードマップ  
**手法**: 全指摘をコードレベルで照合、妥当性を5段階で判定  
**大義**: 短期間での高収益性システム — 計器を直し、正しく測り、そして勝つ

---

## 1. 98#指摘の妥当性判定（コード照合結果）

### 1.1 修正A: BUY:SELL完全対称は「実測」ではない

**判定: ✅ 完全に妥当（Critical）**

コード照合結果:

```python
# ztb/trading/environment/heavy_env/core.py L1618-L1631
@property
def buy_count(self) -> int:
    if hasattr(self, 'position_manager') and hasattr(self.position_manager, 'buy_count'):
        return self.position_manager.buy_count
    return int(self.trades_count * 0.5)  # ← 常にここに到達

@property
def sell_count(self) -> int:
    if hasattr(self, 'position_manager') and hasattr(self.position_manager, 'sell_count'):
        return self.position_manager.sell_count
    return int(self.trades_count * 0.5)  # ← 常にここに到達
```

- `PositionManager`に`buy_count`/`sell_count`属性は**存在しない**（ファイル全検索で0件）
- `hasattr(self.position_manager, 'buy_count')` は常に`False`
- `buy_count = sell_count = int(trades_count * 0.5)` が**必ず**返る
- `total_trades=1036` → `int(1036*0.5) = 518` → BUY:SELL = 518:518

**結論**: 97#の「BUY:SELL完全対称」の記述は**測定値ではなく算術結果**。方向性バイアスの有無は**全く不明**。

### 1.2 修正B: P1-1は「純粋PnL」ではない

**判定: ⚠️ 部分的に妥当（High） — 2/4の指摘が正確、2/4は不正確**

#### ✅ 正しい指摘

**① `hold_penalty_multiplier=0.0` はPnL情報消去**:

```python
# reward_calculator.py L1271-L1281
reward = adjusted_pnl * reward_scaling  # PnLベース報酬
if action == ACTION_HOLD:
    reward *= hold_penalty_multiplier   # 0.0 → reward = 0
```

HOLD時（no-op BUYも含む）、PnLが正でも負でも`reward = 0`。「ペナルティ無効」ではなく**「報酬全消去」**。

さらに重要な発見: `HeavyTradingEnv`のstep()で、no-opのBUY/SELLは`effective_action = ACTION_HOLD`にリマップされる。

```python
# core.py L1148-L1151
if actual_action in [ACTION_BUY, ACTION_SELL]:
    if self.position_manager.trades_count == old_trades_count:
        effective_action = ACTION_HOLD  # no-op → HOLD扱い
```

つまり `hold_penalty_multiplier=0.0` は、**実際に約定しなかった全ステップの報酬をゼロにする**。これは50,000ステップ中、約定した~950回以外の**~49,050ステップ分の学習信号を消している**。

**② `position_change > 0.1 → -0.1` ハードコード残存**:

```python
# reward_calculator.py L1289
if position_change > 0.1:
    reward -= 0.1  # 設定で無効化不可
```

これは設定パラメータでは制御できないハードコードペナルティ。P1-1の「全ペナルティ無効」の意図に反する。

#### ❌ 不正確な指摘

**③ `dynamic_reward_shaper`が残存**:

```python
# dynamic_reward_shaper.py コンストラクタ
enabled: bool = False  # デフォルト無効

# reward_calculator.py L385-388
enabled=self.get_setting_bool("dynamic_reward_shaping.enabled", False)

# shape_reward()内部
if not self.enabled:
    return base_reward  # 素通り
```

デフォルト`enabled=False`であり、P1-1で明示指定なしでも**無効のまま**。98#の「動的シェーピングが残存」は不正確。

**④ `signal_integrator`が残存**:

```python
# signal_integrator.py コンストラクタ
enabled: bool = False  # デフォルト無効

# calculate_reward_simple()内
if self.signal_integrator.enabled:  # False → スキップ
```

同様にデフォルト無効。P1-1では動作しない。

#### まとめ: P1-1残存汚染の正確なリスト

| コンポーネント | 98#の主張 | 実際 | 影響 |
|---|---|---|---|
| `hold_penalty_multiplier=0.0` | PnL情報消去 | **✅ 正しい** | **Critical** — HOLD報酬が全てゼロ |
| `position_change > 0.1 → -0.1` | 残存 | **✅ 正しい** | Medium — 約定時のみ適用 |
| `dynamic_reward_shaper` | 残存 | **❌ 不正確** | None — デフォルト無効 |
| `signal_integrator` | 残存 | **❌ 不正確** | None — デフォルト無効 |
| `asymmetric_reward_scaler` | — | 乗数1.0で中立化済み | None |

### 1.3 修正C: 「HOLD 96%」推定は根拠が弱い

**判定: ✅ 妥当（Medium）**

- no-op BUY（Long中にBUY）時、`effective_action = ACTION_HOLD`にリマップ
- `trades_count`はno-opではインクリメントされない
- 「950取引/50Kステップ → HOLD 96%」は不正確
- 実際のエージェントの行動分布（BUY/SELL/HOLD選択率）は不明

ただし、no-opも結果的にHOLDと同じ経済効果（保有継続）であることは事実。「行動選択」と「約定結果」を区別する必要があるという指摘は正しい。

### 1.4 OOS/データリーク懸念

**判定: ✅ 妥当（High）**

```python
# scripts/v459/run_phase45_p1.py L142
"walk_forward": {"enabled": False}

# sac_trainer.py L920-921（eval_data_path未指定時）
eval_df = df  # Use same dataframe
```

- `train_end_index`未指定
- `walk_forward.enabled=False`
- 訓練と評価で同一dfを使用

ただし、**50Kステップ/100K bufferの実験ではOOSの重要性は相対的に低い**。ランダム行動に近い段階でOOSリークの影響は小さいが、Phase C以降で成績が改善し始めた場合には致命的になりうる。

### 1.5 サブプロセスログ破棄

**判定: ✅ 概ね妥当（Medium）**

```python
# run_phase45_p1_subprocess.py L37-42
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
# 成功時: JSON行のみ抽出、他のstdout/stderrは破棄
# 失敗時: stderr末尾20行のみ表示
```

成功時のログ（学習進捗、リワード統計、アクション分布等）が全て失われている。

### 1.6 ファイルパス不備

**判定: ✅ 妥当（Low）**

| 97#記載パス | 実際 |
|---|---|
| `scripts/v459/run_phase45_p1_subprocess.ps1` | `scripts/v459/run_phase45_p1_subprocess.py` |
| `tests/v459/test_gate05_reward_purity.py` | `tests/unit/trading/components/test_gate05_reward_purity.py` |

---

## 2. 98#ロードマップ（Gate C0-C4）に対する評価

### 2.1 Gate C0（測定の正しさ）— ✅ 採用、最優先

| 提案項目 | 妥当性 | 実装判断 |
|---|---|---|
| `buy_count/sell_count`実測化 | ✅ 必須 | `PositionManager`に属性追加 |
| KPI追加（turnover, fee_rate_effective等） | ✅ 有益 | 段階的に追加 |
| seed別stderr/stdout保存 | ✅ 必須 | サブプロセスランナー修正 |
| `train_end_index`未指定時fail-fast | ⚠️ 過剰 | 警告ログで十分（Phase C以降で必須化） |

**補足**: `train_end_index` fail-fastは50Kステップの探索段階では過剰。現時点ではWARNINGログで注意喚起し、Phase Cでの本格実験開始時にfail-fast化するのが合理的。

### 2.2 Gate C1（真PnL基準の再構築）— ✅ 採用

| 提案項目 | 妥当性 | 実装判断 |
|---|---|---|
| `hold_penalty_multiplier=1.0`に修正 | ✅ 必須 | HOLD時もPnL情報を保持 |
| strict PnLパス追加 | ⚠️ 検討 | 最小限: ハードコード`-0.1`を設定値化 |
| テストで純粋性を証明 | ✅ 必須 | Gate 0.5テスト拡張 |

**重要決定**: `hold_penalty_multiplier`の正しい設定:
- `0.0` = HOLD報酬消去（**誤り** — 97#/Phase Bの設定）
- `1.0` = HOLD報酬はPnLそのまま（**正解** — ペナルティなし）
- `< 1.0` = HOLD報酬を減衰（ペナルティとして機能）

### 2.3 Gate C2（コスト圧縮）— ✅ 採用、ただし前提条件あり

Gate C0/C1完了後に実施。98#提案の3軸は妥当:
- `continuous_threshold`: 0.333 / 0.5 / 0.7
- `min_holding_period`: 0 / 15 / 30
- `allow_reverse`: True / False

ただし、**9条件全てを4seed実行すると36実験（~18時間）**。粗選別（2seed）→拡張（4seed）のアプローチが現実的。

### 2.4 Gate C3（ランダム超過証明）— ✅ 採用、Gate C2と並行可能

ベースライン3種は最低限必要:
1. **Random**（同条件で均等確率BUY/SELL/HOLD）
2. **Buy & Hold**（全期間ロング保有）
3. **Simple Momentum**（RSIベース）

これらはSACとは独立に実行できるため、Gate C2と**並行作業**可能。

### 2.5 Gate C4（ピボット判断）— ✅ 概念として採用

具体的な判断は Gate C2/C3の結果次第。以下の代替案は有力:
- 足種変更（5m/15m）: SNR改善の直接効果が期待できる
- ハイブリッド（ルール+RL）: まず手数料に勝てる頻度を確保
- RL役割縮小（方向は教師あり、サイズ/エグジットのみRL）

### 2.6 98#の48時間実行順に対する修正

98#提案: C0→C1→C2→拡張→C3  
**修正案**: C0とC3ベースラインは並行可能。C1は実装が小さいのでC0と同時。

---

## 3. 実行計画

### Phase 1: 計測基盤修正（Gate C0 + C1）— 推定3-4時間

#### Step 1.1: `PositionManager`に`buy_count`/`sell_count`実測追加

```python
# position_manager.py に追加
self.buy_count: int = 0
self.sell_count: int = 0

# execute_action()内、実際に約定した箇所でインクリメント
# reset()でゼロクリア
```

#### Step 1.2: ハードコードペナルティの設定値化

```python
# reward_calculator.py L1289 修正
# BEFORE:
if position_change > 0.1:
    reward -= 0.1
# AFTER:
position_change_penalty = self.get_setting_float("position_change_penalty", 0.0)
position_change_threshold = self.get_setting_float("position_change_threshold", 0.1)
if position_change_penalty > 0 and position_change > position_change_threshold:
    reward -= position_change_penalty
```

#### Step 1.3: サブプロセスランナーのログ保存

```python
# run_phase45_p1_subprocess.py 修正
# seed別ログファイルを保存
log_path = output_dir / f"{experiment_name}_seed{seed}.log"
with open(log_path, 'w') as f:
    f.write(proc.stdout)
    f.write(proc.stderr)
```

#### Step 1.4: 追加KPI計測

`extract_trainer_env_metrics()`に追加:
- `executed_trade_rate`: 約定率（BUY/SELL行動のうち実際に約定した割合）
- `no_op_rate`: no-op率
- `avg_holding_steps`: 平均保有ステップ数

#### Step 1.5: `hold_penalty_multiplier`修正とGate 0.5テスト拡張

P1-1の設定を `hold_penalty_multiplier=1.0` に修正。
テスト追加: 「HOLD時にPnL報酬が保持されること」を検証。

#### Step 1.6: 完了条件テスト

Gate 0.5テスト拡張版で全テスト合格を確認。

### Phase 2: 真PnL基準再実験（Gate C1確認）— 推定5時間

修正後の設定で4seed × 2条件 × 50Kステップ:
- P1-1（修正版）: `use_simple_reward=True`, `hold_penalty_multiplier=1.0`, ハードコードペナルティ=0
- P1-3（現行設定）: デフォルト（再現性確認用）

### Phase 3: ベースライン確立（Gate C3の一部）— 推定2時間

Phase 2と**並行作業**:
- Random agent（同一環境、均等確率）× 4seed
- Buy & Hold（全期間ロング）× データ期間全体
- Simple Momentum（RSI 30/70）× 4seed

### Phase 4: コスト圧縮実験（Gate C2）— 推定8-10時間

Phase 2の結果を踏まえ:
- 粗選別: 3×3=9条件 × 2seed
- 有望2条件に絞り → 4seed × 50Kステップ

### Phase 5: 判定

Gate C2/C3結果に基づき:
- SACがRandomを統計的に上回る → Phase C4不要、Phase 5準備
- SACがRandomと同等以下 → Gate C4（ピボット検討）

---

## 4. Go/No-Go基準（98#準拠 + 修正）

| # | 基準 | 閾値 | 現状 |
|---|---|---|---|
| 1 | 測定健全性 | KPI定義がコードで監査可能 | ❌（buy_countが推定値） |
| 2 | 純粋PnL検証 | PnL-only経路がテストで保証 | ❌（hold_penalty=0で消去） |
| 3 | ランダム超過 | OOSでRandomを有意に上回る | ❌（ベースライン未設定） |
| 4 | 経済性 | 粗利/取引 > 手数料/取引 × 0.5 | ❌（0.40 vs 15.8 JPY） |
| 5 | 再現性 | 4seed以上で結論が維持 | ⚠️（4seed実施済み、有意差なし） |

**1つでも❌ → Phase 5移行はNo-Go**。現状は全項目が❌または⚠️。

---

## 5. 98#への総合評価

### 5.1 評価サマリー

| カテゴリ | 指摘数 | 正確 | 部分的 | 不正確 |
|---|---|---|---|---|
| 数値整合 | 1 | 1 | 0 | 0 |
| 重大修正A（BUY:SELL） | 1 | **1** | 0 | 0 |
| 重大修正B（PnL純粋性） | 4 | **2** | 0 | **2** |
| 重大修正C（HOLD推定） | 1 | **1** | 0 | 0 |
| OOS/データリーク | 3 | **3** | 0 | 0 |
| ログ破棄 | 1 | 0 | **1** | 0 |
| ファイルパス | 2 | **2** | 0 | 0 |
| **合計** | **13** | **10** | **1** | **2** |

**77%が完全正確、15%が部分的に正確。**

### 5.2 不正確な2件の影響評価

98#が不正確に「残存汚染」とした`dynamic_reward_shaper`と`signal_integrator`は、デフォルト`enabled=False`のため**影響ゼロ**。これらは修正不要であり、実行計画への影響はない。

ただし、98#が**正確に指摘した**項目（特にBUY:SELL推定値問題とhold_penalty_multiplier=0.0の報酬消去）は**Critical**であり、これらが修正されない限りPhase Bの結果は信頼できない。

### 5.3 98#ロードマップの評価

Gate C0→C1→C2→C3→C4の順序は**概ね正しい**。「計器を直してから最適化」は正しい判断。

修正点:
- C0とC3ベースラインは並行可能（効率化）
- `train_end_index` fail-fastは段階的導入（現段階はWARNING）
- C2の全組合せ実験は粗選別→拡張の2段階が現実的

### 5.4 「完全なスイングトレードが出来れば苦労はしない」への所見

98#が指摘する「44倍の不足」は現実を正しく捉えている。1取引あたりの粗利0.40 JPY vs 手数料15.8 JPYの構造は、**取引頻度を10分の1にしただけでは解決しない**。

根本的には:
1. **取引頻度削減**（950→100-200）で手数料総額を圧縮
2. **取引精度向上**（粗利/取引を改善）で損益分岐に接近
3. **両方が同時に改善**しないと収益化は困難

この認識の上で、Gate C2（コスト圧縮）は「頻度削減で手数料を下げる」部分のみ検証する。「精度向上」はSACがそもそも学習できるかどうかの問題であり、ステップ数増加やハイパーパラメータ調整の領域。

---

## 6. 直近の実行順序（具体的タスクリスト）

### 即時着手（Phase 1: Gate C0 + C1）

```
[1] PositionManager に buy_count/sell_count 実測属性を追加
[2] ハードコード position_change ペナルティを設定値化
[3] サブプロセスランナーにseed別ログ保存を追加
[4] P1-1設定の hold_penalty_multiplier=0.0 → 1.0 に修正
[5] Gate 0.5テストを拡張（HOLD報酬保持、buy_count実測の検証）
[6] 追加KPI（executed_trade_rate, no_op_rate）を実装
[7] Gate C0/C1完了のテスト実行・確認
```

### Phase 1完了後（Phase 2 + 3 並行）

```
[8] 修正版P1-1/P1-3で4seed×50K再実験（サブプロセス方式、ログ保存あり）
[9] ベースライン3種（Random/B&H/Momentum）の実装・実行
[10] Phase 2/3結果の比較分析
```

### Phase 4（Phase 2/3結果次第）

```
[11] コスト圧縮AB実験（threshold×holding_period、粗選別2seed）
[12] 有望条件の拡張実験（4seed）
[13] Go/No-Go判定
```

---

## 7. ファイル・コード参照一覧

| 対象 | ファイルパス | 該当行 |
|------|------------|--------|
| BUY/SELLカウントのフォールバック | `ztb/trading/environment/heavy_env/core.py` | L1618-L1631 |
| PositionManagerの属性一覧 | `ztb/trading/environment/components/position_manager.py` | L45-L68 |
| no-op処理 | `ztb/trading/environment/components/position_manager.py` | L225-L258 |
| effective_actionリマップ | `ztb/trading/environment/heavy_env/core.py` | L1148-L1151 |
| hold_penalty_multiplier適用 | `ztb/trading/environment/components/calculators/reward_calculator.py` | L1271-L1281 |
| ハードコードペナルティ | `ztb/trading/environment/components/calculators/reward_calculator.py` | L1289 |
| dynamic_reward_shaper初期化 | `ztb/trading/environment/components/calculators/reward_calculator.py` | L385-L388 |
| signal_integrator初期化 | `ztb/trading/environment/components/calculators/reward_calculator.py` | L433-L440 |
| P1-1設定 | `scripts/v459/run_phase45_p1.py` | L70-L98 |
| walk_forward設定 | `scripts/v459/run_phase45_p1.py` | L142 |
| サブプロセスランナー | `scripts/v459/run_phase45_p1_subprocess.py` | L37-L54 |
| 評価データ処理 | `ztb/training/unified_trainer/algorithms/sac_trainer.py` | L912-L922 |
| Gate 0.5テスト | `tests/unit/trading/components/test_gate05_reward_purity.py` | — |
| Phase B結果JSON | `results/phase45_p1_baseline/p1_results_20260206_150903.json` | — |

---

*本文書は98#レビューの妥当性をコードレベルで検証し、修正すべき点と計画を明確化したもの。  
次のアクションはPhase 1（Gate C0 + C1）の実装着手。*
