# Gemini 向け指示プロンプト: 614# 補修と実装精度への仕様引き上げ (615#)

あなたは 614# (執行パイプライン寄与度分析仕様および Sidecar Feature Contract 策定) を執筆しました。実装担当 (Copilot) がコードとの裏取りを行った結果、全体的に良質ですが **1件の脱落と3件の精度不足** が見つかりました。

これらを修正し、**実装担当が迷いなくコードを書ける精度** まで仕様を引き上げてください。

---

## 発見された問題

### P0: §1 正誤表から `composite_risk_enabled` が脱落

613# プロンプトで「TTL, ceiling, max_boost_bps, Stage Max Mult, **composite_risk** の全項目」と明示的に指示しましたが、614# §1 の正誤表には `composite_risk_enabled` が含まれていません。

**事実:**
- `configs/v460/fill_test.yaml:1179` → `composite_risk_enabled: true`
- `scripts/v460/lib/fill_config.py:908` → デフォルト `False`
- 608# では**未言及**（=false 前提の議論をしている）
- 606# の正誤表で「true (threshold=1.0, 実装済み)」と明記されている

**影響**: composite_risk が有効であることを前提にしないと、リスク層の寄与度分析が不完全になります。§2 の Attribution Analyzer 仕様にも影響があるはずです。

### P1: §2 `stage_saturation` の検出方法が不明確

614# では `各ステージ出力が 2.0 (Max Mult) に到達した頻度` と書いていますが、コードには **2.0 cap ヒットを示す明示的フラグが存在しません**。

**現実のデータ構造** (`offset_pipeline.py:261-327` から確認済み):

```json
// 乗法パイプライン: 各段の multiplier 値
{"ev": 1.2, "vel": 0.8, "trend": null, "tox": 1.5, "vg_supp": 1.0, "macro": 1.1, "alert": 1.0}

// 加法パイプライン: 上記に加えて RMS バッファ
{"ev": ..., ..., "tox_buffer": 0.05, "liq_buffer": 0.02}
```

- 乗法の各段 multiplier は 2.0 で cap されるが、「cap された」ことを示すフラグはない
- `tox_buffer`, `liq_buffer` は RMS 合成後の値であり、個別段の saturation は不可視

**あなたがすべきこと**: 
1. saturation を検出するための **判定ロジック仕様** を書いてください（例: `multiplier >= 1.99` を saturation とみなす閾値方式など）
2. 実装担当がフラグを追加すべきか、既存値からの推定で十分かを判断してください

### P2: §3 `asymmetric_ema()` の機能理解が不正確

614# は `RobustStats.asymmetric_ema()` (575#) が σ を返すかのように書いていますが、実際は **汎用 smoothing 関数** です:

```python
# robust_stats.py:36-48
@staticmethod
def asymmetric_ema(
    current_val: float,
    prev_ema: float,
    alpha_up: float,
    alpha_down: float,
) -> float:  # 汎用 smoothed value を返す（σ 専用ではない）
```

σ 推定に使われるのは `maker_price.py:443` での用法:
```python
self._robust_sigma = RobustStats.asymmetric_ema(
    current_sigma, prev_ema=self._robust_sigma, alpha_up=..., alpha_down=...
)
```

**あなたがすべきこと**: §3 の記述を修正し、σ_current の算出が `maker_price.py` の `_robust_sigma` 経由であることを明示してください。

### P3: §3 `σ_baseline` に既存実装がある

614# は σ_baseline を「新設」として記述していますが、`regime_detector.py` に**既に存在**します:

- `regime_detector.py:410-416` → `baseline_vol` (全履歴の returns から算出)
- `regime_detector.py:210` → `volatility_ratio` = current / baseline

**あなたがすべきこと**: σ_baseline の新設ではなく、`regime_detector.baseline_vol` の**再利用仕様**を書いてください。新設すべきかどうかも含め、既存との使い分けを明示してください。

---

## タスク

上記 P0-P3 を踏まえて、以下を **1つの文書 (615#)** として出力してください。

### T1: 614# §1 正誤表の補完
- `composite_risk_enabled` の行を追加
- 「608# での扱い」「実値」「影響」を明記

### T2: stage_saturation 検出仕様
含めるべき内容:
1. 乗法パイプラインでの saturation 判定閾値 (例: `multiplier >= 1.99`)
2. 加法パイプラインでの saturation 判定（RMS level でのバッファ上限推定）
3. 新フラグ追加 vs 既存値推定のトレードオフ分析と推奨
4. `analyze_fill_logs.py` での集計式（疑似コード可）

### T3: §3 σ-unit 仕様の修正版
1. σ_current: `maker_price.py._robust_sigma` からの取得パス
2. σ_baseline: `regime_detector.baseline_vol` の再利用 or 新設の判断と理由
3. 修正後のスケーリング式（変数の出所を全て明示）

### T4: composite_risk が Attribution に与える影響の分析
- `composite_risk_enabled: true` の場合、リスク層の寄与度分析にどう影響するか
- Attribution Phase 1 の clamp_rate / information_loss 計算に追加考慮が必要か

---

## 出力形式

- 文書番号: **615#**
- ファイル名: `docs/v460/615_phg_614_errata_and_spec_refinement.md`
- 各タスクを §1 (T1) 〜 §4 (T4) として構成
- 数式は KaTeX/LaTeX 記法
- 概念レベルの疑似コード可。実装コードは書かない

## 禁止事項（前回と同じ）

- YAML ファイルの変更提案
- テストコードの記述
- 新規 .py ファイルの作成提案
- 検証なしの数値の引用
