# 406# セルフレビュー: 400#–405# 深堀り分析

> **作成日**: 2026-03-13
> **対象**: 400# (194e30a2e) → 405# (c6ded4a96) + 未コミット 401#/404#
> **分類**: rev (自己監査・深堀り分析)

---

## 0. 総評

400#–405# のセッション群は、**報酬関数の構造的欠陥修正** (400#–401#) と **offset パイプラインの中間キャップデッドロック解消** (402#–405#) の2軸で進行した。
成果としては初の G2+G3 同時 PASS (401#) と sell-side ブースト有効化 (405#) が挙げられるが、**git 管理の破綻** (session037 による 75 ファイル削除→ゴーストファイル化) と **未コミット成果物の散逸** という深刻なプロセス問題が存在する。

**結論**: コード品質は高いが、インフラ（git追跡）と成果物管理に構造的リスクがある。

---

## 1. コミット履歴と未コミット成果物の全容

### 1.1 コミット済み (400#→HEAD)

| SHA | タイトル | 変更ファイル | 品質 |
|-----|---------|------------|------|
| `194e30a2e` | 400# Reward Clean: scale_adjustment fix + balance_shaping BUG FIX | config, reward_calculator, env, yaml, docs | ✅ CRITICAL BUG FIX |
| `3d14f93ce` | 402# 時間帯パフォーマンス根本原因分析 + 397# 再起動後検証 | docs/v460/402, index.md | ✅ 分析として有益 |
| `c6ded4a96` | 405# Offset Ceiling Pipeline Fix | 4 lib ファイル + 4 テスト + docs | ✅ 14 テスト, 4684 全通過 |

### 1.2 未コミット (ディスク上のみ)

| ファイル | 内容 | 状態 | 問題 |
|---------|------|------|------|
| `CHANGELOG.md` | 401# エントリ追加 (400# エントリ差し替え) | modified | CHANGELOG 上書き |
| `docs/v460/401_deep_investigation_findings.md` | F1-F7 発見事項 + 実験結果 | untracked | **コミット必須** |
| `docs/v460/404_ph3_rev_402_403_time_guard_second_opinion.md` | Gemini セカンドオピニオン | untracked | **コミット必須** |
| `ztb/trading/environment/utils/config.py` | F3/F5 修正 | untracked (**ゴースト**) | **§2 参照** |

### 1.3 欠番問題

403# のドキュメント (`403_ph3_rev_402_time_guard_fill_test_multifaceted_review.md`) は **402# コミット (3d14f93ce)** に含まれているが、コミットメッセージには 403# の言及がない。index.md では 402#/403# が別エントリとして登録されているが、git 上は同一コミット。

---

## 2. CRITICAL: session037 ゴーストファイル問題

### 2.1 発見

`session037` (c69611f5f, 2026-03-13 03:20) が `ztb/trading/environment/` 配下の**75ファイルを git から削除**した。しかしディスク上にはファイルが残存し、以下の状態が発生:

```
HEAD 追跡ファイル (ztb/trading/environment/):  3 ファイル
ディスク上のファイル (*.py, __pycache__除外): 68 ファイル
→ 65 ファイルが "ゴースト" (git 未追跡だがディスク上存在)
```

### 2.2 実害

**致命的**: 以下の 2 ファイルは HEAD で追跡されていないが、プロダクションコードから import されている:

| ゴーストファイル | import 元 |
|----------------|----------|
| `ztb/trading/environment/utils/config.py` | `sac_train.py:270`, `heavy_env/core.py:83` |
| `ztb/trading/environment/utils/domain_randomizer.py` | `heavy_env/core.py:84` |

つまり、**SAC訓練パイプラインも HeavyTradingEnv も git 未追跡のゴーストファイルに依存**している。
`git clone` した別環境では `ImportError` で即死する。

### 2.3 401# F3/F5 修正の位置

401# の F3 (`balance_penalty_tolerance` マッピング追加) と F5 (unknown key WARNING) は、この**ゴースト config.py** に適用されている。修正自体は正当だが:

1. このファイルをコミットしない限り修正は永続化しない
2. session037 で削除された他の 64 ファイルも再追跡が必要
3. `.gitignore` にこれらが含まれていないか確認が必要

### 2.4 推奨対応

| 優先度 | アクション |
|--------|----------|
| **P0** | `ztb/trading/environment/utils/config.py` + `domain_randomizer.py` を `git add` して再追跡 |
| **P0** | `heavy_env/core.py` が import する全ファイルが追跡されているか検証 |
| **P1** | session037 で削除された 75 ファイルのうち、実際に import されているもの全てを再追跡 |
| **P2** | 不要な残骸（旧 reward 計算器等）はディスクからも削除 |

---

## 3. 400# Reward Clean のレビュー

### 3.1 良い点

- `scale_adjustment_enabled` フラグ追加は後方互換を維持しつつ問題を制御
- `balance_shaping_enabled` のデフォルト `True` → 明示 `False` 設定は**隠れた BUG FIX**
- YAMLレベルでの設定分離 (`g2_sac_reward_clean.yaml`) は実験管理として適切

### 3.2 懸念点

**400# CHANGELOG が 401# で上書きされている**: 未コミットの CHANGELOG.md diff を見ると、400# の詳細な変更記録 (scale_adjustment, g2_sac_reward_clean.yaml, 分析結果) が 401# エントリで完全に置き換えられている。400# の記録が消失。

```diff
-## 400# Reward Clean — v459知見フル適用 + scale_adjustment修正
+## 401# Deep Investigation + F3/F5 Fix + Reward-Clean Experiment (2026-03-13)
```

→ 400# は独立コミット (`194e30a2e`) なので CHANGELOG にはその記録を残し、401# は**追記**すべき。

---

## 4. 401# 深層調査のレビュー

### 4.1 発見事項の評価

| Finding | 自己評価 | レビュー所見 |
|---------|---------|------------|
| F1: 報酬飽和 | CRITICAL (正しい) | ✅ 分析は完璧。BTC/JPY 0.01 position × 100 → clip[-1,1] で sign(pnl) 二値化。ただし**G2/G3 PASS している事実**と矛盾するため、「致命的だが動作する」という異常な状況の説明が不足 |
| F2: Double Clip | LOW (正しい) | ✅ DEFAULT_REWARD_CLIP_VALUE=10000 で無害。判定妥当 |
| F3: tolerance 無視 | MEDIUM → **LOW** | ⬇️ 現行YAMLが偶然デフォルト値と一致するため顕在化しないと自身で記載。修正自体は正しいが深刻度は LOW |
| F4: デフォルト不一致 | LOW (正しい) | ✅ 明示設定時は無害。整理課題 |
| F5: タイポ黙殺 | HIGH (正しい) | ✅ 防御的プログラミングとして重要。ただしゴーストファイルへの修正 (§2) |
| F6: OOS checkpoint | HIGH (正しい) | ✅ v459 Day9b 50K崩壊の再現防止に必須。設計案も妥当 |
| F7: G3 E3 形骸化 | INFO (正しい) | ✅ cost=0 前提の当然の帰結 |

### 4.2 実験結果の再評価

**G2+G3 同時 PASS は歴史的成果**だが、以下の文脈を付記すべき:

1. **cost=0 (Coincheck Maker 0% fee)**: 手数料ゼロは理想条件。Taker 0.1% 環境では結果が変わりうる
2. **20K steps**: v459 Day9b は 25K→50K で崩壊した。20K での PASS は安全マージンの証明ではない
3. **Seed 456 の r=-0.203**: 報酬とPnLが逆相関するseedが 1/4 存在 → 報酬信号の品質問題 (F1) の傍証
4. **Checkpoint ROI 低下傾向**: Seed 42 は 5K→20K で 0.21%→0.15% に低下。過学習の兆候あり
5. **in-sample vs OOS**: 401# では明示されていないが、sac_train.py の OOS eval が使われていると推察

### 4.3 F6 実装案への補足

401# の設計案 (`best_oos_roi > -inf` → `model.save(best_checkpoint_path)`) は基本的に正しいが:

- **val_env の scaler 同期問題** (384# HIGH-2): 言及はあるが解決策が不明確
- **eval だけでなく early stopping** も検討すべき: OOS ROI が 3 checkpoint 連続低下なら打ち切り
- **checkpoint 間隔**: 5K steps で 4 回は粗い。2.5K steps で 8 回に増やすと過学習検出が早まる

---

## 5. 402# 時間帯分析のレビュー

### 5.1 良い点

- 時間帯防御の7層考古学はプロジェクト全体の見通しを改善
- AS率がPnLの最大ドライバーという結論は403#/404#でもクロスバリデーション済み
- JST 09h sell tail の定量化 (AS=59.8%, sell mean=-3.954) は信頼できる

### 5.2 403# が正しく却下した点

| 402# P0 提案 | 403# 却下理由 | 評価 |
|-------------|-------------|------|
| sell_hour_boost[0] 1.5→2.5 | Mixed-SHA, liveness低下 | ✅ 403#が正しい |
| hard_skip[21] 除外 | 取引データなし=評価不能 | ✅ 403#が正しい |
| sell ceiling 引き上げ | pipeline全体の問題 | ✅ 403#が正しく、405#で対応 |

### 5.3 残る構造的問題

402# → 403# → 404# → 405# の議論の成果として:

1. **パイプライン修正完了** (405#): sell中間cap 0.30→0.50 — 14箇所修正、テスト
2. **パラメータ変更なし** (403# 準拠): 保守的判断として妥当
3. **AS予測モデル**: 404# Action 3 として P1 記録 — 未着手
4. **confidence calibration**: 402# で ≥0.9 最悪 (n=52, mean=-1.69) が判明 — 未対応

---

## 6. 405# Offset Ceiling Pipeline Fix のレビュー

### 6.1 設計判断の評価

404# (Gemini) は「中間キャップ全除去、最終段一括クランプ」を提案。
405# は「side-aware intermediate cap」を選択。**正しい判断**:

- 中間キャップ全除去は暴走防止弁を失う
- side-aware (sell=0.50, buy=0.30) は最小変更原則に従う
- 最終 ceiling は unchanged

### 6.2 懸念: sell_hour_boost の無制限通過

405# セルフレビューで言及されているが、`sell_hour_boost` は `max_ratio` なしで適用される:

```
sell_floor(0.30) → mid_conf×1.2 → high_vol×1.5 → sell_hour_boost×1.5 (no cap)
→ final ceiling: min(result, 0.50) = 0.50
```

sell_hour_boost が上限なしなのは意図的だが、**中間ステップのキャップ緩和 (0.30→0.50) と合わせると、sell offset が 0.50 に到達しやすくなる**。fill rate への影響は次の SHA で監視すべき。

### 6.3 テスト品質

14 テストの構成:
- `_effective_max_ratio` 単体テスト 5 件: ✅ 境界条件を網羅
- sell デッドロック解消 3 件: ✅ high_vol, mid_conf, final ceiling
- buy 不変テスト 1 件: ✅ 既存動作維持を保証
- `_scale_offset_ratio` 4 件: ✅ 基本動作確認

**不足点**:
- `maker_microstructure.py` の 5 箇所修正に対する統合テストがない (as_reservation_shift, delta_star, kyle_lambda, amihud_illiq)
- `maker_risk_guards.py` の volatility_guard, imbalance_risk の修正テストがない
- これらは Protocol stub 更新のみで、実際のブースト×side-aware cap の end-to-end テストが欠落

---

## 7. 横断的構造課題

### 7.1 Git 追跡の信頼性

session037 の大量削除により、**git の state of truth と実際のランタイム依存が乖離**している。

```
git ls-tree HEAD (ztb/trading/environment/): 3 ファイル
Python import 依存: 68+ ファイル (ゴースト)
```

これは「git clone しても動かない」リポジトリであることを意味する。
**プロジェクト存続に関わる P0 の基盤問題**。

### 7.2 ドキュメント・コミット同期

| 問題 | 影響 |
|------|------|
| 401# F3/F5 未コミット | ゴーストファイルへの修正が消失リスク |
| 404# ドキュメント未コミット | Gemini レビューが記録されない |
| CHANGELOG 400#→401# 上書き | 400# の変更記録が消失 |
| 403# が 402# コミットに混在 | 番号体系とコミット粒度の不整合 |

### 7.3 実験管理

401# の実験結果は `results/v460/reward_clean/` に保存されているが、このディレクトリ自体が git 追跡対象か不明。再現性のためには最低限 config + 結果サマリの追跡が必要。

---

## 8. アクション整理

### P0 (即時)

| # | アクション | 理由 |
|---|----------|------|
| 1 | `ztb/trading/environment/` のゴーストファイルを `git add` で再追跡 | clone 不能状態の解消 |
| 2 | 401# ドキュメント + 404# ドキュメントをコミット | 成果物の永続化 |
| 3 | CHANGELOG.md の 400#/401# を両方残す形で修正 | 変更履歴の保全 |

### P1 (次セッション)

| # | アクション | 理由 |
|---|----------|------|
| 4 | F6 OOS best-checkpoint 実装 | 50K 実験前の必須条件 |
| 5 | 405# microstructure/risk_guards の統合テスト追加 | テストカバレッジ gap |
| 6 | session037 削除ファイルの要不要整理 | 68 ゴーストから必要分を選別 |

### P2 (中期)

| # | アクション | 理由 |
|---|----------|------|
| 7 | F1 報酬飽和対策 (reward_scaling/clip 調整) | seed 456 逆相関の根本原因候補 |
| 8 | confidence calibration 調査 | 402# で ≥0.9 最悪が判明 |
| 9 | AS 予測モデル (pre-trade proxy) | 404# Action 3 |

---

## 9. 結論

400#–405# は**技術的には高品質なセッション群**であり、特に:
- 400# balance_shaping BUG FIX は正真正銘の CRITICAL 修正
- 401# G2+G3 同時 PASS は初の里程碑
- 402#→403#→404#→405# の多角的レビューサイクルは模範的

しかし**プロセス上の負債**が蓄積している:
- session037 のゴーストファイル問題は **clone 不能 = CI/CD 不能 = 本番デプロイ不能** を意味する
- 未コミット成果物 (401#, 404#) は揮発性リスクを持つ
- CHANGELOG の上書きは変更追跡の信頼性を毀損する

**一番大事なこと (copilot-instructions.md)**:
> 本プロジェクトは短期間での高収益性システムが大義

この大義に照らせば、ゴーストファイル問題の修正は「短期的な収益性向上」に直接寄与しないが、**100K実験・本番デプロイへの前提条件**である。次セッションの最初の 10 分でゴーストファイルの再追跡を完了すべき。
