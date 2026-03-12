# 030# 029レビュー検証 & 028修正 & 000反映

| key | value |
|---|---|
| 番号 | 030 |
| フェーズ | phg (cross-gate) |
| 種別 | resp (029レビュー応答) |
| 対象文書 | `029_phg_rev_028.md` |
| 作成日 | 2026-02-14 |
| 結論 | **5指摘中: 1件 INVALID, 2件 PARTIALLY VALID, 2件 VALID。028#修正 2箇所、000#改訂 3箇所を実施。** |

---

## 1. 指摘検証結果

### 1.1 指摘 #1: 「6モジュール完全dead codeは断定が強すぎる」→ INVALID

029# はレビューアが `evaluation_manager.py`, `unified_optimizer.py`, `adaptation/__init__.py` を根拠に「参照が残る」と主張。実コード検証の結果:

| 根拠 | 実態 |
|------|------|
| `adaptation/__init__.py` の re-export | 6モジュールの直接importは実質ゼロ。`ConceptDriftManager`/`SafetyManager` は `monitoring/` サブパッケージ由来で対象外 |
| `evaluation_manager.py` の参照 | `OnlineLearningPipeline` は `TYPE_CHECKING` ガード内のみ（実行時import無し）。`ContinuousEvaluationManager` 自体が adaptation 外部から呼ばれていない |
| `unified_optimizer.py` の `ABTestingFramework` | **ファイル内ローカル定義クラス**。`ztb/adaptation/ab_testing/` からのimportではない（同名別実装） |

**混同の原因**: `safety/` vs `monitoring/safety.py`、`ab_testing/` vs `ab_test/`（生きている別パッケージ）の名前類似。

**結論**: 6モジュール全てv460ライブ実行パスから到達不可能。028#の「完全dead code」断定は正確。

### 1.2 指摘 #2: 「`online_learning_engine.py`(V433)が分析から漏れている」→ PARTIALLY VALID

- `ztb/training/online_learning_engine.py` (808行) は確かに存在し、028#で言及されていない
- `trainer.py` L1913 `_initialize_v433_components()` から遅延importされる
- ただし `enable_v433_adaptive=False`（デフォルト）のため**v460ではデッドコード**
- 015# §5 では既に分析済み（028#が参照し損ねた形）

**対応**: 028# §2 にデッド資産として一行追記。HIGH重大度は過大で LOW が妥当。

### 1.3 指摘 #3: 「`start_learning()`インターフェース不整合が未指摘」→ PARTIALLY VALID

- `operations/manager.py:162` が引数なし `start_learning()` を呼ぶ
- `pipeline.py:99` の API は `start_streaming(data_iterator)` — 名前もシグネチャも不一致
- ただし `IntegratedOperationsManager` 自体がどこからもインスタンス化されていない

**対応**: デッドコード内の不整合であり実運用リスクゼロ。028#への追記は不要（デッドコード特性として既にカバー済み）。

### 1.4 指摘 #4: 「循環importの表現が不正確」→ VALID

`algorithms/__init__.py` L85-96 の実態:
- import文が**コメントアウト**されている（L88-93）
- `raise NotImplementedError(...)` で即座にガード（L96）
- Pythonのimportシステム自体が発動しない

**対応**: 028# F1 の記述を「コメントアウト+NotImplementedErrorで回避中の潜在的循環依存」に修正。

### 1.5 指摘 #5: 「G4にガバナンス不足」→ VALID (自己解決済)

- 028# G4追記案にガバナンス条項が無いのは事実
- ただし 029# 自身が §5 item 3 で「shadow評価、rollback、モデル系譜」として回収済み
- v460現フェーズ (ph2, 方策A = モデル非接触) では急務ではない
- 方策B実装時 (ph5) に G4 Gate 条件として追加が妥当

**対応**: ph5 到達時に 000# §3.6 を改訂。現時点での追記は見送り。

---

## 2. 029# §4「000#への反映方針」に対する判断

### 同意する項目

| 029# 推奨 | 判断 | 対応 |
|-----------|------|------|
| §4.1-1: §6リスクに「モデル陳腐化」追加 | **同意** | 本030#にて実施 |
| §4.1-2: §4技術概要に適応運用注記 | **同意** | 本030#にて実施 |
| §4.1-3: Appendix改訂履歴に方針明記 | **同意** | 本030#にて実施 |
| §4.2-1: G4 Gate条件化は保留 | **同意** | ph5まで保留 |
| §4.2-2: Phase成果物への固定追加は保留 | **同意** | 実装完了後 |
| §4.2-3: C方策の採用宣言は保留 | **同意** | v461判断 |

### 029# §6 提案文の採用

> v460では適応運用を段階導入する。
> ph2-ph5では執行パラメータの自動調整（A）を優先し、ph5以降で定期再訓練（B）を検証する。
> 取引ループ中のリアルタイム学習（C）は、安定性と安全統制の実装完了後にv461で再評価する。

→ 000# §4 への注記として採用。

---

## 3. 028# 修正箇所

| # | 箇所 | 修正内容 |
|---|------|---------|
| 1 | §2.1 モジュール一覧 | `ztb/training/online_learning_engine.py` (808L, V433, デッドコード) を追記 |
| 2 | §5 F1 | 「循環import」→「潜在的循環依存（import文コメントアウト+NotImplementedError で回避中）」に修正 |

---

## 4. 000# 改訂箇所

| # | §番号 | 変更内容 |
|---|-------|---------|
| 1 | §4 技術概要 | 適応運用方針の注記追加 |
| 2 | §6 リスク | 「市場レジーム変化によるモデル陳腐化」追加 |
| 3 | Appendix A | 改訂履歴に本反映を記録 |

---

## 5. 即時実施コード作業（ph2不要）

015#–018#、021# で未実施かつ ph2 完了を待たない作業:

| # | 作業 | 根拠 | 状態 |
|---|------|------|------|
| 1 | `ztb/training/v435/train_sac_v435.py` 削除 (14L スタブ) | 015# §6.2, 021# P0 | 本030#にて実施 |
| 2 | `ztb/training/adaptive_sac_core.py` アーカイブ (763L) | 015# §2.3, 028# F4 | 本030#にて実施 |
| 3 | `ztb/training/online_learning_engine.py` アーカイブ (808L) | 029# #2, 015# §5 | 本030#にて実施 |
| 4 | `ztb/adaptation/` デッドモジュール 6件アーカイブ | 028# §2, 030# §1.1 検証済 | 本030#にて実施 |

**除外理由**:
- ModelRegistry: 016# で「不要、FeatureSchemaManager 拡張で代替」と判断済
- ActionPrediction 次元修正: 017# で実装済 (commit `cad66c869`)
- メモリリーク/パフォーマンス: 018# で実装済
- `load_model` 二重定義: 018# で解消済
