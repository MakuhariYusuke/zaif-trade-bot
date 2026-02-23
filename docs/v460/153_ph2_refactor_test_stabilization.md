# 153# P2 品質改善 — テスト安定化 + run_fill_test 分割設計

> 152# §9 で見送った P2 優先度項目 (P2-7, P2-8) の委託実装。
> 別 AI コーディングエージェントが本ドキュメントに従い実行し、結果を末尾に追記する。

---

## §1 背景

152# セッションで Phase C 並行施策を評価した結果、P2 優先度の 2 項目は即時収益寄与が薄いが、
中長期のメンテナンスコストを圧縮するために必要と判断された。

| 項目 | 152# §11.1 評価 | 本セッションの扱い |
|------|----------------|-------------------|
| P2-7 テスト安定化 | `unit` mark 登録済、`--disable-warnings` で抑止中 → 見送り | **実装する** |
| P2-8 run_fill_test 分割設計 | 即時収益寄与なし → メモのみ | **設計メモ + 可能なら初期分割を実装** |

---

## §2 タスク A: テスト安定化 (P2-7)

### 2.1 現状

- テスト総数: ~4,934 (collect 時点)、collect errors: 39
- `pytest.ini` に `--strict-markers` + `--disable-warnings` あり
- `unit` マーカーは `pytest.ini` L25 に登録済み → `PytestUnknownMarkWarning` は `filterwarnings` で抑止
- `conftest.py` L156 で `pytest_collection_modifyitems` によりパス名から自動マーキング
- 実行時は `-n auto` (xdist 並列) + `--maxfail=5`

### 2.2 やること

1. **collect errors 39 件の解消**
   - `pytest --co -q` を実行し、エラーの原因を分類 (import error / fixture missing / syntax)
   - 修正可能なものは修正、不可能なものは `archived/` へ移動または `pytest.ini` の `norecursedirs` に追加
   - **目標: collect errors = 0**

2. **warning ゼロ化**
   - `pytest -v --override-ini="addopts=" -W all` で全 warning を表示
   - `PytestUnknownMarkWarning` 以外の warning があれば修正
   - `filterwarnings` に必要な抑制を追加 (根本修正が難しいもののみ)
   - **目標: `-W error` でもテストが通る** (理想)

3. **不安定テストの安定化**
   - `pytest --co` で collect できるが flaky なテストを特定
   - timing 依存、ファイル I/O 依存、ネットワーク依存のものに `@pytest.mark.slow` または `@pytest.mark.integration` を付与
   - 真に壊れているテストは修正するか `archived/` に移動

### 2.3 制約

- **既存テストの削除は禁止** — 移動 (`archived/`) のみ許可
- **テストの意味を変えない** — assert の条件を緩めてパスさせるのは不可
- `conftest.py` の自動マーキングロジックは変更しない
- `pytest.ini` の `addopts` は変更可能だが、`--strict-markers` と `--cov` は維持

---

## §3 タスク B: run_fill_test 分割設計 (P2-8)

### 3.1 現状

- `scripts/v460/run_fill_test.py`: **2,203 行**、1 クラス (`FillTestRunner`) + ヘルパー
- god object 傾向: lot 管理、regime 判定、SkipGate 評価、注文監視、PnL 計測、状態永続化 etc.
- 既に一部は外部に委譲済み:
  - `BalanceChecker` (残高チェック + ロット管理)
  - `TimeFilter` (時間帯フィルター)
  - `SellDynamicKillManager` (sell kill 判定)
  - `FastFillDefense` (高速約定防御)
  - `FillTestRegimeDetector` (レジーム検知)
  - `SkipGateEvaluator` (SkipGate 評価)

### 3.2 やること

1. **責務マップの作成**
   - `FillTestRunner` の全メソッド (~45 個) を以下のカテゴリに分類:
     - Lot/Position: `_regime_lot_multiplier`, `_regime_adjusted_lot`, `_confidence_lot_factor`, `_effective_order_lot`
     - Order Execution: `_compute_maker_price`, `_monitor_fill_polling`, `_cancel_stale_orders`
     - Measurement: `_measure_post_fill_pnl`, `_compute_orderbook_imbalance`
     - Lifecycle: `run_single_cycle`, `run_continuous`, `_cleanup_sync`, `_acquire_lock`
     - Record/IO: `_make_skip_record`, `_log_event`, `resume_from_existing`
   - 各カテゴリの行数と依存関係を明記

2. **分割候補の提案** (3-5 モジュール)
   - 各モジュールの名前、責務、推定行数、`FillTestRunner` からの呼び出しインターフェース
   - 依存関係図 (テキスト or mermaid)
   - **実装は optional** — 設計メモだけでも可

3. **初期分割の実装** (余力がある場合)
   - 最も独立性が高いカテゴリ (Lot/Position が候補) を抽出
   - `scripts/v460/lib/lot_manager.py` (仮称) として切り出し
   - `FillTestRunner` から委譲呼び出しに変更
   - テスト追加 (切り出したモジュールの単体テスト)

### 3.3 制約

- **FillTestRunner の公開 API は変更しない** — `run_single_cycle()`, `run_continuous()` のシグネチャ維持
- **Phase C 稼働中のため、振る舞い変更は厳禁** — リファクタリングのみ
- 既存テスト (`test_143_regime_utilization.py`, `test_151_confidence_lot.py`, `test_152_parallel_tasks.py`) が全て pass すること
- コミットメッセージ: `refactor(153#): <説明>`

---

## §4 共通ルール

### 4.1 プロジェクト規約

- Python 3.11.9、venv `.venv\Scripts\python.exe`
- 型安全: `Any` 型回避、mypy 活用
- DRY 原則、単一責任原則、SOLID
- 既存ファイル活用の徹底、新規作成は必要最低限
- テストはこまめに実施、回帰確認必須
- PowerShell スクリプトで実行

### 4.2 テスト実行方法

```powershell
# 単体テスト (v460 関連)
.venv\Scripts\python.exe -m pytest tests/unit/v460/ -v --tb=short

# 全テスト (collect 確認)
.venv\Scripts\python.exe -m pytest tests/ --co -q

# 全テスト実行
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
```

### 4.3 コミット

```powershell
git add <files>
git commit --no-verify -m "refactor(153#): <内容>"
```

### 4.4 品質ゲート

| ゲート | 基準 |
|--------|------|
| collect errors | 0 (タスク A 達成後) |
| テスト pass | 既存テストの regression なし |
| mypy | 新規コードに type annotation 必須 |
| 行数 | run_fill_test.py の行数が増えない (タスク B) |

---

## §5 成果物チェックリスト

- [ ] タスク A: collect errors 解消 → `pytest --co -q` が error 0
- [ ] タスク A: warning 削減 → `-W all` 実行結果をログに記載
- [ ] タスク A: 不安定テスト対応 → flaky テストの一覧と対処
- [ ] タスク B: 責務マップ → §6 に追記
- [ ] タスク B: 分割候補の提案 → §6 に追記
- [ ] タスク B (optional): 初期分割の実装 → コミット SHA を §6 に記載
- [ ] 全テスト回帰確認 → `pytest tests/unit/v460/ -v` の結果を §6 に記載
- [ ] コミット完了

---

## §6 実装結果

> ※ 実装担当の AI エージェントが以下に結果を追記すること。

### 6.1 タスク A: テスト安定化

(ここに結果を記載)

### 6.2 タスク B: run_fill_test 分割設計

(ここに結果を記載)

### 6.3 テスト回帰確認

```
(pytest 実行結果を貼付)
```

### 6.4 コミット履歴

```
(git log --oneline を貼付)
```
