# Codex Session 038: 広域課題スキャン — バグ・設計欠陥・技術的負債の包括的発見

## ミッション

本プロジェクト (`zaif-trade-bot`) は BTC/JPY maker bot の SAC 強化学習パイプライン + HFT 執行エンジンです。
408 セッション分の開発で蓄積された潜在的問題を **6 つの異なる視点** から横断的にスキャンし、
カテゴリ別に優先度付きで報告してください。

> **重要**: 「壊れていないから触るな」ではなく、「いつ壊れるか」「壊れたとき何が起こるか」の視点で評価すること。

---

## プロジェクト概要

- **Python 3.11**, SB3 (stable-baselines3), gymnasium, PyTorch, LightGBM
- **SAC 訓練**: `scripts/v460/lib/tasks/sac_train.py` → `HeavyTradingEnv` (1834 行)
- **報酬**: `RewardCalculator` (2252 行, 50 メソッド — God Object)
- **執行エンジン**: `scripts/v460/lib/` (75 ファイル — maker_price, fill_cycle_executor, orchestrator 等)
- **Gate System**: G1 (データ品質) → G2 (SAC 4-seed) → G3 (PnL Monte Carlo)
- **テスト**: 8,662 tests, カバレッジ ~21%
- **HEAD**: `cfc0116de` (408# F-series + blind spot fixes)

---

## 調査対象ディレクトリ

1. **`ztb/trading/environment/`** — 環境・報酬・設定 (最大ファイル: reward_calculator.py 95KB, core.py 83KB)
2. **`scripts/v460/lib/`** — v460 訓練・執行ロジック (75 ファイル)
3. **`scripts/v460/ml/`** — ML パイプライン (26 ファイル)
4. **`ztb/trading/live/`** — ライブトレード基盤
5. **`configs/v460/`** — YAML 設定ファイル
6. **`tests/`** — テストスイート (テスト品質自体も対象)

---

## 6 つの調査視点 (各視点で最低 3 件の発見を目標)

### 視点 1: ロジックバグ・数値計算誤り (CRITICAL)

以下のパターンを重点的にスキャンすること:

- **ゼロ除算**: `/ total`, `/ count`, `/ std`, `/ len(...)` の分母がゼロになりうる箇所
- **Off-by-one**: インデックス範囲、スライス境界、ループ終了条件
- **符号逆転**: ペナルティの加算/減算方向、座標系 (buy/sell, long/short)
- **型の暗黙変換**: `int` と `float` の混在、`bool` を数値として使う箇所
- **NaN/Inf 伝播**: `np.log(0)`, `np.sqrt(negative)`, `np.corrcoef` の結果が NaN のまま使われる箇所
- **条件式の論理ミス**: `and`/`or` の優先順位、`not` の適用範囲

重点ファイル:
- `ztb/trading/environment/components/calculators/reward_calculator.py`
- `scripts/v460/lib/sac_common.py`
- `scripts/v460/lib/maker_price.py`
- `ztb/trading/environment/heavy_env/core.py`

### 視点 2: リソースリーク・パフォーマンス (HIGH)

- **メモリリーク**: `deque`, `list` の無限蓄積、`history` バッファの上限なし成長
- **ファイルハンドルリーク**: `open()` の `with` 文なし使用
- **GPU/CPU リソース**: PyTorch テンソルの `.detach()` 忘れ、不要な `.to(device)` 呼び出し
- **O(N²) ループ**: リスト内リスト検索、繰り返し文字列結合
- **GC 干渉**: `gc.collect()` の過剰呼び出し、大量オブジェクト生成サイクル
- **重複計算**: 同一 DataFrame 加工の繰り返し、キャッシュなし関数呼び出し

重点ファイル:
- `ztb/trading/environment/heavy_env/core.py` (step() ループ)
- `ztb/trading/environment/components/behavioral_penalty_calculator.py`
- `scripts/v460/lib/tasks/sac_train.py` (訓練ループのメモリ管理)

### 視点 3: 設定SSOT違反・デフォルト値ドリフト (MEDIUM-HIGH)

`RewardSettings`、`EnvironmentConfig`、`FillConfig` のデフォルト値と、
コード中のフォールバック値 (`getattr(x, "key", FALLBACK)`) が乖離していないかをスキャン。

- **YAML ↔ Code ドリフト**: `configs/v460/experiments/*.yaml` の値と、Python コードのデフォルト値の不一致
- **`getattr` / `get` のフォールバック値**: SSOT (`RewardSettings.__init__` 等) と異なるデフォルト
- **`from_dict` 漏れ**: YAML キーが `from_dict()` でパースされない (黙殺される)
- **hot-reload 未配線**: `hot_reload_config()` で更新されないフィールド

重点ファイル:
- `ztb/trading/environment/utils/config.py` (EnvironmentConfig, RewardSettings)
- `ztb/trading/environment/components/calculators/reward_calculator.py` (`get_setting_*` 系)
- `scripts/v460/lib/fill_config*.py`

### 視点 4: エラーハンドリング・例外安全性 (MEDIUM)

- **bare except**: `except:` や `except Exception:` で握り潰し
- **サイレント失敗**: `try/except` の `pass` — エラーが検知不能になる箇所
- **例外チェーン切断**: `raise NewError()` vs `raise NewError() from e`
- **不完全なクリーンアップ**: `finally` ブロックなしの `try/except` でリソース未解放
- **assertion の本番使用**: `assert` が本番コードパスに存在する箇所

重点ファイル:
- `scripts/v460/lib/` (執行エンジン全体)
- `ztb/trading/live/` (ライブトレード)

### 視点 5: テスト品質・テストの嘘 (MEDIUM)

- **テストのフラジリティ**: `time.sleep()` や固定タイムスタンプに依存するテスト
- **モック過剰**: 実際のロジックをモックで迂回し、何もテストしていないテスト
- **assert 不在/弱い**: テスト関数に `assert` がない、または `assert True` のみ
- **テスト間の暗黙依存**: グローバル状態の汚染、テスト実行順序依存
- **カバレッジの穴**: 重要な分岐 (エラーパス、境界条件) がテストされていない
- **壊れたテスト**: import エラー、存在しないモジュール参照 (408# で `test_comprehensive_fixes.py` を発見・アーカイブ済み — 他にもあるか?)

対象ディレクトリ: `tests/`

### 視点 6: アーキテクチャ・設計上の負債 (LOW-MEDIUM、ただし増殖リスクは HIGH)

- **God Object**: 1000 行超のクラス/モジュール (reward_calculator.py 2252行, core.py 1834行, initialization.py, config.py)
- **循環依存**: モジュール間の双方向 import
- **Proxy 乱立**: `components/reward_calculator.py` のような re-export shim (408# で発見済み)
- **責務混在**: 1 ファイルに計算・IO・設定解決・ログ出力が混在
- **命名不統一**: `reward/` vs `rewards/` (408# で発見済み)、`env` vs `environment`
- **レガシーコード**: v455/v456/v457 のコードが `ztb/` 内に残存 (408# で調査済み — アーカイブ候補の状況を再確認)
- **DRY 違反**: 同一ロジックの複数実装 (408# §8 で4件発見済み — 他にないか?)

---

## 出力形式

以下のフォーマットで報告してください:

```markdown
## カテゴリ N: [視点名]

### CN-1: [Finding タイトル] — [CRITICAL/HIGH/MEDIUM/LOW]

**ファイル**: `path/to/file.py:LINE`
**現状**: 何が問題か (コード引用 5 行以内)
**影響**: 何が起こりうるか (最悪ケース)
**修正案**: 具体的な修正方針 (1–3 行)
**工数**: S/M/L (Small: ~30min, Medium: ~2h, Large: ~1day)

---
```

各カテゴリは **最低 3 件、最大 10 件** の発見を報告すること。
CRITICAL は即座に修正が必要、LOW は「知っておくべきだが緊急ではない」。

---

## 追加指示

1. **408# 既知事項との重複回避**: 以下は既に修正済みなので報告不要:
   - F4 デフォルト不整合 (balance_penalty, consistency_penalty)
   - F6 OOS best-checkpoint
   - B1 _record_action 二重呼び出し
   - B2 BPC else-branch 属性欠損
   - B3 continuous_action_value シャドーイング
   - B4 avg_gross_per_trade abs()
   - B5 train_val_split 空 DataFrame
   - S4 continuous_action_value tuple バグ
   - `test_comprehensive_fixes.py` 壊れたテスト

2. **リファクタリング提案は「分割の境界」まで具体化**: 「分割すべき」だけでは不十分。どのメソッド群をどのクラスに切り出すかまで提案すること。

3. **テスト提案を含める**: 各 CRITICAL/HIGH の Finding に対して、検証するテストケースのスケルトン (関数名 + assert 1 行) を添える。

4. **横串パターン**: 同一パターンの問題が複数ファイルにまたがる場合、代表 1 件 + 該当全ファイルリストで報告 (各ファイル個別報告は冗長)。

5. **収益性への影響度**: 可能な範囲で「この問題が収益にどう影響するか」を定性的に記述。例: 「報酬の符号逆転 → エージェントが逆方向に学習 → PnL 低下」

---

## 最終ゴール

レポートを `docs/v460/409_phg_rpt_broad_discovery_scan.md` として保存してください。
タイトルは `409# 広域課題スキャン: バグ・設計欠陥・技術的負債の包括的発見` とします。
