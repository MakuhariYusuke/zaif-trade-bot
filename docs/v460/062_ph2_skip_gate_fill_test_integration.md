# 062# ph2: AS SkipGate → fill_test ライブ統合

**日付**: 2026-02-16  
**Phase**: ph2 (G1.1-exec)  
**前提**: 061# SkipGate AS モード実装 (commit `03cd4361b`), walk-forward Skip20% +0.230 bps  
**ステータス**: 完了 — 月曜の fill test 再開時に即使用可能  

---

## §0 エグゼクティブサマリ

061# で実装・検証した AS 分類器 SkipGate を `run_fill_test.py` のライブ注文サイクルに統合。
モデルファイル (`models/v460/skip_gate_as.pkl`) をロードし、注文前に P(AS) を予測、
閾値以上なら注文をスキップすることで adverse selection を低減する。

### 変更の狙い

- **AS 低減**: Skip20% で +0.230 bps (walk-forward 検証済) の改善を実取引に適用
- **安全設計**: `enabled: false` デフォルト、max_skip_rate 安全弁、エラー時自動無効化
- **データ基盤**: SkipGate 判定情報を FillRecord に記録し、後続分析に活用

---

## §1 変更概要

### §1.1 FillTestConfig (S5: SkipGate)

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `skip_gate_enabled` | bool | False | SkipGate 有効化 |
| `skip_gate_mode` | str | "as" | "as" (AS分類器) / "pnl" (PnL回帰) |
| `skip_gate_model_path` | str | models/v460/skip_gate_as.pkl | モデルファイルパス |
| `skip_gate_as_threshold` | float | 0.6 | AS 確率スキップ閾値 (mode=as) |
| `skip_gate_pnl_threshold` | float | 0.0 | PnL 予測スキップ閾値 (mode=pnl) |
| `skip_gate_max_skip_rate` | float | 0.3 | 連続スキップ率上限 (直近20注文中) |

### §1.2 FillRecord (skip_gate フィールド追加)

| フィールド | 型 | 説明 |
|---|---|---|
| `skip_gate_skipped` | Optional[bool] | SkipGate によるスキップ判定 |
| `skip_gate_score` | Optional[float] | 予測スコア (AS確率 or 疑似PnL) |
| `skip_gate_reason` | Optional[str] | 判定理由 (skip/pass/skip_rate_limit/error) |

### §1.3 run_single_cycle 統合ポイント

```
1.   _compute_maker_price()  → order_price, spread, offset
1.5  _skip_gate.evaluate()   → SKIP → FillRecord(cancel_reason="skip_gate") 返却
                              → PASS → 続行
2.   place_order()            → 通常の注文フロー
```

- 板データ (OB depth=5) と直近約定データ (最新50件) からリアルタイム特徴量を構築
- `build_features_from_market_state()` を呼び出し、GATE_FEATURE_COLS (18特徴量) を生成
- レジーム情報は `_regime_detector.current_regime` から取得
- 判定エラー時は非致命的 → ログ出力のみで注文続行

### §1.4 YAML 設定

```yaml
# configs/v460/fill_test.yaml
skip_gate:
  enabled: false               # モデル学習後に true に変更
  mode: as                     # AS 分類器モード (推奨)
  model_path: models/v460/skip_gate_as.pkl
  as_threshold: 0.6
  pnl_threshold: 0.0
  max_skip_rate: 0.3           # 安全弁: 30% 以上のスキップを防止
```

---

## §2 安全設計

1. **デフォルト無効**: `enabled: false` — モデルファイル不在でも起動可
2. **モデル不在時の自動無効化**: パスが存在しない場合は warning ログのみ
3. **評価エラーの graceful degradation**: 例外発生時はスキップ判定をバイパスし注文続行
4. **連続スキップ率制限**: `max_skip_rate=0.3` — 無限スキップループを防止
5. **後方互換**: FillRecord の skip_gate フィールドは Optional[None] — 旧レコード互換

---

## §3 テスト

| テストクラス | 件数 | 内容 |
|---|---|---|
| `Test062SkipGateConfig` | 5 | YAML 解析、from_yaml マッピング、デフォルト値 |
| `Test062SkipGateRunner` | 7 | Runner 統合、FillRecord フィールド、to_dict/from_dict 互換 |
| **合計** | **12** | 全 PASS |

既存テストへの影響: 0 (49件の既存 fill_test テスト + 40件の ML テストすべて PASS)

---

## §4 運用手順 (月曜の fill test 再開時)

### §4.1 前提条件

1. 入金完了 (JSY 残高確認)
2. データ蓄積: `data/v460/raw/` に OB/Trades 3日分以上

### §4.2 SkipGate モデル学習

```powershell
# fill_records データから AS 分類器を学習
.\.venv\Scripts\python.exe -c "
from scripts.v460.ml.skip_gate import train_and_save_as_skip_gate
gate = train_and_save_as_skip_gate(as_threshold=0.6, k=8)
print(f'Model saved: {gate.metadata}')
"
```

### §4.3 有効化

```yaml
# configs/v460/fill_test.yaml
skip_gate:
  enabled: true    # ← false → true に変更
```

### §4.4 fill test 再開

```powershell
.\.venv\Scripts\python.exe scripts/v460/run_fill_test.py --hours 168
```

### §4.5 効果確認

- `cancel_reason="skip_gate"` のレコードを集計
- skip_gate_skipped=True のサイクルでスキップされた注文の AS 率を確認
- skip_gate_skipped=False (PASS) のサイクルの PnL 改善度を計測

---

## §5 変更ファイル一覧

| ファイル | 変更 |
|---|---|
| `scripts/v460/run_fill_test.py` | FillTestConfig S5 fields + from_yaml + Runner init + run_single_cycle gate eval |
| `ztb/metrics/fill_quality.py` | FillRecord skip_gate_* 3フィールド追加 |
| `configs/v460/fill_test.yaml` | skip_gate セクション追加 |
| `tests/unit/v460/test_fill_test_config.py` | Test062 2クラス 12テスト追加 |
| `docs/v460/062_ph2_skip_gate_fill_test_integration.md` | 本ドキュメント |

---

## §6 次のステップ

- [ ] 月曜: 入金 → fill test 再開 → SkipGate 有効化
- [ ] 1日のデータ蓄積後: skip_gate 効果の定量評価
- [ ] ph3: SAC 重複実装の整理 (#2,#5,#6 廃止 → #1 unified_trainer 統一)
- [ ] ph1: G1-info 再検証 (OB/Trades データで XGBoost walk-forward)
