# Gemini 向け指示プロンプト: 仕様策定完了 → 実装フェーズ移行 (618#)

あなたは 614#→615#→616#→617# の仕様策定チェーンを完成させました。Train-Serve Skew も 617# で解消され、数理仕様は十分に成熟しています。

**ここからは仕様ではなく、実装コード (Python) を書いてください。**

---

## 現状の整理

### 仕様の完成状態
| 文書 | 内容 | ステータス |
|:-----|:-----|:-----------|
| 614# | Attribution Analyzer 定義 + Sidecar Feature Contract | ✅ 確定 |
| 615# | 614# errata (composite_risk, stage_saturation 精度向上) | ✅ 確定 |
| 616# §1 | Attribution Phase 2: Euler RMS 分解、Occupancy | ✅ 確定 |
| 616# §2 | Live Feature Builder (deque ベース) | ❌ **617# で破棄** |
| 616# §3 | Adaptive EMA | ✅ 確定 (ただし実装優先度は低い) |
| 617# | Feature Parity: 同期バッチ抽出 + norm.json スキーマ | ✅ 確定 |

### 既に Copilot 側で実装済み (Phase 1)
- `analyze_fill_logs.py` に `section_information_loss()` (clamp_rate / info_loss bps 集計)
- `analyze_fill_logs.py` に `section_stage_saturation()` (multiplier >= 1.99 の飽和率)
- `analyze_fill_logs.py` の `section_clamp_saturation()` に pre_clamp 分布 (p50/p75/p90/p99) を追加済み

### Copilot 側が提供するコードベース情報 (実装に必要な既存 API)

#### 1. `_atomic_deploy_model()` — モデル保存のエントリポイント
```python
# scripts/v460/ml/sac_retrain_scheduler.py L870-910
def _atomic_deploy_model(
    model: SACModelProtocol,
    cfg: SACRetrainConfig,
    model_version: str,
) -> None:
    """モデル + buffer を atomic deploy (tmp → rename)."""
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    # Model: tmp → atomic rename
    fd_m, tmp_model = tempfile.mkstemp(...)
    model.save(tmp_model)
    os.replace(tmp_model, str(cfg.model_path))
    # Buffer: best-effort
    model.save_replay_buffer(tmp_buffer)
    os.replace(tmp_buffer, str(cfg.buffer_path))
```
→ **ここに norm.json 出力を追加する必要がある**

#### 2. `V4FeatureExtractor` / `UnifiedFeatureEngineer` — 特徴量計算の入口
```python
# ztb/features/unified_feature.py
class UnifiedFeatureEngineer:
    def generate_features(self, df: pd.DataFrame, ...) -> pd.DataFrame:
        ...
```
→ **ライブ推論では、このクラス (またはその内部の FeatureRegistry 関数) を直近 N 行の DataFrame に対して呼び出し、最終行を取得する**

#### 3. 市場理論特徴量の関数シグネチャ
```python
# ztb/features/market_theory.py
@register("parkinson_sigma")
def parkinson_sigma(df: pd.DataFrame, window: int = 20) -> pd.Series: ...

@register("vpin_proxy")
def vpin_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series: ...

@register("kyle_lambda_proxy")
def kyle_lambda_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series: ...

@register("amihud_illiq")
def amihud_illiq(df: pd.DataFrame, window: int = 20) -> pd.Series: ...

@register("ema_velocity_bps")
def ema_velocity_bps(df: pd.DataFrame, span: int = 5) -> pd.Series: ...
```
→ **全関数が `df: pd.DataFrame` を受け取り `pd.Series` を返す。ライブでもこの API をそのまま使う**

#### 4. offset pipeline の stages JSON (Attribution Phase 2 の入力)
fill_record に記録される `executor_offset_stages` JSON の実際の構造:
```json
{
  "velocity": 1.05,
  "trending": 0.98,
  "toxicity": 1.12,
  "vg_supp": 1.0,
  "alert": 1.0,
  "ev": 0.95,
  "macro": 1.0,
  "tox_buffer": 0.0015,
  "liq_buffer": 0.0008
}
```
→ **加法パイプライン (additive) のレコードには `tox_buffer` / `liq_buffer` が存在する**

#### 5. モデル保存パス
```python
# scripts/v460/ml/sac_retrain_scheduler.py
model_path: Path = Path("models/v460/sac_sidecar.zip")
buffer_path: Path = Path("models/v460/sac_sidecar.buffer.pkl")
# → norm.json は同ディレクトリに置くのが自然:
#    models/v460/sac_sidecar.norm.json
```

---

## あなたが書くべき実装コード (3 モジュール)

### I1: `norm.json` 出力ロジック — `_export_feature_norms()`

**場所**: `scripts/v460/ml/sac_retrain_scheduler.py` 内の `_atomic_deploy_model()` に追加
**目的**: retrain 成功時にモデルと同時に特徴量の正規化統計を JSON で出力
**仕様**: 617# §3.1 のスキーマに準拠

```python
def _export_feature_norms(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """617# §3.1: 訓練データの特徴量統計を norm.json として出力."""
```

**要件**:
- `df` は訓練に使用した DataFrame（特徴量計算済み）
- 各 `feature_columns` について `mean`, `std`, `min`, `max` を計算
- `std` が 0 の場合は `1e-10` でフロア
- atomic write (tempfile → rename) でファイル破損を防止
- `_atomic_deploy_model()` 末尾から呼び出されるよう統合コードも提示

### I2: ライブ標準化ローダー — `NormLoader`

**場所**: 新規ファイル `ztb/features/norm_loader.py`
**目的**: `norm.json` を読み込み、推論時の Z-score 変換 + NaN imputation + clipping を行う
**仕様**: 617# §3.2 に準拠

```python
class NormLoader:
    """617# §3.2: 推論時の特徴量標準化."""
    
    def __init__(self, norm_path: Path) -> None: ...
    def reload_if_changed(self) -> bool: ...
    def normalize(self, raw_features: dict[str, float]) -> np.ndarray: ...
```

**要件**:
- `normalize()` は NaN → mean 置換、z-score 変換、min/max clipping の 3 ステップ
- `reload_if_changed()` は mtime ベースの hot-reload（retrain 後の自動取り込み）
- feature 順序は `norm.json` 内の `feature_stats` のキー順序に固定

### I3: Attribution Phase 2 分析関数 — `section_attribution_phase2()`

**場所**: `scripts/v460/analysis/analyze_fill_logs.py` に追加する section 関数
**目的**: 616# §1 の Euler RMS 分解 + Occupancy を fill_record から集計
**仕様**: 616# §1.1, §1.2 に準拠

```python
def section_attribution_phase2(records: list[dict[str, Any]]) -> list[str]:
    """616# §1: Euler RMS 分解 + Ceiling Occupancy."""
```

**要件**:
- `executor_offset_stages` JSON から `tox_buffer`, `liq_buffer` および各ステージ multiplier を取得
- 加法パイプラインのレコード (`tox_buffer` キーが存在) のみを対象
- 各 tox ステージの Euler 寄与度 $C_i = (\Delta R_i)^2 / \text{tox\_rms}$ を計算
- Occupancy (%) = tox_rms / (ceiling - R_base) × 100 を集計
- Side 別に集計し、テキストレポート形式の `list[str]` で返す

---

## 制約事項

1. **コードのみ出力**。理論の再説明は不要（614#-617# で十分）
2. **既存 API を使う**。新しいクラスを不必要に生成しない
3. **型アノテーション必須**。`Any` 型は避け、`mypy` で通ること
4. **エラーハンドリング**: JSON パース失敗、ファイル不在は graceful に処理
5. **ドキュメント番号を docstring に含める** (例: `"""617# §3.1: ..."""`)
6. 出力は **618# として 1 つの文書** にまとめてください

---

*以上。仕様フェーズは完了しました。あなたの数理的洞察を、実行可能な Python コードに変換してください。*
