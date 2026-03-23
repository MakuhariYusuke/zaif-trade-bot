# Document 578: Phase 5 監査レポートおよび自己レビュー (Docs 564-577)

## 1. 監査概要
本ドキュメントは、Doc 564から577にかけて行われた「Dynamic Risk Ceiling (eDRC)」および「Robust Stats」の導入に関する実装とアーキテクチャの監査結果をまとめたものです。実装上の重大な欠陥、金融工学的な観点からの構造的盲点、および今後のシステム改善に向けた提言を記載します。

## 2. 実装ミスの指摘と構造的盲点

### 2.1. Telemetryの欠落 (`FillRecord` の不整合)
- **問題**: `Kissell & Glantz` の取引コスト分析において重要な `spread_capture_bps` と `adverse_selection_cost_bps` が、収集されているにもかかわらず永続化されていません。
- **原因**: `FillRecordBuilderMixin` ではこれらの値をペイロードに追加していますが、データクラスである `FillRecord` 自体にこれらのフィールドが定義されておらず、シリアライズ時にサイレントに欠落しています。
- **影響**: eDRCのA/Bテストにおいて、スプレッド確保の実績と逆選択コストの正確な評価が不可能になっています。

### 2.2. Hard Capの論理的破綻 (`hour_ceiling_mult` の適用順序)
- **問題**: `edrc_hard_cap` が期待通りに機能していません。
- **原因**: `fill_config.py` において、`ceiling_dynamic = min(ceiling, edrc_hard_cap)` の評価が `hour_ceiling_mult` （時間帯乗数）の適用**前**に行われています。
- **影響**: 特定の時間帯において、eDRCのハードキャップを突破してリスク限度枠が拡大する（または不当に縮小する）現象が発生します。これは「ハードキャップ」の定義に反します。

### 2.3. 加法パイプラインの名称と実態の乖離
- **問題**: `experimental_additive_pipeline` というフラグが導入されましたが、実態は加法モデル（Additive Model）になっていません。
- **原因**: このフラグが有効になると、eDRCベースの動的ceilingとrobust inputは使用されますが、その後の処理は依然として従来の9段階の**乗法チェーン（Multiplicative Chain）**を通過しています。
- **影響**: 金融工学的な加法モデル ($Offset = Base + \sum(w_i \cdot feature_i)$) が実現されておらず、変動が乗算的に増幅されるリスクが残存しています。

## 3. 金融工学的見地からのシステム改善提言

### 3.1. 真のAdditive Modelへの移行
乗法チェーンは、パラメータ同士の相互作用により予期せぬ極端な値（Fat Tail）を生み出す危険性があります。以下の式に基づく完全な線形加法モデルへの移行を推奨します。
$$
Offset = Base + \sum_{i=1}^{n} (w_i \cdot feature_i)
$$
各特徴量（OFI, VPIN, ATRなど）の寄与を独立して制御・制約（関数によるクリッピング）することで、市場の急変時に対しても頑健な（Robust）クオートが可能になります。

### 3.2. Toxicityと流動性の分離
現在の乗務チェーンでは、市場の「毒性（Adverse Selection Risk）」と「流動性（Volatility/Spread）」が混然一体となって乗算に組み込まれています。今後は以下の2つの独立したバッファとしてアーキテクチャを再編することを提案します。
1. **Liquidity Premium**: ATR等をベースとした基本的なスプレッド幅。
2. **Toxic Flow Discount**: OFIやVPINをベースとした、逆選択に対するペナルティ（非対称なクオートシフト）。

## 4. セルフレビューと今後のアクションプラン

過去のドキュメント（564-577）では、eDRCという先進的な概念の導入に注力するあまり、既存コードの型定義追従（Telemetry）や算術演算の順序（Hard Cap）に対する細部の検証が甘かったと自己批判します。

**直近の具体的な修正アクション**:
1. `ztb/metrics/fill_quality.py` の `FillRecord` dataclass に `spread_capture_bps` と `adverse_selection_cost_bps` を追加。
2. `scripts/v460/lib/fill_config.py` 内の `hour_ceiling_mult` 適用ロジックを修正し、最終的な値に対して `edrc_hard_cap` を適用するように変更。
3. `experimental_additive_pipeline` について、今後のフェーズで真の加法演算アーキテクチャに再構築するための設計に着手。

本ドキュメントをもって、Doc 564-577の総括とコードベースへの反映要件を確定とします。
