# 616# Attribution Phase 2 および Live Feature Builder 数理仕様

- **日付**: 2026-03-24
- **目的**: 581# 加法パイプライン環境下での詳細な寄与度分析、およびインプロセス推論を実現するための実運用側特徴量生成構造を定義する。
- **役割**: 数理仕様・設計文書の策定（コード・YAML 変更・テスト実装は担当外）。

---

## §1 Attribution Phase 2：加法 RMS パイプラインの解剖 (T1)

581# で導入された RMS 加法モデルにおける、各成分の寄与度を数理的に定義する。

### 1.1 RMS 寄与度の分解（Euler 分解）
加法パイプラインにおける総オフセット $R$ は以下で構成される。
$$R = R_{base} + \text{tox\_rms} + \text{liq\_rms}$$
ここで $\text{tox\_rms} = \sqrt{\sum_{i \in \text{tox}} (\Delta R_i)^2}$ である。各 $\Delta R_i$ の実効寄与度 $C_i$ を、オイラーの定理に基づき以下のように定義する。

$$C_i = \frac{(\Delta R_i)^2}{\text{tox\_rms}}$$

- **特性**: $\sum_{i \in \text{tox}} C_i = \text{tox\_rms}$ となり、二乗和の平方根という非線形な結合を線形に配分可能。
- **利点**: 「どのリスク要因が RMS を押し上げたか」を bps 単位で一意に説明できる。

### 1.2 RMS Ceiling 接近率
各バッファが最終的な Ceiling ガードに対してどの程度のリザーブを消費しているかを計測する。

$$\text{occupancy}_{\text{tox}} [\%] = \frac{\text{tox\_rms}}{\text{resolve\_offset\_ceiling}(\text{side}) - R_{base}} \times 100$$

- **飽和判定**: この値が 100% を超えた場合、Ceiling による情報の切り捨て（Information Loss）が発生していると判定する。

### 1.3 加法・乗法の比較評価モデル
A/B テストにおいて、Information Loss (IL) の削減効果を定量化する。
- **指標**: Mean Absolute Information Loss (MAIL)
- **検定**: 加法群と乗法群の IL 分布に対し、Mann-Whitney U 検定を適用し、加法群の IL が統計的に有意に小さいことを実証する。

---

## §2 Live Feature Builder の構造仕様 (T2)

インプロセス推論（方式 B）のために、`fill_test` 内部で動作する特徴量生成器の仕様。

### 2.1 メモリ構造と窓幅管理
各サイクルで以下の 1m 窓（60 サンプル想定 at 1Hz 更新）を保持する。
- **データ構造**: `collections.deque` を用いた固定長リングバッファ。
- **保持対象**: `mid_price`, `bid_volume_L5`, `ask_volume_L5`, `last_trade_side`, `last_trade_qty`。

### 2.2 時間解像度の同期（120s Cycle vs 60s Feature）
`fill_cycle` の発火（~120s）を待たずに特徴量を鮮度高く保つための設計。
- **更新フック**: WebSocket 等のデータ受信スレッドから供給される L2 スナップショット更新ごとに特徴量バッファを更新する（Event-driven）。
- **推論タイミング**: `fill_cycle` の開始直前に、バッファから直近 60s 分をスライスしてベクトル化する。

### 2.3 欠損値 (NaN) および Warm-up 処理
- **起動直後**: バッファが 60s 分充填されるまでの期間は、`SidecarStatus.MISSING` として推論をスキップする。
- **一時的な欠損**: 計算過程で NaN が発生した場合は、`sac_sidecar.norm.json` に記録された訓練セットの `mean` 値で置換する（Mean Imputation）。

---

## §3 適応的平滑化 (Adaptive EMA) の数理仕様 (T3)

モデルの信頼度に基づくシグナルの動的な安定化。

### 3.1 動的平滑化係数 $\alpha_t$ の算出
モデルが出力する `confidence` $c_t \in [0, 1]$ を用い、平滑化係数を以下の線形補間で行う。

$$\alpha_t = \alpha_{min} + (\alpha_{max} - \alpha_{min}) \cdot c_t$$

- **初期推奨値**: $\alpha_{min} = 0.05$ (信頼度低: 強い平滑化), $\alpha_{max} = 0.50$ (信頼度高: 素早い追随)。

### 3.2 EMA 更新公式
既存の `RobustStats.asymmetric_ema()` を利用する場合、以下のようにマッピングする。

$$\text{bias}_{t}^{\text{smoothed}} = \text{asymmetric\_ema}(x_t = \text{bias}_t, \text{prev\_ema} = \text{bias}_{t-1}^{\text{smoothed}}, \alpha_{up} = \alpha_t, \alpha_{down} = \alpha_t)$$

- **注記**: 現段階では方向による非対称性は不要なため、`alpha_up` と `alpha_down` に同一の $\alpha_t$ を与える。

---

*以上。本文書は設計仕様であり、実装担当者はこの数理モデルに基づき既存クラス（OffsetPipelineMixin 等）への組込みを行うこと。*
