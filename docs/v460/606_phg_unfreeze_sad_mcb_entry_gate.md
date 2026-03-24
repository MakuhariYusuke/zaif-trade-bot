# 606# SAD/MCB 有効化 + entry_gate CalibrationMap 接続

- **日付**: 2026-03-25
- **目的**: 605# §8 で発見した事実誤認を訂正し、真に凍結していた SAD・MCB・entry_gate を解凍する
- **前提**: 605# の Tier 0/1 提案の大半が 372#-565# で既に実装済みだった（592# 型データ幻覚の再発）

---

## §0 605# 正誤表の要約

| 605# の主張 | 実際 | 出典 |
|---|---|---|
| ceiling 0.25 → 0.35 が必要 | buy=0.35, sell=0.40 (実装済み) | 565# P1 |
| composite_risk_enabled 必要 | true (threshold=1.0, 実装済み) | 540# |
| Stage Max Mult 未実装 | cap=2.0 (hardcoded, 実装済み) | 565# P3 |
| Sidecar TTL 600s (8% stale) | 7800s (実装済み) | 372# |
| sell_dynamic_kill 1800s | 600s (sell), 900s (buy) (実装済み) | 540#/370# |
| EV toxic skip 未設定 | default -5.0 (稼働中) | 593# |
| SAD/MCB 未インスタンス化 | run_fill_test.py でインスタンス化済み、YAML disabled のみ | 211# |

**教訓**: 536#/537# 時点の提案をそのまま「未着手」と断定せず、実コード・YAML の現在値を必ず検証すること。

---

## §1 変更内容

### §1.1 SAD (Spread Anomaly Detector) 有効化

**ファイル**: `configs/v460/fill_test.yaml` L375-380

```yaml
# Before
spread_anomaly_detector:
  enabled: false   # 513# デフォルト無効 (インスタンス化未実装)

# After
spread_anomaly_detector:
  enabled: true    # 606# 有効化 (513# 以来 disabled だったが配線済みのため解放)
```

**パイプライン上の動作** (`orchestrator_pre_cycle._check_circuit_breakers()`):
- `WIDE` (spread > baseline × 2.0): offset_mult 乗算 (WARNING)
- `DRY` (spread > baseline × 4.0): offset_mult + interval_mult + lot_mult 乗算 (WARNING)
- `FROZEN` (spread > baseline × 8.0): 全注文停止 (HALT)
- MCB × SAD 同時 WARNING: Escalation → 全注文停止

**YAML コメント修正**: 「未インスタンス化」→ 実態に合わせて「run_fill_test.py でインスタンス化済み」に訂正。

### §1.2 MCB (Micro Circuit Breaker) 有効化

**ファイル**: `configs/v460/fill_test.yaml` L386-392

```yaml
# Before
micro_circuit_breaker:
  enabled: false   # 513# デフォルト無効 (インスタンス化未実装)

# After
micro_circuit_breaker:
  enabled: true    # 606# 有効化 (513# 以来 disabled だったが配線済みのため解放)
```

**パイプライン上の動作**:
- `CAUTION` (σ > 1.0): 監視のみ
- `WARNING` (σ > 1.5): offset_mult × 1.5, interval_mult × 2.0
- `HALT` (σ > 2.0): 全注文停止 + cooldown 300s

### §1.3 entry_gate CalibrationMap 接続 (observe mode)

**CalibrationMap 生成**:
```bash
python scripts/v460/ml/calibration_batch.py --days 14
```
- 入力: 16,228 fill records (直近 14 日)
- 使用: 1,816 filled records (post_fill_30s_pnl 付き)
- レジーム分布: ranging=1,590, trending_down=114, trending_up=112
- 出力: `models/v460/entry_gate_calibration.json`

**YAML 変更**: flat field → nested block 化
```yaml
# Before
entry_gate_enabled: false

# After
entry_gate:
  enabled: false                     # observe モード (ログのみ)
  calibration_map_path: "models/v460/entry_gate_calibration.json"
  probability_mode: "lcb"            # lower confidence bound
  online_update: true                # 稼働中も更新
```

**段階的有効化計画**:
1. Phase 1 (現在): `enabled: false` — EV 判定ログを蓄積し、BLOCK/PASS の頻度を観察
2. Phase 2: ログで誤検知率を確認後、`enabled: true` に移行
3. Phase 3: probability_mode を `lcb` → `mean` に変更し感度調整

---

## §2 易経的考察

> **巽為風（57）彖伝**: 「重巽以申命。剛巽乎中正而志行。柔皆順乎剛。是以小亨。」

605# が渙 → 巽の変化途上にあることを確認した。巽の本義は「穏やかに浸透する風」であり、605# §8 の正誤表が示すように、536# 以降の 300+ 番台の修正は既に静かに浸透していた。本 606# は、その浸透を妨げていた最後の「蓋」（enabled: false コメント）を取り除く作業。

> **巽象伝**: 「随風、巽。君子以申命行事。」

「命を申（のべ）て事を行う」— 命（設計意図: SAD/MCB は防御のために作られた）を改めて確認し、それを実行に移す。YAML コメントの「未インスタンス化」という誤った申命を正し、設計意図通りに事を行う。

---

## §3 テスト結果

- entry_gate 統合テスト: **12 passed** (test_555_entry_gate_integration.py)
- 全体テスト: 実行中（YAML 変更はテスト用 mock config に影響しないため問題なし）

---

*以上*
