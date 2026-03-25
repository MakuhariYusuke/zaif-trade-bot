# 623# hard_skip_utc_hours 廃止: 536# 渙原則に基づく動的防御委譲

- **日付**: 2026-03-25
- **著者**: Copilot
- **コミット**: `b8842a1f0` (config), `a591fc687` (docs)
- **種別**: config
- **目的**: 205# §9.4 で導入された固定時刻 Hard Skip を廃止し、536# 原則に基づき動的防御層に委譲

---

## §1 背景

### 導入経緯

| 文書 | 内容 |
|------|------|
| **204#** | 時間帯別分析で JST 01h (UTC 16) AS率64%、JST 06h (UTC 21) PnL -125.8bps/日 を検出 |
| **205# §9.4** | Kyle (1985) 流動性モデルを根拠に「Hard Skip は止血として正しい」と結論。`hard_skip_utc_hours: [16, 21]` 実装 |

### 廃止理由

**536# 渙原則**: 「12時・15時の帯域での売りのみ規制する」等の時間帯ハードコードを「弥縫策のパッチ当て」と批判。「固定のIF文に頼ることでもありません（§3）」と明記。

205# §9.4 時点の止血としては正当であったが、以下の **4層の動的防御** がその後積み上がり、hard_skip の役割は完全に上書きされていた:

1. **skip_gate hour_offsets** (158# P1-6): ML 閾値の時間帯別調整。UTC 16 → +0.5bps、UTC 21 → +0.3bps
2. **hour_ceiling_mult** (467# P2): 危険時間帯での offset ceiling 緩和
3. **sell_hour_offset_boost** (310# A): sell 側の時間帯別 offset 乗数
4. **regime_thresholds** (620# 修正済み): regime 別の skip_gate floor enforcement

### デッドコード問題

hard_skip は pre-cycle **Step 5** で発火し cycle 全体を `continue` でスキップ。skip_gate 評価（Step 14 内部）に到達しないため、以下の設定が **到達不能なデッドコード** になっていた:

| 設定 | 時間帯 | 状態 |
|------|--------|------|
| `skip_gate_hour_offsets[16] = 0.5` | JST 01h | hard_skip が先に発火し未到達 |
| `skip_gate_hour_offsets[21] = 0.3` | JST 06h | 同上 |
| `sell_hour_offset_boost[16] = 1.5` | JST 01h | 同上 |

---

## §2 変更内容

### hard_skip 廃止

```yaml
# Before
hard_skip_utc_hours: [16, 21]

# After (623#)
hard_skip_utc_hours: []  # 536# 渙原則: 固定時刻 IF 文パッチ → 動的防御に委譲
```

### 防御補完（旧デッドコードの有効化 + ceiling 追加）

hard_skip 廃止により pipeline に到達するようになった UTC 16 / 21 に対し、ceiling 防御を追加:

```yaml
hour_ceiling_mult:
  # 既存 ...
  16: 2.0    # 623# JST 01h: AS64% → hard_skip 廃止に伴い ceiling 防御追加
  21: 1.5    # 623# JST 06h: PnL-125.8bps/日 → 同上

sell_hour_offset_boost:
  # 既存 ...
  21: 1.5    # 623# JST 06h → hard_skip 廃止に伴い追加
```

### 防御マトリクス（変更後）

| UTC hour | JST | skip_gate offset | hour_ceiling_mult | sell_hour_boost | 旧 hard_skip |
|----------|-----|----------------:|------------------:|----------------:|:------------:|
| 16 | 01h | +0.5 | ×2.0 | ×1.5 | ~~全停止~~ → 廃止 |
| 21 | 06h | +0.3 | ×1.5 | ×1.5 | ~~全停止~~ → 廃止 |

3層の動的防御が連携し、固定時刻パッチに頼らずマイクロストラクチャに基づくリスク管理を実現。

---

## §3 コード影響

`_check_hard_skip_utc()` （orchestrator_pre_cycle.py L326-361）のコードは残存。`hard_skip_utc_hours: []` により `if not self.config.hard_skip_utc_hours: return False` で即座に抜けるため、実質無効化。コードパスの健全性は維持。

CancelReason `HARD_SKIP_UTC_HOUR` 定数およびテスト（test_168, test_145, test_244, test_286）も残存。将来的に緊急止血が必要な場合に YAML 変更のみで即座に復活可能。

---

## §4 テスト

全 2237 テスト pass（127 skipped, 81 warnings）。既存テストは cancel_reason 定数の存在確認・デフォルト値テストのため影響なし。

---

## §5 536# 渙原則との整合

> 「渙」は「氷解・散らす」の意。凍りついた水（硬直化した固定値や事後対応ルール）に春の風を通して、再び流動化させるべき時を示します。（536# §0）

| 536# の指摘 | 623# での対応 |
|------------|--------------|
| 「時間帯などの固定値にこだわるのをやめ、動的な微視的構造へ移行せよ」 | hard_skip（固定時刻全停止）→ skip_gate ML + ceiling + boost（動的3層） |
| 「固定のIF文（12時だから止める、等）に頼ることでもありません」 | UTC 16/21 の全停止 IF → 予測 PnL ベースの skip_gate 評価に委譲 |
| 「廟（絶対にブレない神髄）だけは強固に残す」 | コード・定数・テストは残存、YAML 変更のみで即座に復活可能 |
