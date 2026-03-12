# 054# — 収益性改善計画: Adverse Selection 低減 + 最適エントリー・エグジット

| key | value |
|---|---|
| 番号 | 054 |
| フェーズ | ph2 (G1.1-exec) |
| 種別 | plan |
| 前提文書 | 000#, 048#, 052#, 053# |
| 作成日 | 2026-02-15 |
| レビュー対象 | 別 AI コーディングエージェント |

---

## §1 エグゼクティブサマリー

G1.1 暫定判定 **FAIL** (053#: E1 fill_rate 76%, E4 PnL -0.620bps, E5 AS 39.1%)。
§3.9 継続中止ルールは **CONTINUE** (fill_rate>70%, 累積損失 -248 JPY << 10K)。

**しかし round-trip ベースでは月次 p50 +19,265 JPY と正** (053# §4)。
根本課題は **Adverse Selection** (AS) であり、AS fill と non-AS fill の PnL 差は **-7.2 bps**。

本計画はこの -7.2 bps の AS impact を構造的に削減し、G1.1 PASS に必要な 3 条件を達成する。

| 基準 | 現在 | 目標 | 改善に必要な量 |
|------|------|------|-------------|
| E1 fill_rate | 76.0% | ≥ 90% | +14 pp |
| E4 PnL (30s) | -0.620 bps | ≥ 0 bps | +0.620 bps |
| E5 AS_ratio | 39.1% | ≤ 20% | -19.1 pp |

---

## §2 問題の構造分析

### §2.1 なぜ今のシステムは損をしているのか

```
[根本原因] Maker 注文は構造的に逆選択 (AS) に晒される
    │
    ├── 即約定 (≤10s) の AS 率 46% → 市場が済動いた方向の注文を拾わされている
    │   (048# §5.1: 速い約定ほど PnL が悪い、遅い約定は +0.96 bps)
    │
    ├── Side が機械的交互 → 市場の方向と無関係に買い/売りを出す
    │   (048# §9.1: 勝率 50% が理論上限)
    │
    └── Exit 戦略が存在しない → 約定後は放置、次サイクルの反対売買が暗黙の exit
        (048# §2.2: ポジション管理なし)

[結果]
    ├── 個別fill 30s mark-to-market: -0.620 bps (E4 FAIL)
    ├── テール 18件 (6.2%) が -265 bps を生成 (全損失の 46%)
    └── ただし Round-trip は +10.74 JPY/pair (暗黙のexit=反対売買が機能)
```

### §2.2 改善レバーの定量ランキング

| 順位 | レバー | 期待改善量 | 難易度 | 根拠 |
|------|--------|----------|--------|------|
| **1** | **AS 回避 (orderbook imbalance ベース)** | **+2.8 bps** (weighted) | 中 | 053# §4.3: AS impact -7.2bps, AS 排除時 +2.8bps 改善 |
| **2** | **テール損失カット** | **+1.8 bps** | 中 | 048# §8.2: SL=1bps で -265bps → -90bps (175bps 改善/291件) |
| **3** | **Side 方向性 (imbalance 追従)** | **+0.5-1.0 bps** | 中 | 048# §9.2 P3: 勝率 50% → 52-55% で +0.5-1.0 bps |
| **4** | **時間帯最適化 (残 skip hours)** | **+0.3 bps** | 低 | 052# 分析: 既に 12/24h をスキップ。追加余地は小 |
| **5** | **オフセット最適化 (spread 適応)** | **+0.2 bps** | 低 | 方策 A 稼働中。微調整の余地あり |

### §2.3 「もう少し待ったら儲かる」仮説

E3 データ (n=26) が示唆:

| 所見 | データ | 含意 |
|------|--------|------|
| 30s 負 → 120s で回復 | 50% (7/14) | **Mean reversion が存在** |
| 30s 正 → 120s でも正 | 75% (9/12) | **利益は持続する** |
| 30s mean → 60s → 120s | +0.21 → -0.62 → +0.10 | **U字曲線**: 60s で底打ち後に回復 |

**解釈**: maker 約定直後は AS で不利な方向に動くが、**120s の時間尺度で mean revert する傾向**がある。つまり「もう少し待てば」仮説は部分的にデータに支持される。

ただし n=26 は統計的に脆弱であり、confidence interval は広い。これを受けて:
1. E3 データ蓄積を継続し、n≥100 で再評価
2. 現時点では **保守的な「待つ」戦略** (offset 深化)で AS を回避し、fill 後の mean reversion に賭ける設計

---

## §3 000# との整合性マトリクス

| 000# 規定 | 本計画との関係 | 整合性 |
|-----------|-------------|--------|
| §2 ph2: maker 執行可能性検証 | fill test の改善は ph2 の範囲内 | ✅ |
| §3.3 G1.1: fill rate, PnL, AS | 全 3 FAIL 項目を直接改善 | ✅ |
| §3.9 CONTINUE | 中止条件に該当せず、改善継続は正当 | ✅ |
| §4 SAC は G1 通過後 | RL/SAC は使用しない。ルールベースのみ | ✅ |
| §4 方策 A: パラメータ適応 | 本計画の改善は方策 A の高度化 | ✅ |
| §4 方策 B: 動的ロット | 変更なし (lot_sizing 維持) | ✅ |
| 034# 行動空間 | エージェントの行動空間は変更しない | ✅ |

**制約**: ph2 では SAC/RL は使用不可。全改善はルールベースまたは設定変更のみ。

---

## §4 改善施策の詳細設計

### §4.1 S1: Orderbook Imbalance ベース AS 予測フィルター

**分類**: ph2 ルールベース改善 (方策 A 高度化)
**期待改善**: E4 +2.8 bps, E5 -15 pp
**優先度**: P0 (最大レバー)

#### 現状の問題

現在の `_compute_maker_price()` は板の best bid/ask のみを参照。板の深さ (depth) 情報を無視しており、**informed flow を事前に検知できない**。

```python
# 現状: depth=1 のみ取得
ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
```

#### 改善設計

**板不均衡 (imbalance)** をリアルタイムに計算し、不利な方向のフローが強い場合にオフセットを自動拡大 or スキップ。

```python
async def _compute_orderbook_imbalance(self, depth: int = 5) -> float:
    """板不均衡を計算.

    Returns:
        imbalance ∈ [-1, +1].
        +1 = bid 側が圧倒的 (買い圧力 = 価格上昇示唆).
        -1 = ask 側が圧倒的 (売り圧力 = 価格下落示唆).
    """
    ob = await self.adapter.get_orderbook(self.config.symbol, depth=depth)
    bid_volume = sum(qty for _, qty in ob.bids[:depth])
    ask_volume = sum(qty for _, qty in ob.asks[:depth])
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    return (bid_volume - ask_volume) / total
```

**AS 予測ロジック** (side と imbalance の不一致を検知):

```python
imbalance = await self._compute_orderbook_imbalance()

# buy 注文だが ask が圧倒的 (売り圧力強い)
#   → 約定後に価格下落 → AS リスク高い → offset 拡大
if side == "buy" and imbalance < -self.config.imbalance_threshold:
    effective_offset_ratio *= self.config.imbalance_offset_boost  # e.g., ×2.0
    logger.info(f"[imbalance] buy side AS risk: imb={imbalance:.3f}, offset boosted")

# sell 注文だが bid が圧倒的 (買い圧力強い)
#   → 約定後に価格上昇 → AS リスク高い → offset 拡大
elif side == "sell" and imbalance > self.config.imbalance_threshold:
    effective_offset_ratio *= self.config.imbalance_offset_boost
    logger.info(f"[imbalance] sell side AS risk: imb={imbalance:.3f}, offset boosted")
```

**設定追加** (fill_test.yaml):

```yaml
imbalance:
  enabled: true
  depth: 5                    # 板深さ (上位 N 段)
  threshold: 0.3              # imbalance 絶対値がこれ以上で「偏り」と判定
  offset_boost: 1.5           # AS リスク時の offset 倍率
  skip_threshold: 0.7         # imbalance がこれ以上なら注文自体をスキップ
```

#### データ収集 (先行実施可能)

fill_test 再開前に `FillRecord` に `orderbook_imbalance` フィールドを追加し、imbalance と AS/PnL の相関を定量化:

```python
# FillRecord 追加フィールド
orderbook_imbalance: Optional[float] = None  # [-1, +1]
bid_depth_total: Optional[float] = None      # bid 側合計数量
ask_depth_total: Optional[float] = None      # ask 側合計数量
```

---

### §4.2 S2: Smart Side Selection (交互→条件付き)

**分類**: ph2 ルールベース改善
**期待改善**: E4 +0.5-1.0 bps (勝率 50% → 52-55%)
**優先度**: P1

#### 現状の問題

```python
def _next_side(self) -> str:
    if self._last_side is None or self._last_side == "sell":
        return "buy"
    return "sell"  # 常に交互
```

市場方向を完全に無視。BTC が上昇トレンド中でも sell を強制発注し、AS を喰らう。

#### 改善設計 — 3 モード

```python
def _next_side(self) -> str:
    """side 決定: imbalance + 交互の hybrid.

    Mode A (default): 交互ベース + imbalance による抑制
    Mode B: imbalance 追従 (トレンド追随)
    Mode C: imbalance 逆張り (mean reversion)
    """
    base_side = "buy" if self._last_side != "buy" else "sell"

    if not self.config.smart_side_enabled:
        return base_side

    imbalance = self._last_imbalance  # 直前の _compute_maker_price で取得済み

    if self.config.smart_side_mode == "suppress":
        # Mode A: 不利な side の場合、スキップして同じ side を繰り返す
        # buy を出そうとしているが、売り圧力が強い → buy をスキップ
        if base_side == "buy" and imbalance < -self.config.imbalance_threshold:
            logger.info(f"[smart_side] Suppressing buy (imb={imbalance:.3f})")
            return self._last_side or "sell"  # 同じ side 継続
        if base_side == "sell" and imbalance > self.config.imbalance_threshold:
            logger.info(f"[smart_side] Suppressing sell (imb={imbalance:.3f})")
            return self._last_side or "buy"
        return base_side

    elif self.config.smart_side_mode == "follow":
        # Mode B: imbalance の方向に素直に追従
        if abs(imbalance) > self.config.imbalance_threshold:
            return "buy" if imbalance > 0 else "sell"
        return base_side  # imbalance が弱い場合は交互

    return base_side
```

**注意**: 009# §4.2 「片側ポジション蓄積禁止」を遵守するため、連続同 side は最大 2 回まで。

```yaml
smart_side:
  enabled: true
  mode: suppress              # suppress / follow
  max_consecutive_same: 2     # 片側蓄積防止 (000# §3.3 安全設計)
```

---

### §4.3 S3: テール損失カット (動的 cycle interval 短縮)

**分類**: ph2 ルールベース改善
**期待改善**: E4 +1.8 bps (テール制御)
**優先度**: P0

#### 現状の問題

048# §3.4: 18 件 (6.2%) の ≤-10bps テール損失が累積損失の 46% を生成。
現在は約定後に 30s/60s/120s の PnL を観察するのみで、**損失遮断の仕組みが存在しない**。

#### 改善設計 — Post-fill Early Warning + Rapid Exit

```python
# 約定後 30s PnL 計測時に、途中 (5s刻み) で mid を監視
if filled and fill_price is not None:
    mid_at_fill = await self._get_mid_price()

    # Early Warning Phase: 5s 刻みで 30s まで監視
    early_exit_triggered = False
    for tick in range(1, 7):  # 5, 10, 15, 20, 25, 30s
        await asyncio.sleep(5.0)
        try:
            mid_now = await self._get_mid_price()
            interim_pnl = self._calc_pnl(side, mid_at_fill, mid_now)

            # テール損失の早期検知: -5 bps 到達で即フラグ
            if interim_pnl < -self.config.early_exit_threshold_bps:
                logger.warning(
                    f"[early_exit] Loss threshold hit at {tick*5}s: "
                    f"{interim_pnl:+.2f} bps < -{self.config.early_exit_threshold_bps}"
                )
                early_exit_triggered = True
                break
        except Exception:
            continue

    if early_exit_triggered:
        # 次サイクルの interval を 0 にして即座に反対注文
        # = 実質的な "exit" (maker ベースなので成行は不可)
        self._rapid_exit_pending = True
        self._rapid_exit_side = "sell" if side == "buy" else "buy"
```

**設定**:

```yaml
early_exit:
  enabled: true
  threshold_bps: 5.0          # この損失に達したら rapid exit フラグ
  monitoring_interval_sec: 5   # 監視刻み
  rapid_exit_interval_sec: 10  # rapid exit 時の cycle interval (通常 120s → 10s)
```

#### Ph3 への接続

この monitoring loop は将来の SAC エージェントに「exit timing」の観察値として渡すことができる。Ph2 ではルールベースの閾値だが、ph3 では RL エージェントが最適な exit タイミングを学習する基盤となる。

---

### §4.4 S4: Spread-Responsive Offset (スプレッド適応型オフセット)

**分類**: ph2 設定最適化 (方策 A 拡張)
**期待改善**: E4 +0.2 bps, E1 +3 pp
**優先度**: P2

#### 現状の問題

- `spread_offset_ratio = 0.05` は**固定値**
- 048# §5.2: 狭スプレッド (<1500 JPY) で PnL -2.79 bps vs 広スプレッド (≥3000) で -0.66 bps
- 狭スプレッド = 流動性集中 = informed flow が多い = AS リスク大

#### 改善設計

```python
def _adaptive_offset_ratio(self, spread: float) -> float:
    """スプレッド幅に応じてオフセット比率を動的調整.

    狭スプレッド → 高比率 (保守的、AS 回避)
    広スプレッド → 低比率 (積極的、fill 優先)
    """
    base = self.config.spread_offset_ratio
    spread_bps = spread / self._last_mid_price * 10000

    if spread_bps < self.config.narrow_spread_bps:
        return min(base * self.config.narrow_spread_boost, 0.30)
    elif spread_bps > self.config.wide_spread_bps:
        return max(base * self.config.wide_spread_ratio, 0.01)
    return base
```

```yaml
spread_adaptive:
  enabled: true
  narrow_spread_bps: 10.0     # これ以下で「狭い」判定
  narrow_spread_boost: 2.0    # 狭い時の offset 倍率
  wide_spread_bps: 25.0       # これ以上で「広い」判定
  wide_spread_ratio: 0.5      # 広い時の offset 割引
```

---

### §4.5 S5: データ拡張 — FillRecord の充実

**分類**: ph2 データ基盤 (先行実施可能)
**期待改善**: 直接なし (分析精度向上)
**優先度**: P0 (他施策の基盤)

`FillRecord` に以下のフィールドを追加し、AS 予測モデルの教師データを蓄積:

```python
# ztb/metrics/fill_quality.py — FillRecord 追加
orderbook_imbalance: Optional[float] = None   # 板不均衡 [-1, +1]
bid_depth_total: Optional[float] = None       # bid 側合計 BTC
ask_depth_total: Optional[float] = None       # ask 側合計 BTC
mid_price_trend_5s: Optional[float] = None    # 直前 5s の mid 変化率 (bps)
spread_bps: Optional[float] = None            # 発注時スプレッド (bps)
effective_offset_ratio: Optional[float] = None # 実際に適用された offset 比率
```

---

## §5 実装フェーズと 000# Gate 整合

### §5.1 Phase 分割

```
Phase 2 (現在)
  │
  ├── Step 1: S5 データ拡張 (FillRecord 新フィールド)
  │   ├── 入金不要 (コード変更のみ)
  │   └── fill test 再開前に実装完了すべき
  │
  ├── Step 2: S1 Imbalance フィルター + S4 Spread 適応
  │   ├── _compute_maker_price 改修
  │   └── fill test 再開と同時に効果検証開始
  │
  ├── Step 3: S2 Smart Side (suppress mode)
  │   ├── _next_side 改修
  │   └── Step 2 の imbalance 計算を流用
  │
  ├── Step 4: S3 テール損失カット
  │   ├── post-fill monitoring loop 追加
  │   └── rapid exit メカニズム
  │
  ├── Step 5: E3 データ分析 (n≥100)
  │   └── 30/60/120s の情報価値を定量評価
  │
  └── Step 6: G1.1 再判定 (新データ n≥200)
      ├── E1-E5 全項目を再評価
      └── PASS → ph3 (SAC), FAIL → 施策追加 or v461

Phase 3 (G1.1 PASS 後)
  │
  ├── SAC に imbalance / spread_bps / regime を特徴量として供給
  ├── S3 の早期監視を RL のリアルタイム exit signal に拡張
  └── SAC が offset / side / exit timing を最適化
```

### §5.2 依存関係

```
S5 (data) ──→ S1 (imbalance) ──→ S2 (smart side)
                    │
                    └──────────→ S4 (spread adaptive)

S3 (tail cut) ← 独立 (他に依存しない)

E3 analysis ← 独立 (fill test 再開 + 時間経過で自動蓄積)
```

### §5.3 先行実施可能なタスク (入金前・fill test 停止中)

| タスク | 内容 | 依存 |
|--------|------|------|
| **T1** | FillRecord 新フィールド追加 (S5) | なし |
| **T2** | `_compute_orderbook_imbalance()` 実装 | なし |
| **T3** | `_compute_maker_price` への imbalance 統合 (S1) | T1, T2 |
| **T4** | `_adaptive_offset_ratio` 実装 (S4) | T1 |
| **T5** | `_next_side` 改修 (S2 suppress mode) | T2, T3 |
| **T6** | Early exit monitoring loop (S3) | なし |
| **T7** | Config 拡張 (fill_test.yaml) | T1-T6 |
| **T8** | 単体テスト追加 (各施策) | T1-T7 |

**全て入金前に実装可能。** fill test 再開時に直ちに新ロジックでデータ収集を開始できる。

---

## §6 リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| Imbalance フィルターが過敏 → fill rate 低下 | E1 悪化 | threshold / boost を保守的に開始 (0.3 / 1.5) |
| Smart side がトレンド方向を誤認 | PnL 悪化 | suppress mode (消極型) で開始。max_consecutive=2 |
| テール損失カットの早期発動 → 利益機会喪失 | Profit Factor 低下 | threshold_bps=5.0 (P99=+12.7 をカバー) |
| 新フィールド追加でレコードサイズ増 | ディスク・保存速度 | float 6個 ≈ 48 bytes / record → 無視可能 |
| 板情報 depth=5 の取得コスト | API レート制限 | 現行 depth=1 → depth=5: 同じ API、追加コストなし |
| 複数施策の同時投入で効果分離不能 | 分析困難 | 各施策に ON/OFF config を設け、段階的に有効化 |

---

## §7 「もう少し待ったら儲かる」— 設計哲学

### §7.1 Maker の本質的優位性

Maker は**待つ者**である。Taker が「今すぐ取引したい」衝動を持つ一方、Maker は板に注文を置き、**有利な価格で取引が来るのを待つ**。この非対称性が maker の唯一のエッジ。

現在のシステムの問題は、この「待つ」本質を活かせていないこと:

| 要素 | 現状 | あるべき姿 |
|------|------|-----------|
| 注文配置 | best bid/ask の近く (offset=0.05) | **板の奥 (offset 拡大)** で更に有利な価格で待つ |
| Side 決定 | 機械的交互 | **市場圧力と逆方向** に注文を置いて待つ (逆張り) |
| 約定後 | 30s 計測して放置 | **mean reversion を待って** exit |
| fill rate | 90% を目標 | fill rate は 70-80% で十分。**約定の質 > 約定の量** |

### §7.2 E4 PnL ≥ 0 と E1 fill rate ≥ 90% のトレードオフ

E4 (PnL≥0) と E1 (fill≥90%) は構造的にトレードオフの関係:

- **攻撃的 offset (小)**: fill rate ↑, PnL ↓, AS ↑
- **保守的 offset (大)**: fill rate ↓, PnL ↑, AS ↓

048# §5.1 のデータがこれを証明:
- 速い約定 (≤10s): AS 46%, PnL -0.88 bps
- 遅い約定 (>60s): AS 27%, PnL **+0.96 bps**

**戦略的判断**: E1 の 90% は**固定オフセットでは不達成可能**。
Imbalance に応じた**条件付き offset**で:
- AS リスクが低い時: 攻撃的 offset → fill rate 貢献
- AS リスクが高い時: 保守的 offset → PnL 保護

この動的制御により **E1 と E4 の同時達成** を狙う。

### §7.3 v451 "Golden Era" の教訓

034# が示す通り、v451 (γ=0.80, PnL only, 3値離散) が歴代最良結果を出した。
v451 の特徴: **シンプルさ**。複雑な行動空間拡張は全て失敗している。

本計画の改善は:
- ルールベース (SAC 不使用)
- 各施策に ON/OFF switch
- 変更点は `_compute_maker_price`, `_next_side`, post-fill monitoring のみ
- **核心ロジックを増やさず、情報入力を増やす** (imbalance, spread_bps)

---

## §8 別 AI エージェントへのレビュー依頼事項

1. **S1 Imbalance フィルターの threshold 設計は妥当か？**
   - imbalance = 0.3 で「偏り」と判定しているが、BTC/JPY 板の典型的な imbalance 分布を踏まえると、この閾値は適切か？
   - depth=5 は十分か？ depth=10 や depth=20 の方が安定した信号が得られるか？

2. **S2 Smart Side の suppress mode は最適か？**
   - Side 抑制 (不利な side をスキップ) と follow mode (imbalance 追従) のどちらが理論的に正しいか？
   - BTC/JPY の typical mean reversion time (数分？数十分？) に依存するが、データはあるか？

3. **S3 テール損失カットの threshold_bps=5.0 は妥当か？**
   - 048# のデータでは P99=+12.7 bps。5 bps で切ると、利益側も +5 以上を喪失するリスクは？
   - asymmetric TP/SL (TP unlimited, SL=5) の方が良いか？

4. **E1 fill rate ≥ 90% は現実的な目標か？**
   - AS 低減のために offset を拡大すると fill rate は下がる。
   - **E4 PnL ≥ 0 が最優先**であり、E1 は 70-80% でも §3.9 CONTINUE (≥70%)。
   - G1.1 のゲート基準自体の改訂 (E1 閾値を 80% に緩和) は、000# §3.3 の精神に反するか？

5. **Round-trip PnL (+10.74 JPY) vs 30s mark-to-market (-0.620 bps) の乖離をどう解釈すべきか？**
   - Round-trip が正なのは「交互売買による暗黙の exit」が効いているから？
   - それとも「30s は短すぎ、実際の PnL 回収には 120-300s 必要」か？
   - post_fill_wait_sec を 120s に延長して E4 を再評価すべきか？

6. **この計画全体の最大の盲点は何か？**

---

## §9 成功基準

| 指標 | 現在 | Step 6 時点の目標 | 判定方法 |
|------|------|----------------|---------|
| E1 fill_rate | 76.0% | ≥ 80% (改善方向) | 新データ n≥200 |
| E4 PnL (30s) | -0.620 bps | **≥ 0 bps** | 新データ mean |
| E5 AS_ratio | 39.1% | **≤ 25%** (大幅改善) | 新データ ratio |
| テール損失 (≤-10bps) | 6.2% (18/291) | ≤ 2% | 新データ分布 |
| Round-trip PnL/pair | +10.74 JPY | ≥ +15 JPY | ペアリング分析 |

**G1.1 PASS には E4/E5 の達成が先決**。E1 は E4/E5 改善に伴う fill rate 低下を S4 (spread 適応) で補填。

---

## Appendix A: 分析データソース

| データ | 出典 | n |
|--------|------|---|
| Fill records (3日間) | results/v460/fill_test/ | 491 (373 filled) |
| E3 multi-timeframe | 同上 (60s/120s fields) | 26 |
| Round-trip pairs | 053# Monte Carlo 分析 | 181 pairs |
| 時間帯 PnL/AS | 048# §4.1 | 291 filled |
| 約定速度 vs PnL | 048# §5.1 | 291 filled |
| TP/SL simulation | 048# §8.2 | 291 filled |

## Appendix B: 実装対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ztb/metrics/fill_quality.py` | FillRecord 新フィールド追加 (S5) |
| `scripts/v460/run_fill_test.py` | S1-S4 全施策の実装 |
| `configs/v460/fill_test.yaml` | 新セクション追加 |
| `tests/unit/v460/test_fill_quality.py` | 新フィールドテスト |
| `tests/unit/v460/test_fill_test_config.py` | config テスト |
