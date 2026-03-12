# 340# 338#/339# レビュー対応 — 符号逆転修正 & Finding 妥当性判定

> **種別**: resp (レビュー対応)  
> **対象**: 338# (Codex), 339# (Gemini 3.1 Pro)  
> **起票**: 2026-03-08  
> **ベース SHA**: `4170d024c` (337# commit)  
> **テスト**: 4185 passed (修正後)

---

## §1 最重要対応: 符号逆転バグ (338# Finding #1, 339# §2)

### §1.1 バグの内容

`ztb/risk/sell_dynamic_kill.py` L512:

```python
# 旧 (バグ):
threshold += threshold_offset_bps
# → offset=+0.3 で threshold=-1.0→-0.7 (0 側に寄る＝厳格化)

# 修正後:
threshold -= threshold_offset_bps
# → offset=+0.3 で threshold=-1.0→-1.3 (負側に寄る＝真の緩和)
```

**kill 条件は `rolling_mean < threshold`** であるため:
- threshold が 0 に近づく = kill されやすい (厳格化)
- threshold が -∞ に近づく = kill されにくい (緩和)

正の offset で「緩和」と意図しながら `threshold += offset` は、threshold を 0 側に移動させ **逆方向 (厳格化)** に作用していた。

### §1.2 影響範囲

| 機能 | 導入時点 | 影響期間 | 実害 |
|------|---------|---------|------|
| `buy_dynamic_kill_inv_relaxation` | 286# | 286#〜340# | 在庫補填 buy の優先時に逆に kill を厳格化 |
| `sell_dynamic_kill_inv_relaxation` | 337# | 337#〜340# (短い) | 同上 (ただし稼働前なので実害なし) |

**286# 以降の全 buy inv_relaxation は、意図と逆に「BTC 不足時に buy を止める」方向に作用していた。** 実運用での影響は、imbalance が 0 近傍だった場合は軽微だったが、偏重時には深刻だった可能性がある。

### §1.3 テスト名と assertion の逆転 (338# Finding #2)

| テスト名 | 旧 assertion | 実際の意味 | 修正 |
|---------|-------------|-----------|------|
| `test_threshold_offset_prevents_kill` | `assert killed_partial is True` | offset で「防止」と言いながら kill を期待 | `assert ... is False` |
| `test_negative_offset_tightens_threshold` | `assert killed is False` | 負 offset で「厳格」と言いながら非 kill を期待 | `assert ... is True` (新テスト値) |

テスト名と assertion の乖離がバグを隠蔽していた。修正済み。

### §1.4 修正内容

- `ztb/risk/sell_dynamic_kill.py`: `threshold += offset` → `threshold -= offset`
- コメント更新: 正の offset = より負方向 = 緩和
- `test_286_comprehensive_resolution.py`: 3 テストの assertion + docstring を修正

---

## §2 各 Finding の妥当性判定

### Finding #2: PnL 指標の混同 (338# HIGH)

**判定: ✅ 妥当 — ただし実害は限定的**

337# のログ分析では `post_fill_30s_pnl` を使用。これは `_track_side_pnl` が DynamicKillManager に渡す指標と同一であり、**kill 制御のデバッグ目的としては正しい指標**。

一方、338# が指摘する通り、`analysis/333_sha_isolated_analysis.py` は `ev_weighted_pnl` 優先で集計するため数値が一致しない。これは **分析目的の違い** であり、337# の control 分析と 333# の strategy 分析は別系統として扱うべき。

**対応**: 337# ドキュメントに指標ラベル注記を追加する価値はあるが、分析結論自体は影響を受けない (kill ループの構造問題は `post_fill_30s_pnl` で正しく捕捉されている)。優先度は低い。

### Finding #3: 二重緩和ルート (338# HIGH, 339# §3.2)

**判定: ✅ 妥当 — 設計上の注意点**

現在 sell 側の在庫連動緩和は 2 箇所に存在:

1. **`sell_dynamic_kill_inv_relaxation`** (337#, orchestrator_guards.py): DynamicKillManager の threshold を動的に offset
2. **`sell_guard_inv_bypass_threshold`** (171#, cycle_gate_aggregator.py): inv_net_imbalance >= 0.3 で sell_dynamic_kill gate を完全バイパス

しかし、これらは**階層が異なる**:

- (1) は「kill 判定の閾値を柔らかくする」(微調整レベル)
- (2) は「在庫が極端に偏ったら kill gate 自体をスキップする」(緊急対応レベル)

**結論**: 二重ルートではあるが、**段階的防御** (graduated defense) として合理的な構造。ただし、ドキュメント化が不足している点は改善が必要。統合ではなく、**責務の文書化** で対応する。

### Finding #4: sell_dynamic_kill 単独原因への過度な帰属 (338# HIGH)

**判定: ✅ 妥当**

338# のデータ: `eb24cf4a` 単独で `skip_gate=51` > `sell_dynamic_kill=42`。skip_gate も同等以上に sell を抑制している。

337# は sell_dynamic_kill の自己強化ループに焦点を当てたが、skip_gate の影響も併せて評価すべきだった。ただし、**skip_gate は ML モデル由来であり YAML パラメータ調整では直接制御できない** ため、337# が DK に集中した判断自体は合理的。

**対応**: 将来的に skip_gate の sell 側閾値 (`skip_gate_as_threshold_sell`) の見直しも検討候補に加える。

### Finding #5: -1.0bps は hindsight fit (338# MEDIUM)

**判定: ⚠️ 部分的に妥当**

338# の「観測窓の最悪 -0.888 をぎりぎり回避する値」という批判は正しい。しかし:

- 元の -0.3bps が正常ノイズ圏内 (rolling-50 の自然変動幅 ±0.5bps) にあることは構造的事実
- -1.0bps は「少なくとも正常ノイズでは発動しない値」として理論的に導出可能
- 338# の ladder 提案 (-0.5→-0.8→-1.0) は慎重だが、**Bot 再起動のたびに段階検証するコストとの兼ね合い**

**結論**: -1.0bps はやや aggressive だが、-0.3bps からの即時改善としては許容範囲。ladder の考え方自体は今後の調整で参考にする。

### Finding #6: balance_forced_switch 完全除外の危険性 (338# MEDIUM, 339# §3.3)

**判定: ✅ 妥当 — 部分的にロールバック検討**

337# §6.3 で forced_switch を rolling PnL から完全除外したが、338# / 339# ともに「システムの実コストを kill 制御から隠蔽する」リスクを指摘。

現状の実装 (`orchestrator_post_cycle.py` L111): buy 側は forced/normal 分離 KPI が既に存在。sell 側には未実装。

**対応方針**:
- 短期: 現在の完全除外を維持 (§1 の符号修正が優先)
- 中期: sell 側にも forced/normal 分離 KPI を追加し、完全除外を「downweight=0.5」に置換
- 338# の「完全に消すな」は正しいが、**今の rolling window 汚染度 (18.7%) は無視できない**ため、何らかの対処は必要

### Finding #7: resume_window の wall-clock 解釈 (338# MEDIUM)

**判定: ✅ 妥当 — 表現の訂正対象**

337# の「約20分」は `120s × 10cycles = 1200s = 20min` の近似。実際は `cycle_interval_sec` が可変 (halt_sleep_multiplier 等) なので固定ではない。

**対応**: 337# ドキュメントの「約20分」を「10 cycle (可変)」に表現修正。ただし分析結論への影響はゼロ。

---

## §3 339# (Gemini) 追加見解の検証

### §3.1 「Hotfix Action Plan」の妥当性

339# は「直ちに Hotfix」を要請。判定:

| 項目 | 339# 推奨 | 対応 | 備考 |
|------|----------|------|------|
| 符号修正 | P0 | **✅ 本 commit で修正済み** | 最優先は正しかった |
| 二重ルート整理 | P1 | 🔄 文書化で対応 | 統合は構造変更が大きすぎる |
| forced 除外ロールバック | P1 | 🔄 中期で downweight 化 | 完全ロールバックは時期尚早 |

### §3.2 「符号修正が他の全対策に先立つ」は正しいか

**正しい。** 符号が逆のまま YAML パラメータを調整しても、inv_relaxation の効果判定自体が汚染される。これは Gemini の指摘通り。

---

## §4 総括: 338#/339# の Finding 品質

| Finding | Codex (338#) | Gemini (339#) | 実検証結果 |
|---------|-------------|---------------|-----------|
| #1 符号逆転 | **CRITICAL** ✅ | 全面同意 ✅ | **確認・修正済み。286# 以降のバグ** |
| #2 PnL 指標混同 | HIGH | 同意 | 妥当だが実害限定的 |
| #3 二重緩和ルート | HIGH | 同意 | 妥当。責務文書化で対応 |
| #4 単独原因帰属 | HIGH | — | 妥当。skip_gate 影響も要評価 |
| #5 hindsight fit | MEDIUM | — | 部分的に妥当 |
| #6 forced 除外 | MEDIUM | 同意 | 妥当。中期で downweight 化 |
| #7 resume clock | MEDIUM | — | 妥当。表現訂正のみ |

**Codex (338#) のレビュー品質は極めて高い。** 特に Finding #1 (符号逆転) は 286# 導入以降見落とされていた致命的バグであり、この発見単独でレビューの価値がある。Gemini (339#) はこれを裏付け、システム工学的視点からの補強を適切に行った。

---

*340# — 2026-03-08 338#/339# レビュー対応完了*
