# 21. v457.1 Phase 2 Results: Frequency Control & Wrapper Implementation

**実施日**: 2026-01-18
**目的**: 取引回数を抑制（Wrapper Control）し、手数料負けを解消する。

## 1. 修正実施報告 (Step 3: Wrapper Control)

当初のバックテスト (Case D/E) 結果に違和感（Cooldown 60 なのに取引が多い）があったため、`backtest_v456.py` を調査・修正しました。

### 修正内容
1.  **Environment Sync Fix**: `info['position']` を使用してトレードを検知するように変更。これにより、環境側で Cooldown 等により拒否されたトレードがカウントされなくなりました。
2.  **Wrapper Logic 追加**: スクリプト側でも `cooldown_steps` や `action_threshold` を強制適用するロジックを追加し、無駄な API コールとログ記録を排除しました。

## 2. 再検証結果 (Case D & F)

修正後のスクリプトで、Case D (Cooldown 60) と Case F (Cooldown 60 + Threat 0.8) を検証しました。

| Metric | Case C (Ref) | Case D (Fixed) | Case F (High Thres) |
| :--- | :--- | :--- | :--- |
| **Config** | `max_ttl: 30` | `cooldown: 60` | `cooldown: 60`, `thres: 0.8` |
| **Trades (/10k)** | 1,057 | **153** ✅ | **153** |
| **Raw PnL** | +297k | **+8,296** | **-1,281** |
| **Net PnL** | -323k | **-847k** ❌ | **-1,015k** ❌ |
| **Result** | Failed | **Failed** | **Failed** |

### 分析
- **Frequency Control**: Wrapper実装により成功。取引回数は 153回 (約65ステップに1回) まで激減し、理論値通りとなりました。
- **Profitability**: **壊滅的**。取引回数を減らして厳選されたはずのトレードでも、1回あたりの期待値がマイナス（手数料負け）です。
- **Action Strength**: Case D (Thres 0.3) と Case F (Thres 0.8) で取引回数が変わらないことから、モデルは常に「確信度Max」のアクションを出しており、スケーリングが機能していません。

## 3. 結論 (Fatal Flaw Identified)

**「Frequency Control (間引き)」では、このモデルを救済できないことが確定しました。**

理由は単純で、**「モデルが持つエッジ（優位性）が、取引コスト (0.1%) よりも圧倒的に小さい」** からです。
間引こうが何しようが、1回取引するたびに期待値として数千円を失う構造になっています。これはパラメータ調整で治るものではなく、**「間違ったことを学習してしまった脳（モデル）」** そのものです。

## 4. 次の手順：v459 (Profit-First Retraining)

既存モデルのパラメータ調整は時間を浪費するだけですので、**「手数料を組み込んだ再学習」** に直ちに着手すべきです。

### 提案プラン
1.  **Reward Function 刷新**:
    - `Net PnL (After Fee)` を厳密に報酬とし、手数料分をマイナス報酬として与える。
    - `Trade Frequency Penalty`: 取引回数自体にペナルティを与え、「無駄撃ち」を抑制する。
2.  **Model Architecture**: 変更なしだが、学習プロセスにおいて `Ent_coef` というよりも確実な利益を重視させる設定にする。

この結果を持って、Phase 3 (再学習フェーズ) へ移行することを提案します。

