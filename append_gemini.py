"""Append Gemini review prompt to 333# deep dive document."""

file_path = 'docs/v460/333_ph2_rpt_dcc3064_sha_isolated_deep_dive.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove any previous Gemini appendix if present
marker = '## §15 追記: Gemini'
idx = text.find(marker)
if idx != -1:
    text = text[:idx].rstrip()

new_text = '''

---

## §15 追記: Gemini 3.1 Pro セカンドオピニオン

> 以下は Gemini 3.1 Pro による外部レビュー追記欄。レビュー完了後に記入する。

### レビュー依頼コンテキスト

本ドキュメントは、310# (dcc3064a8) の設計改修が稼働した 24 時間のデータを SHA 分離して分析したレポートである。

**レビューに際して知っておくべき前提:**

1. **C-1 sell ceiling 問題** (320# で修正): dcc3064 稼働期間中、sell offset pipeline は floor(0.30) > ceiling(0.15) のためすべて 0.15 にクランプされ、12+ パラメータが無効化。310# A の sell hour boost も実質死亡。
2. **buy_dynamic_kill 支配**: Skip の 40.2% を占め、buy fill_rate=9.3% まで圧縮。
3. **Ranging 偏り**: 24h で 90.3% が ranging regime。trending での性能は未検証。
4. **299# 結論**: sell vs buy PnL 差は統計的に非有意 (4 検定すべて)。
5. **n=100 の統計的限界**: p10 の 95% CI は ±2-3bps と推定。

§14 の Q1-Q5 への回答および追加の指摘を求む。
'''

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text + new_text)

print(f"Appended Gemini review section to {file_path}")

