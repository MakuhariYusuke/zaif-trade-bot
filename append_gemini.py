import sys
import re

file_path = 'docs/v460/222_ph2_rev_213_221_deadlock_validation_and_residual_risks.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('## 6. 追記: 213#～221# に対')
if idx != -1:
    text = text[:idx].strip()
    
# Clean up any trailing broken string that might have been appended
idx2 = text.find('---')
if idx2 != -1 and len(text) - idx2 < 200:
    text = text[:idx2].strip()

new_text = '''

---

## 6. 追記: 213#～221# に対するセカンドオピニオンと市場理論に基づく最終審判 (Gemini 3.1 Pro)

### 6.1 総評: 「生存性（Liveness）」と「安全性（Safety）」の致命的混同

Codexの検証と指摘（とりわけ本ドキュメント222#の内容）は、システム工学・データ解析の両面において完璧に正しい。対して、218#〜220#の「デッドロック解消アプローチ」には、金融取引システムとして**「決して越えてはならない一線」**を越えた形跡が見られる。
プログラムが「停止することなく動き続けること（Liveness）」と、「市場の毒性から身を守ること（Safety）」が相反した場合、トレードシステムにおいては**常にSafetyが優先**されねばならない。現在実装されている「Dual-kill bypass」や、「残高不足によるGuardの強行突破」は、警告音が鳴り響く計器盤のランプを叩き割って「車は正常に走っている」と錯覚しているに過ぎない。

### 6.2 Dual-Kill Deadlock Breaker の論理的破綻と「相場のレジーム崩壊」

220#で実装された「BuyとSellの両方がKill判定されたら、両方とも強行突破する（Dual-kill bypass）」というロジックは、HFTや統計的裁定取引の常識から完全に逸脱している。
Mandelbrot（マンデルブロ）のフラクタル市場仮説や、統計的裁定取引（StatArb）の世界では、「双方向同時にエッジがマイナスになる（Dual Killが発動する）」状況は、単なるバグやデッドロックではなく、**「市場のレジームが未知の領域（Uncharted Territory）に入り、手持ちのモデルの前提が崩壊した」**ことを示す最強のシグナルである。
Kelly Criterion（ケリー基準）において、期待値（EV）がマイナス、かつ不確実性が最大化しているときの最適ベットサイズは「厳密にゼロ（完全停止）」である。これを「デッドロックだから」というシステム都合の理由でバイパスするのは、統計的自殺に他ならない。「両側が燃えているなら、取引を止める」のが唯一の正解である。Dual-kill bypassは即刻削除すべきである。

### 6.3 経路依存性バグ（Path Dependency Bug）と Avellaneda-Stoikov の完全喪失

Codexが1.1で指摘した「`per_side_halt` が `balance_forced` によって破られる」問題の深刻さは計り知れない。
Avellaneda-Stoikovモデルにおいて、片側にポジションが偏った際（インベントリ・リスク増大時）は、予約価格（Reservation Price）を極端にシフトさせて非対称なクオートを行う。だが、**「巨額の損失を出して片側Haltが発動しているサイド」**に向かって、単に「そっちのトークンが欲しい（または売りたい）から」という理由だけで発注を強制（Forced）させるのは、インベントリを調整するメリットよりも明らかに逆選択コスト（Adverse Selection）の被害が上回っている証拠である。
制御フロー（Control-flow）上の順番ミスというコード上のバグに留まらず、これは「資金管理（Money Management）」の根幹を否定している状態だ。実装指示通り、`balance_forced` によって `next_side` が反転した直後で必ず再度の `is_side_halted` 検証を行うよう修正せよ。

### 6.4 評価指標の汚染（データ・マージ）：A/Bテストの基本原則違反

Codexが1.3で指摘した「1日の間に5つの異なるコミット（Git SHA）の成果をごちゃ混ぜにして評価している」という事実は、データサイエンスとして致命的だ。
市場の時間的非定常性（Non-stationarity）を扱うシステム開発において、**ロジックの因果関係（Causality）**を正しく推論するためには、エポック（検証期間とSHA）を厳密に分離しなければならない。利益が出ているのか損失が出ているのか、どの変更が効いているのかを「日次集計」で曖昧にする運用は直ちに改め、**「Git SHA × Run ID ベースのマイクロエポック評価」**を標準の分析基盤とせよ。

### 6.5 今後のアクション（妥協なき是正要請）

前回（214#）の反省を踏まえ、純粋な技術と数理・統計ロジックのみを以て要求する。

1. **[P0] Dual-Kill Bypass (Gate 4/5 貫通) の即時廃止**: 両サイドが規定の損失閾値を超えたなら、それは正常な「Halt（全停止）挙動」である。これをデッドロックと呼ぶのをやめ、正しくシステムを休止させろ。
2. **[P0] Balance Forced と Side Halt の順序修復**: `fill_loop_orchestrator.py` の `balance_forced` で `next_side` を上書きした際、必ず `self._daily_drawdown_guard.is_side_halted(next_side)` を評価し、Trueなら取引を完全にSkipさせろ。Haltされているサイドを通すな。
3. **[P1] 状態永続化（State Persistence）の適時保存**: Skipが連続する際にStateが5時間も古いまま放置されるのは、障害復旧のアンチパターンである。Gate BlockでContinueする直前でも、最終保存から一定時間（例: 5分）経過していれば軽量にStateを保存するフックを設けろ。

これらは議論の余地のない、強牢なトレードシステムに必要な絶対要件である。即座の実装反映を推奨する。
'''

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text + new_text)

