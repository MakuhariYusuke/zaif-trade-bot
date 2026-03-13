# 261# Protocol 型安全化 — OrderBookLevelLike, OrderBookSnapshot, BalanceAdapterProtocol

## 概要

259# Sweep Report の P2 残タスク (P2-1, P2-5, P2-6, P2-7) を一括実装。
`getattr` / `adapter: object` / `type: ignore[union-attr]` を Protocol 定義で排除し、
mypy 静的解析通過率と IDE 補完精度を向上。

---

## P2-1: OrderBookLevelLike Protocol

### 背景

`ob_utils.py` の `extract_price()` / `extract_size()` は板データの各レベルを
`tuple[float, float]` と `object` の dual-format で扱い、後者は `getattr(level, "price", 0.0)` で
フォールバック取得していた。型情報が失われ、IDE 補完もスペルミスも検知不能。

### 実装

```python
@runtime_checkable
class OrderBookLevelLike(Protocol):
    @property
    def price(self) -> float: ...
    @property
    def quantity(self) -> float: ...
```

| ファイル | 変更 |
|---|---|
| `ob_utils.py` | Protocol 定義 + `OrderBookLevel` TypeAlias を `Union[tuple, OrderBookLevelLike]` に変更 |
| `ob_utils.py` | `extract_price()`: `getattr(level, "price")` → `level.price` 直接アクセス |
| `ob_utils.py` | `extract_size()`: `getattr(level, "quantity", getattr(...))` → `level.quantity` / `getattr(level, "size")` |
| `ob_recorder.py` | `isinstance(level, OrderBookLevelLike)` ガード追加。非準拠 object をスキップ |

### リスク評価

- **低**: `runtime_checkable` により既存の NamedTuple/dataclass は透過的に適合
- `ob_recorder._normalize_levels()` は `_to_finite_float()` で型検証を維持
  → MagicMock 等の不正 object は `_to_finite_float` → None → skip の経路で排除

---

## P2-5: OrderBookSnapshot Protocol

### 背景

`maker_price.py` の `_last_ob_snapshot: object | None` と
`OrderbookProvider.get_orderbook() -> object` は型情報ゼロ。

### 実装

```python
class OrderBookSnapshot(Protocol):
    @property
    def bids(self) -> Sequence[tuple[float, float]]: ...
    @property
    def asks(self) -> Sequence[tuple[float, float]]: ...
```

| 箇所 | Before | After |
|---|---|---|
| `OrderbookProvider.get_orderbook()` 戻り値 | `object` | `OrderBookSnapshot` |
| `_last_ob_snapshot` 型注釈 | `object \| None` | `OrderBookSnapshot \| None` |

### 市場理論的意義

OrderBook は market microstructure の中核データ構造。
型安全化により bid/ask depth volume 計算の静的検証が可能になり、
AS σ² 推定やスプレッド適応ロジックの信頼性が向上。

---

## P2-6: config_hot_reload getattr 排除

### 背景

`config_hot_reload.py` L400 で `getattr(runner, "_fast_fill_defense", None)` を使用。
しかし `_HotReloadableRunner` Protocol で `_fast_fill_defense: object` は既に宣言済み。

### 実装

```python
# Before
_ffd = getattr(runner, "_fast_fill_defense", None)

# After — 261# P2-6: Protocol で宣言済み → 直接参照
_ffd = runner._fast_fill_defense
```

Protocol 宣言があるにもかかわらず getattr を使用していた矛盾を解消。
`_HotReloadableRunner` の型情報が FFD sync ロジックに伝播する。

---

## P2-7: BalanceAdapterProtocol

### 背景

`balance_checker.py` の `check()` / `_check_sell()` / `_check_buy()` は
`adapter: object` で受け取り、`adapter.get_balance()` / `adapter.get_current_price()` を
`# type: ignore[union-attr]` で呼び出していた。

### 実装

```python
class _BalanceLike(Protocol):
    @property
    def free(self) -> float: ...

class BalanceAdapterProtocol(Protocol):
    async def get_balance(self, currency: str) -> Sequence[_BalanceLike] | None: ...
    async def get_current_price(self, symbol: str) -> float | None: ...
```

| 箇所 | Before | After |
|---|---|---|
| `check()` 引数 | `adapter: object` | `adapter: BalanceAdapterProtocol` |
| `_check_sell()` 引数 | `adapter: object` | `adapter: BalanceAdapterProtocol` |
| `_check_buy()` 引数 | `adapter: object` | `adapter: BalanceAdapterProtocol` |
| `type: ignore[union-attr]` | 3 箇所 | 0 箇所 |

### 理論的根拠

残高チェックはロット自動縮小 (052#) と Kelly Criterion ロットサイジング (P3-2) の
基盤となる。adapter インタフェースの型安全化により、将来の `f* = (pb - q) / b`
統合時に型レベルでの整合性を保証。

---

## P2-4 (保留) の判断根拠

skip_gate_evaluator の getattr は以下の理由で保留:

| 箇所 | 理由 |
|---|---|
| L234/236 `getattr(trade, key)` | dict/object dual-format: `_get_trade_field()` は汎用ユーティリティとして設計されており、Protocol 化すると柔軟性が失われる |
| L830/852 `getattr(self, attr_path)` | メタプログラミング: `_SIDE_MODEL_SLOTS` テーブル駆動の動的属性アクセス。dict 化は可能だが既存テスト資産への影響大 |
| L991/1015 `getattr(adapter, ...)` | GateAdapterProtocol 定義は可能だが、adapter の型が FillTestRunner 内で動的に決まるため影響範囲が広い |

---

## テスト

- `test_261_protocol_type_safety.py` — 19 テスト
  - `TestOrderBookLevelLikeProtocol` (10): Protocol 定義、runtime_checkable、getattr 排除、tuple/object 両対応
  - `TestOrderBookSnapshotProtocol` (3): Protocol 定義、_last_ob_snapshot 型、OrderbookProvider 戻り値
  - `TestConfigHotReloadGetattr` (2): getattr 排除、直接参照確認
  - `TestBalanceAdapterProtocol` (4): Protocol 定義、check() 引数型、type:ignore 排除
- 全 3620 テスト pass

## コミット

`489a77cf8` — `261# Protocol 型安全化: OrderBookLevelLike, OrderBookSnapshot, BalanceAdapterProtocol`
