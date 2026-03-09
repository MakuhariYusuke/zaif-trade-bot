"""Volume-Synchronized VPIN (Easley-López de Prado 2012).

Time-bucket VPIN は低出来高時にノイジーになる。Volume-bucket VPIN は
出来高正規化されているため、BTC/JPY のような不均一な約定流でも安定した
toxicity推定を提供する。

使い方
------
既存の ``_TradeFeatureContext`` が保持する累積配列をそのまま入力に使う。
``compute_vpin_volume_sync()`` は純粋 numpy 関数で副作用なし。

理論
----
VPIN = (1/n) Σ |V_buy^(i) - V_sell^(i)| / V_bucket
where V_bucket は固定出来高バケットサイズ。

References
----------
Easley, D., López de Prado, M. M., & O'Hara, M. (2012).
"Flow Toxicity and Liquidity in a High-Frequency World."
Review of Financial Studies, 25(5), 1457-1493.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_vpin_volume_sync"]

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
#: 推奨バケット数 (Easley+ 2012: 50)
DEFAULT_N_BUCKETS: int = 50
#: バケットサイズのフォールバック (BTC 単位, ~日次出来高の 1/50 相当)
DEFAULT_BUCKET_SIZE_BTC: float = 0.05
#: バケットが確保できない場合のニュートラル値
NEUTRAL_VPIN: float = 0.5


def compute_vpin_volume_sync(
    cumulative_total_volume: np.ndarray,
    cumulative_buy_volume: np.ndarray,
    end_index: int,
    bucket_size: float = DEFAULT_BUCKET_SIZE_BTC,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> float:
    """Volume-bucket VPIN を計算する。

    Parameters
    ----------
    cumulative_total_volume:
        長さ N+1 の累積出来高配列。``[0] = 0.0``, ``[i] = Σ amount[0..i-1]``。
        ``feature_enricher._TradeFeatureContext.cumulative_total_volume`` と同形式。
    cumulative_buy_volume:
        長さ N+1 の累積 buy 出来高配列。同形式。
    end_index:
        現在の約定位置 (1-based index into cumulative arrays,
        i.e. ``cumulative_total_volume[end_index]`` は ``end_index`` 番目までの
        累積出来高)。
    bucket_size:
        一つのバケットに入る出来高 (BTC)。
    n_buckets:
        VPIN 平均に使うバケット数。

    Returns
    -------
    float
        VPIN 値 [0.0, 1.0]。データ不足時は ``NEUTRAL_VPIN`` (0.5)。
    """
    if (
        end_index <= 0
        or bucket_size <= 0.0
        or n_buckets <= 0
        or len(cumulative_total_volume) <= 1
    ):
        return NEUTRAL_VPIN

    # Clamp end_index to array bounds
    end_index = min(end_index, len(cumulative_total_volume) - 1)

    total_vol_end = cumulative_total_volume[end_index]
    total_vol_start = cumulative_total_volume[0]
    available_vol = total_vol_end - total_vol_start

    # 完全なバケット数を算出
    n_full_buckets = int(available_vol / bucket_size)
    actual_n = min(n_full_buckets, n_buckets)
    if actual_n == 0:
        return NEUTRAL_VPIN

    # バケット境界の累積出来高値 (end_index から逆算)
    # bucket_vol_boundaries[0] = 最も古いバケットの先頭
    # bucket_vol_boundaries[-1] = total_vol_end
    start_vol = total_vol_end - actual_n * bucket_size
    bucket_vol_boundaries = start_vol + np.arange(actual_n + 1) * bucket_size

    # 各境界に対応する約定インデックスを searchsorted で取得
    # cumulative_total_volume は単調非減少なので searchsorted が使える
    bucket_trade_indices = np.searchsorted(
        cumulative_total_volume[: end_index + 1],
        bucket_vol_boundaries,
        side="left",
    )

    # バケット毎の buy 出来高と total 出来高
    buy_per_bucket = np.diff(cumulative_buy_volume[bucket_trade_indices])
    total_per_bucket = np.diff(cumulative_total_volume[bucket_trade_indices])

    # VPIN = |V_buy - V_sell| / V_total = |2·V_buy - V_total| / V_total
    safe_total = np.maximum(total_per_bucket, 1e-20)
    vpin_per_bucket = np.abs(2.0 * buy_per_bucket - total_per_bucket) / safe_total

    return float(np.mean(vpin_per_bucket))
