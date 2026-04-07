from __future__ import annotations

VALID_RANGING_OBI_MODES: tuple[str, ...] = (
    "linear",
    "absolute",
    "quadratic",
    "excess",
)


def compute_ranging_obi_multiplier(
    base_multiplier: float,
    *,
    side: str,
    imbalance: float,
    threshold: float,
    factor: float,
    mode: str,
) -> float:
    if factor <= 0.0 or abs(imbalance) <= threshold:
        return base_multiplier

    if mode == "absolute":
        return base_multiplier * (1.0 + abs(imbalance) * factor)
    if mode == "quadratic":
        return base_multiplier * (1.0 + (imbalance**2) * factor)
    if mode == "excess":
        return base_multiplier * (1.0 + max(abs(imbalance) - threshold, 0.0) * factor)

    adjustment = imbalance * factor
    if side == "buy":
        return base_multiplier * (1.0 - adjustment)
    return base_multiplier * (1.0 + adjustment)

