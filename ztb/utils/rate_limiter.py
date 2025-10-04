"""
Rate limiting utilities for API calls and resource protection.

This module provides rate limiting functionality to prevent API abuse
and ensure fair resource usage across different operations.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, cast

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 1.0
    burst_limit: int = 5
    window_seconds: float = 1.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self, config: RateLimitConfig):
        """Initialize token bucket rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.tokens = config.burst_limit
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Add tokens based on time passed
            self.tokens = int(
                min(
                    self.config.burst_limit,
                    self.tokens + time_passed * self.config.requests_per_second,
                )
            )

            if self.tokens >= tokens:
                self.tokens -= tokens
                self.last_update = now
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            # Calculate wait time for next token
            wait_time = max(0.01, 1.0 / self.config.requests_per_second)
            await asyncio.sleep(wait_time)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""

    def __init__(self, config: RateLimitConfig):
        """Initialize sliding window rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Check if request can be made within rate limit.

        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            now = time.time()

            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.config.window_seconds:
                self.requests.popleft()

            if len(self.requests) < self.config.burst_limit:
                self.requests.append(now)
                return True

            return False

    async def wait_for_slot(self) -> None:
        """Wait until a request slot is available."""
        while not await self.acquire():
            # Wait until oldest request expires
            if self.requests:
                wait_time = max(
                    0.01, self.requests[0] + self.config.window_seconds - time.time()
                )
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.01)


class RateLimiter:
    """Unified rate limiter with multiple strategies."""

    def __init__(
        self, config: Optional[RateLimitConfig] = None, strategy: str = "token_bucket"
    ):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
            strategy: Rate limiting strategy ('token_bucket' or 'sliding_window')
        """
        self.config = config or RateLimitConfig()
        self.strategy = strategy

        if strategy == "token_bucket":
            self._limiter = cast(
                Union[TokenBucketRateLimiter, SlidingWindowRateLimiter],
                TokenBucketRateLimiter(self.config),
            )
        elif strategy == "sliding_window":
            self._limiter = cast(
                Union[TokenBucketRateLimiter, SlidingWindowRateLimiter],
                SlidingWindowRateLimiter(self.config),
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire permission for a request.

        Args:
            tokens: Number of tokens/tokens needed

        Returns:
            True if allowed, False if rate limited
        """
        if self.strategy == "token_bucket":
            return await self._limiter.acquire(tokens)
        else:  # sliding_window
            return await self._limiter.acquire()

    async def wait(self, tokens: int = 1) -> None:
        """Wait until request can be made.

        Args:
            tokens: Number of tokens needed
        """
        if isinstance(self._limiter, TokenBucketRateLimiter):
            await self._limiter.wait_for_tokens(tokens)
        else:
            await self._limiter.wait_for_slot()

    def is_allowed(self, tokens: int = 1) -> bool:
        """Check if request would be allowed (synchronous).

        Note: This is a best-effort check and may not be perfectly accurate
        in concurrent scenarios.

        Args:
            tokens: Number of tokens needed

        Returns:
            True if would be allowed
        """
        # For synchronous checks, we use a simple time-based approach
        # This is less accurate but avoids async complexity
        return True  # Placeholder - implement if needed


class MultiRateLimiter:
    """Rate limiter for multiple categories/keys."""

    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """Initialize multi-rate limiter.

        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig()
        self.limiters: Dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    def get_limiter(
        self, key: str, config: Optional[RateLimitConfig] = None
    ) -> RateLimiter:
        """Get or create rate limiter for a key.

        Args:
            key: Identifier for the rate limiter
            config: Specific config for this key

        Returns:
            Rate limiter instance
        """
        if key not in self.limiters:
            self.limiters[key] = RateLimiter(config or self.default_config)
        return self.limiters[key]

    async def acquire(
        self, key: str, tokens: int = 1, config: Optional[RateLimitConfig] = None
    ) -> bool:
        """Acquire permission for a keyed request.

        Args:
            key: Rate limiter key
            tokens: Number of tokens needed
            config: Specific config for this key

        Returns:
            True if allowed
        """
        limiter = self.get_limiter(key, config)
        return await limiter.acquire(tokens)

    async def wait(
        self, key: str, tokens: int = 1, config: Optional[RateLimitConfig] = None
    ) -> None:
        """Wait for permission for a keyed request.

        Args:
            key: Rate limiter key
            tokens: Number of tokens needed
            config: Specific config for this key
        """
        limiter = self.get_limiter(key, config)
        await limiter.wait(tokens)


# Global rate limiter instances
_api_limiter = MultiRateLimiter(
    RateLimitConfig(requests_per_second=2.0, burst_limit=10)
)
_file_limiter = MultiRateLimiter(
    RateLimitConfig(requests_per_second=1.0, burst_limit=3)
)


def get_api_limiter() -> MultiRateLimiter:
    """Get global API rate limiter."""
    return _api_limiter


def get_file_limiter() -> MultiRateLimiter:
    """Get global file operation rate limiter."""
    return _file_limiter


async def rate_limited_api_call(
    key: str, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Make a rate-limited API call.

    Args:
        key: Rate limiter key (e.g., API endpoint)
        func: Function to call
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function call
    """
    await _api_limiter.wait(key)
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"API call failed for {key}: {e}")
        raise


def create_rate_limiter_for_endpoint(
    endpoint: str, requests_per_minute: int = 60
) -> RateLimiter:
    """Create a rate limiter for a specific API endpoint.

    Args:
        endpoint: API endpoint name
        requests_per_minute: Allowed requests per minute

    Returns:
        Configured rate limiter
    """
    config = RateLimitConfig(
        requests_per_second=requests_per_minute / 60.0,
        burst_limit=max(1, requests_per_minute // 10),
        window_seconds=60.0,
    )
    return RateLimiter(config)
