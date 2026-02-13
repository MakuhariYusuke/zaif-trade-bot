class WebSocketException(Exception):
    """Base exception class for our tiny websockets shim."""


class ConnectionClosed(WebSocketException):
    """Raised when a connection is closed."""


class ConnectionClosedError(ConnectionClosed):
    """Alias for third-party code expecting ConnectionClosedError."""


__all__ = ["WebSocketException", "ConnectionClosed"]
"""Minimal subset of websockets.exceptions used by codepaths in tests."""


class InvalidHandshake(Exception):
    pass

class WebSocketException(Exception):
    pass

__all__ = ["InvalidHandshake", "WebSocketException"]
