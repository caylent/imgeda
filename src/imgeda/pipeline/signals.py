"""Graceful Ctrl+C / SIGTERM handling."""

from __future__ import annotations

import signal
import threading


class ShutdownHandler:
    """Thread-safe shutdown coordinator for graceful Ctrl+C handling."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._original_sigint: signal.Handlers | None = None
        self._original_sigterm: signal.Handlers | None = None

    @property
    def is_shutting_down(self) -> bool:
        return self._event.is_set()

    def request_shutdown(self) -> None:
        self._event.set()

    def install(self) -> None:
        """Install signal handlers. First Ctrl+C = graceful, second = immediate."""

        def handler(signum: int, frame: object) -> None:
            if self._event.is_set():
                # Second signal — restore defaults and re-raise
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                raise KeyboardInterrupt
            self._event.set()

        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def uninstall(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)


def worker_init() -> None:
    """Initializer for worker processes — ignore SIGINT so main controls shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
