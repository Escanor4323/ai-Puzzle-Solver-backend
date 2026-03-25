"""Async event bus for decoupled inter-module communication.

Modules register handler coroutines for named events.  When an event
is emitted, all registered handlers run as concurrent ``asyncio`` tasks.
``emit()`` is intentionally synchronous (fire-and-forget) so a slow
handler never blocks the emitter.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type alias for an async event handler
Handler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Lightweight async event bus.

    Usage::

        bus = EventBus()
        bus.on("face:recognized", handle_face)
        bus.emit("face:recognized", {"player_id": "abc"})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = (
            defaultdict(list)
        )

    def on(self, event: str, handler: Handler) -> None:
        """Register a handler for an event.

        Parameters
        ----------
        event : str
            Event name (e.g. ``"face:recognized"``).
        handler : Handler
            Async callable to invoke when the event fires.
        """
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Handler) -> None:
        """Remove a previously registered handler.

        Parameters
        ----------
        event : str
            Event name.
        handler : Handler
            The handler to remove.
        """
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    def emit(self, event: str, data: Any = None) -> None:
        """Fire an event, dispatching to all handlers concurrently.

        Each handler is wrapped in ``asyncio.create_task`` so that
        the emitter is never blocked by handler execution.

        Parameters
        ----------
        event : str
            Event name.
        data : Any
            Payload to pass to each handler.
        """
        for handler in self._handlers.get(event, []):
            try:
                asyncio.create_task(handler(data))
            except RuntimeError:
                # No running event loop — log and skip
                logger.warning(
                    "Cannot emit '%s': no running event loop.",
                    event,
                )
