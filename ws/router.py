"""WebSocket endpoint and connection management.

Handles the ``/ws`` endpoint, maintains active connections, and
routes incoming JSON messages to the appropriate module handler
based on the message ``type`` field.
"""

from __future__ import annotations

from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(
        self, connection_id: str, websocket: WebSocket
    ) -> None:
        """Accept and register a new WebSocket connection.

        Parameters
        ----------
        connection_id : str
            Unique connection identifier.
        websocket : WebSocket
            The FastAPI WebSocket instance.
        """
        await websocket.accept()
        self._connections[connection_id] = websocket

    async def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection from the registry.

        Parameters
        ----------
        connection_id : str
            Connection to remove.
        """
        self._connections.pop(connection_id, None)

    async def send(
        self, connection_id: str, message: dict[str, Any]
    ) -> None:
        """Send a JSON message to a specific connection.

        Parameters
        ----------
        connection_id : str
            Target connection.
        message : dict[str, Any]
            JSON-serialisable message dict.
        """
        ws = self._connections.get(connection_id)
        if ws:
            await ws.send_json(message)

    async def broadcast(
        self, message: dict[str, Any]
    ) -> None:
        """Send a JSON message to all connected clients.

        Parameters
        ----------
        message : dict[str, Any]
            JSON-serialisable message dict.
        """
        for ws in self._connections.values():
            await ws.send_json(message)


def route_message(
    message_type: str, payload: dict[str, Any]
) -> None:
    """Dispatch an incoming message to the appropriate handler.

    Parameters
    ----------
    message_type : str
        The ``type`` field from the WebSocket message envelope.
    payload : dict[str, Any]
        Message payload.
    """
    # Routing is wired during Phase 2 when modules are connected
    pass
