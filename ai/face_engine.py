"""Face detection, recognition, and anti-spoofing engine.

Wraps DeepFace with the ArcFace model and RetinaFace detector backend.
Manages per-player multi-embedding storage, centroid computation, and
liveness checks.  Camera frames arrive as base64-encoded JPEG from the
frontend and are processed in a ``ProcessPoolExecutor`` to avoid
blocking the async event loop.
"""

from __future__ import annotations

from typing import Any


class FaceEngine:
    """Manages face detection, recognition, and enrollment."""

    def __init__(self) -> None:
        pass

    async def process_frame(
        self, frame_b64: str
    ) -> dict[str, Any]:
        """Decode a base64 frame, detect faces, and attempt recognition.

        Parameters
        ----------
        frame_b64 : str
            Base64-encoded JPEG frame from the frontend camera.

        Returns
        -------
        dict[str, Any]
            Detection result with player ID (if matched), confidence,
            emotion data, and anti-spoofing verdict.
        """
        pass

    async def enroll_player(
        self, player_id: str, frames: list[str]
    ) -> None:
        """Capture 3-5 embeddings from different angles for a new player.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        frames : list[str]
            List of base64-encoded JPEG frames.
        """
        pass

    def load_player_embeddings(
        self, player_id: str
    ) -> list[list[float]]:
        """Load stored embeddings for a player from the database.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        list[list[float]]
            List of 512-dimensional ArcFace embeddings.
        """
        pass

    def delete_player_data(self, player_id: str) -> None:
        """Remove all face data for a player (GDPR-style deletion).

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        """
        pass
