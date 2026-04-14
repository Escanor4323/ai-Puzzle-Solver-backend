"""Face detection, recognition, and session management.

Single-pipeline architecture: one DeepFace.represent() call per frame
handles detection (MediaPipe) + alignment + embedding (GhostFaceNet).
Emotion analysis runs on every 5th frame via DeepFace.analyze().

All CPU-bound DeepFace calls run in a ProcessPoolExecutor to avoid
blocking the FastAPI async event loop.

Classes:
    FaceProcessor       — stateless frame processing (runs in executor)
    PlayerMatcher       — in-memory embedding cache + cosine matching
    FaceSessionManager  — state machine for face session lifecycle
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from data.models import (
    FaceDetectionResult,
    FaceEnrollmentResult,
    FaceMatchCandidate,
    FaceSessionState,
)

logger = logging.getLogger(__name__)


# ── FaceProcessor ──────────────────────────────────────────────


class FaceProcessor:
    """Stateless face processing — runs in ProcessPoolExecutor.

    Each method decodes a JPEG frame and calls DeepFace.
    Model loading happens once per worker process on first call.
    """

    RECOGNITION_MODEL = "GhostFaceNet"
    DETECTOR_BACKEND = "opencv"
    DISTANCE_METRIC = "cosine"
    ANTI_SPOOFING = False

    @staticmethod
    def detect_and_embed(frame_bytes: bytes) -> dict[str, Any]:
        """Detect face and generate embedding from JPEG bytes.

        Uses DeepFace.represent() which internally runs MediaPipe
        for detection, then GhostFaceNet for embedding — one call.

        Parameters
        ----------
        frame_bytes : bytes
            Raw JPEG image bytes.

        Returns
        -------
        dict
            {"found": True, "embedding": list[float],
             "facial_area": dict, "is_real": bool}
            or {"found": False} / {"found": False, "error": str}
        """
        try:
            from deepface import DeepFace
            from PIL import Image

            # Decode JPEG to numpy array
            image = Image.open(io.BytesIO(frame_bytes))
            img_array = np.array(image)

            # Single call: detect + align + embed + anti-spoof
            results = DeepFace.represent(
                img_path=img_array,
                model_name=FaceProcessor.RECOGNITION_MODEL,
                detector_backend=FaceProcessor.DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
                anti_spoofing=FaceProcessor.ANTI_SPOOFING,
            )

            if not results:
                return {"found": False}

            # Pick the largest face (closest to camera)
            best = max(
                results,
                key=lambda r: (
                    r.get("facial_area", {}).get("w", 0)
                    * r.get("facial_area", {}).get("h", 0)
                ),
            )

            embedding = best.get("embedding")
            if embedding is None:
                return {"found": False}

            # Sanitize facial_area — DeepFace returns left_eye/right_eye
            # as tuples which aren't JSON-serialisable or Pydantic-friendly.
            raw_area = best.get("facial_area", {})
            if isinstance(raw_area, dict):
                facial_area = {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in raw_area.items()
                }
            else:
                facial_area = {}

            return {
                "found": True,
                "embedding": embedding,
                "facial_area": facial_area,
                "is_real": best.get("is_real", True),
            }

        except Exception as e:
            logger.warning("DeepFace detect_and_embed error: %s", e)
            return {"found": False, "error": str(e)}

    # Lazily initialized FER detector (shared across calls)
    _fer_detector: Any = None

    @staticmethod
    def analyze_emotion(frame_bytes: bytes) -> str | None:
        """Analyze facial emotion from JPEG bytes using FER.

        Uses the ``fer`` library with built-in TFLite model —
        no external weight downloads required.

        Parameters
        ----------
        frame_bytes : bytes
            Raw JPEG image bytes.

        Returns
        -------
        str | None
            Dominant emotion ("happy", "sad", "angry", "neutral",
            "surprise", "fear", "disgust") or None if analysis fails.
        """
        try:
            from fer.fer import FER
            from PIL import Image

            # Lazy-init the FER detector (reuse across calls)
            if FaceProcessor._fer_detector is None:
                FaceProcessor._fer_detector = FER(mtcnn=False)

            image = Image.open(io.BytesIO(frame_bytes))
            img_array = np.array(image)

            results = FaceProcessor._fer_detector.detect_emotions(
                img_array
            )

            if not results:
                return None

            # Pick the face with highest total confidence
            best = max(
                results,
                key=lambda r: max(r.get("emotions", {}).values(), default=0),
            )
            emotions = best.get("emotions", {})
            if not emotions:
                return None

            return max(emotions, key=emotions.get)

        except Exception as e:
            logger.warning("DeepFace emotion analysis error: %s", e)
            return None


# ── PlayerMatcher ──────────────────────────────────────────────


class PlayerMatcher:
    """In-memory embedding cache with cosine distance matching.

    Matching is O(n) over known players — for <100 players this
    completes in <1ms. No executor needed.
    """

    MATCH_THRESHOLD_HIGH = 0.30
    MATCH_THRESHOLD_MEDIUM = 0.45
    MAX_EMBEDDINGS_PER_PLAYER = 20

    def __init__(self) -> None:
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._player_names: dict[str, str] = {}

    async def load_from_database(self, db: Any) -> int:
        """Load all face embeddings from encrypted database.

        Reads face_embeddings table, decrypts each embedding,
        populates in-memory caches.

        Parameters
        ----------
        db : DatabaseManager
            The database manager instance.

        Returns
        -------
        int
            Number of players loaded.
        """
        from security.encryption import decrypt_embedding, EncryptedPayload

        rows = await db.fetch_all(
            "SELECT player_id, embedding_encrypted, dek_encrypted, nonce "
            "FROM face_embeddings"
        )

        for row in rows:
            player_id = row["player_id"]
            try:
                payload = EncryptedPayload(
                    ciphertext=row["embedding_encrypted"],
                    nonce=row["nonce"],
                    dek_encrypted=row["dek_encrypted"],
                )
                raw_bytes = decrypt_embedding(payload)
                embedding = np.frombuffer(raw_bytes, dtype=np.float32)

                if player_id not in self._embeddings:
                    self._embeddings[player_id] = []
                self._embeddings[player_id].append(embedding)
            except Exception as e:
                logger.warning(
                    "Failed to decrypt embedding for %s: %s",
                    player_id,
                    e,
                )

        # Compute centroids
        for player_id, embs in self._embeddings.items():
            self._centroids[player_id] = self._compute_centroid(embs)

        # Load player names
        name_rows = await db.fetch_all(
            "SELECT id, display_name FROM players"
        )
        for row in name_rows:
            self._player_names[row["id"]] = row["display_name"]

        return len(self._embeddings)

    def match(
        self, embedding: np.ndarray
    ) -> FaceMatchCandidate | None:
        """Match an embedding against all known players.

        Parameters
        ----------
        embedding : np.ndarray
            The face embedding to match (will be L2-normalized).

        Returns
        -------
        FaceMatchCandidate | None
            Best match if within threshold, or None.
        """
        if not self._centroids:
            return None

        # L2-normalize input
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        query = embedding / norm

        best_id: str | None = None
        best_distance = float("inf")

        for player_id, centroid in self._centroids.items():
            # Cosine distance = 1 - dot(a, b) for L2-normalized vectors
            distance = 1.0 - float(np.dot(query, centroid))
            if distance < best_distance:
                best_distance = distance
                best_id = player_id

        if best_id is None or best_distance > self.MATCH_THRESHOLD_MEDIUM:
            return None

        return FaceMatchCandidate(
            player_id=best_id,
            player_name=self._player_names.get(best_id, "Unknown"),
            distance=best_distance,
            confidence=1.0 - best_distance,
        )

    async def enroll(
        self,
        player_id: str,
        player_name: str,
        embedding: np.ndarray,
        db: Any | None = None,
    ) -> None:
        """Store an embedding for a player.

        If *db* is provided the embedding is encrypted and persisted.
        Otherwise it is kept only in the in-memory cache (session-only).

        Parameters
        ----------
        player_id : str
            Player identifier.
        player_name : str
            Player display name.
        embedding : np.ndarray
            Face embedding vector.
        db : DatabaseManager | None
            Database manager for persistence (optional).
        """
        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return
        embedding = embedding / norm

        # Check diversity if at capacity
        if player_id in self._embeddings:
            current_count = len(self._embeddings[player_id])
            if current_count >= self.MAX_EMBEDDINGS_PER_PLAYER:
                centroid = self._centroids.get(player_id)
                if centroid is not None:
                    dist = 1.0 - float(np.dot(embedding, centroid))
                    if dist <= 0.1:
                        return  # Not diverse enough

        # Persist to DB if available
        if db is not None:
            try:
                from security.encryption import encrypt_embedding

                raw_bytes = embedding.astype(np.float32).tobytes()
                encrypted = encrypt_embedding(raw_bytes)

                emb_id = str(uuid.uuid4())
                await db.execute(
                    "INSERT INTO face_embeddings "
                    "(id, player_id, embedding_encrypted, "
                    "dek_encrypted, nonce) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        emb_id,
                        player_id,
                        encrypted.ciphertext,
                        encrypted.dek_encrypted,
                        encrypted.nonce,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "DB persistence skipped for %s: %s",
                    player_id, exc,
                )

        # Update in-memory cache
        if player_id not in self._embeddings:
            self._embeddings[player_id] = []
        self._embeddings[player_id].append(embedding)
        self._player_names[player_id] = player_name
        self._centroids[player_id] = self._compute_centroid(
            self._embeddings[player_id]
        )

    async def delete_player(
        self, player_id: str, db: Any
    ) -> None:
        """Remove all face data for a player.

        Parameters
        ----------
        player_id : str
            Player to delete.
        db : DatabaseManager
            Database manager.
        """
        await db.execute(
            "DELETE FROM face_embeddings WHERE player_id = ?",
            (player_id,),
        )
        self._embeddings.pop(player_id, None)
        self._centroids.pop(player_id, None)
        self._player_names.pop(player_id, None)

    @staticmethod
    def _compute_centroid(
        embeddings: list[np.ndarray],
    ) -> np.ndarray:
        """Compute L2-normalized centroid of embeddings."""
        if not embeddings:
            return np.zeros(512, dtype=np.float32)
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            return mean
        return mean / norm


# ── FaceSessionManager ─────────────────────────────────────────


class FaceSessionManager:
    """State machine for face recognition session lifecycle.

    Manages transitions between states (IDLE → SCANNING → DETECTED →
    RECOGNIZED / NEW_PLAYER / etc.) with hysteresis via streak counters
    to prevent flicker.
    """

    RECOGNITION_CONFIRM_FRAMES = 3
    FACE_LOST_FRAMES = 10
    ENROLLMENT_TARGET = 5
    ENROLLMENT_MINIMUM = 3
    EMOTION_EVERY_N_FRAMES = 5

    def __init__(self) -> None:
        self._current_state: FaceSessionState = FaceSessionState.IDLE
        self._current_player_id: str | None = None
        self._current_player_name: str | None = None
        self._recognition_streak: int = 0
        self._loss_streak: int = 0
        self._enrollment_buffer: list[np.ndarray] = []
        self._frame_counter: int = 0
        self._last_emotion: str | None = None
        self._processing_in_flight: bool = False

    async def process_frame(
        self,
        frame_base64: str,
        processor: FaceProcessor,
        matcher: PlayerMatcher,
        executor: ProcessPoolExecutor,
        event_bus: Any | None = None,
    ) -> FaceDetectionResult:
        """Process a single camera frame through the full pipeline.

        Implements the face session state machine:
        1. Detect + embed in executor (non-blocking)
        2. Handle no-face / spoofing / match / no-match
        3. Update streaks and state transitions
        4. Run emotion analysis every N frames

        Parameters
        ----------
        frame_base64 : str
            Base64-encoded JPEG frame.
        processor : FaceProcessor
            Frame processor (runs in executor).
        matcher : PlayerMatcher
            Embedding matcher (runs in main thread).
        executor : ProcessPoolExecutor
            Worker pool for CPU-bound calls.
        event_bus : EventBus | None
            Event bus for emitting state changes.

        Returns
        -------
        FaceDetectionResult
            Current state snapshot.
        """
        # Frame drop: if processing is in flight, discard
        if self._processing_in_flight:
            return self.get_state()

        self._processing_in_flight = True
        self._frame_counter += 1

        try:
            # Decode base64 to bytes
            frame_bytes = base64.b64decode(frame_base64)

            # Run detection in executor (non-blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                processor.detect_and_embed,
                frame_bytes,
            )

            # 2. No face found
            if not result.get("found", False):
                self._loss_streak += 1
                if (
                    self._loss_streak >= self.FACE_LOST_FRAMES
                    and self._current_state == FaceSessionState.RECOGNIZED
                ):
                    self._current_state = FaceSessionState.IDLE
                    self._current_player_id = None
                    self._current_player_name = None
                    self._recognition_streak = 0
                    if event_bus:
                        event_bus.emit("face:lost", {})
                elif self._current_state == FaceSessionState.IDLE:
                    self._current_state = FaceSessionState.SCANNING

                return self.get_state()

            # 3. Face found but not live (anti-spoofing)
            if not result.get("is_real", True):
                self._current_state = FaceSessionState.SPOOFING_DETECTED
                return FaceDetectionResult(
                    state=FaceSessionState.SPOOFING_DETECTED,
                    is_live=False,
                    facial_area=result.get("facial_area"),
                    embedding_dim=len(result.get("embedding", [])),
                )

            # 4. Face found and live
            self._loss_streak = 0
            embedding = np.array(
                result["embedding"], dtype=np.float32
            )
            facial_area = result.get("facial_area")

            # Match against known players
            candidate = matcher.match(embedding)

            if candidate is not None:
                if candidate.distance < PlayerMatcher.MATCH_THRESHOLD_HIGH:
                    # 4a. High confidence match
                    if candidate.player_id == self._current_player_id:
                        self._recognition_streak += 1
                    else:
                        self._current_player_id = candidate.player_id
                        self._current_player_name = candidate.player_name
                        self._recognition_streak = 1

                    if (
                        self._recognition_streak
                        >= self.RECOGNITION_CONFIRM_FRAMES
                    ):
                        self._current_state = FaceSessionState.RECOGNIZED
                        if event_bus:
                            event_bus.emit(
                                "face:recognized",
                                {
                                    "player_id": candidate.player_id,
                                    "player_name": candidate.player_name,
                                    "confidence": candidate.confidence,
                                },
                            )
                    else:
                        self._current_state = FaceSessionState.DETECTED

                else:
                    # 4b. Medium confidence — needs confirmation
                    self._current_state = FaceSessionState.CONFIRM_NEEDED
                    self._current_player_id = candidate.player_id
                    self._current_player_name = candidate.player_name

            else:
                # 4c. No match
                self._current_state = FaceSessionState.NEW_PLAYER
                self._current_player_id = None
                self._current_player_name = None
                self._recognition_streak = 0

            # 5. Emotion analysis every N frames (always, even without recognition)
            if self._frame_counter % self.EMOTION_EVERY_N_FRAMES == 0:
                emotion = await loop.run_in_executor(
                    executor,
                    processor.analyze_emotion,
                    frame_bytes,
                )
                if emotion:
                    self._last_emotion = emotion
                    if event_bus:
                        event_bus.emit(
                            "face:emotion",
                            {
                                "player_id": self._current_player_id or "",
                                "emotion": emotion,
                            },
                        )

            return FaceDetectionResult(
                state=self._current_state,
                player_id=self._current_player_id,
                player_name=self._current_player_name,
                confidence=candidate.confidence if candidate else 0.0,
                emotion=self._last_emotion,
                is_live=True,
                facial_area=facial_area,
                embedding_dim=len(result.get("embedding", [])),
            )

        except Exception as e:
            logger.error("Frame processing error: %s", e)
            return FaceDetectionResult(
                state=FaceSessionState.CAMERA_ERROR,
            )
        finally:
            self._processing_in_flight = False

    async def start_enrollment(
        self, player_name: str
    ) -> str:
        """Start enrollment flow for a new player.

        Parameters
        ----------
        player_name : str
            Display name for the new player.

        Returns
        -------
        str
            The new player_id.
        """
        player_id = str(uuid.uuid4())
        self._current_state = FaceSessionState.ENROLLING
        self._current_player_id = player_id
        self._current_player_name = player_name
        self._enrollment_buffer = []
        return player_id

    async def process_enrollment_frame(
        self,
        frame_base64: str,
        player_id: str,
        processor: FaceProcessor,
        matcher: PlayerMatcher,
        executor: ProcessPoolExecutor,
        db: Any | None = None,
    ) -> FaceEnrollmentResult:
        """Process a single frame during enrollment.

        Captures diverse embeddings (different angles/lighting).

        Parameters
        ----------
        frame_base64 : str
            Base64-encoded JPEG frame.
        player_id : str
            Player being enrolled.
        processor : FaceProcessor
            Frame processor.
        matcher : PlayerMatcher
            For storing embeddings.
        executor : ProcessPoolExecutor
            Worker pool.
        db : DatabaseManager | None
            For persistence (optional — in-memory only if None).

        Returns
        -------
        FaceEnrollmentResult
            Current enrollment progress.
        """
        try:
            frame_bytes = base64.b64decode(frame_base64)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                processor.detect_and_embed,
                frame_bytes,
            )

            if not result.get("found", False):
                return FaceEnrollmentResult(
                    player_id=player_id,
                    embeddings_captured=len(self._enrollment_buffer),
                    embeddings_target=self.ENROLLMENT_TARGET,
                )

            if not result.get("is_real", True):
                return FaceEnrollmentResult(
                    player_id=player_id,
                    embeddings_captured=len(self._enrollment_buffer),
                    embeddings_target=self.ENROLLMENT_TARGET,
                )

            embedding = np.array(
                result["embedding"], dtype=np.float32
            )

            # Check diversity against existing enrollment captures
            is_diverse = True
            if self._enrollment_buffer:
                centroid = PlayerMatcher._compute_centroid(
                    self._enrollment_buffer
                )
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    normed = embedding / norm
                    dist = 1.0 - float(np.dot(normed, centroid))
                    # First few frames always accepted; later need diversity
                    if (
                        len(self._enrollment_buffer) >= 2
                        and dist <= 0.1
                    ):
                        is_diverse = False

            if is_diverse:
                self._enrollment_buffer.append(embedding)
                player_name = self._current_player_name or "Player"
                await matcher.enroll(
                    player_id, player_name, embedding, db
                )

            captured = len(self._enrollment_buffer)
            success = captured >= self.ENROLLMENT_TARGET

            if success:
                self._current_state = FaceSessionState.RECOGNIZED

            return FaceEnrollmentResult(
                player_id=player_id,
                embeddings_captured=captured,
                embeddings_target=self.ENROLLMENT_TARGET,
                success=success,
            )

        except Exception as e:
            logger.error("Enrollment frame error: %s", e)
            return FaceEnrollmentResult(
                player_id=player_id,
                embeddings_captured=len(self._enrollment_buffer),
                embeddings_target=self.ENROLLMENT_TARGET,
            )

    async def confirm_identity(
        self,
        player_id: str,
        confirmed: bool,
        event_bus: Any | None = None,
    ) -> None:
        """Handle player confirmation for medium-confidence match.

        Parameters
        ----------
        player_id : str
            The candidate player_id.
        confirmed : bool
            True if player says "yes that's me".
        event_bus : EventBus | None
            For emitting state change events.
        """
        if confirmed:
            self._current_state = FaceSessionState.RECOGNIZED
            self._current_player_id = player_id
            if event_bus:
                event_bus.emit(
                    "face:recognized",
                    {
                        "player_id": player_id,
                        "player_name": self._current_player_name,
                    },
                )
        else:
            self._current_state = FaceSessionState.NEW_PLAYER
            self._current_player_id = None
            self._current_player_name = None
            self._recognition_streak = 0

    def get_state(self) -> FaceDetectionResult:
        """Return current state snapshot.

        Returns
        -------
        FaceDetectionResult
            Current face session state.
        """
        return FaceDetectionResult(
            state=self._current_state,
            player_id=self._current_player_id,
            player_name=self._current_player_name,
            emotion=self._last_emotion,
        )
