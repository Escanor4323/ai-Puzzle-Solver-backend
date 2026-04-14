"""Extract a short WAV reference clip from a (potentially long) MP3 source.

XTTS-v2 performs best with a 10–30 second mono WAV at 22 050 Hz.
Loading a multi-hour MP3 directly would waste memory and time, so we
slice out a clean segment once and cache the result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

_MIN_DURATION = 3.0    # XTTS-v2 floor
_MAX_DURATION = 30.0   # XTTS-v2 ceiling; longer clips don't improve quality
_TARGET_SR = 22_050    # XTTS-v2 native sample rate


def extract_reference_clip(
    source: Path,
    dest: Path,
    *,
    duration_secs: float = 28.0,
    start_offset_secs: float = 5.0,
) -> None:
    """Load *duration_secs* seconds from *source* and write a 22 050 Hz mono WAV.

    Parameters
    ----------
    source:
        Path to the source audio file (MP3, WAV, etc.).
    dest:
        Output path for the clipped WAV file.
    duration_secs:
        How many seconds to extract.  Must be in [3, 30].
    start_offset_secs:
        Skip the first N seconds (avoids title cards / silence at the start).

    Raises
    ------
    FileNotFoundError
        If *source* does not exist.
    ValueError
        If *duration_secs* is outside [3, 30].
    """
    if not source.exists():
        raise FileNotFoundError(f"Source audio not found: {source}")

    if not (_MIN_DURATION <= duration_secs <= _MAX_DURATION):
        raise ValueError(
            f"duration_secs must be in [{_MIN_DURATION}, {_MAX_DURATION}], "
            f"got {duration_secs}"
        )

    logger.info(
        "Extracting %.1fs reference clip from %s → %s",
        duration_secs, source.name, dest,
    )

    y, _ = librosa.load(
        str(source),
        sr=_TARGET_SR,
        mono=True,
        offset=start_offset_secs,
        duration=duration_secs,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y, _TARGET_SR, subtype="PCM_16")

    logger.info("Reference clip written: %s (%.1f s)", dest, len(y) / _TARGET_SR)
