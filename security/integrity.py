"""SHA-256 file integrity verification.

At build time, ``build_release.sh`` computes SHA-256 hashes for all
protected files (compiled ``.so`` modules, the main entry point, config)
and writes them to a manifest.  At runtime this module verifies that
no files have been modified since the build.

In development (no manifest present) all checks pass through.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Build-time manifest path ─────────────────────────────────────
# Written by build_release.sh after compilation + signing.
_MANIFEST_FILENAME = ".integrity_manifest.json"


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads in 64 KiB chunks to handle large files without
    excessive memory usage.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65_536):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(base_dir: Path, files: list[str]) -> dict[str, str]:
    """Create a SHA-256 manifest for the given files.

    Parameters
    ----------
    base_dir : Path
        Root directory of the application.
    files : list[str]
        Relative paths (from *base_dir*) to hash.

    Returns
    -------
    dict[str, str]
        Mapping of relative path → SHA-256 hex digest.
    """
    manifest: dict[str, str] = {}
    for rel_path in files:
        full = base_dir / rel_path
        if full.is_file():
            manifest[rel_path] = _sha256_file(full)
            logger.debug("Hashed %s", rel_path)
        else:
            logger.warning("Manifest: file not found: %s", rel_path)
    return manifest


def write_manifest(base_dir: Path, manifest: dict[str, str]) -> Path:
    """Write the manifest to the build directory.

    Parameters
    ----------
    base_dir : Path
        Root directory of the application.
    manifest : dict[str, str]
        Mapping produced by ``build_manifest``.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest_path = base_dir / _MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Wrote integrity manifest: %s", manifest_path)
    return manifest_path


def verify_integrity(base_dir: Path | None = None) -> bool:
    """Verify all files in the manifest against their SHA-256 hashes.

    Parameters
    ----------
    base_dir : Path | None
        Root directory.  Defaults to the parent of this module's
        package (i.e. the backend root).

    Returns
    -------
    bool
        True if all files match (or no manifest exists — dev mode).
        False if any file has been tampered with.
    """
    if base_dir is None:
        # security/ is one level down from the backend root
        base_dir = Path(__file__).resolve().parent.parent

    manifest_path = base_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        logger.debug(
            "Integrity check: no manifest (dev mode) — pass through"
        )
        return True

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        # Corrupted manifest → fail
        return False

    for rel_path, expected_hash in manifest.items():
        full = base_dir / rel_path
        if not full.is_file():
            # File removed after build → tampered
            return False

        actual_hash = _sha256_file(full)
        if actual_hash != expected_hash:
            # Content changed → tampered
            return False

    return True
