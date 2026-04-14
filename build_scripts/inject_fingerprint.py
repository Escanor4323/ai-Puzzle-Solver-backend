"""Build-time hardware fingerprint injection.

Collects the target Mac's hardware fingerprint via
``security.hardware_lock.generate_fingerprint()``, then patches the
``EXPECTED_FINGERPRINT`` sentinel in ``security/hardware_lock.py``
with the real double-hash.  The modified source is then compiled by
Cython / PyInstaller.

Usage::

    cd ai-Puzzle-Solver-backend
    python build_scripts/inject_fingerprint.py

This script is NOT used during development — the hardware lock is
disabled by default via ``PUZZLEMIND_HARDWARE_LOCK_ENABLED=false``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

_SENTINEL = '__EXPECTED_FINGERPRINT_SENTINEL__'
_HARDWARE_LOCK_PATH = _BACKEND_ROOT / "security" / "hardware_lock.py"


def collect_fingerprint() -> str:
    """Generate the hardware fingerprint for this Mac.

    Returns
    -------
    str
        SHA3-256 hex digest (double-hashed composite fingerprint).
    """
    from security.hardware_lock import (
        collect_hardware_ids,
        generate_fingerprint,
    )

    # Show collected IDs for build-log verification
    ids = collect_hardware_ids()
    print("Collected hardware identifiers:")
    for name, value in ids.items():
        # Mask the middle of each value for log safety
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else value
        print(f"  {name}: {masked}")

    fingerprint = generate_fingerprint()
    print(f"Double-hash fingerprint: {fingerprint[:16]}...{fingerprint[-16:]}")
    return fingerprint


def inject_fingerprint(fingerprint: str) -> None:
    """Replace the sentinel in hardware_lock.py with the real fingerprint.

    Parameters
    ----------
    fingerprint : str
        The SHA3-256 hex digest to inject.

    Raises
    ------
    RuntimeError
        If the sentinel is not found in the source file (already
        injected, or file was modified).
    """
    source = _HARDWARE_LOCK_PATH.read_text(encoding="utf-8")

    if _SENTINEL not in source:
        raise RuntimeError(
            f"Sentinel '{_SENTINEL}' not found in "
            f"{_HARDWARE_LOCK_PATH}.  Was it already injected?"
        )

    patched = source.replace(_SENTINEL, fingerprint)
    _HARDWARE_LOCK_PATH.write_text(patched, encoding="utf-8")
    print(f"Injected fingerprint into {_HARDWARE_LOCK_PATH}")


def main() -> None:
    """Entry point: collect → inject → verify."""
    print("=" * 55)
    print("PuzzleMind: Hardware Fingerprint Injection")
    print("=" * 55)
    print()

    # Collect
    fingerprint = collect_fingerprint()
    print()

    # Inject
    inject_fingerprint(fingerprint)

    # Verify the sentinel was replaced
    source = _HARDWARE_LOCK_PATH.read_text(encoding="utf-8")
    if _SENTINEL in source:
        print("ERROR: Sentinel still present after injection!")
        sys.exit(1)
    if fingerprint in source:
        print("Verification: fingerprint found in patched source — OK")
    else:
        print("ERROR: Fingerprint not found in patched source!")
        sys.exit(1)

    print()
    print("Fingerprint injection complete.")
    print("Proceed with Cython compilation + PyInstaller packaging.")


if __name__ == "__main__":
    main()
