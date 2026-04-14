"""Hardware fingerprint collection and runtime deadlock verification.

Collects five macOS hardware identifiers via ``ioreg`` / ``system_profiler``,
builds a composite fingerprint with a double-hash (SHA-512 → SHA3-256),
and compares at runtime against a compile-time constant injected by
``build_scripts/inject_fingerprint.py``.

The lock is disabled by default (``HARDWARE_LOCK_ENABLED=false``).
When enabled, ``verify_hardware_lock()`` performs a constant-time
comparison via ``hmac.compare_digest`` — failure exits silently with
code 1 (no error messages, no exceptions).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import subprocess
import sys

logger = logging.getLogger(__name__)

# ── Build-time sentinel — replaced by inject_fingerprint.py ──────
# The string below is a placeholder.  During the production build,
# ``inject_fingerprint.py`` replaces it with the real double-hash
# of the target Mac's hardware fingerprint.
EXPECTED_FINGERPRINT: str = "__EXPECTED_FINGERPRINT_SENTINEL__"

# ── Salt for the outer SHA3-256 layer ────────────────────────────
_OUTER_SALT = b"PuzzleMind::HardwareLock::v1"


# ── Hardware Identifier Collectors ───────────────────────────────

def _run_cmd(cmd: list[str]) -> str:
    """Run a subprocess and return stripped stdout, or '' on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_platform_uuid() -> str:
    """IOPlatformUUID from IORegistry (most reliable Mac identifier)."""
    raw = _run_cmd([
        "ioreg", "-rd1", "-c", "IOPlatformExpertDevice",
    ])
    match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', raw)
    return match.group(1) if match else ""


def _get_serial_number() -> str:
    """Mac serial number via ioreg."""
    raw = _run_cmd([
        "ioreg", "-rd1", "-c", "IOPlatformExpertDevice",
    ])
    match = re.search(r'"IOPlatformSerialNumber"\s*=\s*"([^"]+)"', raw)
    return match.group(1) if match else ""


def _get_mac_address() -> str:
    """Primary (en0) MAC address via ifconfig."""
    raw = _run_cmd(["ifconfig", "en0"])
    match = re.search(r"ether\s+([0-9a-f:]{17})", raw)
    return match.group(1) if match else ""


def _get_boot_rom_version() -> str:
    """Boot ROM version from system_profiler."""
    raw = _run_cmd([
        "system_profiler", "SPHardwareDataType",
    ])
    match = re.search(r"Boot ROM Version:\s*(.+)", raw)
    return match.group(1).strip() if match else ""


def _get_model_identifier() -> str:
    """Model identifier (e.g. MacBookPro18,1) from system_profiler."""
    raw = _run_cmd([
        "system_profiler", "SPHardwareDataType",
    ])
    match = re.search(r"Model Identifier:\s*(.+)", raw)
    return match.group(1).strip() if match else ""


# ── Fingerprint Generation ───────────────────────────────────────

def collect_hardware_ids() -> dict[str, str]:
    """Collect all five hardware identifiers.

    Returns
    -------
    dict[str, str]
        Mapping of identifier name → value.
    """
    return {
        "platform_uuid": _get_platform_uuid(),
        "serial_number": _get_serial_number(),
        "mac_address": _get_mac_address(),
        "boot_rom_version": _get_boot_rom_version(),
        "model_identifier": _get_model_identifier(),
    }


def generate_fingerprint() -> str:
    """Collect hardware identifiers and return a double-hashed fingerprint.

    Double-hash: SHA-512(composite) → SHA3-256(sha512 + salt)

    Returns
    -------
    str
        SHA3-256 hex digest of the double-hashed fingerprint.
    """
    ids = collect_hardware_ids()

    # Composite string — deterministic ordering
    composite = "|".join([
        ids["platform_uuid"],
        ids["serial_number"],
        ids["mac_address"],
        ids["boot_rom_version"],
        ids["model_identifier"],
    ])

    # Layer 1: SHA-512
    sha512_digest = hashlib.sha512(composite.encode("utf-8")).digest()

    # Layer 2: SHA3-256 with salt
    sha3_hash = hashlib.sha3_256()
    sha3_hash.update(sha512_digest)
    sha3_hash.update(_OUTER_SALT)

    return sha3_hash.hexdigest()


# ── Runtime Verification ─────────────────────────────────────────

def verify_hardware_lock() -> bool:
    """Compare the live fingerprint against the build-time expectation.

    Uses ``hmac.compare_digest`` for constant-time comparison to
    prevent timing side-channels.

    If the sentinel is still present (dev mode / not injected), the
    lock passes through — development is not blocked.

    Returns
    -------
    bool
        True if the fingerprint matches (or lock is not active).
        False triggers silent exit in production.
    """
    # Dev mode: sentinel was never replaced → pass through
    if EXPECTED_FINGERPRINT == "__EXPECTED_FINGERPRINT_SENTINEL__":
        logger.debug("Hardware lock: dev mode (sentinel intact)")
        return True

    live_fingerprint = generate_fingerprint()

    # Constant-time comparison
    return hmac.compare_digest(
        live_fingerprint.encode("utf-8"),
        EXPECTED_FINGERPRINT.encode("utf-8"),
    )


def enforce_hardware_lock() -> None:
    """Check the hardware lock and exit silently on mismatch.

    Called from ``main.py`` during startup when
    ``HARDWARE_LOCK_ENABLED`` is True.  On failure, exits with
    code 1 — no error messages, no exceptions, no stack traces.
    """
    if not verify_hardware_lock():
        # Silent exit — no messages to aid reverse-engineering
        sys.exit(1)
