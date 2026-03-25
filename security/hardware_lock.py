"""Hardware fingerprint collection and runtime deadlock verification.

Uses macOS IOKit (via pyobjc) to read the platform UUID, serial number,
and board ID.  A composite SHA-512 hash locks the application to a
specific Mac.  The expected fingerprint is injected at build time by
``build_scripts/inject_fingerprint.py``.
"""


def generate_fingerprint() -> str:
    """Collect hardware identifiers and return a composite hash.

    Returns
    -------
    str
        SHA-512 hex digest of the composite fingerprint.
    """
    pass


def verify_hardware_lock() -> bool:
    """Compare the live fingerprint against the build-time expectation.

    Returns
    -------
    bool
        True if the fingerprint matches, False otherwise.
    """
    pass
