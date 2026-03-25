"""Build-time hardware fingerprint injection.

This script is executed during the production build process.
It collects the target Mac's hardware fingerprint and injects the
hash into ``security/hardware_lock.py`` by replacing the
``EXPECTED_FINGERPRINT`` sentinel string.  The modified source is
then compiled by Nuitka / PyInstaller.

Usage::

    python build_scripts/inject_fingerprint.py

This script is NOT used during development — the hardware lock is
disabled by default via ``PUZZLEMIND_HARDWARE_LOCK_ENABLED=false``.
"""
