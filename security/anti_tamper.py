"""Anti-tamper detection for PuzzleMind.

Three independent detection layers:

1. **Debugger detection** — checks for ``sysctl`` P_TRACED flag,
   ``DYLD_INSERT_LIBRARIES``, and common debugger environment vars.
2. **Environment validation** — verifies the process is running on
   macOS with expected architecture and no virtualisation indicators.
3. **Timing checks** — measures critical-section execution time to
   detect single-step debugging or instrumentation delays.

All checks are designed to fail silently (exit code 1) in production
and pass through in development.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct
import sys
import time

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
_CTL_KERN = 1
_KERN_PROC = 14
_KERN_PROC_PID = 1
_P_TRACED = 0x00000800

# Suspicious environment variables set by debuggers / RE tools
_DEBUGGER_ENV_VARS = frozenset({
    "DYLD_INSERT_LIBRARIES",
    "_LLDB_PROCESS_ENV",
    "LLDB_LAUNCH_FLAG_DISABLE_ASLR",
    "NSZombieEnabled",
    "MallocStackLogging",
    "DYLD_PRINT_LIBRARIES",
    "DYLD_PRINT_INITIALIZERS",
})

# Maximum allowed time (seconds) for a trivial computation.
# If exceeded, something is instrumenting / single-stepping.
_TIMING_THRESHOLD_SECONDS = 2.0


# ── Layer 1: Debugger Detection ──────────────────────────────────

def _is_being_traced() -> bool:
    """Check the P_TRACED flag via sysctl(kern.proc.pid).

    Returns True if a debugger is attached.
    """
    try:
        libc_path = ctypes.util.find_library("c")
        if not libc_path:
            return False
        libc = ctypes.CDLL(libc_path)

        # struct kinfo_proc is large; we only need p_flag at offset
        # Use a 648-byte buffer (macOS kinfo_proc size on arm64/x86_64)
        buf_size = 648
        buf = ctypes.create_string_buffer(buf_size)
        length = ctypes.c_size_t(buf_size)

        # sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()}, 4, ...)
        mib = (ctypes.c_int * 4)(
            _CTL_KERN, _KERN_PROC, _KERN_PROC_PID, os.getpid()
        )
        ret = libc.sysctl(
            mib, 4,
            buf, ctypes.byref(length),
            None, 0,
        )
        if ret != 0:
            return False

        # p_flag is at offset 32 in kinfo_proc (kp_proc.p_flag)
        p_flag = struct.unpack_from("I", buf.raw, 32)[0]
        return bool(p_flag & _P_TRACED)

    except Exception:
        return False


def _has_debugger_env() -> bool:
    """Check for suspicious environment variables."""
    for var in _DEBUGGER_ENV_VARS:
        if os.environ.get(var):
            return True
    return False


def check_debugger() -> bool:
    """Return True if no debugger is detected (safe to proceed).

    Checks both the P_TRACED kernel flag and environment variables.
    """
    if _is_being_traced():
        return False
    if _has_debugger_env():
        return False
    return True


# ── Layer 2: Environment Validation ──────────────────────────────

def check_environment() -> bool:
    """Validate that we're running on genuine macOS hardware.

    Checks:
    - Platform is darwin
    - No virtualisation-related env vars (common VM indicators)
    - Running as a bundled app or normal Python (not Frida/inject)

    Returns True if environment looks legitimate.
    """
    # Must be macOS
    if sys.platform != "darwin":
        return False

    # Check for common VM / instrumentation indicators
    vm_indicators = [
        "QEMU_AUDIO_DRV",
        "VBOX_PROGRAM_NAME",
        "VMWARE_USERNAME",
    ]
    for var in vm_indicators:
        if os.environ.get(var):
            return False

    # Check for Frida (common dynamic instrumentation tool)
    if os.environ.get("FRIDA_SCRIPT"):
        return False

    return True


# ── Layer 3: Timing Check ────────────────────────────────────────

def check_timing() -> bool:
    """Detect single-step debugging via timing measurement.

    Performs a trivial computation and checks whether it took
    unreasonably long — a sign of instrumentation or single-stepping.

    Returns True if timing is within acceptable bounds.
    """
    start = time.monotonic()

    # Trivial computation that should complete in microseconds
    total = 0
    for i in range(10_000):
        total += i * i

    elapsed = time.monotonic() - start

    if elapsed > _TIMING_THRESHOLD_SECONDS:
        return False

    return True


# ── Combined Check ───────────────────────────────────────────────

def run_anti_tamper_checks() -> bool:
    """Run all three anti-tamper layers.

    Returns
    -------
    bool
        True if all checks pass.  False if any layer detects
        tampering / debugging / hostile environment.
    """
    if not check_debugger():
        logger.debug("Anti-tamper: debugger detected")
        return False

    if not check_environment():
        logger.debug("Anti-tamper: environment check failed")
        return False

    if not check_timing():
        logger.debug("Anti-tamper: timing check failed")
        return False

    return True


def enforce_anti_tamper() -> None:
    """Run anti-tamper checks and exit silently on failure.

    Called from ``main.py`` during startup when
    ``HARDWARE_LOCK_ENABLED`` is True.
    """
    if not run_anti_tamper_checks():
        sys.exit(1)
