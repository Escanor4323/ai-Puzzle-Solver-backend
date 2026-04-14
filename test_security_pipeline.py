"""Task 5 verification tests — Hardware Lock / Deadlock System.

Run with:
    python3.12 test_security_pipeline.py
"""

from __future__ import annotations

import hashlib
import hmac
import sys
import tempfile
from pathlib import Path

passed = 0
failed = 0


def ok(name: str):
    global passed
    passed += 1
    print(f"  Test {passed + failed}: {name} — OK")


def fail(name: str, detail: str = ""):
    global failed
    failed += 1
    print(f"  Test {passed + failed}: {name} — FAILED  {detail}")


# ── Test 1: All imports clean ────────────────────────────
try:
    from security.hardware_lock import (
        collect_hardware_ids,
        generate_fingerprint,
        verify_hardware_lock,
        enforce_hardware_lock,
        EXPECTED_FINGERPRINT,
        _OUTER_SALT,
    )
    from security.integrity import (
        build_manifest,
        write_manifest,
        verify_integrity,
        _sha256_file,
    )
    from security.anti_tamper import (
        check_debugger,
        check_environment,
        check_timing,
        run_anti_tamper_checks,
        enforce_anti_tamper,
    )
    from build_scripts.inject_fingerprint import (
        collect_fingerprint,
        inject_fingerprint,
    )
    ok("All imports clean")
except Exception as e:
    fail("All imports clean", str(e))

# ── Test 2: Hardware IDs collected (5 identifiers) ───────
try:
    ids = collect_hardware_ids()
    assert isinstance(ids, dict), f"Expected dict, got {type(ids)}"
    expected_keys = {
        "platform_uuid", "serial_number", "mac_address",
        "boot_rom_version", "model_identifier",
    }
    assert set(ids.keys()) == expected_keys, f"Keys: {set(ids.keys())}"
    non_empty = sum(1 for v in ids.values() if v)
    print(f"    ({non_empty}/5 identifiers found on this Mac)")
    ok(f"Hardware IDs collected ({non_empty}/5 non-empty)")
except Exception as e:
    fail("Hardware IDs collected", str(e))

# ── Test 3: Double-hash fingerprint ──────────────────────
try:
    fp = generate_fingerprint()
    assert isinstance(fp, str), f"Expected str, got {type(fp)}"
    assert len(fp) == 64, f"SHA3-256 should be 64 hex chars, got {len(fp)}"
    # Verify it's valid hex
    int(fp, 16)
    # Verify determinism — calling twice gives same result
    fp2 = generate_fingerprint()
    assert fp == fp2, "Fingerprint not deterministic"
    ok(f"Double-hash fingerprint (sha3-256, {fp[:16]}...)")
except Exception as e:
    fail("Double-hash fingerprint", str(e))

# ── Test 4: Dev mode passthrough ─────────────────────────
try:
    assert EXPECTED_FINGERPRINT == "__EXPECTED_FINGERPRINT_SENTINEL__", \
        "Sentinel was replaced (test must run on uninjected source)"
    result = verify_hardware_lock()
    assert result is True, "Dev mode should pass through"
    ok("Dev mode passthrough (sentinel intact → True)")
except Exception as e:
    fail("Dev mode passthrough", str(e))

# ── Test 5: Constant-time compare (hmac.compare_digest) ──
try:
    # Verify the function uses hmac.compare_digest by checking
    # that it returns False for a wrong fingerprint when sentinel
    # is replaced.  We can't easily test constant-time, but we
    # verify the comparison logic by temporarily patching.
    import security.hardware_lock as hl_mod
    original = hl_mod.EXPECTED_FINGERPRINT
    hl_mod.EXPECTED_FINGERPRINT = "0" * 64  # Wrong fingerprint
    result = hl_mod.verify_hardware_lock()
    assert result is False, "Should fail with wrong fingerprint"
    hl_mod.EXPECTED_FINGERPRINT = original  # Restore
    ok("Constant-time compare (wrong fingerprint → False)")
except Exception as e:
    # Restore on failure too
    try:
        hl_mod.EXPECTED_FINGERPRINT = original
    except Exception:
        pass
    fail("Constant-time compare", str(e))

# ── Test 6: Integrity manifest roundtrip ─────────────────
try:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Create test files
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")

        manifest = build_manifest(tmp_path, ["a.txt", "b.txt"])
        assert len(manifest) == 2, f"Expected 2 entries, got {len(manifest)}"
        assert "a.txt" in manifest
        assert "b.txt" in manifest

        # Write and verify
        write_manifest(tmp_path, manifest)
        assert verify_integrity(tmp_path) is True, "Clean verify should pass"

        # Tamper and re-verify
        (tmp_path / "a.txt").write_text("tampered!")
        assert verify_integrity(tmp_path) is False, "Tampered verify should fail"

    ok("Integrity manifest roundtrip (create → verify → tamper → fail)")
except Exception as e:
    fail("Integrity manifest roundtrip", str(e))

# ── Test 7: Anti-tamper checks pass in dev ───────────────
try:
    assert check_debugger() is True, "Debugger check should pass (no debugger)"
    assert check_environment() is True, "Environment check should pass (macOS)"
    assert check_timing() is True, "Timing check should pass (no instrumentation)"
    assert run_anti_tamper_checks() is True, "Combined check should pass"
    ok("Anti-tamper checks pass (no debugger, macOS, timing OK)")
except Exception as e:
    fail("Anti-tamper checks pass", str(e))

# ── Test 8: Build scripts importable + inject logic ──────
try:
    # Test that inject_fingerprint can find the sentinel
    source = Path("security/hardware_lock.py").read_text()
    assert "__EXPECTED_FINGERPRINT_SENTINEL__" in source, \
        "Sentinel not found in source"

    # Test compile_security has the right module list
    from build_scripts.compile_security import _SECURITY_MODULES
    assert "security/hardware_lock.py" in _SECURITY_MODULES
    assert "security/integrity.py" in _SECURITY_MODULES
    assert "security/anti_tamper.py" in _SECURITY_MODULES

    # Test entitlements.plist exists
    assert Path("build_scripts/entitlements.plist").is_file()

    # Test build_release.sh exists and is executable
    import os
    build_sh = Path("build_scripts/build_release.sh")
    assert build_sh.is_file()
    assert os.access(build_sh, os.X_OK), "build_release.sh not executable"

    ok("Build scripts importable + assets present")
except Exception as e:
    fail("Build scripts importable + assets present", str(e))

# ── Summary ──────────────────────────────────────────────
print()
print(f"{'=' * 50}")
print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)
