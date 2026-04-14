#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# PuzzleMind — Production Build Pipeline
# ─────────────────────────────────────────────────────────
#
# Builds a locked, signed macOS release binary:
#   1. Inject hardware fingerprint into source
#   2. Compile security modules with Cython (.py → .so)
#   3. Generate SHA-256 integrity manifest
#   4. Package with PyInstaller into a single-folder bundle
#   5. Code-sign with entitlements
#
# Usage:
#   cd ai-Puzzle-Solver-backend
#   bash build_scripts/build_release.sh [SIGNING_IDENTITY]
#
# Arguments:
#   SIGNING_IDENTITY  — macOS code signing identity (optional).
#                       Defaults to ad-hoc signing (-).
#
# Prerequisites:
#   pip install cython setuptools pyinstaller
# ─────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$BACKEND_ROOT/dist"
BUILD_DIR="$BACKEND_ROOT/build"
SIGNING_IDENTITY="${1:--}"

echo "======================================================="
echo " PuzzleMind Production Build"
echo "======================================================="
echo
echo " Backend root:     $BACKEND_ROOT"
echo " Signing identity: $SIGNING_IDENTITY"
echo " Python:           $(python3 --version 2>&1)"
echo

cd "$BACKEND_ROOT"

# ── Step 0: Clean previous builds ────────────────────────
echo "--- Step 0: Cleaning previous build artifacts ---"
rm -rf "$DIST_DIR" "$BUILD_DIR" *.spec
echo "  Cleaned."
echo

# ── Step 1: Inject hardware fingerprint ──────────────────
echo "--- Step 1: Injecting hardware fingerprint ---"
python3 build_scripts/inject_fingerprint.py
echo

# ── Step 2: Compile security modules with Cython ─────────
echo "--- Step 2: Compiling security modules (Cython) ---"
python3 build_scripts/compile_security.py
echo

# ── Step 3: Generate integrity manifest ──────────────────
echo "--- Step 3: Generating integrity manifest ---"
python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from security.integrity import build_manifest, write_manifest

base = Path('.')
# Hash all key files (compiled .so modules, main entry, config)
import glob
files = []
# Compiled security modules
for so in glob.glob('security/*.so'):
    files.append(so)
# Core application files
for f in ['main.py', 'config.py']:
    if Path(f).is_file():
        files.append(f)
# AI modules
for f in glob.glob('ai/*.py'):
    files.append(f)

manifest = build_manifest(base, files)
write_manifest(base, manifest)
print(f'  Hashed {len(manifest)} files')
"
echo

# ── Step 4: Package with PyInstaller ─────────────────────
echo "--- Step 4: Packaging with PyInstaller ---"
pyinstaller \
    --name PuzzleMind \
    --onedir \
    --noconfirm \
    --clean \
    --add-data "data:data" \
    --add-data ".integrity_manifest.json:." \
    --add-data "security:security" \
    --hidden-import uvicorn \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import fastapi \
    --hidden-import pydantic \
    --hidden-import pydantic_settings \
    --hidden-import websockets \
    --collect-all sentence_transformers \
    --collect-all transformers \
    --collect-all pymilvus \
    main.py
echo

# ── Step 5: Code sign ───────────────────────────────────
echo "--- Step 5: Code signing ---"
ENTITLEMENTS="$SCRIPT_DIR/entitlements.plist"
APP_DIR="$DIST_DIR/PuzzleMind"

if [ -d "$APP_DIR" ]; then
    # Sign all .so and .dylib files first
    find "$APP_DIR" \( -name "*.so" -o -name "*.dylib" \) -exec \
        codesign --force --sign "$SIGNING_IDENTITY" \
        --entitlements "$ENTITLEMENTS" \
        --timestamp \
        --options runtime {} \;

    # Sign the main executable
    codesign --force --sign "$SIGNING_IDENTITY" \
        --entitlements "$ENTITLEMENTS" \
        --timestamp \
        --options runtime \
        "$APP_DIR/PuzzleMind"

    echo "  Code signing complete."
else
    echo "  WARNING: $APP_DIR not found — skipping signing."
fi
echo

# ── Done ─────────────────────────────────────────────────
echo "======================================================="
echo " Build complete!"
echo " Output: $APP_DIR"
echo "======================================================="
echo
echo " To verify:"
echo "   $APP_DIR/PuzzleMind --verify-only"
echo
