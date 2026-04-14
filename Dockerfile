# ─────────────────────────────────────────────────────────────────────────────
# AI Puzzle Solver – Backend
# OpenShift-compliant image (non-root, UID 1001)
# Packages installed in separate layers so `docker pull` shows real progress
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: PyTorch (sentence-transformers pulls it anyway; pin CPU wheel) ──
# Install CPU-only torch to avoid ~1 GB of CUDA libraries in the image
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: Sentence-transformers + HuggingFace stack ───────────────────────
RUN pip install --no-cache-dir \
        sentence-transformers>=3.0.0 \
        transformers>=4.40.0

# ── Layer 3: TensorFlow / Keras (DeepFace backend) ───────────────────────────
RUN pip install --no-cache-dir \
        tf-keras

# ── Layer 4: CV + inference runtime ──────────────────────────────────────────
RUN pip install --no-cache-dir \
        opencv-python-headless>=4.10.0 \
        onnxruntime>=1.17.0

# ── Layer 5: Vendored DeepFace v0.0.99 ───────────────────────────────────────
COPY vendor/deepface /tmp/deepface
RUN pip install --no-cache-dir /tmp/deepface \
    && rm -rf /tmp/deepface

# ── Layer 6: Remaining application dependencies ──────────────────────────────
RUN pip install --no-cache-dir \
        fastapi>=0.115.0 \
        uvicorn[standard]>=0.30.0 \
        websockets>=13.0 \
        pydantic>=2.0 \
        pydantic-settings>=2.0 \
        anthropic>=0.80.0 \
        openai>=1.50.0 \
        "pymilvus[milvus_lite]>=2.4.0" \
        networkx>=3.3 \
        keyring>=25.0.0 \
        cryptography>=43.0.0 \
        pillow>=10.0.0 \
        numpy>=1.26.0 \
        python-dotenv>=1.0.0 \
        edge-tts>=6.0.0 \
        pytest>=8.0.0 \
        httpx>=0.27.0

# ── Layer 7: Pre-bake CV face models from GitHub Release ─────────────────────
# Stored under /opt (not /tmp) so Docker does NOT overlay them at runtime.
# Source: https://github.com/Escanor4323/ai-Puzzle-Solver-backend/releases/tag/models-v1
ENV DEEPFACE_HOME=/opt/deepface_cache
RUN mkdir -p /opt/deepface_cache/weights && \
    BASE="https://github.com/Escanor4323/ai-Puzzle-Solver-backend/releases/download/models-v1" && \
    curl -fsSL "${BASE}/ghostfacenet_v1.h5" \
         -o /opt/deepface_cache/weights/ghostfacenet_v1.h5 && \
    curl -fsSL "${BASE}/facial_expression_model_weights.h5" \
         -o /opt/deepface_cache/weights/facial_expression_model_weights.h5 && \
    curl -fsSL "${BASE}/2.7_80x80_MiniFASNetV2.pth" \
         -o "/opt/deepface_cache/weights/2.7_80x80_MiniFASNetV2.pth" && \
    curl -fsSL "${BASE}/4_0_0_80x80_MiniFASNetV1SE.pth" \
         -o "/opt/deepface_cache/weights/4_0_0_80x80_MiniFASNetV1SE.pth"

# ── OpenShift restricted-SCC compliance ──────────────────────────────────────
# OpenShift runs containers with an arbitrary UID but always GID 0 (root group).
# Own files as UID 1001 + GID 0, grant group-write so any UID in the project
# range can still read/write the working directory.
RUN groupadd -g 1001 appgroup \
    && useradd -u 1001 -g appgroup -M -s /sbin/nologin appuser \
    && mkdir -p /app /tmp/hf_cache /tmp/torch_cache \
    && chown -R 1001:0 /app /opt/deepface_cache /tmp/hf_cache /tmp/torch_cache \
    && chmod -R g=u /app /opt/deepface_cache /tmp/hf_cache /tmp/torch_cache

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=1001:0 . .

# ── Runtime cache redirects ───────────────────────────────────────────────────
# DEEPFACE_HOME=/opt/deepface_cache — set above, models are baked in at /opt
# HF/Torch caches go to /tmp (writable, ephemeral is fine for these)
ENV HF_HOME=/tmp/hf_cache \
    TORCH_HOME=/tmp/torch_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache

# ── Switch to non-root user ───────────────────────────────────────────────────
USER 1001

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8008/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008", "--workers", "1"]
