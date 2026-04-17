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
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: PyTorch + torchvision (pin both from CPU wheel to prevent downgrade) ──
# fer → facenet-pytorch → torchvision; without pinning torchvision here, pip
# picks torchvision==0.17.2 from PyPI which requires torch==2.2.2 and downgrades.
# torchvision>=0.19.0 is compatible with torch>=2.4.0.
RUN pip install --no-cache-dir \
        "torch>=2.4.0" "torchvision>=0.19.0" --index-url https://download.pytorch.org/whl/cpu

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
# --extra-index-url ensures pip can resolve torchvision from the CPU wheel index
# so fer→facenet-pytorch→torchvision gets 0.26.0+cpu (compat with torch>=2.4)
# instead of torchvision-0.17.2 from PyPI (which requires torch==2.2.2).
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        fastapi>=0.115.0 \
        uvicorn[standard]>=0.30.0 \
        websockets>=13.0 \
        pydantic>=2.0 \
        pydantic-settings>=2.0 \
        anthropic>=0.80.0 \
        openai>=1.50.0 \
        "pymilvus[milvus_lite]>=2.4.0,<2.5.0" \
        networkx>=3.3 \
        keyring>=25.0.0 \
        cryptography>=43.0.0 \
        pillow>=10.0.0 \
        numpy>=1.26.0 \
        python-dotenv>=1.0.0 \
        edge-tts>=6.0.0 \
        fer>=22.5.1 \
        pytest>=8.0.0 \
        httpx>=0.27.0

# ── Layer 6b: Force-pin torch/torchvision in case any dep downgraded them ────
# Safety net: runs after all pip installs so it always has the final say.
RUN pip install --no-cache-dir --force-reinstall \
        "torch>=2.4.0" "torchvision>=0.19.0" --index-url https://download.pytorch.org/whl/cpu

# ── Layer 7: Pre-bake CV face models from GitHub Release ─────────────────────
# DeepFace resolves weights as {DEEPFACE_HOME}/.deepface/weights/
# Stored under /opt (not /tmp) so Docker does NOT overlay them at runtime.
# Source: https://github.com/Escanor4323/ai-Puzzle-Solver-backend/releases/tag/models-v1
ENV DEEPFACE_HOME=/opt/deepface_cache
RUN mkdir -p /opt/deepface_cache/.deepface/weights && \
    BASE="https://github.com/Escanor4323/ai-Puzzle-Solver-backend/releases/download/models-v1" && \
    curl -fsSL "${BASE}/ghostfacenet_v1.h5" \
         -o /opt/deepface_cache/.deepface/weights/ghostfacenet_v1.h5 && \
    curl -fsSL "${BASE}/facial_expression_model_weights.h5" \
         -o /opt/deepface_cache/.deepface/weights/facial_expression_model_weights.h5 && \
    curl -fsSL "${BASE}/2.7_80x80_MiniFASNetV2.pth" \
         -o "/opt/deepface_cache/.deepface/weights/2.7_80x80_MiniFASNetV2.pth" && \
    curl -fsSL "${BASE}/4_0_0_80x80_MiniFASNetV1SE.pth" \
         -o "/opt/deepface_cache/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth"

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
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache \
    # This image has no CUDA runtime — tell TensorFlow explicitly so it
    # skips the GPU probe entirely and doesn't spam CUDA error messages.
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

# ── Switch to non-root user ───────────────────────────────────────────────────
USER 1001

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8008/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008", "--workers", "1"]
