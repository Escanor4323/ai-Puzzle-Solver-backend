# ─────────────────────────────────────────────────────────────────────────────
# AI Puzzle Solver – Backend
# OpenShift-compliant image (non-root, UID 1001)
#
# Supports two build modes via build arg:
#   CPU (default — Mac, Windows without GPU, arm64):
#       docker build .
#   GPU (Windows with NVIDIA CUDA 12.4):
#       docker build --build-arg GPU=true .
#
# Runtime:
#   CPU:  docker run --rm -p 8008:8008 ...
#   GPU:  docker run --rm -p 8008:8008 --gpus all ...
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# GPU=false  → CPU-only wheels, tensorflow (arm64-safe), CUDA disabled
# GPU=true   → CUDA 12.4 PyTorch wheels, tensorflow[and-cuda], GPU enabled
ARG GPU=false

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: PyTorch + torchvision ───────────────────────────────────────────
# GPU=true  → CUDA 12.4 wheels (linux/amd64 only)
# GPU=false → CPU wheels (linux/amd64 + linux/arm64)
# fer → facenet-pytorch → torchvision; pin torchvision to prevent pip from
# picking torchvision==0.17.2 which requires torch==2.2.2 and downgrades.
RUN if [ "$GPU" = "true" ]; then \
        pip install --no-cache-dir \
            "torch>=2.4.0" "torchvision>=0.19.0" \
            --index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install --no-cache-dir \
            "torch>=2.4.0" "torchvision>=0.19.0" \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── Layer 2: Sentence-transformers + HuggingFace stack ───────────────────────
RUN pip install --no-cache-dir \
        sentence-transformers>=3.0.0 \
        transformers>=4.40.0

# ── Layer 3: TensorFlow / Keras ──────────────────────────────────────────────
# GPU=true  → tensorflow[and-cuda] bundles libcudart/libcublas/libcudnn via
#             pip NVIDIA packages — no CUDA base image required.
# GPU=false → plain tensorflow (has arm64 wheels; CPU-only at runtime via
#             CUDA_VISIBLE_DEVICES=-1 set below).
# tf-keras installed in a separate RUN to avoid pip choosing tensorflow-cpu.
RUN if [ "$GPU" = "true" ]; then \
        pip install --no-cache-dir "tensorflow[and-cuda]>=2.16.0"; \
    else \
        pip install --no-cache-dir "tensorflow>=2.16.0"; \
    fi
RUN pip install --no-cache-dir tf-keras

# ── Layer 4: CV + inference runtime ──────────────────────────────────────────
RUN pip install --no-cache-dir \
        opencv-python-headless>=4.10.0 \
        onnxruntime>=1.17.0

# ── Layer 5: Vendored DeepFace v0.0.99 ───────────────────────────────────────
COPY vendor/deepface /tmp/deepface
RUN pip install --no-cache-dir /tmp/deepface \
    && rm -rf /tmp/deepface

# ── Layer 6: Remaining application dependencies ──────────────────────────────
# extra-index-url matches whichever torch variant was installed above.
RUN if [ "$GPU" = "true" ]; then \
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"; \
    else \
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"; \
    fi && \
    pip install --no-cache-dir \
        --extra-index-url "${TORCH_INDEX}" \
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
RUN if [ "$GPU" = "true" ]; then \
        pip install --no-cache-dir --force-reinstall \
            "torch>=2.4.0" "torchvision>=0.19.0" \
            --index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install --no-cache-dir --force-reinstall \
            "torch>=2.4.0" "torchvision>=0.19.0" \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

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
RUN groupadd -g 1001 appgroup \
    && useradd -u 1001 -g appgroup -M -s /sbin/nologin appuser \
    && mkdir -p /app /tmp/hf_cache /tmp/torch_cache \
    && chown -R 1001:0 /app /opt/deepface_cache /tmp/hf_cache /tmp/torch_cache \
    && chmod -R g=u /app /opt/deepface_cache /tmp/hf_cache /tmp/torch_cache

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=1001:0 . .

# ── Runtime environment ───────────────────────────────────────────────────────
# HF/Torch caches go to /tmp (writable, ephemeral).
# TF_ENABLE_ONEDNN_OPTS=0 silences the oneDNN floating-point notice.
# TF_CPP_MIN_LOG_LEVEL=1 suppresses TF C++ INFO noise; errors still surface.
# CUDA_VISIBLE_DEVICES=-1 is set for CPU builds to prevent TF from probing
# for GPU hardware that isn't available/passed through.
ARG GPU=false
ENV HF_HOME=/tmp/hf_cache \
    TORCH_HOME=/tmp/torch_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=1
RUN if [ "$GPU" != "true" ]; then \
        echo "CUDA_VISIBLE_DEVICES=-1" >> /etc/environment; \
    fi

# ── Switch to non-root user ───────────────────────────────────────────────────
USER 1001

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8008/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008", "--workers", "1"]
