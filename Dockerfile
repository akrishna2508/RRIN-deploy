# ============================================================
# Dockerfile — RRIN Inference Server
# ============================================================
#   You do NOT run this file yourself.
#   Render.com / Railway.app / Hugging Face Spaces reads this
#   file automatically and uses it to build your server.
#
# WHAT THIS DOES:
#   1. Starts from a lightweight Python 3.11 base image
#   2. Copies your project code into the container
#   3. Installs all Python dependencies
#   4. When the container starts, it runs the inference server
#      which downloads the model from Hugging Face automatically
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system libraries needed by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (so Docker can cache this layer)
COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the project code
COPY . .

# Create the model cache directory (model downloads here on first start)
RUN mkdir -p model_cache

# Environment variables (override these in Render/HF Spaces secrets):
# HF_MODEL_REPO  → e.g.  yourname/rrin-retina-restoration
# HF_MODEL_FILE  → best.pt  (default)
ENV HF_MODEL_FILE=best.pt
ENV MODEL_CACHE_PATH=model_cache/best.pt
ENV PORT=7860

# Expose the port the server listens on
EXPOSE 7860

# Health check — Render uses this to know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:7860/health || exit 1

# Start the inference server
CMD uvicorn api.inference_server:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1
