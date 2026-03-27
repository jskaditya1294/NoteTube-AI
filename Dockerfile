# =============================================================================
# Dockerfile for YouTube Notes Workflow (CLI application)
# =============================================================================
#
# BUILD:  docker build -t youtube-notes .
#
# RUN (Windows CMD):
#   docker run --rm --env-file .env -v "%cd%/outputs:/app/outputs" youtube-notes "https://www.youtube.com/watch?v=VIDEO_ID"
#
# RUN (Linux / macOS / PowerShell):
#   docker run --rm --env-file .env -v "$(pwd)/outputs:/app/outputs" youtube-notes "https://www.youtube.com/watch?v=VIDEO_ID"
#
# HELP:
#   docker run --rm youtube-notes --help
# =============================================================================


# --- Stage 1: Base image ---
# python:3.12-slim-bookworm is a lightweight Debian image with Python 3.12.
# "slim" means it excludes dev tools/docs to keep the image small (~150 MB vs ~1 GB for full).
# "bookworm" is the Debian 12 codename — stable and well-supported.
FROM python:3.12-slim-bookworm


# --- Working directory inside the container ---
# All subsequent COPY, RUN, and CMD instructions will run relative to /app.
# If /app doesn't exist, Docker creates it automatically.
WORKDIR /app


# --- Install system-level dependencies ---
# These are OS packages needed at runtime (not Python packages).
#   ffmpeg       → yt-dlp uses it to merge/convert downloaded video streams.
#   libgl1       → OpenCV (cv2) needs OpenGL support for image processing.
#   libglib2.0-0 → Low-level C library that OpenCV depends on.
#   libsm6       → X11 Session Management lib, required by OpenCV even in headless mode.
#   libxext6     → X11 extension lib, another OpenCV dependency.
#   libxrender1  → X11 rendering lib, another OpenCV dependency.
# --no-install-recommends keeps the install minimal (no "nice to have" extras).
# The final "rm -rf" cleans the apt cache so it doesn't bloat the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# --- Install Python dependencies ---
# We copy requirements.txt FIRST, separately from the rest of the code.
# Why? Docker caches each layer. If your code changes but requirements.txt
# doesn't, Docker skips the slow "pip install" step entirely on rebuild.
# This is called "layer caching" and makes rebuilds much faster.
COPY requirements.txt .

# --no-cache-dir tells pip not to store download caches (saves ~50-100 MB).
# --upgrade pip ensures we have the latest pip for best compatibility.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Copy application source code ---
# We copy each package folder individually instead of "COPY . /app" because:
#   1. It's explicit — you see exactly what goes into the image.
#   2. Combined with .dockerignore, it prevents secrets (.env), virtual envs
#      (.venv), and large generated files (outputs/) from leaking into the image.
COPY core/     core/
COPY nodes/    nodes/
COPY prompts/  prompts/
COPY services/ services/
COPY utils/    utils/
COPY run.py    .


# --- Environment variables ---
# PYTHONUNBUFFERED=1 → Forces Python to print output immediately (no buffering).
#   Without this, print() inside the container may not appear in "docker logs"
#   until the process exits — very confusing when debugging.
# PYTHONPATH=/app    → Tells Python to look for packages starting from /app.
#   This ensures "from core.state import ..." works correctly because Python
#   can find the core/ folder relative to /app.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app


# --- Container entry point ---
# ENTRYPOINT is the command that ALWAYS runs when the container starts.
# CMD provides default arguments that get appended to ENTRYPOINT.
#
# Together they work like this:
#   docker run youtube-notes                        → runs: python run.py --help
#   docker run youtube-notes "https://youtu.be/..." → runs: python run.py "https://youtu.be/..."
#   docker run youtube-notes VIDEO_ID --skip-interview → runs: python run.py VIDEO_ID --skip-interview
#
# ENTRYPOINT = the fixed part (always "python run.py")
# CMD        = the default args (overridden when you pass your own)
ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
