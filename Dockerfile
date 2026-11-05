FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
# Source pinned by Git submodule's gitlink. Run setup_git before docker build.
COPY external/ultralytics /workspace/external/ultralytics
RUN test -f /workspace/external/ultralytics/pyproject.toml && pip install --no-cache-dir -e /workspace/external/ultralytics
COPY scripts /workspace/scripts
ENTRYPOINT ["python", "scripts/train.py"]
