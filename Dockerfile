# Image for the prediction web service only (no notebook, no training code).
# Same pinned base image and uv version as the ML Zoomcamp 2026 deployment module.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

# uv installs the exact package versions recorded in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.10.11@sha256:3472e43b4e738cf911c99d41bb34331280efad54c73b1def654a6227bb59b2b4 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Install runtime dependencies first: this layer is cached until the lock file changes.
# --no-dev   skips the training / notebook / test packages
# --no-cache keeps uv's download cache out of the image
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-cache

# Application code and the trained model
COPY features.py predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
