# syntax=docker/dockerfile:1
# Multi-stage, multi-arch (amd64 + arm64), non-root runtime.

FROM python:3.12-slim AS base
WORKDIR /app
COPY cache_out.py ./

FROM base AS test
RUN pip install --no-cache-dir pytest
COPY tests/ tests/
RUN python -m pytest tests/ -q

FROM base AS runtime
RUN useradd --create-home --uid 1000 appuser
USER appuser
CMD ["python", "-c", "import cache_out; print('cache_out OK')"]
