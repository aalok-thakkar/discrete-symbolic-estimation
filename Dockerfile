# Hermetic Docker image for DiSE artifact evaluation.
#
# Usage:
#   docker build -t dise .
#   docker run --rm dise dise list
#   docker run --rm -v "$PWD/results:/work/results" dise scripts/reproduce.sh

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /work

# Copy only what's needed to install — for layer-cache friendliness.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY EXPERIMENTS.md INSTALL.md CITATION.cff ./

RUN pip install -e ".[dev,plot]"

# Sanity check: imports succeed and dise CLI works.
RUN dise version && dise list

# Default command: a short demo invocation.
CMD ["dise", "compare", "integer_sqrt_correct_U(1,1023)", \
     "--budget", "500", "--n-seeds", "2", "--mc-samples", "2000"]
