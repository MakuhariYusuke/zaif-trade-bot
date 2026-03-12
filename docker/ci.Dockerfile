FROM python:${PYTHON_VERSION:-3.11}-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY config/requirements/requirements.txt config/requirements/requirements-dev.txt ./

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt || true
RUN pip install -r config/requirements/requirements-dev.txt || true
RUN pip install pytest pytest-cov mypy flake8 ruff

COPY tools/ci-entrypoint.sh /usr/local/bin/ci-entrypoint.sh
RUN chmod +x /usr/local/bin/ci-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/ci-entrypoint.sh"]
