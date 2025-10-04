# Dockerfile for zaif-trade-bot live trading
#
# Production-ready container for live BTC/JPY trading with monitoring,
# health checks, and security features.

FROM python:3.11-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    # Default production settings
    ZTB_LOG_LEVEL=INFO \
    ZTB_ENABLE_METRICS=true \
    ZTB_METRICS_PORT=8000 \
    ZTB_ENABLE_HEALTH_CHECK=true \
    ZTB_HEALTH_PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    # Required for some Python packages
    libgomp1 \
    libatlas-base-dev \
    # Monitoring and networking tools
    netcat-openbsd \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt constraints.txt ./

# Install Python dependencies with security constraints
RUN pip install --no-cache-dir --constraint constraints.txt -r requirements.txt

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 trader && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader

# Create directories for logs and data
RUN mkdir -p /app/logs /app/models /app/data

# Health check for live trading
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports for monitoring
EXPOSE 8080 8000

# Default command for live trading
CMD ["python", "live_trade.py", "--model-path", "/app/models/model.zip", "--duration-hours", "24"]