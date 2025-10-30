# Multi-stage Docker build for ACE RL Evaluation Framework
# Stage 1: Builder - Install dependencies
# Stage 2: Runtime - Minimal production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY ace_task/ ./ace_task/
COPY config.yaml ./
COPY README.md ./

# Create directories for outputs
RUN mkdir -p results/visualizations logs

# Create non-root user for security
RUN useradd -m -u 1000 aceuser && \
    chown -R aceuser:aceuser /app
USER aceuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ace_task; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "ace_task.evaluate", "--runs", "10"]
