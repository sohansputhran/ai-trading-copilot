# AI Trading Copilot - Production Dockerfile
# Multi-stage build: smaller, faster, more secure images

# Stage 1: Builder
# Purpose: Install all dependencies in an isolated environment
# Why separate stage: Keeps compilers, build tools, pip cache out of final image
FROM python:3.11-slim AS builder

# Install system dependencies needed for building Python packages
# - gcc: Required to compile psycopg2 and other C extensions
# - postgresql-dev: Headers for psycopg2 (PostgreSQL driver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for build
WORKDIR /build

# Copy only requirements first (Docker layer caching optimization)
# If requirements.txt doesn't change, Docker reuses this layer
COPY requirements.txt .

# Install Python packages to user site-packages
# --user: Installs to ~/.local (easy to copy to final stage)
# --no-cache-dir: Don't keep pip's download cache (saves space)
# psycopg2-binary is included in requirements.txt
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
# Purpose: Minimal production image with only application code and dependencies
# Why slim base: 400MB smaller than full python:3.11, has only what's needed to run
FROM python:3.11-slim

# Install only runtime dependencies (no compilers, no build tools)
# - libpq5: PostgreSQL client library (for psycopg2 at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# Why: Container shouldn't run as root - limits damage if compromised
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder stage
# This is the magic of multi-stage: we get the packages without the build tools
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
# Order matters for caching: code changes more often than dependencies
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
# configs/ intentionally omitted - directory is currently empty and not tracked by Git.
# Restore this line once config files are added: COPY configs/ ./configs/

# Create data directory for SQLite (if DATABASE_URL not set)
# chown: Give ownership to appuser so it can write trades.db
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add user site-packages to PATH so Python finds our installed packages
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Expose Streamlit default port
EXPOSE 8501

# Health check: Streamlit's health endpoint
# Docker will mark container unhealthy if this fails
# --interval: Check every 30 seconds
# --timeout: Fail if no response in 10 seconds
# --start-period: Grace period after container start
# --retries: Mark unhealthy after 3 failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()"

# Run Streamlit
# --server.address: Listen on all interfaces (required in Docker)
# --server.port: Explicit port (matches EXPOSE)
# --server.headless: No browser launch (we're in a container)
# --server.fileWatcherType: None (auto-reload not needed in production)
CMD ["streamlit", "run", "streamlit_app/app.py", \
    "--server.address", "0.0.0.0", \
    "--server.port", "8501", \
    "--server.headless", "true", \
    "--server.fileWatcherType", "none"]
