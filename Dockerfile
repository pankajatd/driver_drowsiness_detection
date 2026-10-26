# Multi-Stage Production Container for Driver Drowsiness Detection System
# System Codename: DrowsyGuard DMS & Automotive Telemetry Engine
FROM python:3.9-slim AS runtime

# Set non-interactive environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

# Install essential Debian OS runtime packages for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user for defensive security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy dependency manifest and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and preset media
COPY src/ /app/src/
COPY data/sample_presets/ /app/data/sample_presets/

# Set proper file permissions for appuser
RUN chown -R appuser:appgroup /app

# Expose Streamlit default telemetry port
EXPOSE 8501

# Automated Container Healthcheck Probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Drop root privileges and switch to non-root appuser
USER appuser

# Entrypoint to launch DrowsyGuard Streamlit Dashboard
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
