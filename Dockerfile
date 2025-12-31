# SynDX Docker Image
# Preliminary work without clinical validation

FROM python:3.9.16-slim

LABEL maintainer="Chatchai Tritham <chatchai.tritham@nu.ac.th>"
LABEL description="SynDX: Explainable AI-Driven Synthetic Data Generation (Preliminary)"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY syndx/ ./syndx/
COPY notebooks/ ./notebooks/
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install SynDX package
RUN pip install -e .

# Create output directories
RUN mkdir -p /app/outputs/synthetic_patients \
             /app/outputs/figures \
             /app/outputs/metrics \
             /app/data/archetypes \
             /app/models/pretrained

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SYNDX_DATA_DIR=/app/data
ENV SYNDX_OUTPUT_DIR=/app/outputs

# Warning message
RUN echo "#!/bin/bash\n\
echo ''\n\
echo '='*70\n\
echo 'SynDX: Synthetic Data Generation Framework'\n\
echo 'Version 0.1.0'\n\
echo '='*70\n\
echo ''\n\
echo '⚠️  WARNING: Preliminary work without clinical validation'\n\
echo '   Do NOT use for clinical decision-making'\n\
echo ''\n\
echo 'Usage:'\n\
echo '  docker run -it syndx python -m syndx.pipeline'\n\
echo '  docker run -it syndx jupyter lab --ip=0.0.0.0 --allow-root'\n\
echo ''\n\
" > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["/app/entrypoint.sh"]
