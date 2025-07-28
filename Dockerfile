# Use Python 3.9 slim image for AMD64 architecture
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    tk-dev \
    tcl-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
# Use specific index for faster downloads
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create input and output directories with proper permissions
RUN mkdir -p /app/input /app/output && \
    chmod 755 /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Create cache directories
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface

# Set proper permissions for the app directory
RUN chmod -R 755 /app

# Health check to ensure container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ls /app/input /app/output || exit 1

# Default command - runs the main processing script
CMD ["python", "main_simple.py"]

# Labels for better container management
LABEL version="1.0"
LABEL description="PDF Heading Extractor for DotChallenge Round 1A"
LABEL maintainer="your-email@example.com"
LABEL challenge="DotChallenge-Round1A"