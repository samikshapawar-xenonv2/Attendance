FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for dlib/face_recognition/OpenCV
# Note: We don't need camera/display libs since camera runs on client-side
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create directory for temporary files (Excel exports, etc.)
RUN mkdir -p /app/temp && chmod 777 /app/temp

# Expose port for gunicorn
EXPOSE 5000

# Use gunicorn for production with SINGLE worker
# IMPORTANT: Using 1 worker to ensure consistent in-memory session state
# Multiple workers would each have their own attendance_session, causing data inconsistency
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
