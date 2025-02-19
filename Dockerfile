# Base image (smaller alternative to python:3.11)
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy only essential files for dependencies
COPY requirements.txt .

# Install Python dependencies (with caching disabled to save space)
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------
# Final, smaller runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Ensure en_core_web_sm is downloaded
RUN python -m spacy download en_core_web_sm

# Expose Flask's port
EXPOSE 10000

# Run the Flask app
CMD ["python", "app.py"]
