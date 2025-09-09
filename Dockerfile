# Use a lightweight Python base image (CPU-only)
FROM ubuntu:22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

# Copy source
COPY ./app ./app

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
