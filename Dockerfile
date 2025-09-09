# Use a lightweight Python base image
FROM ubuntu:22.04

WORKDIR /app
COPY ./app ./app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch CPU packages first
# Install PyTorch CPU packages (compatible versions)
# Install PyTorch CPU packages (compatible versions)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies without reinstalling torch
RUN pip3 install -r requirements.txt 

# Copy app code
COPY ./app ./app

# Expose FastAPI port
EXPOSE 8000
# before CMD
RUN mkdir -p /app/Pictures

# Run FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
