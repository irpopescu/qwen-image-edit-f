FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Instalăm Python + git
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Setăm alias ca "python"
RUN ln -s /usr/bin/python3 /usr/bin/python

# Instalăm pachetele necesare
RUN pip install --no-cache-dir torch torchvision torchaudio diffusers transformers accelerate safetensors pillow runpod

# Setăm directorul de lucru
WORKDIR /workspace

# Copiem handlerul
COPY handler.py .

# Pornim handlerul
CMD ["python", "-u", "handler.py"]
