FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Instalăm Python și git
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Instalăm pachetele necesare
RUN pip install --no-cache-dir torch torchvision torchaudio diffusers transformers accelerate safetensors pillow runpod

# Setăm directorul de lucru
WORKDIR /workspace

# Copiem fișierul handler.py
COPY handler.py .

# Comandă de pornire
CMD ["python", "-u", "handler.py"]
