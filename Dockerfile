FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision torchaudio diffusers transformers accelerate safetensors pillow runpod

WORKDIR /workspace
COPY handler.py .

CMD ["python", "-u", "handler.py"]
