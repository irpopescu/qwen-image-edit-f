FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod

WORKDIR /workspace
COPY handler.py .

CMD ["python", "-u", "handler.py"]
