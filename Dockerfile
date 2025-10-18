FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# instalăm doar ce e strict necesar
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision torchaudio \
    diffusers==0.30.2 transformers==4.45.2 accelerate==0.33.0 \
    safetensors pillow runpod tqdm modelscope==1.19.0

WORKDIR /workspace
COPY handler.py .

CMD ["python", "-u", "handler.py"]
