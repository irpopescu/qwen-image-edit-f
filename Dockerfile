FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget
RUN pip install --no-cache-dir torch torchvision torchaudio diffusers transformers accelerate safetensors pillow runpod

RUN git lfs install &&     mkdir -p /workspace && cd /workspace &&     git clone https://huggingface.co/Qwen/Qwen-Image

WORKDIR /workspace
COPY handler.py .
CMD ["python", "-u", "handler.py"]
