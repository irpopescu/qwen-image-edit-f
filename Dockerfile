FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision torchaudio transformers==4.45.2 accelerate==0.33.0 \
    safetensors pillow runpod tqdm modelscope==1.19.0 \
    addict requests yapf numpy gradio datasets==2.20.0 evaluate opencv-python oss2 aiohttp pyarrow==16.0.0

# === Cache pentru modele mari ===
ENV MODELSCOPE_CACHE=/workspace/modelscope_cache
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV TORCH_HOME=/workspace/torch_cache
RUN mkdir -p /workspace/modelscope_cache /workspace/hf_cache /workspace/torch_cache
WORKDIR /workspace
COPY handler.py .

CMD ["python", "-u", "handler.py"]
