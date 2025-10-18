FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ——— instalare pachete de sistem fără interacțiune ———
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# ——— instalare librării Python necesare Qwen ———
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision torchaudio transformers==4.45.2 accelerate==0.33.0 \
    safetensors pillow runpod tqdm modelscope==1.19.0 \
    addict requests yapf numpy gradio datasets evaluate opencv-python

ENV HF_HOME=/workspace/hf_cache
WORKDIR /workspace
COPY handler.py .

CMD ["python", "-u", "handler.py"]
