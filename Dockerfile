# ===============================
# ✅ DOCKERFILE PENTRU RUNPOD SERVERLESS - Qwen Image Edit
# ===============================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# === Pachete de bază ===
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# === Instalare pachete Python ===
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision torchaudio transformers==4.45.2 accelerate==0.33.0 \
    safetensors pillow runpod tqdm modelscope==1.19.0 \
    addict requests yapf numpy gradio datasets==2.20.0 evaluate opencv-python oss2 aiohttp pyarrow==16.0.0

# === Cache modele ===
ENV MODELSCOPE_CACHE=/workspace/modelscope_cache
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV TORCH_HOME=/workspace/torch_cache
RUN mkdir -p /workspace/modelscope_cache /workspace/hf_cache /workspace/torch_cache

# === Copiază handlerul ===
WORKDIR /workspace
COPY handler.py .

# Asigură instalarea explicită a pachetului runpod
RUN pip install --no-cache-dir runpod

# === (Opțional) Preîncarcă modelul în build ===
#  – Poți comenta acest bloc dacă vrei build mai rapid
RUN python - <<'PY'
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
print("🚀 Downloading Qwen Image Edit model...")
pipeline(Tasks.image_editing, model="Qwen/Qwen-Image-Edit")
print("✅ Model downloaded & cached.")
PY

# === Comanda de pornire pentru RunPod Serverless ===
CMD ["python", "-u", "handler.py"]
