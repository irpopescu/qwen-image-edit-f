import os, io, base64, torch, runpod
from PIL import Image
from diffusers import DiffusionPipeline

# === Configurare cache local ===
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

print("🚀 Pornim inițializarea modelului Qwen-Image-Edit...")

# === Încărcare model ===
try:
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.float16,
        cache_dir="/workspace/hf_cache",
        local_files_only=False,
        variant="fp16"
    ).to("cuda")
    print("✅ Modelul Qwen-Image-Edit este gata de utilizare!")
except Exception as e:
    print(f"❌ Eroare la încărcarea modelului: {e}")
    raise e


# === Funcția handler principal ===
def handler(job):
    try:
        data = job.get("input", {})
        prompt = data.get("prompt", "photo of a product on white background")
        image_b64 = data.get("image_b64")

        if not image_b64:
            return {"error": "Lipsește imaginea (image_b64)"}

        # Decodăm imaginea primită
        print("📥 Decodăm imaginea inițială...")
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        # Generăm editarea
        print(f"🎨 Aplicăm promptul: {prompt}")
        result = pipe(prompt=prompt, image=image)
        edited_image = result.images[0]

        # Codăm imaginea rezultată
        buf = io.BytesIO()
        edited_image.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print("✅ Editare finalizată cu succes!")
        return {"image_b64": out_b64}

    except Exception as e:
        print(f"❌ Eroare în handler: {e}")
        return {"error": str(e)}


# === Pornim serviciul RunPod ===
print("⚙️ Pornim serverul RunPod...")
runpod.serverless.start({"handler": handler})
