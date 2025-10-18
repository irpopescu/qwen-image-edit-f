import os, io, base64, torch, runpod
from PIL import Image
from diffusers import QwenImageEditPipeline

# === CONFIGURAȚIE MEDIU ===
# Mutăm cache-ul Hugging Face în /workspace (spațiu cu 20GB alocat)
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

print("🚀 Pornim inițializarea modelului Qwen-Image-Edit...")

# === ÎNCĂRCARE MODEL ===
try:
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.float16,
        cache_dir="/workspace/hf_cache"
    ).to("cuda")
    print("✅ Modelul Qwen-Image-Edit este gata de utilizare!")
except Exception as e:
    print(f"❌ Eroare la încărcarea modelului: {e}")
    raise e


# === HANDLER PRINCIPAL ===
def handler(job):
    """
    Primește:
      - input.prompt: textul de editare
      - input.image_b64: imaginea inițială în Base64
    Returnează:
      - image_b64: imaginea editată
      - error: mesaj (dacă apare o eroare)
    """
    try:
        inputs = job.get("input", {})
        prompt = inputs.get("prompt", "photo of product on white background")
        image_b64 = inputs.get("image_b64")

        if not image_b64:
            return {"error": "Lipsește câmpul image_b64 în input."}

        print("📥 Decodăm imaginea inițială...")
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print(f"🎨 Aplicăm promptul: {prompt}")
        result = pipe(prompt=prompt, image=image)

        print("📤 Codăm rezultatul final...")
        output_img = result.images[0]
        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print("✅ Editare finalizată cu succes!")
        return {"image_b64": result_b64}

    except Exception as e:
        print(f"❌ Eroare în handler: {e}")
        return {"error": str(e)}


# === PORNIRE SERVICIU SERVERLESS ===
print("⚙️ Pornim serverul RunPod...")
runpod.serverless.start({"handler": handler})
