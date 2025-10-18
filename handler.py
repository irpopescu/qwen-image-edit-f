import os, io, base64, torch, runpod
from PIL import Image
from diffusers import DiffusionPipeline

# cache local (necesar pt spațiu)
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

print("🚀 Loading Qwen-Image-Edit...")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.float16,
    cache_dir="/workspace/hf_cache",
).to("cuda")
print("✅ Model ready.")

def handler(job):
    try:
        data = job.get("input", {})
        prompt = data.get("prompt", "photo of a product on white background")
        image_b64 = data.get("image_b64")
        if not image_b64:
            return {"error": "Missing image_b64"}

        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        result = pipe(prompt=prompt, image=image)
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return {"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}
    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
