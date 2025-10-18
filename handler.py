import os, io, base64, torch, runpod
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# cache local mare
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

print("🚀 Loading Qwen Image Edit pipeline from ModelScope...")

# Încarcă modelul Qwen oficial
pipe = pipeline(
    Tasks.image_editing,
    model='Qwen/Qwen-Image-Edit',
    device='cuda'
)

print("✅ Modelul Qwen-Image-Edit e gata de lucru!")

def handler(job):
    try:
        data = job.get("input", {})
        prompt = data.get("prompt", "photo of product on white background")
        image_b64 = data.get("image_b64")

        if not image_b64:
            return {"error": "Missing image_b64"}

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        img.save("/workspace/input.png")

        print(f"🎨 Prompt: {prompt}")
        output = pipe(dict(prompt=prompt, image="/workspace/input.png"))

        # ModelScope returnează path către fișier
        result_img = Image.open(output["output_img"]).convert("RGB")
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return {"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}
    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
