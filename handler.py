import os, io, base64, torch, runpod
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

print("🚀 Loading Qwen Image Edit pipeline from ModelScope...")

pipe = pipeline(
    Tasks.image_editing,
    model='Qwen/Qwen-Image-Edit',
    device='cuda'
)

print("✅ Modelul Qwen-Image-Edit e gata de lucru!")

def handler(job):
    try:
        data = job.get("input", {})
        prompt = data.get("prompt", "photo of a product on white background")
        image_b64 = data.get("image_b64")
        if not image_b64:
            return {"error": "Missing image_b64"}

        # salvează imaginea primită
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        img.save("/workspace/input.png")

        print(f"🎨 Prompt: {prompt}")
        output = pipe(dict(prompt=prompt, image="/workspace/input.png"))

        # citește rezultatul returnat de modelscope
        out_path = output["output_img"]
        with open(out_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode("utf-8")

        print("✅ Editare finalizată cu succes!")
        return {"image_b64": result_b64}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
