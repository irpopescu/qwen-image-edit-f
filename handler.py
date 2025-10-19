import os, io, base64, torch, runpod
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# === Cache local persistent ===
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["MODELSCOPE_CACHE"] = "/workspace/modelscope_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)
os.makedirs("/workspace/modelscope_cache", exist_ok=True)

pipe = None  # modelul va fi inițializat o singură dată

def init_model():
    global pipe
    if pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Qwen Image Edit pipeline on device: {device}")
        pipe = pipeline(
            Tasks.image_editing,
            model="Qwen/Qwen-Image-Edit",
            device=device
        )
        print("✅ Modelul Qwen-Image-Edit e gata de lucru!")
    return pipe


def handler(job):
    try:
        model = init_model()
        data = job.get("input", {})
        prompt = data.get("prompt", "photo of a product on white background")
        image_b64 = data.get("image_b64")
        if not image_b64:
            return {"error": "Missing image_b64"}

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        img.save("/workspace/input.png")

        print(f"🎨 Prompt: {prompt}")
        output = model(dict(prompt=prompt, image="/workspace/input.png"))
        out_path = output["output_img"]

        with open(out_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode("utf-8")

        print("✅ Editare finalizată cu succes!")
        return {"image_b64": result_b64}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

# === RunPod entrypoint ===
runpod.serverless.start({"handler": handler})
