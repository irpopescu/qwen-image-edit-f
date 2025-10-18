import base64, io, torch, runpod
from PIL import Image
from diffusers import QwenImageEditPipeline

print("🚀 Loading Qwen-Image-Edit model...")
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.float16
).to("cuda")
print("✅ Model ready!")

def handler(job):
    try:
        inputs = job["input"]
        prompt = inputs.get("prompt", "photo of an object on white background")
        image_b64 = inputs.get("image_b64")
        if not image_b64:
            return {"error": "Missing image_b64 input"}

        print("📥 Decoding input image...")
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        print("🎨 Generating new image...")
        result = pipe(prompt=prompt, image=image)
        out_img = result.images[0]

        print("📤 Encoding result...")
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        output_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print("✅ Done.")
        return {"image_b64": output_b64}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
