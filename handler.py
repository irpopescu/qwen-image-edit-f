import io, base64, torch
from diffusers import DiffusionPipeline
from PIL import Image
import runpod

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float16).to("cuda")

def generate_image(prompt, image_b64):
    init_image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    result = pipe(prompt=prompt, image=init_image, strength=0.3, guidance_scale=5).images[0]
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(event):
    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt", "product photo on white background")
        image_b64 = inp.get("image_b64")
        if not image_b64:
            return {"error": "Missing image_b64"}
        output_b64 = generate_image(prompt, image_b64)
        return {"image_b64": output_b64, "status": "ok"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Local test mode: open sample.jpg if exists.")

runpod.serverless.start({"handler": handler})
