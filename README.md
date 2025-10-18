# Qwen Image Edit (RunPod Serverless)

Serverless endpoint for AI image-to-image editing using the Qwen Image model.

## Run locally
```
python handler.py
```

## Deploy on RunPod
1. Create a Serverless Template from this repo
2. Set entrypoint: `handler.handler`
3. Use GPU: L4 or RTX 4090
4. Test endpoint:
```
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/run -H "Authorization: Bearer <RUNPOD_API_KEY>" -H "Content-Type: application/json" -d '{"input": {"prompt": "bracelet on marble background", "image_b64": "<base64>"}}'
```
