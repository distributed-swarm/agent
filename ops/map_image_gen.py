cat > ops/map_image_gen.py << 'EOF'
import torch
import base64
import io
from diffusers import StableDiffusionPipeline

# Global model cache so we don't reload it every job
MODEL_CACHE = {}

def get_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "pipe" in MODEL_CACHE:
        return MODEL_CACHE["pipe"], device

    print(f"🎨 Loading Model: {model_id} on {device}...")

    # M10s (Maxwell) are safer with float32. 3060 (Ampere) likes float16.
    # Simple heuristic: If capability < 7.0, use float32
    dtype = torch.float32
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 7:
            dtype = torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    # Enable optimizations
    if device == "cuda":
        pipe.enable_attention_slicing() # Saves VRAM for the M10s

    MODEL_CACHE["pipe"] = pipe
    return pipe, device

def map_image_gen(payload):
    prompt = payload.get("prompt", "a cyberpunk city, neon lights")
    steps = int(payload.get("steps", 20)) # Keep steps low for speed test

    pipe, device = get_model()

    # Generate
    image = pipe(prompt, num_inference_steps=steps).images[0]

    # Convert to Base64 for transport
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "status": "success",
        "device": device,
        "image_base64": img_str
    }
EOF
