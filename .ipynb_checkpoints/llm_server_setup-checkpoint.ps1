# --- CONFIGURATION ---
# $AI_PC_IP = "10.211.178.231"
$AI_PC_IP = "10.211.177.52"
$PROXY_URL = "http://proxy-dmz.intel.com:912"
$MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# --- 1. SET PROXY ENVIRONMENT ---
Write-Host "Configuring Proxy Environment..." -ForegroundColor Cyan
$env:HTTP_PROXY = $PROXY_URL
$env:HTTPS_PROXY = $PROXY_URL
$env:NO_PROXY = "localhost,127.0.0.1,$AI_PC_IP"

# --- 2. INITIALIZE PROJECT ---
if (!(Test-Path "pyproject.toml")) {
    Write-Host "Initializing UV Project..." -ForegroundColor Cyan
    uv init
}

# --- 3. INSTALL XPU DEPENDENCIES ---
Write-Host "Installing XPU-enabled Torch and Transformers..." -ForegroundColor Cyan
# Specifically use the Intel XPU index for native backend support
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/xpu
uv add fastapi uvicorn "transformers>=4.48.0" qwen-vl-utils accelerate

# --- 4. CREATE SERVER CODE (main.py) ---
Write-Host "Generating main.py..." -ForegroundColor Cyan
$pythonCode = @"
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import uvicorn

app = FastAPI()
MODEL_ID = "$MODEL_ID"
DEVICE = "xpu"

print(f"Loading {MODEL_ID} to Intel MTL XPU...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)

class Query(BaseModel):
    prompt: str
    image_url: str = None
    max_tokens: int = 100

@app.post("/generate")
async def generate(query: Query):
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": query.prompt}]}]
        if query.image_url:
            messages[0]["content"].insert(0, {"type": "image", "image": query.image_url})
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(DEVICE)
        
        generated_ids = model.generate(**inputs, max_new_tokens=query.max_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return {"response": output_text[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "device": torch.xpu.get_device_name(0)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"@

$pythonCode | Out-File -FilePath "main.py" -Encoding utf8

# --- 5. RUN SERVER ---
Write-Host "Launching LLM Server on Port 8000..." -ForegroundColor Green
uv run python main.py