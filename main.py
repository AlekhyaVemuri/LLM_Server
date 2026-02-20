import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import uvicorn

app = FastAPI()
# MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
# MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking"
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
    max_tokens: int = 1000 # Increased default for slide analysis

@app.post("/generate")
async def generate(query: Query):
    try:
        # Prepare content list
        content = []
        if query.image_url:
            content.append({"type": "image", "image": query.image_url})
        content.append({"type": "text", "text": query.prompt})
        
        messages = [{"role": "user", "content": content}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(DEVICE)
        
        # Use the max_tokens passed from the client
        generated_ids = model.generate(**inputs, max_new_tokens=query.max_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return {"response": output_text[0]}
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)