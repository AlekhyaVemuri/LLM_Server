import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:latest"

print(f"Using Ollama model: {MODEL_NAME}")


# ==========================================
# Request Schema
# ==========================================

class Query(BaseModel):
    prompt: str
    image_base64: str = None
    max_tokens: int = 500


# ==========================================
# Generate Endpoint
# ==========================================

@app.post("/generate")
async def generate(query: Query):

    try:

        payload = {
            "model": MODEL_NAME,
            "prompt": query.prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": query.max_tokens
            }
        }

        # If vision input present
        if query.image_base64:
            payload["images"] = [query.image_base64]

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return {
            "response": result.get("response", ""),
            "device": "Ollama (CPU/GPU depending on config)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "Managed by Ollama"
    }


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
