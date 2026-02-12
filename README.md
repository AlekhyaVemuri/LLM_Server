# Intel AI PC LLM Server (PyTorch XPU Accelerated)

This project automates the deployment of a Vision-Language Model (Qwen3-VL) as a REST API on Intel Meteor Lake hardware. It utilizes the native PyTorch XPU backend for GPU acceleration, bypassing the need for legacy extensions while ensuring compatibility with internal corporate proxies.

## Features

*   **Native XPU Acceleration:** Runs inference on Intel Arc/Iris Xe Graphics using the latest PyTorch 2.5+ XPU backend.
*   **Multimodal Capability:** Supports text and image-based queries using the Qwen3-VL architecture.
*   **uv Optimized:** Managed by the `uv` package manager for lightning-fast dependency resolution and virtual environment handling.
*   **Corporate Proxy Aware:** Pre-configured to handle internal network environments.

## Prerequisites

*   **Hardware:** Intel Core Ultra (Meteor Lake) with integrated Arc Graphics.
*   **OS:** Windows 10/11.
*   **Drivers:** Latest [Intel Graphics Drivers](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).
*   **Package Manager:** [uv](https://github.com/astral-sh/uv) installed (`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`).

## Server Setup

1. **Clone/Save the Automation Script:** Save the provided `llm_server_setup.ps1` to a directory on the AI PC.
2. **Configure IP:** Update the `$AI_PC_IP` variable in the script to match the host machine's internal IP.
3. **Execute:** Run the script in an elevated PowerShell terminal:
   ```powershell
   .\llm_server_setup.ps1
   ```

The script will automatically:
* Configure proxy environment variables.
* Initialize a `uv` project.
* Install XPU-specific PyTorch binaries.
* Download the model weights.
* Launch the **FastAPI** server on port `8000`.

## API Usage

The server exposes a POST endpoint at `/generate`.

### Testing via cURL (Local or Remote)
Ensure you bypass the proxy for local network traffic:
```bash
curl --noproxy "*" -X POST http://<AI_PC_IP>:8000/generate \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"Analyze the benefits of AI PC architecture.\", \"max_tokens\": 100}"
```

### Python Integration
Use the following logic to connect your local applications to the AI PC:
```python
import requests

def call_llm_server(prompt):
    # Disable proxy environment tracking for local network requests
    session = requests.Session()
    session.trust_env = False 
    
    payload = {"prompt": prompt, "max_tokens": 150}
    response = session.post("http://10.211.178.231:8000/generate", json=payload)
    return response.json().get("response")

print(call_llm_server("What is Intel Meteor Lake?"))
```

## Performance Verification

To verify that the model is correctly utilizing the GPU:
1. Open **Task Manager** on the AI PC.
2. Navigate to the **Performance** tab.
3. Select **GPU 0** (Intel Arc Graphics).

## Troubleshooting

*   **403 Forbidden:** Ensure the `NO_PROXY` environment variable includes the AI PC's IP address and `localhost`. 
*   **XPU Not Found:** Verify that you are using the specific XPU wheel index: `https://download.pytorch.org/whl/xpu`.
*   **Out of Memory:** If running large models, ensure sufficient System RAM is available.
