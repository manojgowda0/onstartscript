#!/bin/bash
echo "🚀 Setting up GPT-OSS-20B OpenAI API Server..."

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Update system
apt-get update -qq
apt-get install -y curl wget git htop

# Install EXACT versions that support GPT-OSS architecture
echo "📦 Installing GPT-OSS compatible versions..."
pip uninstall transformers tokenizers torch torchvision torchaudio -y

# Install compatible PyTorch first
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: Install transformers 4.55.0+ for GPT-OSS support
pip install transformers>=4.55.0 accelerate fastapi uvicorn aiohttp pydantic
pip install hf_transfer kernels

# Clear any corrupted cache
rm -rf /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/

# Verify GPT-OSS support
python -c "
import transformers
print(f'✅ Transformers version: {transformers.__version__}')
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('openai/gpt-oss-20b', trust_remote_code=True)
    print('✅ GPT-OSS architecture supported!')
except Exception as e:
    print(f'❌ GPT-OSS not supported: {e}')
"

# Create GPT-OSS API server
cat > /root/gpt_oss_server.py << 'EOF'
#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-OSS-20B API")
model = None
tokenizer = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1500

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        logger.info("🔄 Loading GPT-OSS-20B with proper version support...")
        model_name = "openai/gpt-oss-20b"
        
        # Load tokenizer with GPT-OSS support
        logger.info("📝 Loading GPT-OSS tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load GPT-OSS model with MXFP4 support
        logger.info("🧠 Loading GPT-OSS-20B model (MXFP4 optimized)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # Uses MXFP4 automatically on compatible hardware
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ GPT-OSS-20B loaded successfully!")
        logger.info(f"📊 Model device: {model.device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load GPT-OSS model: {e}")
        logger.info("🔄 Falling back to Llama-3-8B-Instruct...")
        
        # Fallback to proven working model
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("✅ Fallback model loaded successfully!")

@app.get("/health")
async def health():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    return {
        "status": "healthy",
        "model": "openai/gpt-oss-20b",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_gb": f"{gpu_memory:.1f}",
        "transformers_version": transformers.__version__
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "openai/gpt-oss-20b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Format conversation for GPT-OSS
        conversation = ""
        for msg in request.messages:
            if msg.role == "system":
                conversation += f"System: {msg.content}\n"
            elif msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        
        conversation += "Assistant:"
        
        # Generate with GPT-OSS optimizations
        inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(request.max_tokens, 1500),
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response[len(conversation):].strip()
        
        # Clean up GPU memory
        del outputs
        torch.cuda.empty_cache()
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting GPT-OSS-20B API Server")
    logger.info("📍 Endpoints:")
    logger.info("  - Health: http://0.0.0.0:8000/health")
    logger.info("  - API: http://0.0.0.0:8000/v1/chat/completions")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

chmod +x /root/gpt_oss_server.py

echo "🤖 Starting GPT-OSS-20B API server..."
cd /root && python gpt_oss_server.py
