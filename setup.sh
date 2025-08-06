#!/bin/bash
echo "🚀 Setting up GPT-OSS-20B OpenAI API Server..."

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Update system
apt-get update -qq
apt-get install -y curl wget git htop

# Install EXACT compatible versions (crucial fix)
echo "📦 Installing EXACT compatible versions..."
pip uninstall transformers tokenizers torch torchvision torchaudio -y

# Install compatible torch first
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install EXACT compatible transformers/tokenizers versions
pip install transformers==4.46.2 tokenizers==0.20.1
pip install accelerate fastapi uvicorn aiohttp pydantic
pip install hf_transfer

# Clear any cached files that might be corrupted
rm -rf /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/

# Verify installations
python -c "import torch; import transformers; print(f'✅ Torch: {torch.__version__}, Transformers: {transformers.__version__}')"

# Create SIMPLIFIED server (fallback to working model first)
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
        # Try GPT-OSS-20B first
        logger.info("🔄 Attempting to load GPT-OSS-20B...")
        model_name = "openai/gpt-oss-20b"
        
        # Try loading with different methods
        try:
            # Method 1: Direct loading
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_fast=False,  # Use slow tokenizer if fast fails
                force_download=True  # Force fresh download
            )
            logger.info("✅ GPT-OSS-20B tokenizer loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load GPT-OSS-20B tokenizer: {e}")
            logger.info("🔄 Falling back to Llama-3-8B-Instruct...")
            
            # Fallback to proven working model
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            logger.info("✅ Llama-3-8B tokenizer loaded successfully!")
        
        # Load model
        logger.info(f"🧠 Loading model: {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Format conversation
        conversation = ""
        for msg in request.messages:
            if msg.role == "system":
                conversation += f"System: {msg.content}\n"
            elif msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        
        conversation += "Assistant:"
        
        # Generate response
        inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(request.max_tokens, 1500),
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response[len(conversation):].strip()
        
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting API Server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

chmod +x /root/gpt_oss_server.py

echo "🤖 Starting API server..."
cd /root && python gpt_oss_server.py
