#!/bin/bash
echo "🚀 Setting up GPT-OSS-20B OpenAI API Server..."
echo "$(date): Starting setup" >> /root/setup.log

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Update system
apt-get update -qq
apt-get install -y curl wget git htop

# Install compatible PyTorch versions (crucial for avoiding errors)
echo "📦 Installing compatible PyTorch versions..."
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install required packages
echo "📦 Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install transformers==4.44.0 accelerate bitsandbytes optimum
pip install fastapi uvicorn aiohttp pydantic typing-extensions
pip install hf_transfer  # Fixes the hf_transfer error

# Verify installations
echo "🧪 Verifying installations..."
python -c "import torch; import transformers; print(f'✅ Torch: {torch.__version__}, Transformers: {transformers.__version__}')"
python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')"

# Create the GPT-OSS-20B API server
echo "🤖 Creating GPT-OSS-20B API server..."
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
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-OSS-20B API", description="OpenAI-compatible API for GPT-OSS-20B")

# Global variables for model and tokenizer
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
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        logger.info("🔄 Loading GPT-OSS-20B model...")
        model_name = "openai/gpt-oss-20b"
        
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load model with optimizations
        logger.info("🧠 Loading model (this may take 10-15 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "37GB"}  # Leave some headroom on A100
        )
        
        logger.info("✅ GPT-OSS-20B loaded successfully!")
        logger.info(f"📊 Model device: {model.device}")
        logger.info(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "GPT-OSS-20B API Server", "status": "running"}

@app.get("/health")
async def health():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    return {
        "status": "healthy",
        "model": "openai/gpt-oss-20b",
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_gb": f"{gpu_memory:.1f}",
        "model_loaded": model is not None
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
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        # Format conversation for GPT-OSS-20B
        conversation = ""
        for msg in request.messages:
            if msg.role == "system":
                conversation += f"System: {msg.content}\n"
            elif msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        
        conversation += "Assistant:"
        
        # Tokenize input
        inputs = tokenizer(
            conversation, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(request.max_tokens, 2048),
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_response[len(conversation):].strip()
        
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
            }],
            "usage": {
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
                "total_tokens": outputs.shape[1]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting GPT-OSS-20B API Server")
    logger.info("📍 Endpoints:")
    logger.info("  - Health: http://0.0.0.0:8000/health")
    logger.info("  - API: http://0.0.0.0:8000/v1/chat/completions")
    logger.info("  - Models: http://0.0.0.0:8000/v1/models")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
EOF

# Make the script executable
chmod +x /root/gpt_oss_server.py

echo "🤖 Starting GPT-OSS-20B API server..."
echo "$(date): Starting GPT-OSS-20B server" >> /root/setup.log

# Start the server
cd /root && python gpt_oss_server.py

echo "$(date): Setup completed" >> /root/setup.log
