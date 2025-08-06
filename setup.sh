#!/bin/bash
echo "🚀 Setting up GPT-OSS-20B OpenAI API Server..."

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Update system
apt-get update -qq
apt-get install -y curl wget git htop

# Clear any existing GPU processes and memory
echo "🧹 Clearing GPU memory..."
pkill -f python
nvidia-smi --gpu-reset -i 0 || true
sleep 5

# Install exact versions for GPT-OSS MXFP4 support
echo "📦 Installing GPT-OSS compatible versions with proper triton..."
pip uninstall torch torchvision torchaudio transformers triton -y

# Install PyTorch first
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: Install triton >= 3.4.0 and triton_kernels for MXFP4
pip install triton>=3.4.0
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels

# Install transformers with GPT-OSS support
pip install transformers>=4.55.0 accelerate fastapi uvicorn aiohttp pydantic
pip install hf_transfer

# Clear any corrupted cache
rm -rf /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/

# Verify triton installation
python -c "
import triton
print(f'✅ Triton version: {triton.__version__}')
try:
    import triton_kernels
    print('✅ triton_kernels installed')
except:
    print('❌ triton_kernels not found')
"

# Create GPT-OSS ONLY server (no fallback)
cat > /root/gpt_oss_server.py << 'EOF'
#!/usr/bin/env python3
import torch
import gc
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

def clear_gpu_memory():
    """Clear GPU memory completely"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logger.info(f"🧹 GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    # Clear GPU memory first
    clear_gpu_memory()
    
    try:
        logger.info("🚀 Loading GPT-OSS-20B with MXFP4 quantization...")
        model_name = "openai/gpt-oss-20b"
        
        # Load tokenizer
        logger.info("📝 Loading GPT-OSS tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load model with optimized memory settings for MXFP4
        logger.info("🧠 Loading GPT-OSS-20B with MXFP4 quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # Uses MXFP4 automatically
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "36GB"},  # Leave 4GB headroom on A100
            offload_folder="/tmp/offload"  # Emergency offload location
        )
        
        logger.info("✅ GPT-OSS-20B loaded successfully with MXFP4!")
        logger.info(f"📊 Model device: {model.device}")
        
        # Check final GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"📊 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
        
    except Exception as e:
        logger.error(f"❌ Failed to load GPT-OSS-20B: {e}")
        logger.error("❌ NO FALLBACK - GPT-OSS-20B ONLY!")
        raise e

@app.get("/health")
async def health():
    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    
    return {
        "status": "healthy",
        "model": "openai/gpt-oss-20b",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_used_gb": f"{gpu_memory_used:.1f}",
        "gpu_memory_total_gb": f"{gpu_memory_total:.1f}",
        "quantization": "MXFP4"
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
        raise HTTPException(status_code=503, detail="GPT-OSS-20B model not loaded")
    
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
        
        # Generate with GPT-OSS MXFP4 optimizations
        inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(request.max_tokens, 1500),
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                attention_mask=inputs.get('attention_mask')
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response[len(conversation):].strip()
        
        # Clear intermediate GPU memory
        del outputs, inputs
        torch.cuda.empty_cache()
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "openai/gpt-oss-20b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(conversation.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(conversation.split()) + len(response_text.split())
            }
        }
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"GPT-OSS-20B generation failed: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting GPT-OSS-20B API Server (MXFP4 Quantized)")
    logger.info("📍 Endpoints:")
    logger.info("  - Health: http://0.0.0.0:8000/health")
    logger.info("  - API: http://0.0.0.0:8000/v1/chat/completions")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

chmod +x /root/gpt_oss_server.py

echo "🤖 Starting GPT-OSS-20B API server..."
cd /root && python gpt_oss_server.py
