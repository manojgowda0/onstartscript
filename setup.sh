#!/bin/bash
# GPT-OSS-20B Simple vLLM Serve Setup - OpenAI API Compatible
# Copy-paste this into Vast.ai "On-start script" box

set -e

echo "🚀 Setting up GPT-OSS-20B with Simple vLLM Serve..."

# System setup
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget curl git python3 python3-pip

# Upgrade pip and install required packages
echo "📦 Installing packages..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

# Install vLLM and dependencies with fallback
echo "📦 Installing vLLM and dependencies..."
python3 -m pip install --quiet vllm || \
python3 -m pip install --index-url https://pypi.org/simple/ --quiet vllm

# Install supporting packages
python3 -m pip install --quiet transformers accelerate hf-transfer

# Create API directory
mkdir -p /root/api
cd /root/api

# Download GPT-OSS-20B model
echo "📥 Downloading GPT-OSS-20B model..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', local_dir='./model', local_dir_use_symlinks=False)
"

# Create the simple startup script
cat > /root/api/start_vllm.sh << 'VLLM_SCRIPT'
#!/bin/bash
echo "🚀 Starting GPT-OSS-20B with Simple vLLM Serve"
echo "Endpoint: http://143.55.45.86:8000/v1/chat/completions"

cd /root/api

# Set optimization environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Use simple vLLM serve command (avoids compilation issues)
vllm serve ./model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "gpt-oss-20b" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 1024 \
  --trust-remote-code \
  --disable-log-requests
VLLM_SCRIPT

chmod +x /root/api/start_vllm.sh

# Create test script
cat > /root/api/test_api.py << 'TEST_SCRIPT'
#!/usr/bin/env python3
import requests
import json
import time

def wait_for_api(max_retries=30):
    """Wait for API to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            pass
        print(f"⏳ Waiting for API... ({i+1}/{max_retries})")
        time.sleep(10)
    return False

def test_api():
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert JSON converter for Indian property listings."
            },
            {
                "role": "user", 
                "content": "Generate a 3BHK apartment listing in HSR Layout, Bangalore for ₹50,000/month"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    print("🧪 Testing GPT-OSS-20B API...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            print("✅ SUCCESS! API Response:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            # Token usage
            if 'usage' in data:
                usage = data['usage']
                print(f"📊 Tokens: {usage.get('total_tokens', 'N/A')}")
                
            print("\n✅ Your GPT-OSS-20B API is fully functional!")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    if wait_for_api():
        test_api()
    else:
        print("❌ API failed to start within timeout")
TEST_SCRIPT

chmod +x /root/api/test_api.py

# Create README
cat > /root/api/README.txt << 'README'
🚀 GPT-OSS-20B Simple vLLM Setup

YOUR API IS RUNNING AT:
http://YOUR_VAST_IP:8000/v1/chat/completions

INTEGRATION:
Update your code:
API_URL = "http://YOUR_VAST_IP:8000/v1/chat/completions"
MODEL = "gpt-oss-20b"

TESTING:
python3 test_api.py

FEATURES:
- 100% OpenAI API Compatible
- No compilation issues
- Production-ready vLLM serve
- A100 GPU optimized

COMMANDS:
- Restart API: ./start_vllm.sh
- Test API: python3 test_api.py
README

echo ""
echo "✅ SETUP COMPLETE! AUTO-STARTING API SERVER..."
echo ""
echo "🌐 YOUR API ENDPOINT:"
echo "   http://143.55.45.86:8000/v1/chat/completions"
echo ""
echo "🔧 UPDATE YOUR CODE:"
echo "   API_URL = \"http://143.55.45.86:8000/v1/chat/completions\""
echo "   MODEL = \"gpt-oss-20b\""
echo ""
echo "💰 Expected: ~22 hours runtime with $10"
echo "📊 Memory: ~25GB/40GB GPU usage"
echo ""
echo "🎯 Starting vLLM server now..."

# AUTO-START THE API SERVER
cd /root/api
./start_vllm.sh &

# Wait a moment then run test in background
sleep 30 && python3 test_api.py &

# Keep the main process running
wait
