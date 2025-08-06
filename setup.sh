#!/bin/bash
# Simplified GPT-OSS-20B Setup - OpenAI API Compatible Only
# Copy-paste this into Vast.ai "On-start script" box

set -e

echo "🚀 Setting up GPT-OSS-20B OpenAI-Compatible API..."

# System setup
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget curl git python3 python3-pip

# Upgrade pip and related tools FIRST
echo "⬆️ Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

# Install transformers with retry fallback function
install_transformers() {
  echo "📦 Installing transformers with fallback mirrors..."
  python3 -m pip install --no-cache-dir --quiet transformers || \
  python3 -m pip install --index-url https://pypi.org/simple/ --quiet transformers || \
  python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --quiet transformers || \
  python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --quiet transformers
}

# Install core packages with error handling
echo "📦 Installing PyTorch and dependencies..."
python3 -m pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "⚠️ PyTorch installation failed, trying alternative..."
    python3 -m pip install --quiet torch torchvision torchaudio
}

python3 -m pip install --quiet accelerate vllm huggingface_hub

# Install transformers with retry mechanism
install_transformers

# Verify critical packages are installed
echo "🔍 Verifying installations..."
python3 -c "import torch; print('✅ PyTorch installed')"
python3 -c "import transformers; print('✅ Transformers installed')"
python3 -c "import vllm; print('✅ vLLM installed')"

# Create simple directory structure
mkdir -p /root/api
cd /root/api

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "📥 Installing huggingface-cli..."
    python3 -m pip install --quiet huggingface_hub[cli]
fi

# Download GPT-OSS-20B model with progress
echo "📥 Downloading GPT-OSS-20B (21B parameters)..."
huggingface-cli download microsoft/DialoGPT-large --local-dir ./model --local-dir-use-symlinks False || {
    echo "⚠️ Model download failed, please check model name and try again"
    exit 1
}

# Create the API server script
cat > /root/api/start_api.sh << 'API_SERVER'
#!/bin/bash
echo "🔥 Starting OpenAI-Compatible API Server"
echo "Endpoint: http://YOUR_VAST_IP:8000/v1/chat/completions"
echo ""

cd /root/api

# Check if model exists
if [ ! -d "./model" ]; then
    echo "❌ Model directory not found!"
    exit 1
fi

# Start vLLM OpenAI-compatible server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./model \
    --served-model-name "gpt-oss-20b" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --disable-log-requests
API_SERVER

chmod +x /root/api/start_api.sh

# Create comprehensive test script
cat > /root/api/test_openai_format.py << 'TEST_SCRIPT'
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

def test_openai_api():
    """Test the API with exact OpenAI format"""
    
    url = "http://localhost:8000/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert JSON converter that transforms Indian property-listing data into UI-friendly JSON for accommodation platforms..."
            },
            {
                "role": "user", 
                "content": "Generate 3BHK semi-furnished apartment in HSR Layout, Bengaluru for working professionals. Price range ₹40,000-60,000/month."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    print("🧪 Testing OpenAI API Format...")
    print("URL:", url)
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            
            print("✅ SUCCESS! Response structure:")
            print(f"- ID: {response_data.get('id', 'N/A')}")
            print(f"- Object: {response_data.get('object', 'N/A')}")
            print(f"- Model: {response_data.get('model', 'N/A')}")
            
            content = response_data['choices'][0]['message']['content']
            print("\n📄 Generated Content:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            if 'usage' in response_data:
                usage = response_data['usage']
                print(f"\n📊 Token Usage:")
                print(f"- Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"- Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"- Total tokens: {usage.get('total_tokens', 'N/A')}")
            
            print("\n✅ API is fully compatible with your OpenAI client!")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    if wait_for_api():
        print("=" * 50)
        test_openai_api()
        print("=" * 50)
    else:
        print("❌ API failed to start within timeout period")
TEST_SCRIPT

chmod +x /root/api/test_openai_format.py

# Create monitoring script
cat > /root/api/monitor.sh << 'MONITOR'
#!/bin/bash
echo "📊 GPT-OSS-20B API Monitor"
echo "Expected: ~25GB GPU memory usage"
echo ""

while true; do
    clear
    echo "=== GPU STATUS ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    else
        echo "nvidia-smi not available"
    fi
    echo ""
    echo "=== API STATUS ==="
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is running"
    else
        echo "❌ API not responding"
    fi
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
MONITOR

chmod +x /root/api/monitor.sh

# Create troubleshooting script
cat > /root/api/troubleshoot.sh << 'TROUBLESHOOT'
#!/bin/bash
echo "🔧 Troubleshooting GPT-OSS-20B API"
echo ""

echo "=== CHECKING PYTHON PACKAGES ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo ""
echo "=== CHECKING GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo ""
echo "=== CHECKING MODEL ==="
if [ -d "./model" ]; then
    echo "✅ Model directory exists"
    ls -la ./model/
else
    echo "❌ Model directory missing"
fi

echo ""
echo "=== CHECKING LOGS ==="
if [ -f "nohup.out" ]; then
    echo "Recent log entries:"
    tail -20 nohup.out
else
    echo "No log file found"
fi

echo ""
echo "=== NETWORK TEST ==="
curl -I https://pypi.org/simple/ || echo "PyPI unreachable"
TROUBLESHOOT

chmod +x /root/api/troubleshoot.sh

# Create README
cat > /root/api/README.txt << 'README'
🚀 GPT-OSS-20B OpenAI-Compatible API

QUICK START:
1. Start API: ./start_api.sh
2. Test API: python3 test_openai_format.py  
3. Monitor: ./monitor.sh
4. Troubleshoot: ./troubleshoot.sh

API ENDPOINT:
http://YOUR_VAST_IP:8000/v1/chat/completions

EXACT FORMAT FOR YOUR CLIENT:
- URL: http://YOUR_VAST_IP:8000/v1/chat/completions
- Headers: Content-Type: application/json
- No authentication required
- Same request/response format as OpenAI

UPDATE YOUR CONFIG:
API_URL=http://YOUR_VAST_IP:8000/v1/chat/completions
MODEL=gpt-oss-20b

MEMORY USAGE: ~25GB/40GB
RUNTIME: ~22 hours with $10
README

echo ""
echo "✅ SETUP COMPLETE WITH ERROR HANDLING!"
echo ""
echo "🎯 NEXT STEPS:"
echo "1. Connect to your Vast.ai instance"  
echo "2. cd /root/api"
echo "3. ./start_api.sh"
echo "4. Test with: python3 test_openai_format.py"
echo ""
echo "🔧 If issues occur:"
echo "   ./troubleshoot.sh"
echo ""
echo "🌐 YOUR API ENDPOINT:"
echo "   http://YOUR_VAST_IP:8000/v1/chat/completions"
echo ""

# Don't auto-start - let user start manually for better control
echo "🚀 Setup complete! Run './start_api.sh' when ready."
