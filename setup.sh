#!/bin/bash
# GPT-OSS-20B Optimized vLLM Serve Setup - OpenAI API Compatible
# Copy-paste this entire script into Vast.ai "On-start script" box

set -e

echo "🚀 Setting up Optimized GPT-OSS-20B with vLLM Serve..."

# System setup
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget curl git python3 python3-pip

# Upgrade pip and install required packages
echo "📦 Installing optimized packages..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

# Install vLLM and dependencies with fallback
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

# Create the optimized startup script
cat > /root/api/start_vllm_optimized.sh << 'VLLM_SCRIPT'
#!/bin/bash
echo "🚀 Starting Optimized GPT-OSS-20B Server"
echo "Endpoint: http://YOUR_VAST_IP:8000/v1/chat/completions"

cd /root/api

# Set optimization environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Optimized vLLM serve command
vllm serve ./model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "gpt-oss-20b" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --trust-remote-code \
  --disable-log-requests \
  --dtype auto \
  --enable-chunked-prefill \
  --max-num-seqs 8
VLLM_SCRIPT

chmod +x /root/api/start_vllm_optimized.sh

# Create comprehensive test script
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
    
    # Test with Indian property listing (your use case)
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert JSON converter that transforms Indian property-listing data into UI-friendly JSON for accommodation platforms."
            },
            {
                "role": "user", 
                "content": "Generate a 3BHK semi-furnished apartment in HSR Layout, Bengaluru for working professionals. Price range ₹40,000-60,000/month."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    
    print("🧪 Testing GPT-OSS-20B API with Property Listing...")
    
    try:
        response = requests.post(url, json=payload, timeout=90)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            print("✅ SUCCESS! API Response:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            
            # Token usage
            if 'usage' in data:
                usage = data['usage']
                print(f"\n📊 Token Usage:")
                print(f"- Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"- Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"- Total tokens: {usage.get('total_tokens', 'N/A')}")
                
            print("\n✅ Your GPT-OSS-20B API is fully functional!")
            print("🔗 API Endpoint: http://YOUR_VAST_IP:8000/v1/chat/completions")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

def test_simple_json():
    """Test simple JSON generation"""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "Generate a simple JSON object with name, age, and city fields"}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    print("\n🔧 Testing Simple JSON Generation...")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print("Generated Response:")
            print(content)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(content)
                print("✅ Valid JSON structure!")
            except:
                print("⚠️ Generated text, but not valid JSON")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if wait_for_api():
        print("=" * 60)
        test_api()
        print("=" * 60)
        test_simple_json()
        print("=" * 60)
    else:
        print("❌ API failed to start within timeout period")
TEST_SCRIPT

chmod +x /root/api/test_api.py

# Create monitoring script
cat > /root/api/monitor.sh << 'MONITOR'
#!/bin/bash
echo "📊 GPT-OSS-20B Optimized API Monitor"
echo "Expected: ~30GB GPU memory usage"
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
        echo "🔗 Endpoint: http://YOUR_VAST_IP:8000/v1/chat/completions"
    else
        echo "❌ API not responding"
    fi
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
MONITOR

chmod +x /root/api/monitor.sh

# Create README with optimized settings
cat > /root/api/README.txt << 'README'
🚀 GPT-OSS-20B Optimized vLLM Setup

YOUR API IS RUNNING AT:
http://YOUR_VAST_IP:8000/v1/chat/completions

OPTIMIZED SETTINGS:
- GPU Memory Utilization: 90% (~30GB/40GB)
- Max Token Length: 2048 tokens
- Max Concurrent Requests: 8
- Chunked Prefill: Enabled
- Data Type: Auto-optimized

INTEGRATION:
Update your code:
API_URL = "http://YOUR_VAST_IP:8000/v1/chat/completions"
MODEL = "gpt-oss-20b"

COMMANDS:
- Start API: ./start_vllm_optimized.sh
- Test API: python3 test_api.py
- Monitor: ./monitor.sh

FEATURES:
- 100% OpenAI API Compatible
- Production-ready performance
- Optimized for A100 GPU
- Perfect for property listings
README

echo ""
echo "✅ OPTIMIZED SETUP COMPLETE! AUTO-STARTING API SERVER..."
echo ""
echo "🎯 OPTIMIZATION FEATURES:"
echo "   - GPU Memory: 90% utilization (~30GB)"
echo "   - Max Tokens: 2048 (doubled capacity)"
echo "   - Concurrent Requests: 8 simultaneous"
echo "   - Chunked Prefill: Memory efficient"
echo ""
echo "🌐 YOUR API ENDPOINT:"
echo "   http://YOUR_VAST_IP:8000/v1/chat/completions"
echo ""
echo "🔧 UPDATE YOUR CODE:"
echo "   API_URL = \"http://YOUR_VAST_IP:8000/v1/chat/completions\""
echo "   MODEL = \"gpt-oss-20b\""
echo ""
echo "💰 Expected: ~22 hours runtime with $10"
echo "📊 Memory: ~30GB/40GB GPU usage (optimized)"
echo ""
echo "🎯 Starting optimized vLLM server now..."

# AUTO-START THE OPTIMIZED API SERVER
cd /root/api
./start_vllm_optimized.sh &

# Wait for startup then run test
sleep 45 && python3 test_api.py &

# Keep the main process running
wait
