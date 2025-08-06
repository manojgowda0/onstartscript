#!/bin/bash
# Simplified GPT-OSS-20B Setup - OpenAI API Compatible Only
# Copy-paste this into Vast.ai "On-start script" box

set -e
echo "🚀 Setting up GPT-OSS-20B OpenAI-Compatible API..."

# System setup
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget curl git python3 python3-pip

# Install only required packages
echo "📦 Installing packages..."
pip3 install --upgrade pip --quiet
pip3 install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers accelerate \
    vllm \
    huggingface_hub

# Create simple directory structure
mkdir -p /root/api
cd /root/api

# Download GPT-OSS-20B model
echo "📥 Downloading GPT-OSS-20B (21B parameters)..."
huggingface-cli download openai/gpt-oss-20b --local-dir ./model --local-dir-use-symlinks False

# Create the API server script
cat > /root/api/start_api.sh << 'API_SERVER'
#!/bin/bash
echo "🔥 Starting OpenAI-Compatible API Server"
echo "Endpoint: http://YOUR_VAST_IP:8000/v1/chat/completions"
echo ""

cd /root/api

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

# Create test script that matches your format exactly
cat > /root/api/test_openai_format.py << 'TEST_SCRIPT'
#!/usr/bin/env python3
import requests
import json

def test_openai_api():
    """Test the API with exact OpenAI format from your codebase"""
    
    # Your exact API format
    url = "http://localhost:8000/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        # No Authorization needed for local deployment
    }
    
    # Exact payload structure from your code
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
    print("Testing with your exact payload structure...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Verify exact response structure
            print("✅ SUCCESS! Response structure:")
            print(f"- ID: {response_data.get('id', 'N/A')}")
            print(f"- Object: {response_data.get('object', 'N/A')}")
            print(f"- Model: {response_data.get('model', 'N/A')}")
            print(f"- Created: {response_data.get('created', 'N/A')}")
            
            # Extract content like your code does
            content = response_data['choices'][0]['message']['content']
            print("\n📄 Generated Content:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            # Check usage stats
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

# Test with simpler JSON generation
def test_simple_json():
    """Test simple JSON generation"""
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "Generate a JSON object with name, age, and city fields"}
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
            print("Generated JSON:")
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
    print("=" * 50)
    test_openai_api()
    print("=" * 50)
    test_simple_json()
    print("=" * 50)
TEST_SCRIPT

chmod +x /root/api/test_openai_format.py

# Create monitoring script
cat > /root/api/monitor.sh << 'MONITOR'
#!/bin/bash
echo "📊 GPT-OSS-20B API Monitor"
echo "Expected: ~25GB GPU memory usage"
echo ""
watch -n 5 'echo "=== GPU STATUS ===" && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits && echo "" && echo "=== API STATUS ===" && curl -s http://localhost:8000/health | head -1 2>/dev/null || echo "API not responding" && echo ""'
MONITOR

chmod +x /root/api/monitor.sh

# Create simple README
cat > /root/api/README.txt << 'README'
🚀 GPT-OSS-20B OpenAI-Compatible API

QUICK START:
1. Start API: ./start_api.sh
2. Test API: python3 test_openai_format.py  
3. Monitor: ./monitor.sh

API ENDPOINT:
http://YOUR_VAST_IP:8000/v1/chat/completions

EXACT FORMAT FOR YOUR CLIENT:
- URL: http://YOUR_VAST_IP:8000/v1/chat/completions
- Headers: Content-Type: application/json
- No authentication required
- Same request/response format as OpenAI

UPDATE YOUR CONFIG:
Replace in your code:
API_URL=http://YOUR_VAST_IP:8000/v1/chat/completions
MODEL=gpt-oss-20b

MEMORY USAGE: ~25GB/40GB
RUNTIME: ~22 hours with $10
README

echo ""
echo "✅ SIMPLE API SETUP COMPLETE!"
echo ""
echo "🎯 NEXT STEPS:"
echo "1. Connect to your Vast.ai instance"  
echo "2. cd /root/api"
echo "3. ./start_api.sh"
echo "4. Test with: python3 test_openai_format.py"
echo ""
echo "🌐 YOUR API ENDPOINT:"
echo "   http://YOUR_VAST_IP:8000/v1/chat/completions"
echo ""
echo "🔧 UPDATE YOUR CODE:"
echo "   API_URL=http://YOUR_VAST_IP:8000/v1/chat/completions"
echo "   MODEL=gpt-oss-20b"
echo ""
echo "💰 Expected cost: ~22 hours with $10"
echo "📊 Memory usage: ~25GB/40GB (efficient)"

# Auto-start the API
echo "🚀 Starting API server..."
/root/api/start_api.sh
