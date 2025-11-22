# LLM Interpretability Workshop - macOS Installation Guide

## Prerequisites

Before starting, make sure you have:
- macOS 10.15 or later
- At least 8GB of RAM (16GB recommended)
- At least 10GB of free disk space
- Terminal access

---

## Step 1: Install Homebrew (if not already installed)

Homebrew is macOS's package manager. Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, verify:
```bash
brew --version
```

---

## Step 2: Install Python 3.10+

```bash
# Install Python via Homebrew
brew install python@3.11

# Verify installation
python3 --version
```

You should see: `Python 3.11.x`

---

## Step 3: Install Ollama

Ollama is your local LLM server.

### Method 1: Direct Download (Recommended)
1. Visit https://ollama.com/download
2. Download the macOS installer
3. Drag Ollama to Applications folder
4. Open Ollama from Applications

### Method 2: Homebrew
```bash
brew install ollama
```

### Start Ollama
```bash
# Start Ollama service
ollama serve
```

Leave this terminal running and open a new terminal tab for the next steps.

### Pull Models
In a new terminal:
```bash
# Pull the models we'll use (this will take a few minutes)
ollama pull llama3.2:3b
ollama pull phi3:mini
```

### Test Ollama
```bash
# Quick test
ollama run llama3.2:3b "Hello, test"
```

If you see a response, Ollama is working! Press `Ctrl+D` or type `/bye` to exit.

---

## Step 4: Set Up Python Environment

### Create a project directory
```bash
mkdir ~/interpretability-workshop
cd ~/interpretability-workshop
```

### Create virtual environment
```bash
python3 -m venv venv
```

### Activate virtual environment
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 5: Install Python Packages

### Create requirements file
```bash
cat > requirements.txt << 'EOF'
# Core ML frameworks
torch>=2.0.0
transformers>=4.35.0

# LangChain for agents
langchain>=0.1.0
langchain-community>=0.0.10

# Streamlit for dashboard
streamlit>=1.29.0

# Visualization
plotly>=5.18.0
matplotlib>=3.8.0

# Data processing
pandas>=2.1.0
numpy>=1.24.0

# Ollama integration
ollama>=0.1.0
EOF
```

### Install all packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 5-10 minutes depending on your internet speed.

### Verify installations
```bash
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import langchain
print(f"LangChain: {langchain.__version__}")

import streamlit
print(f"Streamlit: {streamlit.__version__}")

import plotly
print(f"Plotly: {plotly.__version__}")

print("\n✅ All packages installed successfully!")
EOF
```

---

## Step 6: Test HuggingFace Model Download

Let's download a small model for testing:

```bash
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Downloading microsoft/phi-2 model...")
print("This is a 2.7B parameter model (~5GB download)")
print("This will take 5-10 minutes depending on your connection...")

# Download model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="cpu",  # Use CPU for Mac
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

print("\n✅ Model downloaded successfully!")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
EOF
```

---

## Step 7: Verify Everything Works

Create a test script:

```bash
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify all components are working
"""

print("🔍 Testing LLM Interpretability Workshop Setup\n")

# Test 1: PyTorch
print("1️⃣ Testing PyTorch...")
import torch
print(f"   ✅ PyTorch {torch.__version__} installed")
print(f"   Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")

# Test 2: Transformers
print("\n2️⃣ Testing Transformers...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer("Hello world")
print(f"   ✅ Tokenization works: {tokens}")

# Test 3: LangChain
print("\n3️⃣ Testing LangChain...")
from langchain.callbacks.base import BaseCallbackHandler
print("   ✅ LangChain imports successful")

# Test 4: Ollama
print("\n4️⃣ Testing Ollama connection...")
try:
    import ollama
    # Try to list models
    models = ollama.list()
    print(f"   ✅ Ollama connected. Models available: {len(models.get('models', []))}")
except Exception as e:
    print(f"   ⚠️  Ollama connection failed: {e}")
    print("   Make sure Ollama is running: ollama serve")

# Test 5: Streamlit
print("\n5️⃣ Testing Streamlit...")
import streamlit
print(f"   ✅ Streamlit {streamlit.__version__} installed")

# Test 6: Plotly
print("\n6️⃣ Testing Plotly...")
import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])])
print("   ✅ Plotly visualization working")

print("\n" + "="*50)
print("✨ All systems operational! Ready for the workshop!")
print("="*50)
EOF

chmod +x test_setup.py
python3 test_setup.py
```

---

## Common Issues & Solutions

### Issue 1: "command not found: ollama"
**Solution:** Restart your terminal or add Ollama to PATH:
```bash
export PATH="/Applications/Ollama.app/Contents/MacOS:$PATH"
```

### Issue 2: PyTorch installation fails
**Solution:** Try installing with specific version:
```bash
pip install torch==2.0.0
```

### Issue 3: "Cannot connect to Ollama"
**Solution:** Make sure Ollama is running:
```bash
# In one terminal
ollama serve

# In another terminal
ollama list
```

### Issue 4: Model download is too slow
**Solution:** Use a smaller model for testing:
```python
# Instead of phi-2, use gpt2 (124M params, much smaller)
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Issue 5: Out of memory errors
**Solution:** Use smaller batch sizes or models:
```python
# Use float16 instead of float32
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16
)
```

---

## Hardware Considerations

### For Mac with Apple Silicon (M1/M2/M3):
- You can use the MPS backend for GPU acceleration
- PyTorch will automatically detect and use it
- Expect 2-3x faster inference than CPU

### For Intel Macs:
- Will use CPU only
- Inference will be slower but still functional
- Consider using smaller models (gpt2, distilgpt2)

---

## Next Steps

Once everything is installed:

1. **Clone the workshop repo** (when available):
   ```bash
   git clone https://github.com/Albinator3000/PennCBCClaudeCode
   cd PennCBCClaudeCode
   ```

2. **Keep Ollama running** in a separate terminal:
   ```bash
   ollama serve
   ```

3. **Activate your environment** before each session:
   ```bash
   cd ~/interpretability-workshop
   source venv/bin/activate
   ```

4. **Launch the demo** (we'll create this next):
   ```bash
   streamlit run interpretability_dashboard.py
   ```

---

## Quick Reference Commands

```bash
# Start Ollama server
ollama serve

# List available models
ollama list

# Pull a new model
ollama pull <model-name>

# Activate Python environment
source venv/bin/activate

# Deactivate environment
deactivate

# Run Streamlit app
streamlit run app.py

# Check Python packages
pip list
```

---

## Estimated Installation Time

- Homebrew & Python: 5-10 minutes
- Ollama: 2-5 minutes
- Ollama models: 5-15 minutes
- Python packages: 5-10 minutes
- HuggingFace model: 5-10 minutes

**Total: 20-50 minutes** depending on internet speed

---

## Storage Requirements

- Python packages: ~2GB
- Ollama models (llama3.2:3b + phi3:mini): ~4GB
- HuggingFace phi-2: ~5GB
- Workspace: ~1GB

**Total: ~12GB**

---

## Ready to Build!

Once you see all green checkmarks ✅ from the test script, you're ready to build the interpretability dashboard!

Next up: Creating the demo application 🚀
