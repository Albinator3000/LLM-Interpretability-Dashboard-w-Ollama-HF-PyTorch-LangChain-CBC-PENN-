from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

def download_model():
    print("=" * 60)
    print("Downloading microsoft/phi-2 model...")
    print("Size: ~5GB | Time: 5-10 minutes")
    print("=" * 60)
    
    try:
        # Download tokenizer
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        print("✅ Tokenizer downloaded")
        
        # Download model
        print("\n[2/2] Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
        print("✅ Model downloaded")
        
        # Test
        print("\n[Test] Running quick test...")
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test output: {result}")
        
        print("\n" + "=" * 60)
        print("✨ Ready! Run: streamlit run app.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()