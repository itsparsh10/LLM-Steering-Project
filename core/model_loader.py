from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def load_model(local=True, verbose=True):
    """
    Load GPT-2 small (local) or Phi-3 (Colab) model.
    
    Args:
        local: If True, load GPT-2 small. If False, load Phi-3.
        verbose: If True, print loading progress.
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"   Device: {device}")
    
    if local:
        model_id = "gpt2"  # Small model: ~124M params, ~500MB (faster download than gpt2-medium)
        if verbose:
            print(f"   Loading GPT-2 model from Hugging Face...")
            print(f"   Model ID: {model_id}")
            print(f"   (This will download ~500MB on first run)")
        
        try:
            # Load tokenizer
            if verbose:
                print("   Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # GPT-2 doesn't have a pad token, set it to eos_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            if verbose:
                print("   Loading model (this may take a few minutes on first run)...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                output_hidden_states=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Move to device
            if verbose:
                print(f"   Moving model to {device}...")
            model = model.to(device)
            
            # Set to eval mode
            model.eval()
            
            if verbose:
                print("   ✅ Model loaded successfully!")
                print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise
    
    else:
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        if verbose:
            print(f"   Loading Phi-3 model from Hugging Face...")
            print(f"   Model ID: {model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True
            )
            model.eval()
            
            if verbose:
                print("   ✅ Phi-3 model loaded successfully!")
        
        except Exception as e:
            print(f"   ❌ Error loading Phi-3 model: {e}")
            raise

    return model, tokenizer
