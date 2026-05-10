from transformers import pipeline
import torch

# Global variable to hold the pipeline so we don't load it multiple times
_generator = None

def init_llm():
    global _generator
    if _generator is None:
        print("Loading local LLM model (TinyLlama)... This may take a moment on the first run.")
        # We use device_map="auto" if torch has CUDA, else CPU
        device = 0 if torch.cuda.is_available() else -1
        try:
            _generator = pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device=device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            print("LLM loaded successfully.")
        except Exception as e:
            print(f"Error loading LLM: {e}")

def generate_response(query, products):
    """
    products: list of dicts representing the filtered products
    """
    if _generator is None:
        init_llm()
        
    if _generator is None:
        return "I found some products for you, but my conversational module is currently unavailable."
        
    product_list_str = ""
    for p in products:
        product_list_str += f"- {p['title']} (${p['price']}, Rating: {p['rating']})\n"
        
    # TinyLlama chat format
    prompt = f"""<|system|>
You are a helpful shopping assistant. Write a short, friendly 2-3 sentence recommendation explaining why the products match the user's need. Keep it concise.
<|user|>
The user asked: '{query}'.
Based on semantic search, here are the top matching products:
{product_list_str}
<|assistant|>
"""
    
    # Generate text
    results = _generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    generated_text = results[0]['generated_text']
    
    # Extract only the assistant's reply
    reply = generated_text.split("<|assistant|>")[-1].strip()
    return reply
