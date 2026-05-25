import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_models():
    api_key = os.getenv("AI_KEY")
    if not api_key:
        print("Error: AI_KEY not found in .env")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    
    print("--- FETCHING MODELS FROM NVIDIA NIM REGISTRY ---")
    try:
        models = client.models.list()
        print(f"Total models found: {len(models.data)}\n")
        
        # Lọc ra các model chat chính quy
        chat_models = []
        for m in models.data:
            mid = m.id.lower()
            # Bỏ qua các model embedding, vlm, ranking, rerank
            if any(kw in mid for kw in ["embed", "rerank", "ranking", "vision", "clip", "whisper"]):
                continue
            chat_models.append(m.id)
            
        chat_models.sort()
        for i, model_id in enumerate(chat_models, start=1):
            print(f"{i:2d}. {model_id}")
            
    except Exception as e:
        print(f"Failed to retrieve models: {e}")

if __name__ == "__main__":
    get_models()
