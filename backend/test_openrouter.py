import os
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

api_key = os.getenv("AI_KEY")
base_url = os.getenv("NVIDIA_BASE_URL")
model_name = os.getenv("NVIDIA_MODEL")

print(f"=== CAU HINH KIEM TRA ===")
print(f"API Key: {api_key[:10]}...{api_key[-5:] if api_key else ''}")
print(f"Base URL: {base_url}")
print(f"Model Name: {model_name}")
print(f"=========================\n")

try:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    print("PINGING OpenRouter...")
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Reply very briefly."},
            {"role": "user", "content": "Hello!"}
        ],
        max_tokens=250
    )
    
    print("\nSUCCESS!")
    print(f"AI Completion Object: {completion}")
    if completion.choices[0].message.content:
        print(f"Content: {completion.choices[0].message.content}")
    else:
        print("Warning: content field is empty or None. Checking message properties...")
        print(f"Message keys: {dir(completion.choices[0].message)}")
        # Nếu có reasoning_content, trích xuất nó
        reasoning = getattr(completion.choices[0].message, "reasoning", None) or getattr(completion.choices[0].message, "reasoning_content", None)
        print(f"Reasoning content: {reasoning}")
    
except Exception as e:
    print(f"\nFAILED!")
    print(f"Error: {str(e)}")
