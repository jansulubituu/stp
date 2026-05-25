import os
import sys
import logging
from dotenv import load_dotenv

# Thiết lập hiển thị Log Console để debug chi tiết
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Nạp biến môi trường từ .env trước khi import backend core
load_dotenv()

# Thêm thư mục hiện tại vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_e2e_test():
    print("\n" + "="*60)
    print("--- STARTING END-TO-END PIPELINE TEST ---")
    print("="*60 + "\n")

    try:
        print("[STEP 0] Initializing Analyzer Service...")
        from app.services.analyzer import analyzer_service
        print("SUCCESS: Analyzer Imported!\n")
        
        test_query = (
            "A method and device for wireless power transmission and receiving. "
            "The system includes an inductive coupling coil and a smart safety controller "
            "that adjusts frequency and current dynamically using machine learning algorithms "
            "to optimize charging efficiency for electric vehicles."
        )
        
        print(f"[STEP 1] Test input prepared:")
        print(f"Query snippet: '{test_query[:100]}...'\n")
        
        print("[STEP 2] Executing Full Pipeline (Profiler -> ES Cloud -> Graph -> Nvidia NIM)...")
        result = analyzer_service.analyze(test_query)
        
        print("\n" + "-"*40)
        print("--- RESULTS ---")
        print("-"*40)
        
        print(f"Summary: {result.summary}")
        print(f"Key Points: {result.key_points}")
        print(f"Suggestions: {result.suggestions}")
        
        print("\nMarkdown Analysis Output Sample (first 400 chars):")
        print("="*50)
        # Encode safely just in case LLM returned UTF-8 that the terminal hates
        safe_print = result.analysis.encode('ascii', 'replace').decode('ascii')
        print(safe_print[:400] + "...")
        print("="*50)
        
        # Verify
        if "error" in result.summary.lower() or "failed" in result.summary.lower():
            print("\nFAIL: The pipeline encountered an error.")
            return False
        else:
            print("\nSUCCESS: The pipeline ran flawlessly end-to-end!")
            return True


    except Exception as e:
        print(f"\n💥 LỖI NGHIÊM TRỌNG XẢY RA TRONG QUÁ TRÌNH CHẠY:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
