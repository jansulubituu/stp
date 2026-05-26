import logging
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_e2e_test() -> bool:
    print("\n" + "=" * 60)
    print("--- STARTING MULTI-AGENT E2E PIPELINE TEST ---")
    print("=" * 60 + "\n")

    try:
        print("[STEP 0] Initializing multi-agent service...")
        from app.services.multiagent_adapter import multiagent_service

        runtime_check = multiagent_service.validate_runtime()
        print("Runtime ok:", runtime_check.get("ok"))
        if runtime_check.get("issues"):
            print("Runtime issues:")
            for issue in runtime_check["issues"]:
                print(" -", issue)
            print()

        test_query = (
            "A method and device for wireless power transmission and receiving. "
            "The system includes an inductive coupling coil and a smart safety controller "
            "that adjusts frequency and current dynamically using machine learning algorithms "
            "to optimize charging efficiency for electric vehicles."
        )

        print("[STEP 1] Executing full PB4 multi-agent pipeline...")
        result = multiagent_service.analyze(test_query)

        print("\n" + "-" * 40)
        print("--- RESULTS ---")
        print("-" * 40)
        print(f"Summary: {result.summary}")
        print(f"Key Points: {result.key_points}")
        print(f"Suggestions: {result.suggestions}")
        print("\nMarkdown Analysis Output Sample:")
        print("=" * 50)
        print(result.analysis[:600] + "...")
        print("=" * 50)

        if "failed" in result.summary.lower() or "error" in result.analysis.lower():
            print("\nFAIL: The multi-agent pipeline returned an error result.")
            return False

        print("\nSUCCESS: The multi-agent pipeline ran end-to-end.")
        return True
    except Exception:
        print("\nFATAL: Unhandled exception while running the multi-agent pipeline.")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
