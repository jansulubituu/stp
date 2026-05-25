import pandas as pd

def extract_query():
    df = pd.read_parquet("pac_test_plan_views.parquet")
    full_text = df.iloc[0]["query_text"]
    
    with open("extracted_query_ep_1225393.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print("SUCCESS: EXTRACTED TO extracted_query_ep_1225393.txt")

if __name__ == "__main__":
    extract_query()
