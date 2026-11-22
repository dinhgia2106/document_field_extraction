import os
import math
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    # Try to verify if it is in environment anyway
    if "OPENAI_API_KEY" not in os.environ:
         raise ValueError("OPENAI_API_KEY not set in .env file or environment variables")
    api_key = os.environ["OPENAI_API_KEY"]

# 2. Initialize Client
client = OpenAI(api_key=api_key)

def test_chatgpt_logprobs():
    try:
        print("--- Sending request to ChatGPT... ---")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Write exactly 1 random sentence about the universe."}
            ],
            logprobs=True,
            top_logprobs=3,
            temperature=0,
            max_tokens=100
        )

        # Process result
        content_text = response.choices[0].message.content
        print(f"\nContent: {content_text}\n")
        print("--- Logprobs Data ---")
        
        # Access logprobs
        logprobs_content = response.choices[0].logprobs.content
        
        if logprobs_content:
            print(f"{'Token':<20} | {'Prob':<10} | LogProb")
            print("-" * 50)
            
            for i, item in enumerate(logprobs_content):
                print(f"--- Step {i+1} ---")
                
                # Print the chosen token first
                prob = math.exp(item.logprob) * 100
                print(f"CHOSEN: {repr(item.token):<12} | {prob:.2f}%    | {item.logprob:.4f}")
                
                # Print top candidates
                if item.top_logprobs:
                    print("  Top Candidates:")
                    for cand in item.top_logprobs:
                        cand_prob = math.exp(cand.logprob) * 100
                        print(f"  {repr(cand.token):<18} | {cand_prob:.2f}%    | {cand.logprob:.4f}")
        else:
            print("No logprobs returned.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chatgpt_logprobs()