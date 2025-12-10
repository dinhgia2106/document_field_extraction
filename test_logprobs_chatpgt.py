import os
import math
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env file")

# 2. Initialize Client
client = OpenAI(api_key=api_key)

def test_openai_logprobs():
    try:
        print("--- Sending request to OpenAI... ---")
        
        # 3. Call API
        # Should use gpt-3.5-turbo or gpt-4o-mini for cost-effective testing
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "user", "content": "Write the correct name of the capital of Vietnam."}
            ],
            logprobs=True,      # Required: Enable logprobs
            top_logprobs=2,     # Optional: Get 2 additional alternatives
            temperature=0       # For the most stable results possible
        )

        # 4. Process results
        choice = response.choices[0]
        content = choice.message.content
        logprobs_data = choice.logprobs.content

        print(f"\nReturned content: {content}\n")
        print("--- Logprobs Details per Token ---")
        print(f"{'Token':<15} | {'Logprob':<10} | {'Probability (%)':<15} | {'Top 2 Alternatives'}")
        print("-" * 85)

        if logprobs_data:
            for item in logprobs_data:
                token_text = item.token
                log_prob = item.logprob
                
                # Convert natural logarithm to %: e^logprob * 100
                probability = math.exp(log_prob) * 100
                
                # Get list of alternatives (if any)
                alternatives = []
                if item.top_logprobs:
                    for top in item.top_logprobs:
                        alt_prob = math.exp(top.logprob) * 100
                        alternatives.append(f"{repr(top.token)}({alt_prob:.1f}%)")
                
                alt_str = ", ".join(alternatives)

                # Print table
                # Use repr() to clearly show whitespace or special characters
                print(f"{repr(token_text):<15} | {log_prob:<10.4f} | {probability:<15.2f}% | {alt_str}")
        else:
            print("No logprobs data returned.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_openai_logprobs()