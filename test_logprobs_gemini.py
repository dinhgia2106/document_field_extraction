import os
import math
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

genai.configure(api_key=api_key)

def test_gemini_logprobs():
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        print("--- Sending request... ---")
        
        generation_config = {
            "response_logprobs": True,
            "logprobs": 3,
            "max_output_tokens": 100, # Short limit for easier viewing
            "temperature": 0
        }

        response = model.generate_content(
            "Write exactly 1 random sentence about the universe.",
            generation_config=generation_config
        )

        # Process result
        print(f"\nContent: {response.text}\n")
        print("--- Logprobs Data ---")
        
        # Get first candidate
        candidate = response.candidates[0]

        # Safely check if logprobs_result exists
        # Note: Some old SDK versions return different attribute names,
        # but it is usually logprobs_result or inside candidates[0].content
        if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
            # logprobs_result.chosen_candidates: Contains only the CHOSEN token (1 unique sequence).
            # logprobs_result.top_candidates: Contains list of candidates (top K) at each step.
            
            top_candidates = candidate.logprobs_result.top_candidates
            
            if top_candidates:
                print(f"{'Token':<20} | {'Prob':<10} | LogProb")
                print("-" * 50)
                for i, step in enumerate(top_candidates):
                    print(f"--- Step {i+1} ---")
                    for cand in step.candidates:
                        prob = math.exp(cand.log_probability) * 100
                        # Mark chosen token (for easier viewing)
                        # Note: More complex logic is needed to map correctly with chosen_candidates if highlighting is desired,
                        # but here we just print to see "3 candidates" as requested.
                        print(f"{repr(cand.token):<20} | {prob:.2f}%    | {cand.log_probability:.4f}")
            else:
                 # Fallback if API hasn't returned top_candidates as expected
                 print("top_candidates not found, fallback to chosen_candidates:")
                 for item in candidate.logprobs_result.chosen_candidates:
                    token = item.token
                    log_prob = item.log_probability
                    prob = math.exp(log_prob) * 100
                    print(f"Token: {repr(token):<15} | Prob: {prob:.2f}%")
        else:
            # If this line prints, try printing dir(candidate) to see the structure
            print("API did not return logprobs (or this SDK version maps data differently).")
            # Debug: print entire candidate to inspect
            # print(candidate)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini_logprobs()