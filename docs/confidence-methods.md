# Methods for Evaluating Confidence of Extracted Fields

**No official "confidence" from LLMs:** Currently, neither ChatGPT nor Gemini (whether GPT-4 or Gemini Pro) returns internal probability scores for individual result fields. In chat interfaces, there is no built-in "confidence" field (GPT also does not "expose" internal confidence as a numerical value). Therefore, it is necessary to build custom evaluation solutions, such as through prompting or post-processing checks.

## (1) Using a second LLM as a "judge"

A common approach is to call another LLM (either the same model or a different one) to evaluate the extracted fields. For example, you send the original text and the extracted results back to a secondary LLM, asking it to verify if each field is correct or to estimate the probability of correctness.

The OpenAI community recommends this **"LLM-as-a-judge"** method: the secondary LLM analyzes the primary LLM's output and marks it as "Correct"/"Incorrect" or assigns a score. Similarly, a guide on Label Studio describes a prompt for a second LLM: it receives the input text and classification (or extraction) results, then returns a conclusion based on the content. This can be extended to have the secondary LLM return a confidence score based on the prompt's intent.

*   **Example Prompt:** _"Evaluate extraction results: DateOfBirth = 01/01/1980. Based on the original text, please indicate if the DateOfBirth field is correct. If possible, rate the confidence (0-100 score) of this result."_

The limitation lies in creating prompt sets and processing results. This method incurs additional LLM costs but can be useful for detecting significant errors by comparing multiple LLM "observers" if necessary.

**Sources:**
*   [OpenAI Community: Evaluating Confidence Levels](https://community.openai.com/t/evaluating-the-confidence-levels-of-outputs-generated-by-large-language-models-gpt-4o/1127104)
*   [Label Studio: LLM Evaluation Methods](https://labelstud.io/blog/llm-evaluation-comparing-four-methods-to-automatically-detect-errors/#:~:text=This%20is%20an%20innovative%20approach,see%20the%20original%20research%20paper)

## (2) Using token probabilities (logprobs) – if API is available

If using an LLM API like GPT-4, you can leverage logprobs to calculate confidence. For example, structure the prompt to request a JSON key-value return, then retrieve the log-probabilities of the output tokens. Afterward, sum the log-probs of the tokens within each field to derive a general probability for that field. This method is highlighted in the article "Confidence Unlocked...": after parsing the JSON, sum the log-prob of each token in the value, then convert it to a probability. The result is a confidence score (0–1) for each key-value pair. Tools like the `llm-confidence` library also perform this calculation based on logprobs.

*   **Note:** GPT-4o provides more useful logprobs compared to GPT-4o-mini, while Gemini/Claude did not support logprobs up to 2024. However, Google introduced the Gemini Pro API (Vertex AI), allowing logprobs retrieval starting in 2025. If using the GPT-4 API, this probability data can be utilized immediately; if using only the chat interface, this information cannot be accessed directly.

**Sources:**
*   [OpenAI Community: Evaluating Confidence Levels](https://community.openai.com/t/evaluating-the-confidence-levels-of-outputs-generated-by-large-language-models-gpt-4o/1127104)
*   [Google Developers: Unlock Gemini Reasoning with Logprobs](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/#:~:text=Ever%20wished%20you%20could%20see,view%20into%20the%20model%27s%20choices)

## (3) Multiple runs / Consistency checks (Self-consistency)

Another measure is to compare results by running the model multiple times with similar prompts. The **self-consistency** technique is often used: paraphrase the prompt (or make minor context changes) and have the LLM execute the task multiple times. If the result for the same field is consistently identical (high consensus), you can trust that value more; conversely, if the LLM frequently returns different results (low consensus), that field may be unreliable.

*   **Example:** Perform 3–5 runs with slightly different phrasing, then see if the "DateOfBirth" field always yields "01/01/1980". Self-consistency helps isolate areas where the LLM is prone to contradiction or instability. High consensus across runs signals relatively high confidence; if the majority disagrees, it should be marked for review.

**Source:**
*   [Label Studio: LLM Evaluation Methods](https://labelstud.io/blog/llm-evaluation-comparing-four-methods-to-automatically-detect-errors/#:~:text=The%20self,the%20reliability%20of%20its%20predictions)

## (4) Technical Prompting – Requesting LLM self-estimation

You can engineer prompts to force the LLM to report confidence or "self-check" before answering. For example, in a JSON prompt, add a request: "Return each field accompanied by a confidence field as a percentage." Or use language such as: "Please answer and state how confident you are in each answer on a scale 0–100."

*   **Note:** This is merely the LLM's assumption, not a true probability. Users on forums note that GPT does not "reveal" internal confidence; if asked to estimate, it relies on context and may be overconfident (research shows chatbots often appear overconfident even when incorrect). However, prompts reminding the LLM to evaluate or ask clarification questions if unsure are still effective in making the model answer more carefully.
*   **Example:** Add to the prompt: *"If you are unsure, please report this to me and ask for more information."* Or prompt the LLM to answer honestly rather than guessing. Keywords like "[Inference]" or "[Unverified]" can be used to annotate speculative content. Nevertheless, self-estimated confidence scores are often unverified and require additional manual evaluation.