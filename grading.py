import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

def get_token(model_name):
    """Return the API token for a model."""
    name = model_name.lower()
    if "gemini" in name or "gemma" in name:
        return os.getenv("GEMMA_TOKEN")
    elif "deepseek" in name:
        return os.getenv("DEEPSEEK_TOKEN")
    else:
        return os.getenv("HF_TOKEN")

def extract_json(text):
    """Extract first JSON object from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw_response": text}
    return {"error": "No JSON found", "raw_response": text}

def grade_report(text, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Grade a report using the specified LLM."""
    token = get_token(model_name)
    if not token:
        return {"error": f"Token for {model_name} not set!"}

    prompt = f"""
You are a computer engineering professor. Grade this report using:
1. Technical Accuracy (25%)
2. Clarity (25%)
3. Code/Docs (25%)
4. Originality (25%)

Return JSON like this:
{{
  "scores": {{"technical": X, "clarity": X, "code": X, "originality": X}},
  "overall": X.X,
  "feedback": {{
    "strengths": ["max 3 bullets"],
    "improvements": ["max 3 priority items"]
  }}
}}

Report:
{text}
"""

    # Meta-Llama (Hugging Face)
    if "meta" in model_name.lower():
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=model_name, token=token)
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return extract_json(resp.choices[0].message["content"])
        except Exception as e:
            return {"error": str(e)}

    # Gemini / Gemma (Google API)
    elif "gemini" in model_name.lower() or "gemma" in model_name.lower():
        import requests
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": token}
        payload = {"contents":[{"parts":[{"text": prompt}]}]}
        try:
            resp = requests.post(url, headers=headers, json=payload).json()
            text_out = ""
            for c in resp.get("candidates", []):
                content = c.get("content", [])
                if isinstance(content, dict):
                    content = [content]
                for part in content:
                    p = part.get("parts", [])
                    if isinstance(p, dict):
                        p = [p]
                    for pp in p:
                        text_out += pp.get("text", "") + "\n"
                    if isinstance(part, str):
                        text_out += part + "\n"
            # remove code fences
            text_out = re.sub(r"```(?:json)?\n(.*?)```", r"\1", text_out, flags=re.DOTALL)
            return extract_json(text_out)
        except Exception as e:
            return {"error": str(e)}

    # DeepSeek 7B Chat (Hugging Face Inference API)
    elif "deepseek" in model_name.lower():
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=model_name, token=token)
            resp = client.text_generation(prompt, max_new_tokens=400)
            output_text = resp.generated_text
            return extract_json(output_text)
        except Exception as e:
            return {"error": str(e)}

    else:
        return {"error": f"Unknown model: {model_name}"}
