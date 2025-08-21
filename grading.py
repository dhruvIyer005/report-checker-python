import os
import json
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


def get_token():
    """Return the HF token for inference."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("⚠️ Please set HF_TOKEN in .env")
    return token


def extract_json(text):
    """Extract JSON object from model output."""
    if not text:
        return {"error": "Empty response", "raw_response": ""}

    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\n(.*?)```", r"\1", text, flags=re.DOTALL)

    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw_response": text}

    return {"error": "No JSON found", "raw_response": text}


def grade_report(text, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Grade a report using HF models (LLaMA, Mistral, Gemma)."""
    token = get_token()
    client = InferenceClient(model=model_name, token=token)

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
    "strengths": ["exactly 3 strong points"],
    "improvements": ["exactly 3 areas to improve"]
  }}
}}

Report:
{text}
"""

    try:
        # Mistral models only support text generation
        if "mistral" in model_name.lower():
            output_text = client.text_generation(prompt, max_new_tokens=500)
        else:  # llama & gemma
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            output_text = resp.choices[0].message["content"]

        parsed = extract_json(output_text)
        if "error" in parsed:
            return {"error": "Failed to parse JSON", "raw_response": output_text}
        return parsed

    except Exception as e:
        return {"error": str(e)}
