from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json5
import re

# loading HF
load_dotenv()
client = InferenceClient(model="openai/gpt-oss-20b", token=os.getenv("HF_TOKEN"))

def grade_report(text):
    system_prompt = """You are a computer engineering professor. Return STRICT JSON ONLY inside ```json ...``` fences.
    
Return JSON in the format:
{
  "scores": {"technical": X, "clarity": X, "code": X, "originality": X},
  "overall": X.X,
  "feedback": {
    "strengths": ["max 3 bullets"],
    "improvements": ["max 3 priority items"]
  }
}
"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=400
        )

      
        # Hugging Face clients sometimes give .output_text, sometimes .choices
        output_text = getattr(response, "output_text", None)
        if output_text is None and hasattr(response, "choices"):
            msg = response.choices[0].message
            output_text = msg["content"] if isinstance(msg, dict) else msg.content

        # Try to extract JSON from code fences first
        match = re.search(r"```json(.*?)```", output_text, re.DOTALL)
        if not match:
            # fallback: try to grab first {...}
            match = re.search(r"\{.*\}", output_text, re.DOTALL)

        if match:
            try:
                return json5.loads(match.group(1) if "```" in match.group(0) else match.group(0))
            except Exception as e:
                return {
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": output_text
                }
        else:
            return {
                "error": "No JSON found in response",
                "raw_response": output_text
            }

    except Exception as e:
        return {
            "error": str(e),
            "raw_response": None
        }
