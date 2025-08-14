from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import re

# loading HF
load_dotenv()
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))

def grade_report(text):
    system_prompt = """You are a computer engineering professor. Grade this report using:
1. Technical Accuracy (25%): Correctness of implementations
   -90-100: Flawless with advanced concepts
   -80-89: Minor errors
   -<80: Fundamental mistakes
2. Clarity (25%): Organization and explanation
   -90-100: Professional quality
   -80-89: Needs minor improvement
   -<80: Hard to follow 
3. Code/Docs (25%): Quality of code comments/diagrams
   -90-100: Excellent documentation
   -80-89: Basic
   -<80: Poor/no comments
4. Originality (25%): Novelty
   -90-100: Innovative
   -80-89: Standard Implementation
   -<80: Not original

Return JSON in the format:
{
  "scores": {"technical": X, "clarity": X, "code": X, "originality": X},
  "overall": X.X,
  "feedback": {
    "strengths": ["max 3 bullets"],
    "improvements": ["max 3 priority items"]
  }
}

Example Response:
{
  "scores": {"technical":88, "clarity":90, "code":80, "originality":75},
  "overall": 82.5,
  "feedback": {
    "strengths": ["Clear pseudocode", "Good theoretical foundation"],
    "improvements": ["Add runtime analysis", "Compare with alternative algorithms"]
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

        # extract the  reply
        output_text = response.choices[0].message["content"]

        # extract JSON 
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return {
                "error": "No valid JSON found in model output",
                "raw_response": output_text
            }

    except Exception as e:
        return {
            "error": str(e),
            "raw_response": None
        }
