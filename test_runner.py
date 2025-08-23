import os
import json
from grading import grade_report
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------
# MODELS CONFIG (HuggingFace)
# ----------------------------
MODELS = {
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "phi-3.5-mini": "microsoft/phi-3.5-mini-instruct"
}

# ----------------------------
# LOAD REPORTS
# ----------------------------
with open("reference_feedback.json", "r") as f:
    reports = json.load(f)

results = {}

# ----------------------------
# CLEAN JSON OUTPUT
# ----------------------------
def clean_json_output(text):
    # Remove comments like // ...
    lines = []
    for line in text.splitlines():
        if "//" in line:
            line = line.split("//")[0]
        lines.append(line)
    cleaned = "\n".join(lines)
    # Remove trailing commas
    cleaned = cleaned.replace(",}", "}").replace(",]", "]")
    return cleaned.strip()

# ----------------------------
# EVALUATE EACH MODEL
# ----------------------------
for model_name, model_id in MODELS.items():
    print(f"\n=== Evaluating {model_id} ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        results[model_name] = {}
        for report_id, report_data in reports.items():
            print(f"--- {report_id} ---")

            try:
                raw_output = grade_report(generator, report_data["report_text"])
                print("[DEBUG raw output]", raw_output)

                if not raw_output:
                    raise ValueError("Empty model output")

                cleaned = clean_json_output(raw_output)

                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    parsed = {"error": "⚠️ Failed to parse JSON from model response.", "raw_response": raw_output}

                results[model_name][report_id] = parsed

            except Exception as e:
                print(f"[ERROR] Failed to grade {report_id} with {model_id}\nReason: {e}")
                results[model_name][report_id] = {"error": str(e)}

    except Exception as e:
        print(f"Skipping {model_id} due to error:\n{'*'*60}\n{e}\n{'*'*60}")
        continue

# ----------------------------
# SAVE RESULTS
# ----------------------------
with open("evaluation_results_microsoft.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Evaluation completed. Results saved to evaluation_results.json")
