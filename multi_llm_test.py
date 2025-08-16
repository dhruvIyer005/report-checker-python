from grading import grade_report

# Sample report text
sample_text = "This report explains Python loops but misses examples for nested loops."

# List of models to test
models = {
    "Meta-Llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Gemma": "gemma/gemma-llm-model",
    "Deepseek": "deepseek/deepseek-llm-model"
}

for name, model_name in models.items():
    print(f"\n=== Testing {name} ===")
    try:
        result = grade_report(sample_text, model_name=model_name)
        if isinstance(result, dict) and "feedback" in result:
            print("Scores:", result.get("scores", {}))
            print("Strengths:", result["feedback"].get("strengths", []))
            print("Improvements:", result["feedback"].get("improvements", []))
        else:
            print("Error or unexpected output:", result)
    except Exception as e:
        print("Exception:", e)
