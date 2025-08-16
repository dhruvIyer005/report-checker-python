from grading import grade_report

sample_text = "This report explains Python loops but misses examples for nested loops."

# Use Meta-Llama
result = grade_report(sample_text, model_name="meta-llama/Meta-Llama-3-8B-Instruct")

print("Result:")
print(result)
