# test_runner.py
import json
import nltk
import numpy as np
from pathlib import Path
from grading import grade_report
from rouge_score import rouge_scorer

# Download punkt tokenizer for word_tokenize
nltk.download("punkt")

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def rouge_scores(gen, ref):
    try:
        gen_text = " ".join(nltk.word_tokenize(gen.lower()))
        ref_text = " ".join(nltk.word_tokenize(ref.lower()))
        scores = scorer.score(ref_text, gen_text)
        return {k: v.fmeasure for k, v in scores.items()}
    except Exception as e:
        print(f"[WARNING] ROUGE scoring failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# Load reference feedback
refs = json.load(open("reference_feedback.json"))

# List of models to evaluate
models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-9b-it",
    "microsoft/Phi-3.5-mini-instruct"
]

for model in models:
    print(f"\n=== Evaluating {model} ===")
    results = []

    for rid, data in refs.items():
        print(f"\n--- {rid} ---")
        out = grade_report(data.get("report_text", ""), model_name=model)

        if "error" in out:
            print(f"[ERROR] Failed to grade {rid} with {model}")
            print(f"Reason: {out.get('error')}")
            continue

        fb = out.get("feedback", {})
        if not fb:
            print(f"[ERROR] No feedback returned for {rid} with {model}")
            continue

        # Combine strengths and improvements
        gen_text = " ".join(fb.get("strengths", []) + fb.get("improvements", []))
        ref_text = data.get("reference", "")
        scores = rouge_scores(gen_text, ref_text)

        results.append({
            "id": rid,
            "gen": gen_text,
            "ref": ref_text,
            **scores
        })

        print("Generated feedback:", gen_text)
        for k, v in scores.items():
            print(f"{k.upper()} F1: {v:.4f}")

    # Save results per model
    filename = f"evaluation_results_{model.split('/')[0]}.json"
    Path(filename).write_text(json.dumps(results, indent=2))

    if results:
        avg = {k: np.mean([r[k] for r in results]) for k in ["rouge1", "rouge2", "rougeL"]}
        print("\n=== Average ROUGE ===")
        for k, v in avg.items():
            print(f"{k.upper()} F1: {v:.4f}")

    print(f"\nResults saved to {filename}\n")
