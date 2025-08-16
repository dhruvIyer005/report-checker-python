import nltk
nltk.download("punkt_tab")
nltk.download("punkt")
import sys, os
sys.path.append(os.path.dirname(__file__))
import json, nltk, numpy as np
from pathlib import Path
from grading import grade_report
from rouge_score import rouge_scorer



scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

def rouge_scores(gen, ref):
    gen, ref = map(lambda t: " ".join(nltk.word_tokenize(t.lower())), [gen, ref])
    return {k: v.fmeasure for k,v in scorer.score(ref, gen).items()}


refs = json.load(open("reference_feedback.json"))
results = []

for rid, data in refs.items():
    print(f"\n=== {rid} ===")
    out = grade_report(data.get("report_text","This is the student’s report text"))
    fb = out.get("feedback", {})
    if not fb: continue

    gen = " ".join(fb.get("strengths",[])+fb.get("improvements",[]))
    scores = rouge_scores(gen, data["reference"])
    results.append({"id":rid,"gen":gen,"ref":data["reference"],**scores})

    print("Generated:", gen)
    for k,v in scores.items(): print(f"{k.upper()} F1: {v:.4f}")


Path("evaluation_results.json").write_text(json.dumps(results, indent=2))
if results:
    avg = {k: np.mean([r[k] for r in results]) for k in ["rouge1","rouge2","rougeL"]}
    print("\n=== Average ROUGE ===")
    for k,v in avg.items(): print(f"{k.upper()} F1: {v:.4f}")

print("\n Results saved to evaluation_results.json")
