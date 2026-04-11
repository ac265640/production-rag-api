from evaluation.runner import evaluate_mode
import json
from config import settings

# Wait, if I just run evaluate_mode("multi_rerank"), I can get the dict.
res = evaluate_mode("multi_rerank")
print()
print("HYBRID RESULT:", res)

# Let me append it to results.json if it exists.
try:
    with open(settings.EVAL_RESULTS_PATH, "r") as f:
        existing = json.load(f)
except Exception:
    existing = {}

existing["multi_rerank"] = res

with open(settings.EVAL_RESULTS_PATH, "w") as f:
    json.dump(existing, f, indent=2)

print("Saved to results.json!")
