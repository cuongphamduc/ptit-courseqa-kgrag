"""Record the S5 router's dev cross-validation accuracy (same features, folds and C grid as rag/router.py Router.train)
without overwriting the trained router. Output: results/dev/router.json"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data  # noqa: E402
from rag.router import Router  # noqa: E402

r = Router.load()
qs = data.questions("ptit", "dev")
y = np.array([int(q["type"].startswith("multi")) for q in qs])
X = r.features([q["question"] for q in qs])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
grid = {C: float(np.mean(cross_val_score(LogisticRegression(C=C, max_iter=5000), X, y, cv=cv))) for C in (0.1, 0.3, 1.0, 3.0, 10.0)}
routes = {}
for ds in ("ptit_test", "viquad_test"):
    rows = [json.loads(l) for l in open(os.path.join(data.ROOT, "results", "retrieval", ds, "S5.jsonl"))]
    routes[ds] = {k: sum(x["route"] == k for x in rows) for k in ("single", "multi")}
qt = {q["id"]: q for q in data.questions("ptit", "test")}
rows = {json.loads(l)["id"]: json.loads(l)["route"] for l in open(os.path.join(data.ROOT, "results", "retrieval", "ptit_test", "S5.jsonl"))}
acc_test = float(np.mean([(rows[i] == "multi") == q["type"].startswith("multi") for i, q in qt.items()]))
out = {"embed_model": r.embed_model, "C": float(r.clf.C), "cv_accuracy_by_C": grid, "cv_accuracy": grid[float(r.clf.C)],
       "n_dev": len(qs), "routes_test": routes, "ptit_test_route_accuracy": round(acc_test, 4)}
json.dump(out, open(os.path.join(data.ROOT, "results", "dev", "router.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
