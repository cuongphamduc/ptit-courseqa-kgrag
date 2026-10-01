"""S5 question router: single-hop vs multi-hop, trained on PTIT DEV only.

Features: query embedding (embedding chosen on dev) + lexical cue flags for comparison / combination questions.
Model: L2-regularised logistic regression; C chosen by 5-fold cross-validation on dev.
Labels: multi = multi_intra | multi_cross; single = single | unanswerable.
"""
import json
import os
import pickle
import re

import numpy as np

from . import data, retrieval

CUES = [r"so sánh", r"khác (nhau|với|biệt)", r"giống", r"\bvà\b", r"đồng thời", r"cả hai", r"mối (quan hệ|liên hệ)",
        r"kết hợp", r"dựa (vào|trên)", r"trước khi|sau khi", r"tương ứng", r"thuộc", r"của .* (có|là)", r"bao nhiêu", r"những"]
PATH = os.path.join(data.WORK, "index", "router.pkl")


def cue_features(q):
    ql = q.lower()
    return np.array([float(bool(re.search(c, ql))) for c in CUES] + [len(ql.split()) / 40.0], dtype=np.float32)


class Router:
    def __init__(self, embed_model, clf=None, threshold=0.5):
        self.embed_model, self.clf, self.threshold = embed_model, clf, threshold

    def features(self, questions, qvecs=None):
        if qvecs is None:
            qvecs = retrieval.encode(self.embed_model, questions, "q")
        return np.hstack([qvecs, np.stack([cue_features(q) for q in questions])])

    def prob_multi(self, question, qvec):
        return float(self.clf.predict_proba(self.features([question], qvec[None, :]))[0, 1])

    def route(self, question, qvec):
        return "multi" if self.prob_multi(question, qvec) >= self.threshold else "single"

    @classmethod
    def train(cls, embed_model):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        qs = data.questions("ptit", "dev")
        y = np.array([int(q["type"].startswith("multi")) for q in qs])
        r = cls(embed_model)
        X = r.features([q["question"] for q in qs])
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        grid = {C: float(np.mean(cross_val_score(LogisticRegression(C=C, max_iter=5000), X, y, cv=cv))) for C in (0.1, 0.3, 1.0, 3.0, 10.0)}
        best = max(grid, key=grid.get)
        r.clf = LogisticRegression(C=best, max_iter=5000).fit(X, y)
        info = {"embed_model": embed_model, "cv_accuracy_by_C": grid, "C": best, "cv_accuracy": grid[best],
                "n_dev": len(qs), "multi_share_dev": float(y.mean())}
        pickle.dump(r, open(PATH, "wb"))
        return r, info

    @staticmethod
    def load():
        return pickle.load(open(PATH, "rb"))
