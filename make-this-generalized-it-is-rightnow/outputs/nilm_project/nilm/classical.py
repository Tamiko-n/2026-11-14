"""Non-neural reference methods - "different logic" for the comparison.

  * Rule:  ON if the aggregate jumped above its own recent baseline by more than
           the appliance threshold (classic edge/threshold NILM heuristic).
  * LogReg: linear logistic regression + ridge on hand-crafted window statistics.
  * GBoost: histogram gradient boosting (strong non-ANN tabular learner).
All use the same windows, the same train/val/test split and the same metrics.
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .engine import tune_thresholds


def features(X):
    """Window statistics (X is already standardised whole-house power)."""
    last = X[:, -1]
    d = np.diff(X, axis=1)
    f = [last, X.mean(1), X.std(1), X.min(1), X.max(1),
         last - X.min(1), last - X[:, -15:].mean(1), last - X[:, -5:].mean(1),
         np.abs(d).max(1), d[:, -1], d[:, -5:].sum(1), np.percentile(X, 10, axis=1),
         np.percentile(X, 90, axis=1), X[:, -15:].mean(1), X[:, -30:].std(1)]
    return np.stack(f, 1).astype(np.float32)


def run_classical(kind, data, seed=42, max_train=200_000, log=print):
    tr, va = data["splits"]["train"], data["splits"]["val"]
    rs = np.random.RandomState(seed)
    idx = rs.choice(len(tr["X"]), min(max_train or len(tr["X"]), len(tr["X"])), replace=False)
    A = data["appliances"]
    t0 = time.time()
    clfs, regs = [], []
    if kind == "Rule":
        # learn, per appliance, the aggregate step size (W) that best separates ON/OFF on train
        agg_w = tr["X"][idx] * data["x_std"] + data["x_mean"]
        step = agg_w[:, -1] - np.percentile(agg_w, 5, axis=1)
        for j, a in enumerate(A):
            y = tr["Y"][idx, j]
            valid = np.isfinite(y)
            cands = np.percentile(step, np.linspace(1, 99, 60))
            best = max(cands, key=lambda c: _bacc(y[valid], step[valid] > c))
            on_mean = tr["P"][idx][y == 1, j].mean() if y.sum() else 0.0
            clfs.append(float(best))
            regs.append(float(on_mean))
        model = {"steps": clfs, "on_mean": regs}
        thr = np.full(len(A), 0.5, np.float32)
        log(f"[Rule] fitted in {time.time() - t0:.1f}s")
        return model, thr, time.time() - t0
    Ftr = features(tr["X"][idx])
    sc = StandardScaler().fit(Ftr)
    inputs = sc.transform(Ftr) if kind == "LogReg" else Ftr
    for j, a in enumerate(A):
        y = tr["Y"][idx, j]
        p = tr["P"][idx, j]
        valid = np.isfinite(y)
        x, y, p = inputs[valid], y[valid], p[valid]
        if not len(y):
            c = DummyClassifier(strategy="constant", constant=0).fit(inputs[:1], [0])
            r = DummyRegressor(strategy="constant", constant=0).fit(inputs[:1], [0])
        elif np.unique(y).size == 1:
            c = DummyClassifier(strategy="most_frequent").fit(x, y)
            r = DummyRegressor().fit(x, p)
        elif kind == "LogReg":
            c = LogisticRegression(max_iter=300, class_weight="balanced").fit(x, y)
            r = Ridge(alpha=1.0).fit(x, p)
        else:
            c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                               random_state=seed).fit(x, y)
            r = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(x, p)
        clfs.append(c)
        regs.append(r)
        log(f"[{kind}] {a} fitted")
    model = {"kind": kind, "scaler": sc, "clf": clfs, "reg": regs}
    vprob, _ = predict_classical(model, va["X"], data)
    thr = tune_thresholds(vprob, va["Y"])
    return model, thr, time.time() - t0


def _bacc(y, yhat):
    tpr = (yhat & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = (~yhat & (y == 0)).sum() / max((y == 0).sum(), 1)
    return (tpr + tnr) / 2


def _rule_prob(model, X, data):
    agg_w = X * data["x_std"] + data["x_mean"]
    step = agg_w[:, -1] - np.percentile(agg_w, 5, axis=1)
    return np.stack([(step > s).astype(np.float32) for s in model["steps"]], 1)


def predict_classical(model, X, data):
    if "steps" in model:
        prob = _rule_prob(model, X, data)
        pw = prob * np.array(model["on_mean"], np.float32)
        return prob, pw
    F = features(X)
    Fs = model["scaler"].transform(F) if model["kind"] == "LogReg" else F
    prob = np.stack([c.predict_proba(Fs)[:, -1] if c.classes_[-1] == 1 else np.zeros(len(Fs))
                     for c in model["clf"]], 1)
    pw = np.stack([r.predict(Fs) for r in model["reg"]], 1)
    return prob.astype(np.float32), pw.astype(np.float32)
