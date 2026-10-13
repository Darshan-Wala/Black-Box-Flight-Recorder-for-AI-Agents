"""Train the diagnosis model and rank suspect steps."""
import os

import numpy as np

try:  # the real package is needed to train; inference falls back to a pure-Python reader
    import lightgbm as lgb
except (ImportError, OSError):  # OSError: libgomp.so.1 missing (e.g. on Vercel)
    lgb = None

from .signals import EVIDENCE, FEATURES, decisive, extract

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt")


# A step is flagged as faulty when its final score reaches this. Chosen on validation only
# (grouped cross-validation over the real training tasks): F1 was flat from 0.7 to 0.9.
THRESHOLD = 0.8


def features_of(booster):
    """The feature list a model was trained with (an older model.txt has fewer features)."""
    return booster.feature_name()


def _matrix(rows, features=FEATURES):
    return np.array([[r[f] for f in features] for r in rows], dtype=float)


def train(runs, rounds=300, features=FEATURES, real_weight=1.0):
    """Each step of each failed run is one row; label 1 = the faulty step.
    `real_weight` multiplies the weight of rows from recorded (non-simulated) runs."""
    X, y, w = [], [], []
    for run in runs:
        if run.get("outcome") != "fail" or not run.get("faulty_step"):
            continue
        weight = 1.0 if "sim_spec" in run else real_weight
        for step, row in zip(run["steps"], extract(run)):
            X.append([row[f] for f in features])
            y.append(int(step["step_no"] == run["faulty_step"]))
            w.append(weight)
    X, y = np.array(X, dtype=float), np.array(y)
    if lgb is None:
        raise RuntimeError("training needs the lightgbm package: pip install lightgbm")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=10,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  scale_pos_weight=(len(y) - y.sum()) / max(1, y.sum()), seed=7, verbose=-1)
    return lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=list(features)), num_boost_round=rounds)


def save(booster, path=MODEL_PATH):
    booster.save_model(path)


def load(path=MODEL_PATH):
    """lightgbm.Booster when the package is importable, else the pure-Python reader
    (same scores and contributions; used on Vercel). BLACKBOX_LGB_LITE=1 forces the reader."""
    if lgb is not None and os.environ.get("BLACKBOX_LGB_LITE", "") not in ("1", "true"):
        return lgb.Booster(model_file=path)
    from .lgb_lite import Booster
    return Booster(path)


def scores(booster, run, rows=None):
    """Stage 1: the model's score (0..1) for every step of a run, in step order."""
    rows = extract(run) if rows is None else rows
    return [float(x) for x in booster.predict(_matrix(rows, features_of(booster)))] if rows else []


def final_scores(booster, run, rerank=True, rows=None):
    """Stage 2: model scores, with steps that carry decisive evidence moved to the top.
    A decisive step scores 1 + its model score, so it always outranks a step without such
    evidence and decisive steps keep the model's order among themselves (diagnose() shows it
    capped at 1.0). rerank=False is stage 1 alone."""
    rows = extract(run) if rows is None else rows
    raw = scores(booster, run, rows)
    if not rerank:
        return raw
    return [1.0 + p if decisive(r) else p for p, r in zip(raw, rows)]


def rank(booster, run, k=3, rerank=True):
    """step_no of the top-k suspects."""
    s = np.array(final_scores(booster, run, rerank))
    return [run["steps"][i]["step_no"] for i in np.argsort(-s, kind="stable")[:k]]


def diagnose(booster, run, k=3, rerank=True):
    """Return the top-k suspects: step_no, confidence, score, evidence, flagged."""
    rows = extract(run)
    if not rows:
        return []
    feats = features_of(booster)
    X = _matrix(rows, feats)
    final = np.array(final_scores(booster, run, rerank, rows))
    score = np.minimum(final, 1.0)
    order = np.argsort(-final, kind="stable")
    top = list(order[:k])
    contrib = dict(zip(top, booster.predict(X[top], pred_contrib=True)))  # only the rows shown
    share = score / score.sum() if score.sum() > 0 else np.full(len(score), 1 / len(score))
    suspects = []
    for i in order[:k]:
        sure = decisive(rows[i]) if rerank else []
        fired = [(contrib[i][feats.index(f)], label) for f, label, test in EVIDENCE
                 if f in feats and label not in sure and test(rows[i])]
        pushing = [label for c, label in sorted(fired, reverse=True) if c > 0]
        evidence = (sure + pushing)[:3] or [label for _, label in fired][:2] or ["position_and_dependents_only"]
        suspects.append(dict(step_no=run["steps"][i]["step_no"],
                             confidence=round(float(share[i]), 3),
                             score=round(float(score[i]), 3),
                             evidence=evidence,
                             flagged=bool(score[i] >= THRESHOLD)))
    return suspects
