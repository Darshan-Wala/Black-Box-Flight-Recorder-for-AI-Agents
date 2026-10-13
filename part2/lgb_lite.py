"""Pure-Python reader for a LightGBM text model (part2/model.txt).

Used when the `lightgbm` package cannot be imported, e.g. on Vercel, whose serverless
runtime has no system `libgomp.so.1` (the LightGBM wheel needs it). It gives the same
numbers as `lightgbm.Booster` for the parts the diagnosis API uses:

    booster.feature_name()
    booster.predict(X)                      # probabilities (binary objective)
    booster.predict(X, pred_contrib=True)   # exact TreeSHAP, last column = expected value

Training still needs the real `lightgbm` package (python -m part2.train, locally).
Only numerical splits are supported, which is what part2/model.py trains.
"""
from __future__ import annotations

import math

import numpy as np

_ZERO = 1e-35  # LightGBM's kZeroThreshold


class _Tree:
    __slots__ = ("num_leaves", "split_feature", "threshold", "decision_type", "left", "right",
                 "leaf_value", "internal_count", "leaf_count")

    def __init__(self, f: dict):
        self.num_leaves = int(f["num_leaves"])

        def arr(key, typ):
            return [typ(x) for x in f[key].split()] if f.get(key) else []

        self.leaf_value = arr("leaf_value", float)
        self.leaf_count = arr("leaf_count", float)
        if self.num_leaves > 1:
            if int(f.get("num_cat", "0")):
                raise ValueError("categorical splits are not supported by lgb_lite")
            self.split_feature = arr("split_feature", int)
            self.threshold = arr("threshold", float)
            self.decision_type = arr("decision_type", int)
            self.left = arr("left_child", int)
            self.right = arr("right_child", int)
            self.internal_count = arr("internal_count", float)

    # child index: >= 0 internal node, < 0 leaf ~idx
    def _decide(self, x, node: int) -> int:
        fval = x[self.split_feature[node]]
        dt = self.decision_type[node]
        missing = (dt >> 2) & 3            # 0 none, 1 zero, 2 NaN
        default_left = bool(dt & 2)
        if math.isnan(fval) and missing != 2:
            fval = 0.0
        if (missing == 1 and abs(fval) <= _ZERO) or (missing == 2 and math.isnan(fval)):
            return self.left[node] if default_left else self.right[node]
        return self.left[node] if fval <= self.threshold[node] else self.right[node]

    def leaf(self, x) -> int:
        if self.num_leaves == 1:
            return 0
        node = 0
        while node >= 0:
            node = self._decide(x, node)
        return ~node

    def _count(self, node: int) -> float:
        return self.internal_count[node] if node >= 0 else self.leaf_count[~node]

    def expected_value(self) -> float:
        if self.num_leaves == 1:
            return self.leaf_value[0]
        total = self.internal_count[0]
        return sum(c / total * v for c, v in zip(self.leaf_count, self.leaf_value))

    # ---- TreeSHAP (same algorithm as LightGBM's Tree::TreeSHAP) ----
    def shap(self, x, phi) -> None:
        if self.num_leaves > 1:
            self._recurse(x, phi, 0, [], 1.0, 1.0, -1)

    def _recurse(self, x, phi, node, path, zero_frac, one_frac, feat):
        path = [p[:] for p in path]
        _extend(path, zero_frac, one_frac, feat)
        depth = len(path) - 1
        if node < 0:
            value = self.leaf_value[~node]
            for i in range(1, depth + 1):
                w = _unwound_sum(path, i)
                el = path[i]
                phi[el[0]] += w * (el[2] - el[1]) * value
            return
        hot = self._decide(x, node)
        cold = self.right[node] if hot == self.left[node] else self.left[node]
        w = self._count(node)
        hot_zero, cold_zero = self._count(hot) / w, self._count(cold) / w
        inc_zero = inc_one = 1.0
        split = self.split_feature[node]
        for k, p in enumerate(path):
            if p[0] == split:
                inc_zero, inc_one = p[1], p[2]
                _unwind(path, k)
                break
        self._recurse(x, phi, hot, path, hot_zero * inc_zero, inc_one, split)
        self._recurse(x, phi, cold, path, cold_zero * inc_zero, 0.0, split)


# path element: [feature_index, zero_fraction, one_fraction, pweight]
def _extend(path, zero_frac, one_frac, feat):
    d = len(path)
    path.append([feat, zero_frac, one_frac, 1.0 if d == 0 else 0.0])
    for i in range(d - 1, -1, -1):
        path[i + 1][3] += one_frac * path[i][3] * (i + 1) / (d + 1)
        path[i][3] = zero_frac * path[i][3] * (d - i) / (d + 1)


def _unwind(path, k):
    d = len(path) - 1
    one, zero = path[k][2], path[k][1]
    nxt = path[d][3]
    for i in range(d - 1, -1, -1):
        if one != 0:
            tmp = path[i][3]
            path[i][3] = nxt * (d + 1) / ((i + 1) * one)
            nxt = tmp - path[i][3] * zero * (d - i) / (d + 1)
        else:
            path[i][3] = path[i][3] * (d + 1) / (zero * (d - i))
    for i in range(k, d):
        path[i][0], path[i][1], path[i][2] = path[i + 1][0], path[i + 1][1], path[i + 1][2]
    path.pop()


def _unwound_sum(path, k):
    d = len(path) - 1
    one, zero = path[k][2], path[k][1]
    nxt = path[d][3]
    total = 0.0
    if one != 0:
        for i in range(d - 1, -1, -1):
            tmp = nxt * (d + 1) / ((i + 1) * one)
            total += tmp
            nxt = path[i][3] - tmp * zero * (d - i) / (d + 1)
    else:
        for i in range(d - 1, -1, -1):
            total += path[i][3] * (d + 1) / (zero * (d - i))
    return total


class Booster:
    """Drop-in for the subset of lightgbm.Booster that part2/model.py uses at inference."""

    def __init__(self, model_file: str):
        with open(model_file, encoding="utf-8") as fh:
            text = fh.read()
        header, trees, cur = {}, [], None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Tree="):
                cur = {}
                trees.append(cur)
                continue
            if line in ("end of trees",) or line.startswith("feature_importances:") or line.startswith("parameters:"):
                cur = None
                if line != "end of trees":
                    break
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            (cur if cur is not None else header)[k] = v
        obj = header.get("objective", "")
        if not obj.startswith("binary") or int(header.get("num_class", "1")) != 1:
            raise ValueError(f"lgb_lite only supports binary models, got objective={obj!r}")
        self._sigmoid = 1.0
        for part in obj.split()[1:]:
            if part.startswith("sigmoid:"):
                self._sigmoid = float(part.split(":", 1)[1])
        self._features = header["feature_names"].split()
        self._trees = [_Tree(t) for t in trees]

    def feature_name(self):
        return list(self._features)

    def num_trees(self):
        return len(self._trees)

    def _raw(self, x) -> float:
        return sum(t.leaf_value[t.leaf(x)] for t in self._trees)

    def predict(self, X, pred_contrib: bool = False, raw_score: bool = False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rows = X.tolist()
        if pred_contrib:
            n = len(self._features)
            out = np.zeros((len(rows), n + 1))
            base = sum(t.expected_value() for t in self._trees)
            for r, x in enumerate(rows):
                phi = [0.0] * (n + 1)
                for t in self._trees:
                    t.shap(x, phi)
                phi[n] += base
                out[r] = phi
            return out
        raw = np.array([self._raw(x) for x in rows])
        return raw if raw_score else 1.0 / (1.0 + np.exp(-self._sigmoid * raw))
