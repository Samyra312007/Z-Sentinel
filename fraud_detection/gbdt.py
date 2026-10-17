"""Gradient-boosted decision trees, implemented in pure Python.

Why hand-rolled instead of LightGBM/XGBoost (plan 2.2)? The Phase 1 pipeline
must run on IBM LinuxONE (s390x) with no wheels available. These trees train
and score anywhere CPython runs, and the artifact is plain JSON — easy to
audit, version and later convert to ONNX for ML for z/OS.

Model details:
  * binary log-loss, Newton (second-order) leaf values
  * histogram splits over quantile-binned features
  * class weighting for the ~3% fraud prevalence
  * per-prediction feature attributions (Saabas-style path contributions),
    sufficient for the plan's "top 3 contributing signals" requirement
"""

from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass, field


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TreeNode:
    feature: int | None = None  # None => leaf
    bin_threshold: int | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    value: float = 0.0  # leaf output (only meaningful on leaves)
    gain: float = 0.0

    def is_leaf(self) -> bool:
        return self.feature is None

    def to_dict(self) -> dict:
        if self.is_leaf():
            return {"v": round(self.value, 6)}
        return {
            "f": self.feature,
            "b": self.bin_threshold,
            "g": round(self.gain, 6),
            "l": self.left.to_dict(),  # type: ignore[union-attr]
            "r": self.right.to_dict(),  # type: ignore[union-attr]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        if "v" in d:
            return cls(value=float(d["v"]))
        return cls(
            feature=int(d["f"]),
            bin_threshold=int(d["b"]),
            gain=float(d.get("g", 0.0)),
            left=cls.from_dict(d["l"]),
            right=cls.from_dict(d["r"]),
        )


# ---------------------------------------------------------------------------
# GBDT
# ---------------------------------------------------------------------------

@dataclass
class GBDT:
    n_estimators: int = 60
    learning_rate: float = 0.12
    max_depth: int = 3
    min_samples_leaf: int = 25
    l2: float = 1.0
    n_bins: int = 16
    subsample: float = 0.7
    pos_weight: float = 20.0
    seed: int = 42

    feature_names: list[str] = field(default_factory=list)
    bin_edges: list[list[float]] = field(default_factory=list)
    trees: list[TreeNode] = field(default_factory=list)
    base_score: float = 0.0
    train_info: dict = field(default_factory=dict)

    # -- training ----------------------------------------------------------

    def fit(self, X: list[list[float]], y: list[int],
            feature_names: list[str] | None = None) -> "GBDT":
        if not X:
            raise ValueError("empty training data")
        nf = len(X[0])
        self.feature_names = feature_names or [f"f{i}" for i in range(nf)]
        if len(self.feature_names) != nf:
            raise ValueError("feature_names length does not match X")

        # 1. global quantile binning
        B: list[list[int]] = [[0] * nf for _ in range(len(X))]
        self.bin_edges = []
        for j in range(nf):
            col = [row[j] for row in X]
            edges = _bin_edges(col, self.n_bins)
            self.bin_edges.append(edges)
            for i, v in enumerate(col):
                B[i][j] = bisect.bisect_right(edges, v)

        n_bins_per_feature = [len(e) + 1 for e in self.bin_edges]

        # 2. class-weighted base rate
        rng = random.Random(self.seed)
        w_pos = max(1.0, self.pos_weight)
        sum_w = sum(w_pos if yi else 1.0 for yi in y)
        p0 = sum(w_pos * yi for yi in y) / sum_w
        self.base_score = logit(p0)
        F = [self.base_score] * len(X)

        # 3. boosting rounds
        self.trees = []
        for _ in range(self.n_estimators):
            p = [sigmoid(f) for f in F]
            w = [w_pos if yi else 1.0 for yi in y]
            g = [w[i] * (p[i] - y[i]) for i in range(len(y))]
            h = [w[i] * p[i] * (1.0 - p[i]) for i in range(len(y))]
            if self.subsample < 1.0:
                idx = [i for i in range(len(y)) if rng.random() < self.subsample]
                if not idx:
                    idx = list(range(len(y)))
            else:
                idx = list(range(len(y)))
            tree = self._build(B, n_bins_per_feature, idx, g, h, 0)
            self.trees.append(tree)
            for i in range(len(X)):  # update all rows, not just the subsample
                F[i] += self.learning_rate * _tree_value(tree, B[i])

        self.train_info = {
            "rows": len(X),
            "features": nf,
            "positives": sum(y),
            "base_score": round(self.base_score, 6),
            "trees": self.n_estimators,
        }
        return self

    def _build(self, B, nb, idx, g, h, depth) -> TreeNode:
        G = sum(g[i] for i in idx)
        H = sum(h[i] for i in idx)
        n = len(idx)
        leaf = TreeNode(value=-G / (H + self.l2))

        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return leaf

        best = None  # (gain, feature, bin, left_idx, right_idx)
        for j in range(len(nb)):
            bins = nb[j]
            gs = [0.0] * bins
            hs = [0.0] * bins
            cs = [0] * bins
            for i in idx:
                b = B[i][j]
                gs[b] += g[i]
                hs[b] += h[i]
                cs[b] += 1
            gl = hl = 0.0
            cl = 0
            for b in range(bins - 1):
                gl += gs[b]
                hl += hs[b]
                cl += cs[b]
                cr = n - cl
                if cl < self.min_samples_leaf or cr < self.min_samples_leaf:
                    continue
                gr = G - gl
                hr = H - hl
                gain = (gl * gl / (hl + self.l2)
                        + gr * gr / (hr + self.l2)
                        - G * G / (H + self.l2))
                if best is None or gain > best[0]:
                    best = (gain, j, b)

        if best is None or best[0] <= 1e-9:
            return leaf

        gain, j, b = best
        left_idx = [i for i in idx if B[i][j] <= b]
        right_idx = [i for i in idx if B[i][j] > b]
        node = TreeNode(feature=j, bin_threshold=b, gain=gain)
        node.left = self._build(B, nb, left_idx, g, h, depth + 1)
        node.right = self._build(B, nb, right_idx, g, h, depth + 1)
        return node

    # -- inference ----------------------------------------------------------

    def _bins_for(self, row: list[float]) -> list[int]:
        return [bisect.bisect_right(self.bin_edges[j], row[j])
                for j in range(len(self.bin_edges))]

    def predict_logit(self, X: list[list[float]]) -> list[float]:
        out = []
        for row in X:
            b = self._bins_for(row)
            out.append(self.base_score
                       + self.learning_rate * sum(_tree_value(t, b) for t in self.trees))
        return out

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [sigmoid(v) for v in self.predict_logit(X)]

    # -- explanations --------------------------------------------------------

    def contributions(self, row: list[float]) -> dict[int, float]:
        """Saabas-style attribution: spread each tree's leaf output across the
        features on its decision path, weighted by split gain."""
        b = self._bins_for(row)
        total: dict[int, float] = {}
        for tree in self.trees:
            node = tree
            path: list[tuple[int, float]] = []
            while not node.is_leaf():
                path.append((node.feature, max(node.gain, 1e-9)))
                node = node.left if b[node.feature] <= node.bin_threshold else node.right
            if not path:
                continue
            leaf_value = node.value * self.learning_rate
            weight_sum = sum(w for _, w in path)
            for feat, w in path:
                total[feat] = total.get(feat, 0.0) + leaf_value * (w / weight_sum)
        return total

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "l2": self.l2,
                "n_bins": self.n_bins,
                "subsample": self.subsample,
                "pos_weight": self.pos_weight,
                "seed": self.seed,
            },
            "feature_names": self.feature_names,
            "bin_edges": self.bin_edges,
            "base_score": self.base_score,
            "trees": [t.to_dict() for t in self.trees],
            "train_info": self.train_info,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GBDT":
        m = cls(**{k: v for k, v in d["params"].items()})
        m.feature_names = list(d["feature_names"])
        m.bin_edges = [list(e) for e in d["bin_edges"]]
        m.base_score = float(d["base_score"])
        m.trees = [TreeNode.from_dict(t) for t in d["trees"]]
        m.train_info = dict(d.get("train_info", {}))
        return m


def _tree_value(node: TreeNode, bins: list[int]) -> float:
    while not node.is_leaf():
        node = node.left if bins[node.feature] <= node.bin_threshold else node.right
    return node.value


def _bin_edges(values: list[float], n_bins: int) -> list[float]:
    """Midpoint edges between unique values (when few), else quantile edges."""
    uniq = sorted(set(values))
    if len(uniq) == 0:
        return []
    if len(uniq) <= n_bins:
        return [(uniq[k] + uniq[k + 1]) / 2.0 for k in range(len(uniq) - 1)]
    n = len(values)
    edges: list[float] = []
    for k in range(1, n_bins):
        v = values[min(n - 1, int(k * n / n_bins))]
        if not edges or v > edges[-1]:
            edges.append(v)
    return edges


def dump_model(gbdt: GBDT, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gbdt.to_dict(), fh, separators=(",", ":"))
