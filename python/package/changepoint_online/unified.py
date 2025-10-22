"""
Unified detector (single-side, candidate-only comp_create_fn)

- comp_create_fn(): should be a zero-argument function that returns an initial candidate dict:
    {"st": 0.0 or np.array([...]), "tau": 0, "theta0": ...}
  The Detector will infer the initial CUSUM from this candidate.
- Detector is right-side only (one candidate list).
- Costs are split per family (univariate: gaussian/bernoulli/poisson/gamma; multivariate: gaussian/poisson).
- Multivariate pruning uses ConvexHull (with optional 2D projections) and is robust to an initial scalar st.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# -------------------------
# State
# -------------------------
@dataclass
class CUSUM:
    sn: any = 0.0   # scalar or numpy array
    n: int = 0


# -------------------------
# Candidate-only component factories
# -------------------------
def comp_univariate(theta0=None):
    """
    Zero-arg factory returning an initial univariate candidate dict.
    Subsequent candidates produced by Detector.append (using current cs) will not come from this factory;
    the factory is just for the initial candidate required at construction-time.
    """
    def create():
        return {"st": 0.0, "tau": 0, "theta0": theta0}
    return create


def comp_multivariate(theta0=None):
    """
    Zero-arg factory returning an initial multivariate candidate dict.
    We return scalar 0.0 for st (Detector handles first vector update and converts cs.sn to array).
    """
    def create():
        return {"st": 0.0, "tau": 0, "theta0": theta0}
    return create


# -------------------------
# Univariate family-specific costs (return costs)
# -------------------------
def compute_costs_uni_gaussian(candidates, cs: CUSUM):
    K = len(candidates)
    costs = np.full(K, -1e300, dtype=float)
    S_n = float(cs.sn)
    n = cs.n
    for i, c in enumerate(candidates):
        tau = int(c["tau"])
        S_i = float(c["st"])
        right_len = n - tau
        if right_len <= 0 or tau <= 0 or n <= 0:
            #costs[i] = -1e300
            costs[i] = ((S_n - S_i) ** 2) / float(right_len) - (S_n * S_n) / float(n)
            continue
        costs[i] = (S_i * S_i) / float(tau) + ((S_n - S_i) ** 2) / float(right_len) - (S_n * S_n) / float(n)
    return costs


def compute_costs_uni_bernoulli(candidates, cs: CUSUM):
    K = len(candidates)
    costs = np.full(K, -1e300, dtype=float)
    S_n = float(cs.sn)
    n = cs.n
    eps = 1e-9
    for i, c in enumerate(candidates):
        tau = int(c["tau"])
        S_i = float(c["st"])
        theta0 = c.get("theta0", None)
        right_len = n - tau
        if right_len <= 0:
            costs[i] = -1e300
            continue
        s = S_n - S_i
        p_hat = s / float(right_len)
        p_hat = max(eps, min(1 - eps, p_hat))
        if theta0 is None:
            costs[i] = s * math.log(p_hat) + (right_len - s) * math.log(1 - p_hat)
        else:
            costs[i] = s * math.log(p_hat / theta0) + (right_len - s) * math.log((1 - p_hat) / (1 - theta0))
    return costs


def compute_costs_uni_poisson(candidates, cs: CUSUM):
    K = len(candidates)
    costs = np.full(K, -1e300, dtype=float)
    S_n = float(cs.sn)
    n = cs.n
    eps = 1e-9
    for i, c in enumerate(candidates):
        tau = int(c["tau"])
        S_i = float(c["st"])
        theta0 = c.get("theta0", None)
        right_len = n - tau
        if right_len <= 0:
            costs[i] = -1e300
            continue
        s = S_n - S_i
        lam_hat = max(eps, s / float(right_len))
        if theta0 is None:
            costs[i] = - right_len * lam_hat + s * math.log(lam_hat)
        else:
            costs[i] = - right_len * (lam_hat - theta0) + s * math.log(lam_hat / theta0)
    return costs


def compute_costs_uni_gamma(candidates, cs: CUSUM, shape=1.0):
    K = len(candidates)
    costs = np.full(K, -1e300, dtype=float)
    S_n = float(cs.sn)
    n = cs.n
    eps = 1e-9
    for i, c in enumerate(candidates):
        tau = int(c["tau"])
        S_i = float(c["st"])
        theta0 = c.get("theta0", None)
        right_len = n - tau
        if right_len <= 0:
            costs[i] = -1e300
            continue
        s = S_n - S_i
        arg = s / (shape * float(right_len))
        arg = max(eps, arg)
        if theta0 is None:
            costs[i] = - right_len * shape * math.log(arg) - s * (1.0 / arg)
        else:
            costs[i] = right_len * shape * math.log(theta0 / arg) - s * (1.0 / arg - 1.0 / theta0)
    return costs


# -------------------------
# Multivariate-specific costs (return costs)
# -------------------------
def compute_costs_multi_gaussian(candidates, cs: CUSUM):
    K = len(candidates)
    costs = np.full(K, -1e300, dtype=float)
    S_n = np.array(cs.sn)
    n = cs.n
    for i, c in enumerate(candidates):
        tau = int(c["tau"])
        S_i = np.atleast_1d(np.array(c["st"]))
        right_len = n - tau
        if tau <= 0 or right_len <= 0 or n <= 0:
            costs[i] = -1e300
            continue
        term1 = np.sum((S_i * S_i) / float(tau))
        term2 = np.sum(((S_n - S_i) * (S_n - S_i)) / float(right_len))
        term3 = np.sum((S_n * S_n) / float(n))
        costs[i] = term1 + term2 - term3
    return costs


def compute_costs_multi_poisson(candidates, cs: CUSUM):
    K = len(candidates)
    costs = np.full(K, -1e300, dtype=float)
    S_n = np.array(cs.sn)
    n = cs.n
    eps = 1e-9
    for i, c in enumerate(candidates):
        tau = int(c["tau"])
        S_i = np.atleast_1d(np.array(c["st"]))
        right_len = n - tau
        if right_len <= 0:
            costs[i] = -1e300
            continue
        r = S_n - S_i
        lam_hat = np.maximum(eps, r / float(right_len))
        term = np.sum(- float(right_len) * lam_hat + r * np.log(lam_hat))
        costs[i] = term
    return costs


# -------------------------
# Prune functions
# -------------------------
def make_prune_univariate(compute_costs_fn):
    """Prune using monotone ordering of MLEs (univariate)."""
    def prune(candidates, cs):
        K = len(candidates)
        if K <= 1:
            return candidates
        i = K
        while i > 1 and (cs.sn - candidates[i-1]["st"])/(cs.n - candidates[i-1]["tau"]) <= (cs.sn - candidates[i-2]["st"])/(cs.n - candidates[i-2]["tau"]):
            i -= 1
            if i == 1:
                break
        return candidates[:i]
    return prune


def make_prune_multivariate(dim_indexes=None):
    """
    Prune using ConvexHull or projected 2D approximations.
    Robust to scalar initial st dummy (converts to zero-vector for hull calculation).
    """
    def prune(candidates, cs):
        K = len(candidates)
        if K <= 1:
            return candidates

        sn_arr = np.atleast_1d(np.array(cs.sn))
        target_dim = sn_arr.size

        st_rows = []
        for c in candidates:
            st_c = np.atleast_1d(np.array(c["st"]))
            if st_c.size == target_dim:
                st_rows.append(st_c.copy())
            elif st_c.size == 1 and target_dim > 1:
                # initial dummy: convert scalar to zero-vector
                st_rows.append(np.zeros(target_dim, dtype=float))
            else:
                raise ValueError(
                    "Candidate 'st' dimensionality (%d) incompatible with current CUSUM dimension (%d)."
                    % (st_c.size, target_dim)
                )

        st_stack = np.vstack(st_rows)
        taus = np.array([c["tau"] for c in candidates])[:, None]
        points = np.hstack([taus, st_stack])  # (K, 1 + d)

        if dim_indexes is None:
            try:
                hull = ConvexHull(points)
                idx = np.unique(hull.vertices)
            except Exception:
                idx = np.arange(K)
        else:
            on_hull = []
            for pair in dim_indexes:
                cols = np.append(0, np.array(pair) + 1)
                sub = points[:, cols]
                try:
                    hull = ConvexHull(sub)
                    on_hull.extend(hull.vertices)
                except Exception:
                    on_hull.extend(range(K))
            idx = np.unique(on_hull)

        pruned = [candidates[i] for i in idx]
        pruned.sort(key=lambda d: d["tau"])
        return pruned

    return prune


# -------------------------
# Detector (expects comp_create_fn() -> candidate dict)
# -------------------------
class Detector:
    """
    Single-side detector. Expects comp_create_fn to be a zero-argument callable that returns
    an initial candidate dict: {"st": ..., "tau": ..., "theta0": ...}
    """

    def __init__(self, comp_create_fn, prune_fn, compute_costs_fn):
        # comp_create_fn must be zero-arg returning a candidate dict
        try:
            initial = comp_create_fn()
        except TypeError:
            raise TypeError(
                "comp_create_fn must be a zero-argument function that returns an initial candidate dict. "
                "Use e.g. comp_univariate(theta0=None) which returns such a callable."
            )
        if not isinstance(initial, dict):
            raise RuntimeError("comp_create_fn() must return a candidate dict.")

        # infer CUSUM from candidate
        st_val = initial.get("st", 0.0)
        tau_val = int(initial.get("tau", 0))
        st_arr = np.atleast_1d(np.array(st_val))
        if st_arr.size == 1:
            cs_sn = float(st_arr[0])
        else:
            cs_sn = st_arr.copy()
        self.cs = CUSUM(sn=cs_sn, n=tau_val)

        # store fns and initialise candidate list
        self.comp_create = comp_create_fn
        self.prune_fn = prune_fn
        self.compute_costs_fn = compute_costs_fn
        self.qr = [dict(initial)]
        self.qr_opt = None

    def _update_cs(self, y):
        # update cs.n and cs.sn; convert cs.sn to array on first vector y if necessary
        self.cs.n += 1
        if isinstance(y, np.ndarray):
            if not isinstance(self.cs.sn, np.ndarray):
                # convert scalar cs.sn to zero-array matching y's shape and add previous scalar
                self.cs.sn = np.zeros_like(y, dtype=float) + float(self.cs.sn)
            self.cs.sn = np.array(self.cs.sn) + np.array(y)
        else:
            if isinstance(self.cs.sn, np.ndarray):
                # broadcast scalar y to vector
                self.cs.sn = np.array(self.cs.sn) + float(y)
            else:
                self.cs.sn += y

    def update(self, y):
        """
        Process new observation y (scalar or 1-D array).
        Steps:
         - update cs
         - prune candidates
         - compute costs, store opt
         - append new candidate corresponding to current (sn,n)
        """
        # update cs
        self._update_cs(y)

        # prune using provided prune_fn
        self.qr = self.prune_fn(self.qr, self.cs)

        # compute costs (compute_costs_fn returns costs array)
        vals = self.compute_costs_fn(self.qr, self.cs)
        self.qr_opt = float(np.max(vals)) if len(vals) > 0 else -1e300

        # append new candidate representing current time (create dict based on current cs)
        last_template = self.qr[-1] if len(self.qr) > 0 else {"st": 0.0, "tau": 0, "theta0": None}
        new_cand = dict(last_template)
        # set st and tau from current cs
        new_cand["st"] = np.array(self.cs.sn) if isinstance(self.cs.sn, np.ndarray) else float(self.cs.sn)
        new_cand["tau"] = int(self.cs.n)
        self.qr.append(new_cand)

    def statistic(self):
        return self.qr_opt if self.qr_opt is not None else 0.0

    def changepoint(self):
        """
        Return most-likely changepoint (tau) and stat based on current costs.
        Exclude the very last candidate (it's the dummy for the current time).
        """
        if len(self.qr) <= 1:
            return {"stopping_time": self.cs.n, "changepoint": None, "stat": None}
        vals = self.compute_costs_fn(self.qr[:-1], self.cs)
        i = int(np.argmax(vals))
        return {"stopping_time": self.cs.n, "changepoint": self.qr[i]["tau"], "stat": float(vals[i])}


# -------------------------
# Example usage (traces + plots)
# -------------------------
if __name__ == "__main__":
    np.random.seed(0)

    data = np.concatenate((np.random.normal(0, 1, 200), np.random.normal(3.5, 1, 200)))


    # --- Univariate Gaussian example ---
    comp_uni = comp_univariate(theta0=None)                # zero-arg factory, returns candidate dict
    prune_uni = make_prune_univariate(compute_costs_uni_gaussian)
    detector_uni = Detector(comp_uni, prune_uni, compute_costs_uni_gaussian)

    uni_stat_trace = []
    uni_cp_trace = []
    for y in data:
        detector_uni.update(float(y))
        uni_stat_trace.append(detector_uni.statistic())
        cp = detector_uni.changepoint().get("changepoint", None)
        uni_cp_trace.append(np.nan if cp is None else cp)

    # --- Multivariate Gaussian example ---
    D = 3
    comp_multi = comp_multivariate(theta0=None)            # zero-arg factory
    dim_pairs = [(0, 1), (0, 2), (1, 2)]
    prune_multi = make_prune_multivariate(dim_indexes=dim_pairs)
    detector_multi = Detector(comp_multi, prune_multi, compute_costs_multi_gaussian)

    Y_pre = np.random.normal(0.0, 1.0, size=(100, D))
    Y_post = np.random.normal([4.0, 4.0, 0.0], 1.0, size=(100, D))
    Y = np.vstack([Y_pre, Y_post])

    multi_stat_trace = []
    multi_cp_trace = []
    for y in Y:
        detector_multi.update(y)
        multi_stat_trace.append(detector_multi.statistic())
        cp = detector_multi.changepoint().get("changepoint", None)
        multi_cp_trace.append(np.nan if cp is None else cp)

    # -------------------------
    # Plotting
    # -------------------------
    plt.figure()
    plt.plot(np.arange(1, len(uni_stat_trace) + 1), uni_stat_trace)
    plt.title("Univariate statistic over time")
    plt.xlabel("n")
    plt.ylabel("statistic")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(1, len(uni_cp_trace) + 1), uni_cp_trace)
    plt.title("Univariate estimated changepoint over time")
    plt.xlabel("n")
    plt.ylabel("tau_hat")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(1, len(multi_stat_trace) + 1), multi_stat_trace)
    plt.title("Multivariate statistic over time")
    plt.xlabel("n")
    plt.ylabel("statistic")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(1, len(multi_cp_trace) + 1), multi_cp_trace)
    plt.title("Multivariate estimated changepoint over time")
    plt.xlabel("n")
    plt.ylabel("tau_hat")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()

    plt.show()
