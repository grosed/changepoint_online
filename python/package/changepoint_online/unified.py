from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# -------------------------
# State (CUSUM with behaviour)
# -------------------------
@dataclass
class CUSUM:
    sn: any = 0.0  # scalar or numpy array
    n: int = 0
    theta0: any = None

    # Return the initial candidate dict used by Detector at construction-time
    def initial_candidate(self):
        st_val = 0.0 if not isinstance(self.sn, np.ndarray) else np.array(self.sn)
        return {"st": st_val, "tau": int(self.n), "theta0": self.theta0}

    # Update CUSUM state with new observation y (scalar or 1D array)
    def update(self, y):
        self.n += 1
        # If y is an ndarray, ensure sn is an array of same shape
        if isinstance(y, np.ndarray):
            if not isinstance(self.sn, np.ndarray):
                # convert scalar sn to a zero-array with previous scalar added
                self.sn = np.zeros_like(y, dtype=float) + float(self.sn)
            self.sn = np.array(self.sn) + np.array(y)
        else:
            # y scalar
            if isinstance(self.sn, np.ndarray):
                # broadcast scalar y across vector sn
                self.sn = np.array(self.sn) + float(y)
            else:
                self.sn = float(self.sn) + float(y)

    # Default prune: no-op (return candidates unchanged).
    # Subclasses override this with strategy-specific pruning.
    def prune(self, candidates):
        return candidates


class UnivariateCUSUM(CUSUM):
    """
    Univariate CUSUM with monotone-MLE pruning.
    """
    def __init__(self, theta0=None, sn=0.0, n=0):
        super().__init__(sn=sn, n=n, theta0=theta0)

    def prune(self, candidates):
        K = len(candidates)
        if K <= 1:
            return candidates
        i = K
        # prune based on monotone ordering of right-segment MLEs
        while i > 1:
            c1 = candidates[i - 1]
            c0 = candidates[i - 2]
            tau1 = int(c1["tau"])
            tau0 = int(c0["tau"])
            denom1 = self.n - tau1
            denom0 = self.n - tau0

            # compute ratios safely; if denominator <= 0, treat ratio as +inf so it won't cause pruning
            try:
                num1 = (np.array(self.sn) - np.array(c1["st"])).astype(float)
            except Exception:
                num1 = float(self.sn) - float(c1["st"])
            try:
                num0 = (np.array(self.sn) - np.array(c0["st"])).astype(float)
            except Exception:
                num0 = float(self.sn) - float(c0["st"])

            ratio1 = (num1 / float(denom1)) if denom1 > 0 else float("inf")
            ratio0 = (num0 / float(denom0)) if denom0 > 0 else float("inf")

            # comparison: if newest ratio <= previous then drop newest
            # (behaviour preserved from original make_prune_univariate)
            if ratio1 <= ratio0:
                i -= 1
                if i == 1:
                    break
            else:
                break
        return candidates[:i]


class MultivariateCUSUM(CUSUM):
    """
    Multivariate CUSUM with ConvexHull-based pruning.
    If dim_indexes is provided, project to those 2D subspaces (pairs) plus tau
    and take union of hull vertices across projections.
    Robust to scalar initial st (interprets scalar initial st as zero-vector).
    """
    def __init__(self, theta0=None, sn=0.0, n=0, dim_indexes=None):
        super().__init__(sn=sn, n=n, theta0=theta0)
        # dim_indexes: list of pairs of dimension indices to project onto for 2D hulls
        self.dim_indexes = dim_indexes

    def prune(self, candidates):
        K = len(candidates)
        if K <= 1:
            return candidates

        sn_arr = np.atleast_1d(np.array(self.sn))
        target_dim = sn_arr.size

        # Build matrix of st rows, converting scalar dummy to zero-vector if needed
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

        st_stack = np.vstack(st_rows)  # (K, d)
        taus = np.array([int(c["tau"]) for c in candidates])[:, None]  # (K, 1)
        points = np.hstack([taus, st_stack])  # (K, 1 + d)

        if self.dim_indexes is None:
            # full-dim hull
            try:
                hull = ConvexHull(points)
                idx = np.unique(hull.vertices)
            except Exception:
                idx = np.arange(K)
        else:
            # project to each 2D subspace (tau + each pair of dims)
            on_hull = []
            for pair in self.dim_indexes:
                cols = np.append(0, np.array(pair) + 1)  # include tau at col 0
                sub = points[:, cols]
                try:
                    hull = ConvexHull(sub)
                    on_hull.extend(hull.vertices)
                except Exception:
                    # if hull fails (e.g. degenerate), include all indices
                    on_hull.extend(range(K))
            idx = np.unique(on_hull)

        pruned = [candidates[i] for i in idx]
        pruned.sort(key=lambda d: d["tau"])
        return pruned


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
# Detector (expects a CUSUM instance)
# -------------------------
class Detector:
    """
    Single-side detector. Expects a CUSUM instance (UnivariateCUSUM or MultivariateCUSUM)
    and a compute_costs_fn(candidates, cs) function.
    """

    def __init__(self, cs: CUSUM, compute_costs_fn):
        if not isinstance(cs, CUSUM):
            raise TypeError("cs must be an instance of CUSUM (or subclass).")
        initial = cs.initial_candidate()
        if not isinstance(initial, dict):
            raise RuntimeError("cs.initial_candidate() must return a candidate dict.")

        # store CUSUM instance and functions
        self.cs = cs
        self.compute_costs_fn = compute_costs_fn

        # candidate list and best-stat
        self.qr = [dict(initial)]
        self.qr_opt = None

    def update(self, y):
        """
        Process new observation y (scalar or 1-D array).
        Steps:
         - update cs (self.cs.update)
         - prune candidates (self.cs.prune)
         - compute costs, store opt
         - append new candidate corresponding to current (sn,n)
        """
        # update cs
        self.cs.update(y)

        # prune using CUSUM's prune method
        self.qr = self.cs.prune(self.qr)

        # compute costs (compute_costs_fn returns costs array)
        vals = self.compute_costs_fn(self.qr, self.cs)
        self.qr_opt = float(np.max(vals)) if len(vals) > 0 else -1e300

        # append new candidate representing current time (clone last template)
        last_template = self.qr[-1] if len(self.qr) > 0 else self.cs.initial_candidate()
        new_cand = dict(last_template)
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
    cs_uni = UnivariateCUSUM(theta0=None)                # univariate CUSUM initializer
    detector_uni = Detector(cs_uni, compute_costs_uni_gaussian)

    uni_stat_trace = []
    uni_cp_trace = []
    for y in data:
        detector_uni.update(float(y))
        uni_stat_trace.append(detector_uni.statistic())
        cp = detector_uni.changepoint().get("changepoint", None)
        uni_cp_trace.append(np.nan if cp is None else cp)

    # --- Multivariate Gaussian example ---
    D = 3
    # multivariate CUSUM: start scalar sn=0.0; will convert on first vector update
    dim_pairs = [(0, 1), (0, 2), (1, 2)]
    cs_multi = MultivariateCUSUM(theta0=None, dim_indexes=dim_pairs)
    detector_multi = Detector(cs_multi, compute_costs_multi_gaussian)

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
