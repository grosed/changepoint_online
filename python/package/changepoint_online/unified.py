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

    # Return the initial candidate dict used by Detector at construction-time.
    # Subclasses may return a list of candidates (e.g., two-side returns two).
    def initial_candidate(self):
        st_val = 0.0 if not isinstance(self.sn, np.ndarray) else np.array(self.sn)
        return {"st": st_val, "tau": int(self.n), "theta0": self.theta0}

    # Create the new candidate(s) representing the current state (after an update).
    # Return a dict (single candidate) or a list of dicts (multiple candidates).
    def new_candidate(self):
        # default single-side candidate
        return {"st": np.array(self.sn) if isinstance(self.sn, np.ndarray) else float(self.sn),
                "tau": int(self.n),
                "theta0": self.theta0}

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


class OneSideUnivariateCUSUM(CUSUM):
    """
    One side univariate CUSUM (positive change if side == 'right', negative if side == 'left').
    Implements monotone-MLE pruning for candidates that belong to this side.
    """

    def __init__(self, theta0=None, sn=0.0, n=0, side="right"):
        super().__init__(sn=sn, n=n, theta0=theta0)
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.side = side

    def initial_candidate(self):
        # include side marker for clarity
        base = super().initial_candidate()
        base["side"] = self.side
        return base

    def new_candidate(self):
        base = super().new_candidate()
        base["side"] = self.side
        return base

    def prune(self, candidates):
        """
        Prune a list of candidates that are assumed to belong to this side (i.e., their 'side' matches).
        This function will behave like the previous make_prune_univariate: monotone ordering of MLEs.
        """
        # Work only on candidates for this side. (If provided mixed list, filter first.)
        side_candidates = [c for c in candidates if c.get("side", "right") == self.side]
        K = len(side_candidates)
        if K <= 1:
            # If there were mixed candidates, return original order filtered by side + others preserved.
            # But typical use: we pass only the side-specific list.
            return candidates

        i = K
        while i > 1:
            c1 = side_candidates[i - 1]
            c0 = side_candidates[i - 2]
            tau1 = int(c1["tau"])
            tau0 = int(c0["tau"])
            denom1 = self.n - tau1
            denom0 = self.n - tau0

            # compute numerators safely
            num1 = float(self.sn) - float(c1["st"])
            num0 = float(self.sn) - float(c0["st"])

            ratio1 = (num1 / float(denom1)) if denom1 > 0 else float("inf")
            ratio0 = (num0 / float(denom0)) if denom0 > 0 else float("inf")

            if ratio1 <= ratio0:
                i -= 1
                if i == 1:
                    break
            else:
                break

        # return pruned list for this side
        pruned_side = side_candidates[:i]
        # But the detector expects us to return a list of candidates for the whole detector.
        # To keep minimal changes and flexibility, if the input 'candidates' were exactly the
        # side list, return pruned_side; otherwise replace matching side entries in the original list.
        if len(side_candidates) == len(candidates):
            return pruned_side
        else:
            # rebuild full list: keep the non-side candidates as-is and put pruned for this side back in.
            other = [c for c in candidates if c.get("side", "right") != self.side]
            combined = other + pruned_side
            # sort by tau to be deterministic
            combined.sort(key=lambda d: (int(d["tau"]), d.get("side", "")))
            return combined


class UnivariateCUSUM(CUSUM):
    """
    Two-sided univariate CUSUM composed of two OneSideUnivariateCUSUM instances:
      - right: detects positive changes using y
      - left : detects positive changes on -y (i.e. negative changes on y)
    This class coordinates updates/pruning and exposes combined initial/new candidates for Detector.
    """

    def __init__(self, theta0=None, sn=0.0, n=0, prune_dim=None):
        # We still set sn and n at the top-level for compatibility, but internal sides will be used.
        super().__init__(sn=sn, n=n, theta0=theta0)
        # create internal side-specific CUSUMs; they will be updated with y and -y respectively
        self.right = OneSideUnivariateCUSUM(theta0=theta0, sn=sn, n=n, side="right")
        self.left = OneSideUnivariateCUSUM(theta0=theta0, sn=sn, n=n, side="left")

    def initial_candidate(self):
        # return list of two side-marked candidates
        return [self.right.initial_candidate(), self.left.initial_candidate()]

    def new_candidate(self):
        # return a list of two new candidates (right and left) for current time
        return [self.right.new_candidate(), self.left.new_candidate()]

    def update(self, y):
        """
        Update both sides: right uses y, left uses -y. Keep n consistent.
        """
        # update right with y
        self.right.update(y)
        # update left with -y
        negy = -y if not isinstance(y, np.ndarray) else -1.0 * np.array(y)
        self.left.update(negy)
        # keep top-level sn/n as informational (not used by univariate cost functions for two-side,
        # but keep consistent)
        self.n = self.right.n
        # For top-level sn, we keep the canonical cumulative sum of original (right) data
        self.sn = self.right.sn

    def prune(self, candidates):
        """
        Prune candidates by splitting into right and left, pruning separately, then re-combining.
        Return combined pruned candidate list sorted by tau (and side for determinism).
        """
        # split
        right_cands = [c for c in candidates if c.get("side", "right") == "right"]
        left_cands  = [c for c in candidates if c.get("side") == "left"]

        pr_right = self.right.prune(right_cands)
        pr_left = self.left.prune(left_cands)

        # ensure both are lists
        if not isinstance(pr_right, list):
            pr_right = [pr_right]
        if not isinstance(pr_left, list):
            pr_left = [pr_left]

        combined = pr_right + pr_left
        combined.sort(key=lambda d: (int(d["tau"]), d.get("side", "")))
        return combined


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
            # replicate prior logic: produce same expression as original
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
# Multivariate-specific costs
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
# Two-sided cost wrapper
# -------------------------
def make_two_sided_cost_fn(base_cost_fn):
    """
    Wrap a univariate base_cost_fn(candidates, cs_side) so that it can operate on the
    combined candidate list produced by UnivariateCUSUM. The wrapper dispatches
    each candidate to the appropriate side's CUSUM (cs.left or cs.right).
    """
    def fn(candidates, cs):
        costs = np.full(len(candidates), -1e300, dtype=float)
        for i, c in enumerate(candidates):
            side = c.get("side", "right")
            if side == "left":
                # evaluate cost of this single candidate using the left-side cs (which uses -y)
                single = base_cost_fn([c], cs.left)
                costs[i] = float(single[0]) if len(single) > 0 else -1e300
            else:
                single = base_cost_fn([c], cs.right)
                costs[i] = float(single[0]) if len(single) > 0 else -1e300
        return costs
    return fn


# -------------------------
# Detector (expects a CUSUM instance)
# -------------------------
class Detector:
    """
    Single-side detector (or two-side if cs provides multiple candidates).
    Expects a CUSUM instance (OneSideUnivariateCUSUM, UnivariateCUSUM, MultivariateCUSUM, ...).
    compute_costs_fn must accept (candidates, cs) and return a numpy array of costs.
    """

    def __init__(self, cs: CUSUM, compute_costs_fn):
        if not isinstance(cs, CUSUM):
            raise TypeError("cs must be an instance of CUSUM (or subclass).")
        initial = cs.initial_candidate()
        if isinstance(initial, dict):
            self.qr = [dict(initial)]
        elif isinstance(initial, list):
            # copy list of candidate dicts
            self.qr = [dict(x) for x in initial]
        else:
            raise RuntimeError("cs.initial_candidate() must return a candidate dict or a list of candidate dicts.")

        # store CUSUM instance and cost function
        self.cs = cs
        self.compute_costs_fn = compute_costs_fn
        self.qr_opt = None

    def update(self, y):
        """
        Process new observation y (scalar or 1-D array).
        Steps:
         - update cs (self.cs.update) -> for TwoSide this updates both sides
         - prune candidates (self.cs.prune)
         - compute costs, store opt
         - append new candidate(s) corresponding to current (sn,n) via cs.new_candidate()
        """
        # update cs to add a new observation (may update side-internals)
        self.cs.update(y)

        # prune using CUSUM's prune method (may accept combined list)
        self.qr = self.cs.prune(self.qr)

        # compute costs (compute_costs_fn returns costs array)
        vals = self.compute_costs_fn(self.qr, self.cs)
        self.qr_opt = float(np.max(vals)) if len(vals) > 0 else -1e300

        # append new candidate(s) representing current time using cs.new_candidate()
        new_cand = self.cs.new_candidate()
        if isinstance(new_cand, dict):
            self.qr.append(dict(new_cand))
        elif isinstance(new_cand, list):
            # this is the typical case for a two-side CUSUM test
            for nc in new_cand:
                self.qr.append(dict(nc))
        else:
            raise RuntimeError("cs.new_candidate() must return a dict or a list of dicts.")

    def statistic(self):
        return self.qr_opt if self.qr_opt is not None else 0.0

    def changepoint(self):
        """
        Return most-likely changepoint (tau) and stat based on current costs.
        Exclude the very last candidate(s) (they are the dummy(s) for the current time).
        """
        if len(self.qr) <= 1:
            return {"stopping_time": self.cs.n, "changepoint": None, "stat": None}
        # exclude the last candidate entry (for One-side we exclude last one; for Two-side we exclude
        # as many trailing candidates as new_candidate() returns)
        last_candidates = self.cs.new_candidate()
        exclude_count = 1 if isinstance(last_candidates, dict) else len(last_candidates)
        considered = self.qr[:-exclude_count]
        if len(considered) == 0:
            return {"stopping_time": self.cs.n, "changepoint": None, "stat": None}
        vals = self.compute_costs_fn(considered, self.cs)
        i = int(np.argmax(vals))
        return {"stopping_time": self.cs.n, "changepoint": considered[i]["tau"], "stat": float(vals[i])}


# -------------------------
# Example usage (traces + plots)
# -------------------------
if __name__ == "__main__":
    np.random.seed(0)

    data = np.concatenate((np.random.normal(0, 1, 200), np.random.normal(3.5, 1, 200)))

    # --- One-side Univariate Gaussian example (previous univariate behavior) ---
    cs_one = OneSideUnivariateCUSUM(theta0=None)                # one-side CUSUM (right)
    detector_one = Detector(cs_one, compute_costs_uni_gaussian)

    one_stat_trace = []
    one_cp_trace = []
    for y in data:
        detector_one.update(float(y))
        one_stat_trace.append(detector_one.statistic())
        cp = detector_one.changepoint().get("changepoint", None)
        one_cp_trace.append(np.nan if cp is None else cp)

    # --- Two-side Univariate Gaussian example ---
    # Use wrapper so we can reuse the same univariate cost fn for both sides
    two_cost_fn = make_two_sided_cost_fn(compute_costs_uni_gaussian)
    cs_two = UnivariateCUSUM(theta0=None)
    detector_two = Detector(cs_two, two_cost_fn)

    two_stat_trace = []
    two_cp_trace = []
    for y in data:
        detector_two.update(float(y))
        two_stat_trace.append(detector_two.statistic())
        cp = detector_two.changepoint().get("changepoint", None)
        two_cp_trace.append(np.nan if cp is None else cp)

    # --- Multivariate Gaussian example (unchanged) ---
    D = 3
    comp_multi = None  # we use MultivariateCUSUM from previous code if needed; here we keep example minimal
    # For demonstration, reuse prior MultivariateCUSUM from earlier version if desired

    # -------------------------
    # Plotting
    # -------------------------
    plt.figure()
    plt.plot(np.arange(1, len(one_stat_trace) + 1), one_stat_trace, label="one-side")
    plt.plot(np.arange(1, len(two_stat_trace) + 1), two_stat_trace, label="two-side", alpha=0.7)
    plt.title("Univariate statistics over time (one-side vs two-side)")
    plt.xlabel("n")
    plt.ylabel("statistic")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(1, len(one_cp_trace) + 1), one_cp_trace, label="one-side")
    plt.plot(np.arange(1, len(two_cp_trace) + 1), two_cp_trace, label="two-side", alpha=0.7)
    plt.title("Estimated changepoint over time")
    plt.xlabel("n")
    plt.ylabel("tau_hat")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
