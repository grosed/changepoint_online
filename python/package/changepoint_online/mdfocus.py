from scipy.spatial import ConvexHull
import numpy as np
from focus import CompFunc

class MDGaussianClass(CompFunc):
    """
    This function represents a Multidimentional Gaussian component function. For more details, see `help(CompFunc)`.
    """
    def get_max(self, cs):

        if self.theta0 is None:
            r_tau = cs.n - self.tau
            r_st = cs.sn - self.st
            right_cusum_row_sums = np.sum(r_st[:-1]**2, axis=1) / r_tau[:-1]
            left_cusum_row_sums  = np.sum(self.st[1:]**2, axis=1) / self.tau[1:]
            tot_cusum            = right_cusum_row_sums[0]


            return left_cusum_row_sums[:-1] + right_cusum_row_sums[1:] - tot_cusum

        else:
            # TODO: fix centering with the theta0
            r_tau = cs.n - self.tau
            r_st = cs.sn - self.st - np.outer(r_tau, self.theta0)
            right_cusum_row_sums = np.sum(r_st[:-1]**2, axis=1) / r_tau[:-1]

            return right_cusum_row_sums[1:]
        
def MDGaussian(loc=None):
    """
    This function returns a function that creates an instance of the GaussianClass, for 
    Gaussian change-in-mean.

    Parameters
    ----------
    loc (float): The pre-change location (mean) parameter, if known. Defaults to None for pre-change mean unkown.

    Returns
    -------
    function: A function that takes three arguments (st, tau, m0) and returns an instance of GaussianClass.
    """
    return lambda st, tau: MDGaussianClass(st, tau, loc)

class MDFocus:
    """
    The MDFocus class implements the MDFocus method.

    """

    def __init__(self, comp_func, pruning_params = (2, 1)) :
        """
        Focus(comp_func)

        Initializes the Focus detector.

        Parameters
        ----------
        comp_func: A constructor for the component function given an exponential family model to use for the change detection.
                Currently implemented: Gaussian(), Bernoulli(), Poisson(), Gamma() or Exponential().
                For more details, check documentation of each component function, e.g. `help(Gaussian)`.

        Returns
        -------
        MDFocus:
            An instance of class MDFocus, our changepoint detector.

        Examples
        --------

        """
        self.cs = MDFocus._CUSUM()
        self.q = MDFocus._Cost(ps = comp_func(
            np.array([]),
            np.array([0])
            )) # list storing the component functions
        self.comp_func = comp_func
        self.pruning_in = None
        self.pruning_params = pruning_params
        
    def statistic(self) :

        return MDFocus._get_max_all(self.q, self.cs)

    def changepoint(self) :
        pass
        # def _argmax(x) :
        #     return max(zip(x,range(len(x))))[1]
        # if self.ql.opt > self.qr.opt:
        #     i = _argmax([p.get_max(self.cs) - 0.0 for p in self.ql.ps[:-1]])
        #     most_likely_changepoint_location = self.ql.ps[i].tau
        # else:
        #     i = _argmax([p.get_max(self.cs) - 0.0 for p in self.qr.ps[:-1]])
        #     most_likely_changepoint_location = self.qr.ps[i].tau
        # return {"stopping_time": self.cs.n,"changepoint": most_likely_changepoint_location}
        
    def update(self, y):

        # getting the dimention of y to initialize the first pruning and list of pieces
        if self.cs.n == 0:
            self.pruning_in = y.shape[0] + 2
            self.q.ps.st = np.array([np.zeros(y.shape[0])])
            

        # updating the total cumulative sum and time step
        self.cs.n += 1
        self.cs.sn += y

        if self.pruning_in == 0:
            self.q.ps = MDFocus._prune(self.q.ps)

            self.pruning_in = len(self.q.ps.tau) * (self.pruning_params[0] - 1) + self.pruning_params[1]
        
        # add a new point
        self.q.ps.st = np.row_stack([self.q.ps.st, self.cs.sn.copy()])
        self.q.ps.tau = np.append(self.q.ps.tau, self.cs.n)

        # update the pruning iteration counter
        self.pruning_in -= 1

    
    class _Cost:
        def __init__(self, ps, opt=-1.0):
            self.ps = ps  # a list containing the various pieces
            self.opt = opt  # the global optimum value for ease of access
    class _CUSUM:
        def __init__(self, sn=0, n=0):
            self.sn = sn
            self.n = n

    def _prune(ps):
        points = np.column_stack([ps.tau, ps.st])
        on_the_hull = ConvexHull(points)

        ps.tau = ps.tau[on_the_hull.vertices]
        ps.st  = ps.st[on_the_hull.vertices]

        return ps
    
    def _get_max_all(q, cs):
        if cs.n == 1:
            return 0.0
        else:
            locals = q.ps.get_max(cs)
            return max(locals)


if __name__ == "__main__":
  import numpy as np
  import time

  np.random.seed(123)

  # Define means and standard deviations for pre-change and post-change periods (independent dimensions)
  mean_pre = np.array([0.0, 0.0, 5.0])
  mean_post = np.array([10.0, 10.0, 0.0])

  std_pre = np.array([1.0, 1.0, 1.0])
  std_post = np.array([1.0, 1.0, 1.0])

  # Sample sizes for pre-change and post-change periods
  size_pre = 5000
  size_post = 500

  # Generate pre-change data (independent samples for each dimension)
  Y_pre = np.random.normal(mean_pre, std_pre, size=(size_pre, 3))

  # Generate post-change data (independent samples for each dimension)
  Y_post = np.random.normal(mean_post, std_post, size=(size_post, 3))

  # Concatenate data with a changepoint in the middle
  changepoint = size_pre
  Y = np.concatenate((Y_pre, Y_post))

  # Assuming Focus and Gaussian classes are defined elsewhere (replace with your implementation)
  detector = MDFocus(MDGaussian(), pruning_params = (2, 1))
  threshold = 50.0
  t = time.perf_counter()
  for y in Y:
      detector.update(y)
      if detector.statistic() >= threshold:
        break
  print(time.perf_counter() - t)
  print(len(detector.q.ps.tau))
  print(detector.cs.n)

  # pre-change known 
  detector = MDFocus(MDGaussian(loc=mean_pre), pruning_params = (2, 1))
  threshold = 50.0
  t = time.perf_counter()
  for y in Y:
      detector.update(y)
      if detector.statistic() >= threshold:
        break
  print(time.perf_counter() - t)
  print(len(detector.q.ps.tau))
  print(detector.cs.n)
