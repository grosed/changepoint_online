from scipy.spatial import ConvexHull
import numpy as np
from focus import CompFunc

class MDGaussianClass(CompFunc):
    """
    This function represents a Multidimentional Gaussian component function. For more details, see `help(CompFunc)`.
    """
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st

        pass

        # if self.theta0 is None:
        #     return -0.5 * c * x ** 2 + s * x + self.m0
        # else:
        #     out = c * x ** 2 - 2 * s * x - (c * self.theta0 ** 2 - 2 * s * self.theta0)
        #     return -out / 2
        
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

    def __init__(self, comp_func) :
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
        self.q = MDFocus._Cost(ps = []) # list storing the component functions
        self.comp_func = comp_func

    def statistic(self) :

        return max(self.ql.opt, self.qr.opt)

    def changepoint(self) :

        def _argmax(x) :
            return max(zip(x,range(len(x))))[1]
        if self.ql.opt > self.qr.opt:
            i = _argmax([p.get_max(self.cs) - 0.0 for p in self.ql.ps[:-1]])
            most_likely_changepoint_location = self.ql.ps[i].tau
        else:
            i = _argmax([p.get_max(self.cs) - 0.0 for p in self.qr.ps[:-1]])
            most_likely_changepoint_location = self.qr.ps[i].tau
        return {"stopping_time": self.cs.n,"changepoint": most_likely_changepoint_location}
        
    def update(self, y):

        # getting the dimention of y
        p = y.shape[0]

        # updating the total cumulative sum and time step
        self.cs.n += 1
        self.cs.sn += y

        if self.cs.n > p + 2:
            # slower
            # points = np.array([np.append(p.tau, p.st) for p in self.q.ps])

            points = np.column_stack([[p.tau for p in self.q.ps], np.array([p.st for p in self.q.ps])])
            on_the_hull = ConvexHull(points)

            self.q.ps = self.q.ps[on_the_hull.vertices]

        #self.qr.opt = Focus._get_max_all(self.qr, self.cs, m0)
        
        # add a new point
        self.q.ps = np.append(self.q.ps, self.comp_func(self.cs.sn.copy(), self.cs.n))

        



    class _Cost:
        def __init__(self, ps, opt=-1.0):
            self.ps = ps  # a list containing the various pieces
            self.opt = opt  # the global optimum value for ease of access
    class _CUSUM:
        def __init__(self, sn=0, n=0):
            self.sn = sn
            self.n = n

    def _prune(q, cs, side="right"):
        i = len(q.ps)
        if i <= 1:
            return q
        if side == "right":
            def cond(q1, q2):
                return q1.argmax(cs) <= q2.argmax(cs)
        elif side == "left":
            def cond(q1, q2):
                return q1.argmax(cs) >= q2.argmax(cs)
        while cond(q.ps[i - 1], q.ps[i - 2]):
            i -= 1
            if i == 1:
                break
        q.ps = q.ps[:i]
        return q
    
    def _get_max_all(q, cs, m0):
        return max(p.get_max(cs) - m0 for p in q.ps)



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
  size_post = 5

  # Generate pre-change data (independent samples for each dimension)
  Y_pre = np.random.normal(mean_pre, std_pre, size=(size_pre, 3))

  # Generate post-change data (independent samples for each dimension)
  Y_post = np.random.normal(mean_post, std_post, size=(size_post, 3))

  # Concatenate data with a changepoint in the middle
  changepoint = size_pre
  Y = np.concatenate((Y_pre, Y_post))

  # Assuming Focus and Gaussian classes are defined elsewhere (replace with your implementation)
  detector = MDFocus(MDGaussian())
  threshold = 10.0
  t = time.perf_counter()
  for y in Y:
      detector.update(y)
  print(time.perf_counter() - t)
  print(len(detector.q.ps))

    #   if detector.statistic() >= threshold:
    #       break