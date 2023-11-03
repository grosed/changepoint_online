# Copyright (C) 2023 Gaetano Romano, Dan Grose
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.txt>.

import math,sys


################
##  Families  ##
################

class CompFunc:
    def __init__(self, st, tau, m0):
        self.st = st  # sum of the data from 1 to tau
        self.tau = tau  # tau, point at which one piece was introduced
        self.m0 = m0

    def argmax(self, cs):
        return (cs.sn - self.st) / (cs.n - self.tau)

    def get_max(self, cs):
        return self.eval(self.argmax(cs), cs)


class Guassian(CompFunc):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        return -0.5 * c * x ** 2 + s * x + self.m0
    
class Bernoulli(CompFunc):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        return S * math.log(x) + (c - s) * math.log(1 - x) + self.m0

    def argmax(self, cs):
        agm = (cs.sn - self.st) / (cs.n - self.tau)
        if agm == 0:
            return sys.float_info.min
        elif agm == 1:
            return 1 - sys.float_info.min
        else:
            return agm

class Poisson(CompFunc):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        return -c * x + s * math.log(x) + self.m0

    def argmax(self, cs):
        agm = (cs.sn - self.st) / (cs.n - self.tau)
        return agm if agm != 0 else sys.float_info.min


class GammaClass(CompFunc):
    def __init__(self, st, tau, m0, shape):
        super().__init__(st, tau, m0)
        self.shape = shape

    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        return -c * self.shape * math.log(x) - s * (1 / x) + self.m0

    def argmax(self, cs):
        return (cs.sn - self.st) / (self.shape * (cs.n - self.tau))

def Gamma(shape) : return lambda st,tau,m0 : GammaClass(st,tau,m0,shape)

def Exponential() : return lambda st,tau,m0 : GammaClass(st,tau,m0,1)
    
class AR1Class(CompFunc):
    def __init__(self, st, tau, m0, phi):
        super().__init__(st, tau, m0)
        self.phi = phi

    def eval(self, x, cs):
        c = (cs.n - self.tau) * (1 - self.phi) ** 2
        s = (cs.sn - self.st) * (1 - self.phi)
        out = c * x ** 2 - 2 * s * x + (1 - self.phi) * self.m0
        return -out

    def argmax(self, cs):
        return (cs.sn - self.st) / ((cs.n - self.tau) * (1 - self.phi))


def AR1(phi) : return lambda st,tau,m0 : AR1Class(st,tau,m0,phi)
    

################################
##########   FOCUS   ###########
################################


class Focus:
    """
    The Focus class implements the FOCuS method, an algorithm for detecting changes in data streams on one-parameter exponential family models.
    For instance, FOCuS can detect changes-in-mean in a Gaussian data stream (white noise). 
    It can be applied to settings where either the pre-change parameter is known or unknown.
        
    FOCuS solves the CUSUM likelihood-ratio test exactly in O(log(n)) time per iteration, where n is the current iteration. 
    The method is equivalent to running a rolling window (MOSUM) simultaneously for all sizes of window, or the Page-CUSUM for all possible values
    of the size of change (an infinitely dense grid). 

    DISCLAIMER: Albeit the FOCuS algorithm decreases the per-iteration cost from O(n) to O(log(n)), this 
    implementation is not technically online as for n->infty this code would inherently overflow. 
    True online implementations are described in the references below.

    References
    ----------
    Fast online changepoint detection via functional pruning CUSUM statistics
        G Romano, IA Eckley, P Fearnhead, G Rigaill - Journal of Machine Learning Research, 2023
    A Constant-per-Iteration Likelihood Ratio Test for Online Changepoint Detection for Exponential Family Models
        K Ward, G Romano, I Eckley, P Fearnhead - arXiv preprint arXiv:2302.04743, 2023

            
    Examples
    --------
    ```python

    ### Simple gaussian change in mean case ###
    import numpy as np
    
    np.random.seed(0)
    Y = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=5000), np.random.normal(loc=10.0, scale=1.0, size=5000)))

    detector = Focus(Gaussian)
    threshold = 10.0
    for y in Y:
        detector.update(y)
        if detector.statistics() >= threshold:
            break
    ```

    Attributes
    ----------
    cs: 
        An instance of the `_CUSUM` class that keeps track of the cumulative sum and count.
    ql: 
        An instance of the `_Cost` class, that keeps track of the cost function on the left side of the pre-change parameter.
    qr: 
        An instance of the `_Cost` class, that keeps track of the cost function on the right side of the pre-change parameter.
    comp_func: 
        The constructor for the component function given a specific distribution (e.g. Gaussian).

    """

    def __init__(self, comp_func) :
        """
        Focus(comp_func)

        Initializes the Focus detector.

        Parameters
        ----------
        comp_func: A constructor for the component function given an exponential family model to use for the change detection.
                Currently implemented: Gaussian, Bernoulli, Poisson, GammaClass, or AR1Class.
                For more details, check help(comp_func).

        Returns
        -------
        Focus:
            An instance of class Focus, our changepoint detector.


        """
        self.cs = Focus._CUSUM()
        self.ql = Focus._Cost(ps = [comp_func(0.0, 0, 0.0)])
        self.qr = Focus._Cost(ps = [comp_func(0.0, 0, 0.0)])
        self.comp_func = comp_func

    def statistics(self) :
        """
        statistics()

        Computes the value of the CUSUM test statistics at the current iteration.

        Parameters
        ----------
        Null

        Returns
        -------
        float:
            The value of the CUSUM test statistics at the current iteration.
        """
        return max(self.ql.opt, self.qr.opt)

    def changepoint(self) :
        """
        changepoint()

        Returns the most likely changepoint location.
        

        Parameters
        -----
        Null

        Returns
        -------
        dict:
            A dictionary containing the stopping time and the most likely changepoint location.
        """
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
        """
        update(y)

        Updates the FOCuS statistics with a new observation (data point).

        Parameters
        -----
        y: The new data point, either a single integer or a double.

        Returns
        -------
        None
        """
        # updating the cusums and count with the new point
        self.cs.n += 1
        self.cs.sn += y

        # updating the value of the max of the null (for pre-change mean unkown)
        m0 = self.qr.ps[0].get_max(self.cs)

        # pruning step
        Focus._prune(self.qr, self.cs, "right")  # true for the right pruning
        Focus._prune(self.ql, self.cs, "left")  # false for the left pruning

        # check the maximum
        self.qr.opt = Focus._get_max_all(self.qr, self.cs, m0)
        self.ql.opt = Focus._get_max_all(self.ql, self.cs, m0)

        # add a new point
        self.qr.ps.append(self.comp_func(self.cs.sn, self.cs.n, m0))
        self.ql.ps.append(self.comp_func(self.cs.sn, self.cs.n, m0))

    class _Cost:
        def __init__(self, ps, opt=0):
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

    
