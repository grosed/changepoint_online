# Copyright (C) 2023 Gaetano Romano
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

import math


##############
##  Pieces  ##
##############

class Family:
    def __init__(self, St, tau, m0):
        self.St = St  # sum of the data from 1 to tau
        self.tau = tau  # tau, point at which one piece was introduced
        self.m0 = m0

    def argmax(self, cs):
        return (cs.Sn - self.St) / (cs.n - self.tau)

    def get_max(self, cs):
        return self.eval(self.argmax(cs), cs)


class Guassian(Family):
    def eval(self, x, cs):
        c = cs.n - self.tau
        S = cs.Sn - self.St
        return -0.5 * c * x ** 2 + S * x + self.m0
    
class Bernoulli(Family):
    def eval(self, x, cs):
        c = cs.n - self.tau
        S = cs.Sn - self.St
        return S * math.log(x) + (c - S) * math.log(1 - x) + self.m0

    def argmax(self, cs):
        agm = (cs.Sn - self.St) / (cs.n - self.tau)
        if agm == 0:
            return 1e-8
        elif agm == 1:
            return 1 - 1e-8
        else:
            return agm

class Poisson(Family):
    def eval(self, x, cs):
        c = cs.n - self.tau
        S = cs.Sn - self.St
        return -c * x + S * math.log(x) + self.m0


    def argmax(self, cs):
        agm = (cs.Sn - self.St) / (cs.n - self.tau)

        return agm if agm != 0 else 0.000000000001


class classGamma(Family):
    def __init__(self, St=0, tau=0, m0=0, shape=1):
        super().__init__(St, tau, m0)
        self.shape = shape

    def eval(self, x, cs):
        c = cs.n - self.tau
        S = cs.Sn - self.St
        return -c * self.shape * math.log(x) - S * (1 / x) + self.m0

    def argmax(self, cs):
        return (cs.Sn - self.St) / (self.shape * (cs.n - self.tau))

def Gamma(shape) : return lambda St,tau,m0 : classGamma(St,tau,m0,shape)

    
class classAR1(Family):
    def __init__(self, St=0, tau=0, m0=0, phi=0):
        super().__init__(St, tau, m0)
        self.phi = phi

    def eval(self, x, cs):
        c = (cs.n - self.tau) * (1 - self.phi) ** 2
        S = (cs.Sn - self.St) * (1 - self.phi)
        out = c * x ** 2 - 2 * S * x + (1 - self.phi) * self.m0
        return -out

    def argmax(self, cs):
        return (cs.Sn - self.St) / ((cs.n - self.tau) * (1 - self.phi))


def AR1(phi) : return lambda St,tau,m0 : classAR1(St,tau,m0,phi)
    
import numpy as np



################################
##### EX-FOCUS ITERATION #######
################################




class Focus:
    def __init__(self, newP):

        self.cs = Focus._CUSUM()
        self.Ql = Focus._Cost(ps=[newP(0.0, 0, 0.0)])
        self.Qr = Focus._Cost(ps=[newP(0.0, 0, 0.0)])
        self.newP = newP


    def threshold(self) :
        return max(self.Ql.opt, self.Qr.opt)

    def changepoint(self) :
        if self.Ql.opt > self.Qr.opt:
            i = np.argmax([p.get_max(self.cs) - 0.0 for p in self.Ql.ps[:-1]])
            most_likely_changepoint_location = self.Ql.ps[i].tau
        else:
            i = np.argmax([p.get_max(self.cs) - 0.0 for p in self.Qr.ps[:-1]])
            most_likely_changepoint_location = self.Qr.ps[i].tau
        return {"stopping_time": self.cs.n,"changepoint": most_likely_changepoint_location}
        
    def update(self, y):

        # updating the cusums and count with the new point
        self.cs.n += 1
        self.cs.Sn += y

        # updating the value of the max of the null (for pre-change mean unkown)
        m0val = self.Qr.ps[0].get_max(self.cs)

        # pruning step
        Focus._prune(self.Qr, self.cs, "right")  # true for the right pruning
        Focus._prune(self.Ql, self.cs, "left")  # false for the left pruning

        # check the maximum
        self.Qr.opt = Focus._get_max_all(self.Qr, self.cs, m0val)
        self.Ql.opt = Focus._get_max_all(self.Ql, self.cs, m0val)

        # add a new point
        self.Qr.ps.append(self.newP(self.cs.Sn, self.cs.n, m0val))
        self.Ql.ps.append(self.newP(self.cs.Sn, self.cs.n, m0val))

    class _Cost:
        def __init__(self, ps, opt=0):
            self.ps = ps  # a list containing the various pieces
            self.opt = opt  # the global optimum value for ease of access
    class _CUSUM:
        def __init__(self, Sn=0, n=0):
            self.Sn = Sn
            self.n = n

    def _prune(Q, cs, side="right"):
        i = len(Q.ps)
        if i <= 1:
            return Q
        if side == "right":
            def cond(q1, q2):
                return q1.argmax(cs) <= q2.argmax(cs)
        elif side == "left":
            def cond(q1, q2):
                return q1.argmax(cs) >= q2.argmax(cs)
        while cond(Q.ps[i - 1], Q.ps[i - 2]):
            i -= 1
            if i == 1:
                break
        Q.ps = Q.ps[:i]
        return Q

    
    def _get_max_all(Q, cs, m0val):
        return max(p.get_max(cs) - m0val for p in Q.ps)

    
