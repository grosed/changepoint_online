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

import math,sys


##############
##  Families  ##
##############

class Family:
    def __init__(self, st, tau, m0):
        self.st = st  # sum of the data from 1 to tau
        self.tau = tau  # tau, point at which one piece was introduced
        self.m0 = m0

    def argmax(self, cs):
        return (cs.sn - self.st) / (cs.n - self.tau)

    def get_max(self, cs):
        return self.eval(self.argmax(cs), cs)


class Guassian(Family):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        return -0.5 * c * x ** 2 + s * x + self.m0
    
class Bernoulli(Family):
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

class Poisson(Family):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        return -c * x + s * math.log(x) + self.m0

    def argmax(self, cs):
        agm = (cs.sn - self.st) / (cs.n - self.tau)
        return agm if agm != 0 else sys.float_info.min


class GammaClass(Family):
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

    
class AR1Class(Family):
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
    
import numpy as np



################################
##########   FOCUS   ##########
################################




class Focus:
    def __init__(self, family):

        self.cs = Focus._CUSUM()
        self.ql = Focus._Cost(ps = [family(0.0, 0, 0.0)])
        self.qr = Focus._Cost(ps = [family(0.0, 0, 0.0)])
        self.family = family


    def threshold(self) :
        return max(self.ql.opt, self.qr.opt)

    def changepoint(self) :
        if self.ql.opt > self.qr.opt:
            i = np.argmax([p.get_max(self.cs) - 0.0 for p in self.ql.ps[:-1]])
            most_likely_changepoint_location = self.ql.ps[i].tau
        else:
            i = np.argmax([p.get_max(self.cs) - 0.0 for p in self.qr.ps[:-1]])
            most_likely_changepoint_location = self.qr.ps[i].tau
        return {"stopping_time": self.cs.n,"changepoint": most_likely_changepoint_location}
        
    def update(self, y):

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
        self.qr.ps.append(self.family(self.cs.sn, self.cs.n, m0))
        self.ql.ps.append(self.family(self.cs.sn, self.cs.n, m0))

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

    
