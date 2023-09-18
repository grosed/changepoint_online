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

### global cost ###
class Cost:
    def __init__(self, ps, opt=0):
        self.ps = ps  # a list containing the various pieces
        self.opt = opt  # the global optimum value for ease of access


### cusum class ###
class CUSUM:
    def __init__(self, Sn=0, n=0):
        self.Sn = Sn
        self.n = n


##############
##  Pieces  ##
##############

class Family:
    def __init__(self, St=0, tau=0, m0=0, Mdiff=0):
        self.St = St  # sum of the data from 1 to tau
        self.tau = tau  # tau, point at which one piece was introduced
        self.m0 = m0
        self.Mdiff = Mdiff

    # generic of the eval
    def eval(self, x, cs, theta0=None):
        raise NotImplementedError("Subclasses should implement eval")

    def argmax(self, cs):
        return (cs.Sn - self.St) / (cs.n - self.tau)

    def get_max(self, cs, theta0):
        return self.eval(self.argmax(cs), cs, theta0)


class Guassian(Family):
    def eval(self, x, cs, theta0=None):
        c = cs.n - self.tau
        S = cs.Sn - self.St

        if theta0 is None:
            return -0.5 * c * x ** 2 + S * x + self.m0

        else:
            out = c * x ** 2 - 2 * S * x - (c * theta0 ** 2 - 2 * S * theta0)
            return -out / 2



class Bernoulli(Family):
    def eval(self, x, cs, theta0=None):
        c = cs.n - self.tau
        S = cs.Sn - self.St

        if theta0 is None:
            return S * math.log(x) + (c - S) * math.log(1 - x) + self.m0
        else:
            return S * math.log(x / theta0) + (c - S) * math.log((1 - x) / (1 - theta0))

    def argmax(self, cs):
        agm = (cs.Sn - self.St) / (cs.n - self.tau)

        if agm == 0:
            return 1e-8
        elif agm == 1:
            return 1 - 1e-8
        else:
            return agm


class Poisson(Family):
    def eval(self, x, cs, theta0=None):
        c = cs.n - self.tau
        S = cs.Sn - self.St

        if theta0 is None:
            return -c * x + S * math.log(x) + self.m0
        else:
            return -c * (x - theta0) + S * math.log(x / theta0)

    def argmax(self, cs):
        agm = (cs.Sn - self.St) / (cs.n - self.tau)

        return agm if agm != 0 else 0.000000000001


class Gamma(Family):
    def __init__(self, St=0, tau=0, m0=0, Mdiff=0, shape=1):
        super().__init__(St, tau, m0, Mdiff)
        self.shape = shape

    def eval(self, x, cs, theta0=None):
        c = cs.n - self.tau
        S = cs.Sn - self.St

        if theta0 is None:
            return -c * self.shape * math.log(x) - S * (1 / x) + self.m0

        else:
            return c * self.shape * math.log(theta0 / x) - S * (1 / x - 1 / theta0)

    def argmax(self, cs):
        return (cs.Sn - self.St) / (self.shape * (cs.n - self.tau))


class AR1(Family):
    def __init__(self, St=0, tau=0, m0=0, Mdiff=0, phi=0):
        super().__init__(St, tau, m0, Mdiff)
        self.phi = phi

    def eval(self, x, cs, theta0=None):
        c = (cs.n - self.tau) * (1 - self.phi) ** 2
        S = (cs.Sn - self.St) * (1 - self.phi)

        if theta0 is None:
            out = c * x ** 2 - 2 * S * x + (1 - self.phi) * self.m0
            return -out

        else:
            out = c * x ** 2 - 2 * S * x - (c * theta0 ** 2 - 2 * S * theta0)
            return -out

    def argmax(self, cs):
        return (cs.Sn - self.St) / ((cs.n - self.tau) * (1 - self.phi))


#############
### prune ###
#############
def prune(Q, cs, side="right", theta0=None):
    i = len(Q.ps)
    if i <= 1:
        return Q

    if side == "right":
        def cond(q1, q2):
            return q1.argmax(cs) <= (max(theta0, q2.argmax(cs)) if theta0 is not None else q2.argmax(cs))
    elif side == "left":
        def cond(q1, q2):
            return q1.argmax(cs) >= (min(theta0, q2.argmax(cs)) if theta0 is not None else q2.argmax(cs))

    while cond(Q.ps[i - 1], Q.ps[i - 2]):
        i -= 1
        if i == 1:
            break

    Q.ps = Q.ps[:i]

    return Q


#############
## get max ##
#############
# find max overall cost function
def get_max_all(Q, cs, theta0, m0val):
    return max(p.get_max(cs, theta0) - m0val for p in Q.ps)


################################
##### EX-FOCUS ITERATION #######
################################

class Focus:
    def __init__(self, newP):
        self.cs = CUSUM()
        self.Ql = Cost(ps=[newP(0.0, 0, 0.0)])
        self.Qr = Cost(ps=[newP(0.0, 0, 0.0)])
        self.newP = newP

    def update(self, y, theta0, adp_max_check):

        # updating the cusums and count with the new point
        self.cs.n += 1
        self.cs.Sn += y

        # updating the value of the max of the null (for pre-change mean unkown)
        m0val = 0.0
        if theta0 is None:
            m0val = self.Qr.ps[0].get_max(self.cs, theta0)

        # pruning step
        prune(self.Qr, self.cs, "right", theta0)  # true for the right pruning
        prune(self.Ql, self.cs, "left", theta0)  # false for the left pruning

        # check the maximum
        if adp_max_check:
            pass
        else:
            self.Qr.opt = get_max_all(self.Qr, self.cs, theta0, m0val)
            self.Ql.opt = get_max_all(self.Ql, self.cs, theta0, m0val)

        # add a new point
        self.Qr.ps.append(self.newP(self.cs.Sn, self.cs.n, m0val))
        self.Ql.ps.append(self.newP(self.cs.Sn, self.cs.n, m0val))
