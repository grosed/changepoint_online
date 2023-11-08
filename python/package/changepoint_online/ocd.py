from sortedcontainers import SortedList
from scipy.stats import norm
from math import sqrt,log

def add(X,Ux,Y,Uy,T,val) :
    lb = X.bisect_left(val)
    rb = X.bisect_right(val)
    t = rb - lb
    lb = Y.bisect_left(val)
    rb = Y.bisect_right(val)
    t = t + rb - lb
    T = T + 3*(t*t + t)   
    if Ux == None :
        Ux = 0.0
    if lb != rb :
        Ux += 0.5*(rb - lb)
    Ux += lb
    X.add(val)  
    if len(Y) > 0 :
        Uy = len(X)*len(Y) - Ux
    return (X,Ux,Y,Uy,T)

def remove(X,Ux,Y,Uy,T,val) :
    X.remove(val)
    lb = X.bisect_left(val)
    rb = X.bisect_right(val)
    t = rb - lb
    lb = Y.bisect_left(val)
    rb = Y.bisect_right(val)
    t = t + rb - lb
    T = T - 3*(t*t + t)  
    if len(X) == 0 :
         Ux = None
    else :
        if lb != rb :
            Ux -= 0.5*(rb - lb)
        Ux -= lb
        if len(Y) > 0 :
            Uy = len(X)*len(Y) - Ux
    return (X,Ux,Y,Uy,T)

class seq_mann_whitney_U :
    def __init__(self) :
        self.X = SortedList()
        self.Y = SortedList()
        self.Ux = None
        self.Uy = None
        self.T = 0.0
    def add_x(self,val) :
        self.X,self.Ux,self.Y,self.Uy,self.T = add(self.X,self.Ux,self.Y,self.Uy,self.T,val)
    def add_y(self,val) :
        self.Y,self.Uy,self.X,self.Ux,self.T = add(self.Y,self.Uy,self.X,self.Ux,self.T,val)
        return self 
    def remove_x(self,val) :
        self.X,self.Ux,self.Y,self.Uy,self.T = remove(self.X,self.Ux,self.Y,self.Uy,self.T,val)
    def remove_y(self,val) :
        self.Y,self.Uy,self.X,self.Ux,self.T = remove(self.Y,self.Uy,self.X,self.Ux,self.T,val)
    def asymptotic_z(self) :
        if self.Ux == None or self.Uy == None :
            return None
        nx = len(self.X)
        ny = len(self.Y)
        n = nx + ny
        mu = nx*ny/2
        U = self.Ux
        if self.Uy > U :
            U = self.Uy  
        sigma = sqrt((mu/6)*(n+1-self.T/(n*(n-1))))
        if sigma == 0.0 :
            return 0.0
        return (U - mu)/sigma

class ocd_mwu :
    def __init__(self,n,m) :
        self.X = list()
        self.Y = list()
        self.Z = list()
        self.L = seq_mann_whitney_U()
        self.R = seq_mann_whitney_U()
        self.n = n
        self.m = m
        self.z = None
    def push(self,x) :
        if len(self.Y) == self.m :
            popped = self.Y.pop(0)
            self.R.remove_y(popped)
            self.X.append(popped)
            self.L.add_y(popped)
        if len(self.X) == self.n :
            popped = self.X.pop(0)
            self.L.remove_y(popped)
            self.Z.pop(0)
            self.L.remove_x(popped)
            self.R.remove_x(popped)
        self.Y.append(x)
        self.Z.append(x)
        self.R.add_y(x)
        self.L.add_x(x)
        self.R.add_x(x)
        if len(self.X) > 1 :
            pl = 2.0*norm.sf(self.L.asymptotic_z())
            pr = 2.0*norm.sf(self.R.asymptotic_z())
            self.z = log(pl) + log(pr)
        else :
            self.z = None
