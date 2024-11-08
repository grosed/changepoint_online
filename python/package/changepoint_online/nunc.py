# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:04:05 2024

@author: austine
"""

from sortedcontainers import SortedList
from collections import deque
import math

def nunc_default_quantiles(sorted_data, num_quantiles) :
    
    '''
    This function computes the quantiles from a set of sorted data using the
    method proposed in Haynes (2017) and used by the NUNC method.
    
    The quantiles are weighted towards the tails of the distribution, with the
    aim being that this will reduce detection delay and increase power
    compared to fixed quantiles.
    
    The function takes a sorted list as input as NUNC stores the window of data
    as a self-balancing red-black tree to allow efficient quantile updating.
    
    **Parameters:**
    -----------
    
    sorted_data (listlike) : The list-like object data with the to calculate 
                             the quantiles from, sorted in ascending order.
    num_quantiles (int): The number of quantiles to compute.
    
    Raises:
    -----------
    
    ValueError: If the num_quantiles arguments are non-positive.
    
    TypeError: If num_quantiles is not an integer, or sorted_data not a list.
    
    References
    ----------

    Haynes K, Fearnhead P, Eckley IA (2017). “A computationally efficient 
    nonparametric approach for changepoint detection.” Statistics and 
    Computing, 27(5), 1293–1305. ISSN 1573-1375, doi:10.1007/s1122201696875.

    Online non-parametric changepoint detection with application to monitoring 
    operational performance of network devices. / Austin, Edward; Romano, 
    Gaetano; Eckley, Idris et al. In: Computational Statistics and Data 
    Analysis, Vol. 177, 107551, 31.01.2023.
    '''

    if not (isinstance(num_quantiles, int)):
        raise TypeError("num_quantiles must be a positive integer")
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer") 

    def quantile(prob) :
        #this function works out the quantile value
        #it uses linear interpolation, ie quantile type = 7
        h = (len(sorted_data) - 1) * prob
        h_floor = int(h)
        if h_floor == h:
            return sorted_data[int(h)]
        else:
            non_int_part = h - h_floor 
            lower = sorted_data[h_floor]
            upper = sorted_data[h_floor + 1]
            return lower + non_int_part * (upper - lower)
    w = len(sorted_data)  #number of points in window
    c = math.log(2*w-1)   #weight as in Zou (2014)
    probs = [(1/(1+(2*(w-1)*math.exp((-c/num_quantiles)*(2*i-1)))))
             for i in range(num_quantiles)]
    return [quantile(p) for p in probs]    
        

class nunc:
    
    '''
    Implements the NUNC (Non-parametric UNbounded Change), a windowed algorithm
    for online changepoint detection in high frequency data streams.
    
    NUNC makes no parametric assumptions on the data under the null, and so is
    well suited to setting where this distribution is unknown or cannot be 
    easily modelled. 
    
    NUNC detects changes inside a sliding window of data by using a likelihood
    ratio test to determine when the distribution of the window segmented into
    two pieces (at the changepoint) offers a significantly better fit to the 
    data than a single distribution. The test aggregates across num_quantiles
    quantiles of the empirical CDF.
    
    Due to the fact that NUNC operates on a sliding window, and updates the
    quantiles of the null as new points are received, it is well suited to 
    operating on data streams where the null distribution is non-stationary.
    
    Various methods can be used to update the quantiles. The default NUNC
    quantiles are weighted towards the tail probabilities, however any
    function that takes in a list of sorted data (in ascending order) and a 
    number of quantiles to compute can be used. 
    The function takes a sorted list as input as NUNC stores the window of data
    as a self-balancing red-black tree to allow efficient quantile updating.
    In the examples an instance where fixed quantiles are specified, a la
    NP-FOCuS, is also presented. 
    
    
    **Parameters:**
    ----------
    
    window_size (int): The size of the NUNC window.
    num_quantiles (int): The number of quantiles used by NUNC.
    get_quantile_vals (callable): A function to compute the quantiles. Takes
                                 two arguments: the sorted data used to
                                 compute the quantiles, and the number of 
                                 quantiles to compute.
    
    Raises
   -------

   ValueError: If the window_size or num_quantiles arguments are non-positive.
       
   TypeError: If the window_size or num_quantiles arguments are not integers,
              or if the get_quantile_vals is not a callable function that
              computes the quantiles from the data in the NUNC window.

   Attributes
   -----------
   tree (SortedList): This is a SortedList, in the form of a red-black tree
                      which stores the data in the current window in ascending
                      order. The benefit of this is that it reduces the cost
                      of computing the quantiles for each new window of data.
    window (deque):   A deque of length window_size containing the current window
                      of data.
    max_cost (float): The current value of the test statistic - the max of the
                      cost function in the current window.
    max_cost_in_window (float): The index of the max_cost in the window.
    num_quantiles (int): The number of quantiles.
    get_quantile_vals (callable): The function to compute the quantiles.
                                Has two arguments: the sorted data to use to
                                compute the quantiles, and the number of 
                                quantiles to compute.
    quantiles (list): The list of quantiles for the current window.
                

   Methods
   --------

   update(y):     Updates the NUNC window and test stat with the new point `y`.
   statistic():   Returns the NUNC test statistic.
   changepoint(): Returns a dictionary containing information about the 
                  detected changepoint inside the window.
                  This includes the stopping time, which for NUNC will always
                  be the right hand edge of the window as all points in the
                  window are checked; the changepoint index in the window;
                  and the maximum statistic.

   References
   ----------

   Online non-parametric changepoint detection with application to monitoring 
   operational performance of network devices. / Austin, Edward; Romano, 
   Gaetano; Eckley, Idris et al. In: Computational Statistics and Data 
   Analysis, Vol. 177, 107551, 31.01.2023.


   Examples:
   ---------
    
   ```python
   from changepoint_online.nunc import nunc, nunc_default_quantiles
   import numpy as np
   
   np.random.seed(1)
   # Generate data with change
   X1 = np.random.normal(1, 1, 3000)
   X2 = np.random.normal(-2, 4, 5000)
   Y = np.concatenate((X1, X2))

   # Create and use NUNC detector
   detector = nunc(300, 5, nunc_default_quantiles)

   stat_over_time = []

   for y in Y:
       detector.update(y)
       stat_over_time.append(detector.max_cost)
       if detector.statistic() > 30:
           break


   changepoint_info = detector.changepoint()

   print(f"Output information:\n{changepoint_info}")

   import matplotlib.pyplot as plt
   plt.plot(detector.window)
   plt.axvline(changepoint_info["changepoint"], color = "red")
   ##########
   #Example 2
   ##########

   #In this second example we show NUNC detecting another change, but we show
   #how the setup can be structured to keep track of the change location in the 
   #data stream starting from the beginning:
       
   import numpy as np
   from changepoint_online.nunc import nunc, nunc_default_quantiles
   
   np.random.seed(10)
   # Generate data with change and random walk component
   X1 = np.random.gumbel(0, 1, 500)
   X2 = np.random.gumbel(-3, 4, 500)
   gaussian_noise = np.random.normal(0, 0.01, 1000)
   rw_noise = np.cumsum(gaussian_noise)
   Y = np.concatenate((X1, X2))
   Y = Y + rw_noise


   # Create and use NUNC detector
   detector = nunc(200, 10, nunc_default_quantiles)

   stat_over_time = []
   points_observed = 0

   for y in Y:
       points_observed += 1
       detector.update(y)
       stat_over_time.append(detector.max_cost)
       if detector.statistic() > 30:
           break

   #Adjust change info for position in overall data stream
   changepoint_info = detector.changepoint()
   changepoint_info["changepoint"] += (points_observed - len(detector.window))
   changepoint_info["stopping_time"] += (points_observed - len(detector.window))

   print(f"Output information:\n{changepoint_info}")

   import matplotlib.pyplot as plt
   plt.plot(Y)
   plt.axvline(changepoint_info["changepoint"], color = "red")

   ############
   #Example 3
   ############

   #Here we show how to detect a change using fixed quantiles:
   
   from changepoint_online.nunc import nunc, nunc_default_quantiles
   import numpy as np
   
   np.random.seed(10)
   # Generate data with change
   X1 = np.random.gamma(4, 2, 500)
   X2 = np.random.gamma(4, 4, 500)
   Y = np.concatenate((X1, X2))

   # Create and use NUNC detector

   quantiles =  [np.quantile(Y[:100], q) for q in [0.01, 0.25, 0.5, 0.75, 0.99]]
   def fixed_quantiles(sorted_data, num_quantiles):
       return quantiles

   detector = nunc(200, 5, fixed_quantiles)

   for y in Y:
       detector.update(y)
       stat_over_time.append(detector.max_cost)
       if detector.statistic() > 15:
           break


   changepoint_info = detector.changepoint()

   print(f"Output information:\n{changepoint_info}")

   import matplotlib.pyplot as plt
   plt.plot(detector.window)
   plt.axvline(changepoint_info["changepoint"], color = "red")

   ```
    '''
    
    def __init__(self, window_size, num_quantiles, 
                 get_quantile_vals = nunc_default_quantiles):

        if not (isinstance(window_size, int)):
            raise TypeError("window_size must be a positive integer")
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")   
            
        if not (isinstance(num_quantiles, int)):
            raise TypeError("num_quantiles must be a positive integer")
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be a positive integer")   
        
        if not callable(get_quantile_vals):
            raise TypeError("get_quantile_vals must be a function")

        
        self.tree = SortedList()
        self.window = deque( maxlen = window_size)
        self.max_cost = 0
        self.max_cost_location_in_window = None
        self.num_quantiles = num_quantiles
        self.get_quantile_vals = get_quantile_vals
        self.quantiles = None
        
    def update(self, x):
        
        ###########
        #Functions Needed For Update Step
        ###########
        
        
        def argmax(l) :
            pos = max(range(len(l)),key=lambda i: l[i])
            return (l[pos],pos) 
        
        def eCDF_vals(data,quantile):    
            #used to return value of eCDF, not cost, at a given quantile  
            #data is the tree of data used to compute the ecdf
            #quantile is the numeric quantile value
            
            left = data.bisect_left(quantile)
            right = data.bisect_right(quantile)
            #value is number of points to left of quantile, plus 0.5 times
            #the points equal to the quantile
            val = (left+0.5*(right-left))/len(data)
            return val
        
        def one_point_emp_dist(data, quantile):
            #function for computing empirical CDF for data at a set quantile
            #ie, is a point less than, equal to, or greater than the quantile
            #data is an array of numerics
            #quantile is the quantile to evaluate the eCDF at.
            if(data < quantile):
                return(1)
            elif (data == quantile):
                return(0.5)
            else: 
                return(0)
            
        def cdf_cost(cdf_val, seg_len):
            #function for computing the likelihood function
            #cdf_val is the value of the eCDF at a set quantile
            #seg_len is the length of the data used 
            if(cdf_val <= 0 or cdf_val >= 1):
                return(0) #avoids rounding error, does not affect result
            conj = 1 - cdf_val
            cost = seg_len * (cdf_val * math.log(cdf_val)
                              - conj * math.log(conj))
            return(cost)
        
        def update_window_ecdf_removal(data_to_remove, quantiles, current_ecdf,
                                       current_len):
            #this function takes a set of K current eCDFs values
            #computed from a data set of length current_len, and removes a 
            #point from this data. 
            #The function then returns the updated CDF values following removal
            #of this point.
            num_quantiles = len(quantiles)
            for i in range(num_quantiles):
                current_ecdf[i] *= current_len
                current_ecdf[i] -= one_point_emp_dist(data_to_remove,
                                                      quantiles[i]) 
                current_ecdf[i] /= (current_len - 1)  
            return current_ecdf

    ##############
    #Update
    ##############
    
        if len(self.window) == self.window.maxlen :
            self.tree.remove(self.window[0])
        self.window.append(x)
        self.tree.add(x)
        #only test once the window is full
        if len(self.window) == self.window.maxlen:
            
            w = len(self.window)  #get window size
            Q = self.get_quantile_vals(self.tree, self.num_quantiles)
            self.quantiles = Q  #update quantiles
            
            full_cdf_vals = [eCDF_vals(self.tree, q) for q in Q] #full data eCDF
            right_cdf_vals = full_cdf_vals.copy() #updates as we search for change 
            full_cost = sum(cdf_cost(val, w) for val in full_cdf_vals) 
            segment_costs = list() #used for storing costs of segmented window
            current_len = w #current length of right segment, updates iteratively
            left_cdf_vals = [0] * len(Q) #update as we search window for points
            for i in range(0, w-1): #window updates are O(K)
            #as we loop over the window we "move" points from the right to
            #left segment, and update the eCDFs. This provides an O(K) cost
            #for updating the eCDFs for each segment.
                right_cdf_vals = update_window_ecdf_removal(self.window[i], Q,
                                                    right_cdf_vals, current_len)
                #remove points fromRHS iteratively and update eCDF
                current_len -= 1
                for j in range(len(Q)): #update LHS using RHS and full eCDFs
                    left_cdf_vals[j] = ((full_cdf_vals[j]*w - right_cdf_vals[j]*current_len) / (w - current_len))
                #compute costs of segmented data
                left_cost = sum([cdf_cost(val, w - current_len)
                                 for val in left_cdf_vals])
                right_cost = sum([cdf_cost(val, current_len) 
                                  for val in right_cdf_vals])
                segment_costs.append(left_cost + right_cost)
            #update cost function
            costs = [2*(cost - full_cost) for cost in segment_costs]
            self.max_cost, self.max_cost_location_in_window = argmax(costs)
            
    def statistic(self):
        return self.max_cost
    
    def changepoint(self):
        return {"stopping_time" : len(self.window),
                "changepoint" : self.max_cost_location_in_window,
                "max_cost" : self.max_cost,
                }
