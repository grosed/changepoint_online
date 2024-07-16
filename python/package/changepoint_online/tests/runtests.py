# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:39:42 2024

@author: austine
"""

from changepoint_online import Focus, Gaussian, Gamma, Poisson, Bernoulli, NPFocus, MDFocus, MDGaussian, MDPoisson, nunc, nunc_default_quantiles
import numpy as np
import unittest



class test_exponential(unittest.TestCase):

    def test_gaussian(self):
        # generating some data with a change at 50,000
        np.random.seed(0)
        Y = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=50000), np.random.normal(loc=2.0, scale=1.0, size=5000)))

        # initialize a Focus Gaussian detector
        detector = Focus(Gaussian())
        threshold = 13.0

        for y in Y:
            # update your detector sequentially with
            detector.update(y)
            if detector.statistic() >= threshold:
                break

        expected = {'stopping_time': 50013, 'changepoint': 50000}
        actual = detector.changepoint()

        self.assertEqual(expected, actual)

    def test_gamma(self):
        # generating some data with a change at 50,000
        np.random.seed(0)
        Y = np.concatenate((np.random.gamma(4.0, scale=3.0, size=50000), 
                            np.random.gamma(4.0, scale=6.0, size=5000)))

        # initialize a Gamma change-in-scale detector (with shape = 4)
        detector = Focus(Gamma(shape=4.0))
        threshold = 12.0
        for y in Y:
            detector.update(y)
            if detector.statistic() >= threshold:
                break

        expected = {'stopping_time': 50012, 'changepoint': 50000}
        actual = detector.changepoint()

        self.assertEqual(expected, actual)

    def test_poisson(self):
        # generating some data with a change at 50,000
        np.random.seed(0)
        Y = np.concatenate((np.random.poisson(lam=4.0, size=5000),
        np.random.poisson(lam=4.2, size=5000)))
        detector = Focus(Poisson())
        threshold = 10.0
        for y in Y:
            detector.update(y)
            if detector.statistic() >= threshold:
                break
        detector.changepoint()

        expected = {'stopping_time': 6488, 'changepoint': 4973}
        actual = detector.changepoint()

        self.assertEqual(expected, actual)

    def test_bernoulli(self):
        # generating some data with a change at 50,000
        np.random.seed(0)
        Y = np.concatenate((np.random.binomial(n=1, p=.4, size=5000),
        np.random.binomial(n=1, p=.8, size=5000)))
        detector = Focus(Bernoulli())
        threshold = 15.0
        for y in Y:
            detector.update(y)
            if detector.statistic() >= threshold:
                break
        detector.changepoint()

        expected = {'stopping_time': 5024, 'changepoint': 4999}
        actual = detector.changepoint()

        self.assertEqual(expected, actual)

class test_nonparametric(unittest.TestCase):
    def test_npfocus(self):
        
        # Define a simple Gaussian noise function
        def generate_gaussian_noise(size):
            return np.random.normal(loc=0.0, scale=1.0, size=size)
        
        np.random.seed(0)
        # Generate mixed data with change in gamma component
        gamma_1 = np.random.gamma(4.0, scale=3.0, size=5000)
        gamma_2 = np.random.gamma(4.0, scale=6.0, size=5000)
        gaussian_noise = generate_gaussian_noise(10000)
        Y = np.concatenate((gamma_1 + gaussian_noise[:5000], gamma_2 + gaussian_noise[5000:]))

        # Create and use NPFocus detector
        ## One needs to provide some quantiles to track the null distribuition over
        quantiles = [np.quantile(Y[:100], q) for q in [0.25, 0.5, 0.75]]
        ## the detector can be initialised with those quantiles
        detector = NPFocus(quantiles)

        for y in Y:
            detector.update(y)
            # we can sum the statistics over to get a detection
            # see  (Romano, Eckley, and Fearnhead 2024) for more details
            if np.sum(detector.statistic()) > 25:
                break


        actual = detector.changepoint()
        expected = {'stopping_time': 5018, 'changepoint': 4999, 'max_stat': 15.700167233855154}
        self.assertAlmostEqual(expected, actual, places=6)




class test_multivariate(unittest.TestCase):
    def test_mvgaussian(self):

        np.random.seed(123)

        # Define means and standard deviations for pre-change and post-change periods (independent dimensions)
        mean_pre = np.array([0.0, 0.0, 5.0])
        mean_post = np.array([1.0, 1.0, 4.5])

        # Sample sizes for pre-change and post-change periods
        size_pre = 5000
        size_post = 500

        # Generate pre-change data (independent samples for each dimension)
        Y_pre = np.random.normal(mean_pre, size=(size_pre, 3))

        # Generate post-change data (independent samples for each dimension)
        Y_post = np.random.normal(mean_post, size=(size_post, 3))

        # Concatenate data with a changepoint in the middle
        changepoint = size_pre
        Y = np.concatenate((Y_pre, Y_post))

        # Initialize the Gaussian detector 
        # (for change in mean known, as in univariate case, write MDGaussian(loc=mean_pre))
        detector = MDFocus(MDGaussian(), pruning_params = (2, 1))
        threshold = 25
        for y in Y:
            detector.update(y)
            if detector.statistic() >= threshold:
                break
        expected = {'stopping_time': 5014, 'changepoint': 5000}
        actual = detector.changepoint()

        self.assertEqual(expected, actual)

    def test_mvpoisson(self):
        np.random.seed(123)

        # Define rates (lambda) for pre-change and post-change periods (independent dimensions)
        rate_pre = np.array([1.0, 2.0, 3.0])
        rate_post = np.array([2.0, 3.0, 4.0])

        # Sample sizes for pre-change and post-change periods
        size_pre = 5000
        size_post = 500

        # Generate pre-change data (independent samples for each dimension)
        Y_pre = np.random.poisson(rate_pre, size=(size_pre, 3))

        # Generate post-change data (independent samples for each dimension)
        Y_post = np.random.poisson(rate_post, size=(size_post, 3))

        # Concatenate data with a changepoint in the middle
        changepoint = size_pre
        Y = np.concatenate((Y_pre, Y_post))

        # Initialize the Poisson detector
        detector = MDFocus(MDPoisson(), pruning_params = (2, 1))
        threshold = 45  # Adjust the threshold as needed for the Poisson distribution

        for y in Y:
            detector.update(y)
            if detector.statistic() >= threshold:
                break
        expected = {'stopping_time': 5056, 'changepoint': 5000}
        actual = detector.changepoint()

        self.assertEqual(expected, actual)
        
class test_nunc(unittest.TestCase):
    def test_nunc(self):
        
       np.random.seed(10)
       # Generate data with change
       X1 = np.random.normal(1, 1, 500)
       X2 = np.random.normal(-2, 4, 500)
       Y = np.concatenate((X1, X2))
    
       # Create and use NUNC detector
       detector = nunc(300, 5, nunc_default_quantiles)
    
       stat_over_time = []
    
       for y in Y:
           detector.update(y)
           stat_over_time.append(detector.max_cost)
           if detector.statistic() > 40:
               break

       actual = detector.changepoint()
       expected = {'stopping_time': 300, 'changepoint': 286, 'max_cost': 41.166820004265354}
       self.assertEqual(expected['stopping_time'], actual['stopping_time'])
       self.assertEqual(expected['changepoint'], actual['changepoint'])
       self.assertAlmostEqual(expected['max_cost'], actual['max_cost'], places=6)

if __name__ == '__main__':
    unittest.main()
