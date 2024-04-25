"""changepoint_online - A Collection of Methods for Online Changepoint Detection

The changepoint_online package provides efficient algorithms for detecting 
changes in data streams. It implements the focus algorithm for one-parameter exponential family models
including the changes-in-mean. It's versatile enough to be applied in scenarios where the pre-change parameter is either known or unknown, 
and you can even constrain it to detect specific types of changes (such as increases or decreases in parameter values).

Key Features
------------
Contains all focus exponential family algorithms as well as the NPFOCuS algorithm for non-parametric changepoint detection.
The Focus algorithm solves the CUSUM likelihood-ratio test exactly in O(log(n))
time per iteration, where n represents the current iteration.
The method is equivalent to running a rolling window (MOSUM) simultaneously 
for all sizes of windows or the Page-CUSUM for all possible values of the size of change (an infinitely dense grid).

References
-----------

    Fast online changepoint detection via functional pruning CUSUM statistics
        G Romano, IA Eckley, P Fearnhead, G Rigaill - Journal of Machine Learning Research, 2023
    A Constant-per-Iteration Likelihood Ratio Test for Online Changepoint Detection for Exponential Family Models
        K Ward, G Romano, I Eckley, P Fearnhead - Statistics and Computing, 2024
    A log-linear non-parametric online changepoint detection algorithm based on functional pruning.
        G Romano, IA Eckley, P Fearnhead, IEEE Transactions on Signal Processing, 2024

Quick Example
-------------
    ```python

    ### Simple gaussian change in mean case ###
    from changepoint_online import Focus, Gaussian
    import numpy as np

    np.random.seed(0)
    Y = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=5000), np.random.normal(loc=10.0, scale=1.0, size=5000)))

    detector = Focus(Gaussian())
    threshold = 10.0
    for y in Y:
        detector.update(y)
        if detector.statistic() >= threshold:
            break

    result = detector.changepoint()
    print(f"We detected a changepoint at time {result['stopping_time']}.")
    ```

Copyright (C) 2023 Gaetano Romano, Dan Grose

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.txt>.
"""



from .focus import Focus, Gaussian, Bernoulli, Poisson, Gamma, Exponential, NPFocus
