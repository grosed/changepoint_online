# `changepoint_online`


- [**A Collection of Methods for Online Changepoint
  Detection**](#a-collection-of-methods-for-online-changepoint-detection)
- [Installation](#installation)
  - [using pip](#using-pip)
- [Examples](#examples)
  - [Simple Univariate Gaussian
    Change-in-mean](#simple-univariate-gaussian-change-in-mean)
  - [Change in one-parameter exponential family
    distributions](#change-in-one-parameter-exponential-family-distributions)
  - [Non-Parametric changepoint
    detection](#non-parametric-changepoint-detection)
  - [Multivariate Data](#multivariate-data)
  - [Real-data examples](#real-data-examples)
- [License](#license)
- [GitHub Repository](#github-repository)
- [How to Cite This Work](#how-to-cite-this-work)
- [References](#references)

## **A Collection of Methods for Online Changepoint Detection**

The changepoint_online package provides efficient algorithms for
detecting changes in data streams, based on the `Focus` algorithm. The
`Focus` algorithm solves the CUSUM likelihood-ratio test exactly in
$O(\log(n))$ time per iteration, where n represents the current
iteration. The method is equivalent to running a rolling window (MOSUM)
simultaneously for all sizes of windows or the Page-CUSUM for all
possible values of the size of change (an infinitely dense grid).

**Key Features**

- Contains all `Focus` exponential family algorithms as well as the
  `NPFocus` algorithm for non-parametric changepoint detection.

- It’s versatile enough to be applied in scenarios where the pre-change
  parameter is either known or unknown.

- It is possible to apply constraints to detect specific types of
  changes (such as increases or decreases in parameter values).

## Installation

### using pip

#### installing from PyPI

    pip install changepoint-online

#### installing from github with pip

    python -m pip install 'git+https://github.com/grosed/changepoint_online/#egg=changepoint_online&subdirectory=python/package'

## Examples

### Simple Univariate Gaussian Change-in-mean

``` python
from changepoint_online import Focus, Gaussian
import numpy as np

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

print(detector.changepoint())
```

    {'stopping_time': 50013, 'changepoint': 50000}

If the pre-change location is known (in case of previous training data),
this can be specified with:

``` python
# initialize a Focus Gaussian detector (with pre-change location known)
detector = Focus(Gaussian(loc=0))
threshold = 13.0

for y in Y:
    # update your detector sequentially with
    detector.update(y)
    if detector.statistic() >= threshold:
        break

print(detector.changepoint())
```

    {'stopping_time': 50013, 'changepoint': 50000}

See `help(FamilyName)` for distribution specific parameters,
e.g. `help(Gaussian)`.

### Change in one-parameter exponential family distributions

As in the Gaussian case, we can detect:

- Poisson change-in-rate

- Gamma change-in-scale (or rate). Exponential change-in-rate
  implemented as a Gamma `shape=1`.

- Bernoulli change-in-probability

For example:

``` python
from changepoint_online import Focus, Gamma
import numpy as np 

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
        
print(detector.changepoint())
```

    {'stopping_time': 50012, 'changepoint': 50000}

### Non-Parametric changepoint detection

If we do not know the underlying distribution, or if the nature of the
change is unkown *a priori,* we can then use `NPFocus`.

``` python
from changepoint_online import NPFocus
import numpy as np

# Define a simple Gaussian noise function
def generate_gaussian_noise(size):
    return np.random.normal(loc=0.0, scale=1.0, size=size)

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

stat_over_time = []

for y in Y:
    detector.update(y)
    # we can sum the statistics over to get a detection
    # see  (Romano, Eckley, and Fearnhead 2024) for more details
    if np.sum(detector.statistic()) > 25:
        break


changepoint_info = detector.changepoint()
print(changepoint_info["stopping_time"])
```

    5014

### Multivariate Data

It is possible to run a multivariate analysis via MDFocus by feeding, at
each iteration, a `numpy` array.

#### Gaussian Change-in-mean

``` python
from changepoint_online import MDFocus, MDGaussian
import numpy as np

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
print(detector.changepoint())
```

    {'stopping_time': 5014, 'changepoint': 5000}

#### Poisson Change-in-rate

``` python
from changepoint_online import MDFocus, MDPoisson
import numpy as np

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
print(detector.changepoint())
```

    {'stopping_time': 5056, 'changepoint': 5000}

### Real-data examples

More examples, including real-world applications, are found in the
[examples](https://github.com/grosed/changepoint_online/tree/main/examples)
folder, including:

- Change in the tails of Energy Wholesale Price.
  ([markdown](https://github.com/grosed/changepoint_online/blob/main/examples/energy_wholesale.md),
  [quarto
  notebook](https://github.com/grosed/changepoint_online/blob/main/examples/energy_wholesale.qmd))

- Constrained Spike inference in calcium imaging data
  ([markdown](https://github.com/grosed/changepoint_online/blob/main/examples/constrained_spike_inference.md),
  [quarto
  notebook](https://github.com/grosed/changepoint_online/blob/main/examples/constrained_spike_inference.qmd))

## License

Copyright (C) 2023 Gaetano Romano, Daniel Grose

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see
<https://www.gnu.org/licenses/gpl-3.0.txt>.

## GitHub Repository

Source files for the packages can be found at
<https://github.com/grosed/changepoint_online>

## How to Cite This Work

A possible BibTeX entry for this package could be:

    @software{changepoint_online,
      author       = {Daniel Grose, Gaetano Romano},
      title        = {changepoint_online: A Collection of Methods for Online Changepoint Detection.},
      month        = Apr,
      year         = 2024,
      version      = {v1.0.0},
      url          = {https://https://github.com/grosed/changepoint_online}
    }

For citing the methodologies:

- Gaussian FOCuS: (Romano et al. 2023)

- Other Exponential Family detectors: (Ward et al. 2024)

- Multi-dimentional FOCuS (Pishchagina et al. 2023)

- NPFocus: (Romano, Eckley, and Fearnhead 2024)

See references below.

## References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-pishchagina2023online" class="csl-entry">

Pishchagina, Liudmila, Gaetano Romano, Paul Fearnhead, Vincent Runge,
and Guillem Rigaill. 2023. “Online Multivariate Changepoint Detection:
Leveraging Links with Computational Geometry.” *arXiv Preprint
arXiv:2311.01174*.

</div>

<div id="ref-romano2024" class="csl-entry">

Romano, Gaetano, Idris A. Eckley, and Paul Fearnhead. 2024. “A
Log-Linear Nonparametric Online Changepoint Detection Algorithm Based on
Functional Pruning.” *IEEE Transactions on Signal Processing* 72:
594–606. <https://doi.org/10.1109/tsp.2023.3343550>.

</div>

<div id="ref-romano2023fast" class="csl-entry">

Romano, Gaetano, Idris A Eckley, Paul Fearnhead, and Guillem Rigaill.
2023. “Fast Online Changepoint Detection via Functional Pruning CUSUM
Statistics.” *Journal of Machine Learning Research* 24 (81): 1–36.
<https://www.jmlr.org/papers/v24/21-1230.html>.

</div>

<div id="ref-ward2024constant" class="csl-entry">

Ward, Kes, Gaetano Romano, Idris Eckley, and Paul Fearnhead. 2024. “A
Constant-Per-Iteration Likelihood Ratio Test for Online Changepoint
Detection for Exponential Family Models.” *Statistics and Computing* 34
(3): 1–11.

</div>

</div>
