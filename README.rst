Fit exponential and harmonic functions using Chebyshev polynomials
==================================================================

Chebyfit is a Python library that implements the algorithms described in:

    Analytic solutions to modelling exponential and harmonic functions using
    Chebyshev polynomials: fitting frequency-domain lifetime images with
    photobleaching. G C Malachowski, R M Clegg, and G I Redford.
    J Microsc. 2007; 228(3): 282-295. doi: 10.1111/j.1365-2818.2007.01846.x

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.6.6

Requirements
------------
* `CPython >= 3.7 <https://www.python.org>`_
* `Numpy 1.15 <https://www.numpy.org>`_

Revisions
---------
2021.6.6
    Fix compile error on Python 3.10.
    Remove support for Python 3.6 (NEP 29).
2020.1.1
    Remove support for Python 2.7 and 3.5.
2019.10.14
    Support Python 3.8.
    Fix numpy 1type FutureWarning.
2019.4.22
    Fix setup requirements.
2019.1.28
    Move modules into chebyfit package.
    Add Python wrapper for _chebyfit C extension module.
    Fix static analysis issues in _chebyfit.c.

Examples
--------
Fit two-exponential decay function:

>>> deltat = 0.5
>>> t = numpy.arange(0, 128, deltat)
>>> data = 1.1 + 2.2 * numpy.exp(-t / 33.3) + 4.4 * numpy.exp(-t / 55.5)
>>> params, fitted = fit_exponentials(data, numexps=2, deltat=deltat)
>>> numpy.allclose(data, fitted)
True
>>> params['offset']
array([1.1])
>>> params['amplitude']
array([[4.4, 2.2]])
>>> params['rate']
array([[55.5, 33.3]])

Fit harmonic function with exponential decay:

>>> tt = t * (2 * math.pi / (t[-1] + deltat))
>>> data = 1.1 + numpy.exp(-t / 22.2) * (3.3 - 4.4 * numpy.sin(tt)
...                                          + 5.5 * numpy.cos(tt))
>>> params, fitted = fit_harmonic_decay(data, deltat=0.5)
>>> numpy.allclose(data, fitted)
True
>>> params['offset']
array([1.1])
>>> params['rate']
array([22.2])
>>> params['amplitude']
array([[3.3, 4.4, 5.5]])

Fit experimental time-domain image:

>>> data = numpy.fromfile('test.b&h', dtype='float32').reshape((256, 256, 256))
>>> data = data[64:64+64]
>>> params, fitted = fit_exponentials(data, numexps=1, numcoef=16, axis=0)
>>> numpy.allclose(data.sum(axis=0), fitted.sum(axis=0))
True
