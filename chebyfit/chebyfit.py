# chebyfit.py

# Copyright (c) 2008-2024, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Fit exponential and harmonic functions using Chebyshev polynomials.

Chebyfit is a Python library that implements the algorithms described in:

    Analytic solutions to modelling exponential and harmonic functions using
    Chebyshev polynomials: fitting frequency-domain lifetime images with
    photobleaching. G C Malachowski, R M Clegg, and G I Redford.
    J Microsc. 2007; 228(3): 282-295. doi: 10.1111/j.1365-2818.2007.01846.x

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.4.24

Quickstart
----------

Install the chebyfit package and all dependencies from the
`Python Package Index <https://pypi.org/project/chebyfit/>`_::

    python -m pip install -U chebyfit

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/chebyfit>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.9, 3.12.3
- `NumPy <https://pypi.org/project/numpy/>`_ 1.26.4

Revisions
---------

2024.4.24

- Support NumPy 2.

2024.1.6

- Support Python 3.12.

2023.4.22

- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

2022.9.29

- Add type hints.
- Convert to Google style docstrings.

2022.8.26

- Update metadata.
- Remove support for Python 3.7 (NEP 29).

2021.6.6

- Fix compile error on Python 3.10.
- Remove support for Python 3.6 (NEP 29).

2020.1.1

- Remove support for Python 2.7 and 3.5.

2019.10.14

- Support Python 3.8.
- Fix numpy 1type FutureWarning.

2019.4.22

- Fix setup requirements.

2019.1.28

- Move modules into chebyfit package.
- Add Python wrapper for _chebyfit C extension module.
- Fix static analysis issues in _chebyfit.c.

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

"""

from __future__ import annotations

__version__ = '2024.4.24'

__all__ = [
    'fit_exponentials',
    'fit_harmonic_decay',
    'chebyshev_forward',
    'chebyshev_invers',
    'chebyshev_norm',
    'chebyshev_polynom',
    'polynom_roots',
]

import numpy

try:
    from . import _chebyfit
except ImportError:
    import _chebyfit  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

MAXEXPS = 8
MAXCOEF = 64
DEFCOEF = 32


def fit_exponentials(
    data: ArrayLike,
    numexps: int,
    deltat: float = 1.0,
    numcoef: int = DEFCOEF,
    axis: int = -1,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Fit multi-exponential functions.

    Can be used to fit time-domain fluorescence image data.

    Parameters:
        data :
            Experimental data (observed values of the dependent variable).
            Will be converted to float64.
        numexps:
            Number of exponentials to fit (1-8).
        deltat:
            Time difference in seconds between data points along time-axis
            (defines the independent variable).
        numcoef:
            Number of polynomial coefficients to use (default: 32).
        axis:
            Index of time-axis along which to fit the data.

    Returns:
        params:
            Numpy.recarray of fitting parameters:
            offset, amplitude[numexps], rate[numexps], frequency[numexps].
        fitted:
            Fitted data (predicted value of the dependent variable).
            Same shape as data.

    """
    params, fitted = _chebyfit.fitexps(data, numexps, numcoef, deltat, axis)
    if numexps == 1:
        dtype = numpy.dtype(
            [
                ('offset', 'f8'),
                ('amplitude', 'f8'),
                ('rate', 'f8'),
                ('frequency', 'f8'),
            ]
        )
    else:
        dtype = numpy.dtype(
            [
                ('offset', 'f8'),
                ('amplitude', f'{numexps}f8'),
                ('rate', f'{numexps}f8'),
                ('frequency', f'{numexps}f8'),
            ]
        )
    return params.view(dtype), fitted


def fit_harmonic_decay(
    data: ArrayLike,
    deltat: float = 1.0,
    numcoef: int = DEFCOEF,
    axis: int = -1,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Fit harmonic functions with exponential decay.

    Can be used to fit frequency-domain fluorescence image data with
    photobleaching.

    Parameters:
        data:
            Experimental data (observed values of the dependent variable).
            Will be converted to float64.
        deltat:
            Time difference in seconds between data points along time-axis
            (defines the independent variable).
        numcoef:
            Number of polynomial coefficients to use (default: 32).
        axis:
            Index of time-axis along which to fit the data.

    Returns:
        params:
            Numpy.recarray of fitting parameters: offset, rate, amplitude[3].
        fitted:
            Fitted data (predicted value of the dependent variable).
            Same shape as data.

    """
    params, fitted = _chebyfit.fitexpsin(data, numcoef, deltat, axis)
    dtype = numpy.dtype(
        [('offset', 'f8'), ('rate', 'f8'), ('amplitude', '3f8')]
    )
    return params.view(dtype), fitted


def chebyshev_forward(
    data: ArrayLike, numcoef: int = DEFCOEF
) -> numpy.ndarray:
    """Return coefficients dj of forward Chebyshev transform from data.

    >>> data = 1.1 + 2.2 * numpy.exp(-numpy.arange(32) / 3.3)
    >>> chebyshev_forward(data, 16)[:8]
    array([1.36, 0.61, 0.61, 0.41, 0.2 , 0.08, 0.03, 0.01])

    """
    return _chebyfit.chebyfwd(data, numcoef)


def chebyshev_invers(coef: ArrayLike, numdata: int) -> numpy.ndarray:
    """Return reconstructed data from Chebyshev coefficients dj.

    >>> data = 1.1 + 2.2 * numpy.exp(-numpy.arange(32) / 3.3)
    >>> data2 = chebyshev_invers(chebyshev_forward(data, 16), len(data))
    >>> numpy.allclose(data, data2)
    True

    """
    return _chebyfit.chebyinv(coef, numdata)


def chebyshev_norm(numdata: int, numcoef: int = DEFCOEF) -> numpy.ndarray:
    """Return Chebyshev polynomial normalization factors Rj.

    >>> chebyshev_norm(4, 4)
    array([ 4.  ,  2.22,  4.  , 20.  ])

    """
    return _chebyfit.chebynorm(numdata, numcoef)


def chebyshev_polynom(
    numdata: int, numcoef: int = DEFCOEF, norm: bool = False
) -> numpy.ndarray:
    """Return Chebyshev polynomials Tj(t) / Rj.

    >>> chebyshev_polynom(numdata=4, numcoef=2, norm=False)
    array([[ 1.  ,  1.  ,  1.  ,  1.  ],
           [ 1.  ,  0.33, -0.33, -1.  ]])

    """
    return _chebyfit.chebypoly(numdata, numcoef, norm)


def polynom_roots(coeffs: ArrayLike) -> numpy.ndarray:
    """Return complex roots of complex polynomial using Laguerre's method.

    Complex polynomial coefficients ordered from smallest to largest power.

    >>> polynom_roots([-250., 155., -9., -5., 1.])
    array([ 2.+0.j,  4.-3.j,  4.+3.j, -5.+0.j])

    """
    return _chebyfit.polyroots(coeffs)


if __name__ == '__main__':
    import doctest
    import math  # noqa: required by doctests
    import os

    try:
        os.chdir('tests')
    except Exception:
        pass
    numpy.set_printoptions(suppress=True, precision=2)
    doctest.testmod(optionflags=doctest.ELLIPSIS)
