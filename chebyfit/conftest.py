"""Pytest configuration."""

import math

import numpy
import pytest


@pytest.fixture(autouse=True)
def doctest_config(doctest_namespace):
    """Add numpy to doctest namespace."""
    numpy.set_printoptions(suppress=True, precision=2)
    doctest_namespace['numpy'] = numpy
    doctest_namespace['math'] = math
