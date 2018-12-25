import sys
import pytest
from numpy.testing import (
    assert_raises, assert_warns, suppress_warnings, assert_,
    assert_equal, assert_almost_equal)
import numpy as np
import numpy
from numpy.core.numeric import pickle
from functools import reduce
import textwrap
import operator
import warnings
from ArrayCollection import ArrayCollection, CollectionScalar

class TestConstruction:
    def test_simple_dict(self):
        a = ArrayCollection({'age': [8, 10, 7, 8], 'weight': [10, 20, 13, 15]})
        assert_equal(a.dtype, np.dtype([('age', int), ('weight', int)]))
        assert_equal(a['age'], [8, 10, 7, 8])
        assert_equal(a.shape, (4,))

if __name__ == '__main__':
    a = np.arange(4, dtype='u2')
    b = np.arange(4, 8, dtype='f8')
    A = ArrayCollection({'a': a, 'b': b})

    print(A[0])
    print(repr(A))

    B = ArrayCollection(np.ones((2,3), dtype='u1,S3'))
    print(repr(B.reshape((3,2))))
    print(repr(empty_collection((3,4), 'f4,u1,u1')))

    # does not work yet
    #print("Concatenate:")
    print(np.concatenate([A, A]))

