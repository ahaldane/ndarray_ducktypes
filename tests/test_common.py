import sys
import pytest
from numpy.testing import (
    assert_raises, assert_warns, suppress_warnings, assert_,
    assert_equal, assert_almost_equal)
import numpy as np
import numpy

from ndarray_ducktypes.common import get_duck_cls, ducktype_link

class Test_get_duck_cls:
    def test(self):
        class A:
            def __array_function__(self, func, types, arg, kwarg):
                pass
        class A_scalar:
            def __array_function__(self, func, types, arg, kwarg):
                pass
        ducktype_link(A, A_scalar)

        class B:
            def __array_function__(self, func, types, arg, kwarg):
                pass
        class B_scalar:
            def __array_function__(self, func, types, arg, kwarg):
                pass
        ducktype_link(B, B_scalar, known_types=(A, A_scalar))

        class C(A): pass
        class C_scalar(A_scalar): pass
        ducktype_link(C, C_scalar)
        
        nd, sc = np.array([1]), np.float64(1)
        a, asc = A(), A_scalar()
        b, bsc = B(), B_scalar()
        c, csc = C(), C_scalar()

        assert_equal(get_duck_cls(a, a), A)
        assert_equal(get_duck_cls(a, asc), A)
        assert_equal(get_duck_cls(asc, asc), A)
        assert_equal(get_duck_cls(asc, a), A)

        assert_equal(get_duck_cls(a, nd), A)
        assert_equal(get_duck_cls(a, sc), A)
        assert_equal(get_duck_cls(asc, nd), A)
        assert_equal(get_duck_cls(asc, sc), A)

        assert_equal(get_duck_cls(nd, nd), np.ndarray)
        assert_equal(get_duck_cls(nd, sc), np.ndarray)

        assert_equal(get_duck_cls(a, b), B)
        assert_equal(get_duck_cls(b, a), B)
        assert_equal(get_duck_cls(a, bsc), B)

        assert_equal(get_duck_cls(a, c), C)
        assert_equal(get_duck_cls(c, a), C)
        assert_equal(get_duck_cls(a, csc), C)

        # TODO/question:
        #assert_raises(get_duck_cls(b, c), TypeError)

        # should we not make parent classes automatically be "known_types"?
        # Eg so someone can say that they don't want their derived class
        # to be mixed with the parent class. Maybe add a kwd arg
        # know_parents=True to ducktype_link
