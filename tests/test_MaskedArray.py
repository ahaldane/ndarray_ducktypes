import sys
import pickle
import pytest
from functools import reduce
import textwrap
import operator
import warnings

from numpy.testing import (
    assert_raises, assert_warns, suppress_warnings, assert_,
    assert_equal, assert_almost_equal)
import numpy as np
import numpy

from ndarray_ducktypes.MaskedArray import (MaskedArray, MaskedScalar, X,
    replace_X, _minvals, _maxvals)
from ndarray_ducktypes.common import ducktype_link

pi = np.pi

################################################################################
#                         MaskedArray Testing setup
################################################################################

# For parametrized numeric testing
num_dts = [np.dtype(dt_) for dt_ in '?bhilqBHILQefdgFD']
num_ids = [dt_.char for dt_ in num_dts]

def getdata(a):
    if isinstance(a, (MaskedArray, MaskedScalar)):
        return a.filled()
    return a

def getmask(a):
    if isinstance(a, (MaskedArray, MaskedScalar)):
        return a.mask
    return np.bool_(False)

# XXX make a less hacky version of this
def assert_masked_equal(actual, desired, err_msg='', anymask=False):
    __tracebackhide__ = True  # Hide traceback for py.test

    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_masked_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError("%s not in %s" % (k, actual))
            assert_masked_equal(actual[k], desired[k],
                                'key=%r\n%s' % (k, err_msg))
    elif (isinstance(desired, (list, tuple)) and
            isinstance(actual, (list, tuple))):
        assert_masked_equal(len(actual), len(desired), err_msg)
        for k in range(len(desired)):
            assert_masked_equal(actual[k], desired[k],
                                'item=%r\n%s' % (k, err_msg))
    else:
        dx, dy = getdata(actual), getdata(desired)
        mx, my = getmask(actual), getmask(desired)

        if anymask:
            dx = dx.copy()
            m = mx | my
            dx[m] = dy[m]
            assert_equal(dx, dy, err_msg)
        else:
            assert_equal(dx, dy, err_msg)
            assert_equal(mx, my, err_msg)

def assert_almost_masked_equal(actual, desired, err_msg='', anymask=False):
    __tracebackhide__ = True  # Hide traceback for py.test

    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_almost_masked_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError("%s not in %s" % (k, actual))
            assert_almost_masked_equal(actual[k], desired[k],
                                'key=%r\n%s' % (k, err_msg))
    elif (isinstance(desired, (list, tuple)) and
            isinstance(actual, (list, tuple))):
        assert_masked_equal(len(actual), len(desired), err_msg)
        for k in range(len(desired)):
            assert_almost_masked_equal(actual[k], desired[k],
                                'item=%r\n%s' % (k, err_msg))
    else:
        dx, dy = getdata(actual), getdata(desired)
        mx, my = getmask(actual), getmask(desired)

        if anymask:
            dx = dx.copy()
            m = mx | my
            dx[m] = dy[m]
            assert_almost_equal(dx, dy, err_msg=err_msg)
        else:
            assert_almost_equal(dx, dy, err_msg=err_msg)
            assert_equal(mx, my, err_msg)

################################################################################
#                         Tests ported from numpy.ma
################################################################################
# includes some new tests too

class TestMaskedArray:
    # Base test class for MaskedArrays.

    def setup(self):
        # Base data definition.
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        m2 = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        xm = MaskedArray(x.copy(), mask=m1.copy())
        ym = MaskedArray(y.copy(), mask=m2.copy())
        z = np.array([-.5, 0., .5, .8])
        zm = MaskedArray(z.copy(), mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)

    def test_basicattributes(self):
        # Tests some basic array attributes.
        a = MaskedArray([1, 3, 2])
        b = MaskedArray([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 1)
        assert_equal(a.size, 3)
        assert_equal(b.size, 3)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (3,))

    def test_basic0d(self):
        # Checks masking a scalar
        x = MaskedArray(0)
        assert_equal(str(x), '0')
        x = MaskedArray(0, mask=True)
        assert_equal(str(x), 'X')
        x = MaskedArray(0, mask=False)
        assert_equal(str(x), '0')
        x = MaskedArray(0, mask=1)
        assert_(x.filled().dtype is x._data.dtype)

    def test_basic1d(self):
        # Test of basic array creation and properties in 1 dimension.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_((xm - ym).filled(0).any())
        assert_((xm.mask.astype(int) != ym.mask.astype(int)).any())
        s = x.shape
        assert_equal(np.shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(zm.dtype, z.dtype)
        assert_equal(xm.size, reduce(lambda x, y:x * y, s))
        assert_equal(xm.count(), len(m1) - reduce(lambda x, y:x + y, m1))
        assert_equal(xm.filled(1.e20), xf)

    def test_basic2d(self):
        # Test of basic array creation and properties in 2 dimensions.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        cnt = len(m1) - reduce(lambda x, y:x + y, m1)
        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s
            m1.shape = s

            assert_equal(np.shape(xm), s)
            assert_equal(xm.shape, s)
            assert_equal(xm.size, reduce(lambda x, y:x * y, s))
            assert_equal(xm.count(), cnt)
            assert_masked_equal(xm, MaskedArray(xf.copy(), m1))
            assert_masked_equal(xm, MaskedArray(x.copy(), m1))
            assert_equal(xm.filled(1.e20), xf)

    def test_concatenate_basic(self):
        # Tests concatenations.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # basic concatenation
        assert_equal(MaskedArray(np.concatenate((x, y)),
                                 np.concatenate((m1, m2))),
                                 np.concatenate((xm, ym)))
        assert_equal(MaskedArray(np.concatenate((x, y)),
                                 np.concatenate((m1, np.zeros_like(m2)))),
                                 np.concatenate((xm, y)))
        z1 = np.zeros_like(m1)
        assert_equal(MaskedArray(np.concatenate((x, y, x)),
                                 np.concatenate((z1, m2, z1))),
                                 np.concatenate((x, ym, x)))

    def test_concatenate_alongaxis(self):
        # Tests concatenations.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # Concatenation along an axis
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = m1.shape = m2.shape = s
        assert_equal(xm.mask, np.reshape(m1, s))
        assert_equal(ym.mask, np.reshape(m2, s))
        xmym = np.concatenate((xm, ym), 1)
        assert_masked_equal(MaskedArray(np.concatenate((x, y), 1),
                                        np.concatenate((m1,m2),1)), xmym)
        assert_masked_equal(np.concatenate((xm.mask, ym.mask), 1), xmym.mask)

        x = MaskedArray(np.zeros(2))
        y = MaskedArray(np.ones(2), mask=[False, True])
        z = np.concatenate((x, y))
        assert_masked_equal(z, MaskedArray([0, 0, 1, X]))
        z = np.concatenate((y, x))
        assert_masked_equal(z, MaskedArray([1, X, 0, 0]))

    def test_concatenate_flexible(self):
        # Tests the concatenation on flexible arrays.
        data = MaskedArray(list(zip(np.random.rand(10),
                                     np.arange(10))),
                            dtype=[('a', float), ('b', int)])

        test = np.concatenate([data[:5], data[5:]])
        assert_equal(test.filled(0), data.filled(0))
        # XXX: this used to be assert_equal(test, data),
        # but there is a bug in np.equal for structured arrays exposed by
        # NDArrayOperatorMixin.

    def test_creation_ndmin(self):
        # Check the use of ndmin
        x = MaskedArray([1, 2, 3], mask=[1, 0, 0], ndmin=2)
        assert_equal(x.shape, (1, 3))
        assert_equal(x._data, [[1, 2, 3]])
        assert_equal(x._mask, [[1, 0, 0]])

    def test_creation_ndmin_from_maskedarray(self):
        # Make sure we're not losing the original mask w/ ndmin
        x = MaskedArray([1, 2, 3])
        x[-1] = X
        xx = MaskedArray(x, ndmin=2, dtype=float)
        assert_equal(x.shape, x._mask.shape)
        assert_equal(xx.shape, xx._mask.shape)

    def test_creation_maskcreation(self):
        # Tests how masks are initialized at the creation of Maskedarrays.
        data = MaskedArray(np.arange(24, dtype=float))
        data[[3, 6, 15]] = X
        dma_1 = MaskedArray(data)
        assert_equal(dma_1.mask, data.mask)
        dma_2 = MaskedArray(dma_1)
        assert_equal(dma_2.mask, dma_1.mask)
        dma_3 = MaskedArray(dma_1.filled(0), mask=[1, 0, 0, 0] * 6)
        assert_((dma_3.mask != dma_1.mask).any())

        x = MaskedArray([1, 2, 3], mask=True)
        assert_equal(x.mask, [True, True, True])
        x = MaskedArray([1, 2, 3], mask=False)
        assert_equal(x.mask, [False, False, False])
        y = MaskedArray([1, 2, 3], mask=x.mask, copy=False)
        # XXX todo: may_share_memory
        #assert_(not np.may_share_memory(x.mask, y.mask))
        #y = MaskedArray([1, 2, 3], mask=x.mask, copy=True)
        #assert_(not np.may_share_memory(x.mask, y.mask))

    def test_creation_with_list_of_maskedarrays(self):
        # Tests creating a masked array from a list of masked arrays.
        x = MaskedArray(np.arange(5), mask=[1, 0, 0, 0, 0])
        data = MaskedArray([x, x[::-1]])
        assert_equal(data.filled(9), [[9, 1, 2, 3, 4], [4, 3, 2, 1, 9]])
        assert_equal(data.mask, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

    def test_creation_from_ndarray_with_padding(self):
        x = np.array([('A', 0)], dtype={'names':['f0','f1'],
                                        'formats':['S4','i8'],
                                        'offsets':[0,8]})
        data = MaskedArray(x)
        # used to fail due to 'V' padding field in x.dtype.descr

    def test_maskedelement(self):
        # Test of masked element
        x = MaskedArray(np.arange(6))
        x[1] = X
        assert_equal(x[1].mask, True)
        assert_equal(x.filled(0)[1], 0)

    def test_set_element_as_object(self):
        # Tests setting elements with object
        a = MaskedArray(np.empty(1, dtype=object))
        x = (1, 2, 3, 4, 5)
        a[0] = x
        assert_equal(a[0].filled(), x)
        assert_(a[0].filled() is x)

        import datetime
        dt = datetime.datetime.now()
        a[0] = dt
        assert_(a[0].filled() is dt)

    def test_indexing(self):
        # Tests conversions and indexing
        x1 = np.array([1, 2, 4, 3])
        x2 = MaskedArray(x1, mask=[1, 0, 0, 0])
        x3 = MaskedArray(x1, mask=[0, 1, 0, 1])
        x4 = MaskedArray(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        assert_masked_equal(np.sort(x1)[1:], np.sort(x2)[:-1])
        # tests of indexing
        assert_(type(x2[1].filled()) is type(x1[1]))
        assert_((x1[1] == x2[1]).filled(True).all())
        assert_(x2[0].mask == True)
        assert_((x1[2] == x2[2]).filled(True).all())
        assert_((x1[2:5] == x2[2:5]).filled(True).all())
        assert_((x1[:] == x2[:]).filled(True).all())
        assert_((x1[1:] == x3[1:]).filled(True).all())
        x1[2] = 9
        x2[2] = 9
        assert_((x1 == x2).filled(True).all())
        x1[1:3] = 99
        x2[1:3] = 99
        assert_((x1 == x2).filled(True).all())
        x2[1] = X
        assert_((x1 == x2).filled(True).all())
        x2[1:3] = X
        assert_((x1 == x2).filled(True).all())
        x2[:] = x1
        x2[1] = X
        assert_equal(x2.mask, np.array([0, 1, 0, 0]))
        x3[:] = MaskedArray([1, 2, 3, 4], [0, 1, 1, 0])
        assert_equal(x3.mask, np.array([0, 1, 1, 0]))
        x4[:] = MaskedArray([1, 2, 3, 4], [0, 1, 1, 0])
        assert_equal(x4.mask, np.array([0, 1, 1, 0]))
        assert_((x4 == np.array([1, 2, 3, 4])).filled(True).all())

        x1 = MaskedArray([1, 'hello', 2, 3], dtype=object)
        x2 = np.array([1, 'hello', 2, 3], dtype=object)
        s1 = x1[1].filled()
        s2 = x2[1]
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_masked_equal(s1, s2)
        assert_(x1[1:1].shape == (0,))

    def test_copy(self):
        # Tests of some subtle points of copying and sizing.
        n = np.array([0, 0, 1, 0, 0], dtype=bool)

        x1 = np.arange(5)
        m = n.copy()
        y1 = MaskedArray(x1, mask=m)
        assert_equal(y1._data.__array_interface__, x1.__array_interface__)
        assert_equal(x1, y1._data)
        assert_equal(y1._mask.__array_interface__, m.__array_interface__)

        y1a = MaskedArray(y1)
        # Default for masked array is not to copy; see gh-10318.
        assert_(y1a._data.__array_interface__ ==
                        y1._data.__array_interface__)
        assert_(y1a._mask.__array_interface__ ==
                        y1._mask.__array_interface__)

        m = n.copy()
        y2 = MaskedArray(x1, mask=m)
        assert_(y2._data.__array_interface__ == x1.__array_interface__)
        assert_(y2._mask.__array_interface__ == m.__array_interface__)
        assert_(y2[2].mask)
        y2[2] = 9
        assert_(not y2[2].mask)
        assert_(y2._mask.__array_interface__ == m.__array_interface__)
        assert_((y2.mask ==  0).all())

        m = n.copy()
        y2a = MaskedArray(x1, mask=m, copy=1)
        assert_(y2a._data.__array_interface__ != x1.__array_interface__)
        #assert_( y2a.mask is not m)
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        assert_(y2a[2].mask)
        y2a[2] = 9
        assert_(not y2a[2].mask)
        #assert_( y2a.mask is not m)
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        assert_((y2a.mask == 0).all())

        y3 = MaskedArray(x1 * 1.0, mask=m)
        assert_(y3.filled().dtype is (x1 * 1.0).dtype)

        x4 = MaskedArray(np.arange(4))
        x4[2] = X
        y4 = np.resize(x4, (8,))
        assert_equal(np.concatenate([x4, x4]), y4)
        assert_equal(y4.mask, [0, 0, 1, 0, 0, 0, 1, 0])
        y5 = np.repeat(x4, (2, 2, 2, 2), axis=0)
        assert_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        y6 = np.repeat(x4, 2, axis=0)
        assert_equal(y5, y6)
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        assert_equal(y5, y7)
        y8 = x4.repeat(2, 0)
        assert_equal(y5, y8)

        y9 = x4.copy()
        assert_equal(y9._data, x4._data)
        assert_equal(y9._mask, x4._mask)

        x = MaskedArray([1, 2, 3], mask=[0, 1, 0])
        # Copy is False by default
        y = MaskedArray(x)
        assert_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_equal(y._mask.ctypes.data, x._mask.ctypes.data)
        y = MaskedArray(x, copy=True)
        assert_(y._data.ctypes.data != x._data.ctypes.data)
        assert_(y._mask.ctypes.data != x._mask.ctypes.data)

    def test_copy_0d(self):
        # gh-9430
        x = MaskedArray(43, mask=True)
        xc = x.copy()
        assert_equal(xc.mask, True)

    def test_copy_immutable(self):
        # Tests that the copy method is immutable, GitHub issue #5247
        a = np.ma.array([1, 2, 3])
        b = np.ma.array([4, 5, 6])
        a_copy_method = a.copy
        b.copy
        assert_equal(a_copy_method(), [1, 2, 3])

    def test_deepcopy(self):
        from copy import deepcopy
        a = MaskedArray([0, 1, 2], mask=[False, True, False])
        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        assert_(id(a._mask) != id(copied._mask))

        copied[1] = 1
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        copied[1] = 0
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

    def test_mask_readonly_view(self):
        a = MaskedArray([[1,X,3], [X,4,X], [1,X,6]], dtype='u4')
        m = a.mask
        assert_raises(ValueError, m.__setitem__, (0,0), 1)
        a[0,0] = X
        assert_equal(m[0,0], True)

    def test_str_repr(self):
        a = MaskedArray([0, 1, 2], mask=[False, True, False])
        assert_equal(str(a), '[0 X 2]')
        assert_equal(repr(a), 'MaskedArray([0, X, 2])')

        # arrays with a continuation
        a = MaskedArray(np.arange(2000))
        a[1:50] = X
        assert_equal(repr(a),
            'MaskedArray([   0,    X,    X, ..., 1997, 1998, 1999])')

        # line-wrapped 1d arrays are correctly aligned
        a = MaskedArray(np.arange(20))
        assert_equal(
            repr(a),
            textwrap.dedent('''\
        MaskedArray([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19])'''))

        # 2d arrays cause wrapping
        a = MaskedArray([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        a[1,1] = X
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            MaskedArray([[1, 2, 3],
                         [4, X, 6]], dtype=int8)''')
        )

        # but not it they're a row vector
        assert_equal( repr(a[:1]), 'MaskedArray([[1, 2, 3]], dtype=int8)')

        # dtype=int is implied, so not shown
        assert_equal(
            repr(a.astype(int)),
            textwrap.dedent('''\
            MaskedArray([[1, 2, 3],
                         [4, X, 6]])''')
        )

    def test_0d_unicode(self):
        u = u'caf\xe9'
        utype = type(u)

        arr_nomask = MaskedArray(u)
        arr_masked = MaskedArray(u, mask=True)

        assert_equal(utype(arr_nomask), u)
        assert_equal(utype(arr_masked), u'X')

    def test_pickling(self):
        # Tests pickling
        for dtype in (int, float, str, object):
            dat = np.arange(10).astype(dtype)

            masks = ([0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # partially masked
                     True,                            # Fully masked
                     False)                           # Fully unmasked

            for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
                for mask in masks:
                    a = MaskedArray(dat, mask)
                    a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
                    assert_equal(a_pickled._mask, a._mask)
                    assert_equal(a_pickled._data, a._data)
                    assert_equal(a_pickled.mask, mask)
# XXX
#    def test_pickling_subbaseclass(self):
#        # Test pickling w/ a subclass of ndarray
#        x = np.array([(1.0, 2), (3.0, 4)],
#                     dtype=[('x', float), ('y', int)]).view(np.recarray)
#        a = MaskedArray(x, mask=[(True, False), (False, True)])
#        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
#            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
#            assert_equal(a_pickled._mask, a._mask)
#            assert_equal(a_pickled, a)
#            assert_(isinstance(a_pickled._data, np.recarray))

    def test_pickling_wstructured(self):
        # Tests pickling w/ structured array
        a = MaskedArray([(1, 1.), (2, 2.)], mask=[0, 1],
                  dtype=[('a', int), ('b', float)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)

    def test_pickling_keepalignment(self):
        # Tests pickling w/ F_CONTIGUOUS arrays
        a = MaskedArray(np.arange(10))
        a.shape = (-1, 2)
        b = a.T
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            test = pickle.loads(pickle.dumps(b, protocol=proto))
            assert_equal(test, b)

    def test_single_element_subscript(self):
        # Tests single element subscripts of Maskedarrays.
        a = MaskedArray([1, 3, 2])
        b = MaskedArray([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a[0].shape, ())
        assert_equal(b[0].shape, ())
        assert_equal(b[1].shape, ())

    def test_topython(self):
        # Tests some communication issues with Python.
        assert_equal(1, int(MaskedArray(1).filled()))
        assert_equal(1.0, float(MaskedArray(1).filled()))
        assert_equal(1, int(MaskedArray([[[1]]]).filled()))
        assert_equal(1.0, float(MaskedArray([[1]]).filled()))
        assert_raises(TypeError, float, MaskedArray([1, 1]).filled())

        a = MaskedArray([1, 2, 3], mask=[1, 0, 0])
        assert_raises(TypeError, lambda: float(a))
        assert_equal(float(a[-1].filled()), 3.)
        assert_raises(TypeError, int, a)
        assert_equal(int(a[-1].filled()), 3)

    def test_oddfeatures_1(self):
        # Test of other odd features
        x = MaskedArray(np.arange(20))
        x = x.reshape((4, 5))
        z = x + 10j * x
        assert_equal(z.real, x)
        assert_equal(z.imag, 10 * x)
        assert_equal((z * np.conjugate(z)).real, 101 * x * x)
        z.imag[...] = 0.0

        x = MaskedArray(np.arange(10))
        x[3] = X
        assert_(str(x[3]) == 'X')
        c = x >= 8
        assert_(np.where(c, X(int), X).count() == 0)
        assert_(np.where(c, X(int), X).shape == c.shape)

    def test_oddfeatures_2(self):
        # Tests some more features.
        x = MaskedArray([1., 2., 3., 4., 5.])
        c = MaskedArray([1, 1, 1, 0, 0])
        x[2] = X
        z = np.where(c.filled(), x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = X
        z = np.where(c.filled(), x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(not z[0].mask)
        assert_(not z[1].mask)
        assert_(z[2].mask)

    def test_oddfeatures_3(self):
        # Tests some generic features
        atest = MaskedArray([10], mask=True)
        btest = MaskedArray([20])
        idx = atest.mask
        atest[idx] = btest[idx]
        assert_equal(atest.filled(), [20])

    def test_filled_with_object_dtype(self):
        a = np.ma.masked_all(1, dtype='O')
        assert_equal(a.filled('x')[0], 'x')

    def test_filled_with_flexible_dtype(self):
        # Test filled w/ flexible dtype
        flexi = MaskedArray([(1, 1, 1)],
                      dtype=[('i', int), ('s', '|S8'), ('f', float)])
        flexi[0] = X
        assert_equal(flexi.filled(), np.array([0], dtype=flexi.dtype))
        assert_equal(flexi.filled(1), np.array([1], dtype=flexi.dtype))

    def test_filled_with_mvoid(self):
        # Test filled w/ mvoid
        ndtype = [('a', int), ('b', float)]
        a = MaskedScalar((1, 2.), mask=True, dtype=ndtype)
        # Filled using default
        test = a.filled()
        assert_equal(tuple(test), (0, 0))
        # Explicit fill_value
        test = a.filled((-1, -1))
        assert_equal(tuple(test), (-1, -1))

    def test_filled_with_f_order(self):
        # Test filled w/ F-contiguous array
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "order parameter of MaskedArray")
            a = MaskedArray(np.array([(0, 1, 2), (4, 5, 6)], order='F'),
                      mask=np.array([(0, 0, 1), (1, 0, 0)], order='F'),
                      order='F')  # this is currently ignored
        assert_(a._data.flags['F_CONTIGUOUS'])
        assert_(a.filled(0).flags['F_CONTIGUOUS'])

    def test_fancy_printoptions(self):
        # Test printing a masked array w/ fancy dtype.
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = MaskedArray([(1, (2, 3.0)), (4, (5, 6.0))],
                           mask=[0, 1], dtype=fancydtype)
        control = "[(1, (2, 3.))            X]"
        assert_equal(str(test), control)

        # Test 0-d array with multi-dimensional dtype
        t_2d0 = MaskedArray((0, [[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]],
                              0.0),
                             mask = False,
                             dtype = "int, (2,3)float, float")
        control = "(0, [[0., 0., 0.], [0., 0., 0.]], 0.)"
        assert_equal(str(t_2d0), control)

    def test_void0d(self):
        # Test creating a mvoid object
        ndtype = [('a', int), ('b', int)]
        a = np.array([(1, 2,)], dtype=ndtype)[0]
        f = MaskedScalar(a)
        assert_(isinstance(f, MaskedScalar))
        assert_(isinstance(f.filled(), np.void))

    def test_mvoid_getitem(self):
        # Test mvoid.__getitem__
        ndtype = [('a', int), ('b', int)]
        a = MaskedArray([(1, 2,), (3, 4)], mask=[0, 1], dtype=ndtype)
        # w/o mask
        f = a[0].filled()
        assert_(isinstance(f, np.void))
        assert_equal((f[0], f['a']), (1, 1))
        assert_equal(f['b'], 2)
        # w/ mask
        f = a[1]
        assert_(isinstance(f.filled(), np.void))
        assert_(f[0].mask)
        assert_(f['a'].mask)

        #XXX not sure how to deal with subarrays yet
        ## exotic dtype
        #A = MaskedArray([([0,1],),([0,1],)],
        #                mask=[([True, False],)],
        #                dtype=[("A", ">i2", (2,))])
        #assert_equal(A[0]["A"], A["A"][0])
        #assert_equal(A[0]["A"], MaskedArray([0, 1],
        #                 mask=[True, False], dtype=">i2"))

    #def test_mvoid_iter(self):
    #    # Test iteration on __getitem__
    #    ndtype = [('a', int), ('b', int)]
    #    a = MaskedArray([(1, 2,), (3, 4)], mask=[0, 1],
    #                     dtype=ndtype)
    #    # w/o mask
    #    assert_equal(list(a[0].filled()), [1, 2])
    #    # w/ mask
    #    assert_equal(list(a[1]), [masked, 4])

    #def test_mvoid_print(self):
    #    # Test printing a mvoid
    #    mx = MaskedArray([(1, 1), (2, 2)], dtype=[('a', int), ('b', int)])
    #    assert_equal(str(mx[0]), "(1, 1)")
    #    mx['b'][0] = X
    #    ini_display = masked_print_option._display
    #    masked_print_option.set_display("-X-")
    #    try:
    #        assert_equal(str(mx[0]), "(1, -X-)")
    #        assert_equal(repr(mx[0]), "(1, -X-)")
    #    finally:
    #        masked_print_option.set_display(ini_display)

    #    # also check if there are object datatypes (see gh-7493)
    #    mx = MaskedArray([(1,), (2,)], dtype=[('a', 'O')])
    #    assert_equal(str(mx[0]), "(1,)")

#    def test_mvoid_multidim_print(self):

#        # regression test for gh-6019
#        t_ma = MaskedArray(data = [([1, 2, 3],)],
#                            mask = [([False, True, False],)],
#                            fill_value = ([999999, 999999, 999999],),
#                            dtype = [('a', '<i4', (3,))])
#        assert_(str(t_ma[0]) == "([1, --, 3],)")
#        assert_(repr(t_ma[0]) == "([1, --, 3],)")

#        # additional tests with structured arrays

#        t_2d = MaskedArray(data = [([[1, 2], [3,4]],)],
#                            mask = [([[False, True], [True, False]],)],
#                            dtype = [('a', '<i4', (2,2))])
#        assert_(str(t_2d[0]) == "([[1, --], [--, 4]],)")
#        assert_(repr(t_2d[0]) == "([[1, --], [--, 4]],)")

#        t_0d = MaskedArray(data = [(1,2)],
#                            mask = [(True,False)],
#                            dtype = [('a', '<i4'), ('b', '<i4')])
#        assert_(str(t_0d[0]) == "(--, 2)")
#        assert_(repr(t_0d[0]) == "(--, 2)")

#        t_2d = MaskedArray(data = [([[1, 2], [3,4]], 1)],
#                            mask = [([[False, True], [True, False]], False)],
#                            dtype = [('a', '<i4', (2,2)), ('b', float)])
#        assert_(str(t_2d[0]) == "([[1, --], [--, 4]], 1.0)")
#        assert_(repr(t_2d[0]) == "([[1, --], [--, 4]], 1.0)")

#        t_ne = MaskedArray(data=[(1, (1, 1))],
#                            mask=[(True, (True, False))],
#                            dtype = [('a', '<i4'), ('b', 'i4,i4')])
#        assert_(str(t_ne[0]) == "(--, (--, 1))")
#        assert_(repr(t_ne[0]) == "(--, (--, 1))")

    def test_object_with_array(self):
        mx1 = MaskedArray([1.], mask=[True])
        mx2 = MaskedArray([1., 2.])
        mx = MaskedArray([mx1, mx2], mask=[False, True], dtype='O')
        assert_(mx[0].filled() is mx1)
        assert_(mx[1].filled() is not mx2)
        assert_(np.all(mx[1]._data == mx2._data))
        assert_(np.all(mx[1].mask))
        # check that we return a view.
        z = mx[0].filled()
        z[0] = 0.
        assert_(mx1[0].filled() == 0.)


class TestMaskedArrayArithmetic:
    # Base test class for MaskedArrays.

    def setup(self):
        # Base data definition.
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        m2 = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        xm = MaskedArray(x.copy(), mask=m1)
        ym = MaskedArray(y.copy(), mask=m2)
        z = np.array([-.5, 0., .5, .8])
        zm = MaskedArray(z.copy(), mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown(self):
        np.seterr(**self.err_status)

    def test_basic_arithmetic(self):
        # Test of basic arithmetic.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        a2d = MaskedArray([[1, 2], [0, 4]])
        a2dm = MaskedArray(a2d.filled(), [[0, 0], [1, 0]])
        assert_equal(a2d * a2d, a2d * a2dm)
        assert_equal(a2d + a2d, a2d + a2dm)
        assert_equal(a2d - a2d, a2d - a2dm)
        for s in [(12,), (4, 3), (2, 6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            m1 = m1.reshape(s)
            m2 = m2.reshape(s)
            m12 = m1 | m2
            assert_masked_equal(MaskedArray(-x, m1), -xm)
            assert_masked_equal(MaskedArray(x + y, m12), xm + ym)
            assert_masked_equal(MaskedArray(x - y, m12), xm - ym)
            assert_masked_equal(MaskedArray(x * y, m12), xm * ym)
            assert_masked_equal(MaskedArray(x / y, m12), xm / ym, anymask=True)
            assert_masked_equal(MaskedArray(a10 + y, m2), a10 + ym)
            assert_masked_equal(MaskedArray(a10 - y, m2), a10 - ym)
            assert_masked_equal(MaskedArray(a10 * y, m2), a10 * ym)
            assert_masked_equal(MaskedArray(a10 / y, m2), a10 / ym,anymask=True)
            assert_masked_equal(MaskedArray(x + a10, m1), xm + a10)
            assert_masked_equal(MaskedArray(x - a10, m1), xm - a10)
            assert_masked_equal(MaskedArray(x * a10, m1), xm * a10)
            assert_masked_equal(MaskedArray(x / a10, m1), xm / a10,anymask=True)
            assert_masked_equal(MaskedArray(x ** 2, m1), xm ** 2)
            assert_masked_equal(MaskedArray(abs(x) ** 2.5, m1), abs(xm) ** 2.5)
            assert_masked_equal(MaskedArray(x ** y, m12), xm ** ym)
            assert_masked_equal(MaskedArray(np.add(x, y), m12), np.add(xm, ym))
            assert_masked_equal(MaskedArray(np.subtract(x, y), m12),
                                            np.subtract(xm, ym))
            assert_masked_equal(MaskedArray(np.multiply(x, y), m12),
                                            np.multiply(xm, ym))
            assert_masked_equal(MaskedArray(np.divide(x, y), m12),
                                            np.divide(xm, ym), anymask=True)

    def test_divide_on_different_shapes(self):
        x = MaskedArray(np.arange(6, dtype=float))
        x[0] = X
        x.shape = (2, 3)
        y = MaskedArray(np.arange(3, dtype=float))
        y[0] = X

        z = x / y
        assert_masked_equal(z, MaskedArray([[X, 1., 1.], [X, 4., 2.5]]))

        z = x / y[None,:]
        assert_masked_equal(z, MaskedArray([[X, 1., 1.], [X, 4., 2.5]]))

        y = MaskedArray(np.arange(2, dtype=float))
        y[0] = X
        z = x / y[:, None]
        assert_masked_equal(z, MaskedArray([[X, X, X], [3., 4., 5.]]))

    def test_masked_singleton_arithmetic(self):
        # Tests some scalar arithmetics on MaskedArrays.
        # Masked singleton should remain masked no matter what
        xm = MaskedArray(0, mask=1)
        assert_(not (1 / MaskedArray(0)).mask)
        assert_((1 + xm).mask)
        assert_((-xm).mask)
        assert_(np.maximum(xm, xm).mask)
        assert_(np.minimum(xm, xm).mask)

    def test_masked_singleton_equality(self):
        # Tests (in)equality on masked singleton
        a = MaskedArray([1, 2, 3], mask=[1, 1, 0])
        assert_((a[0] == 0).mask)
        assert_((a[0] != 0).mask)
        assert_equal((a[-1] == 0).filled(), False)
        assert_equal((a[-1] != 0).filled(), True)

    def test_arithmetic_with_masked_singleton(self):
        # Checks that there's no collapsing to masked
        x = MaskedArray([1, 2])
        y = x * X
        assert_equal(y.shape, x.shape)
        assert_equal(y.mask, [True, True])
        y = x[0] * X
        assert_(y.mask)
        y = x + X
        assert_equal(y.shape, x.shape)
        assert_equal(y.mask, [True, True])

    def test_arithmetic_with_masked_singleton_on_1d_singleton(self):
        # Check that we're not losing the shape of a singleton
        x = MaskedArray([1, ])
        y = x + X
        assert_equal(y.shape, x.shape)
        assert_equal(y.mask, [True, ])

    def test_basic_ufuncs(self):
        # Test various functions such as sin, cos.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_masked_equal(MaskedArray(np.cos(x), m1), np.cos(xm))
        assert_masked_equal(MaskedArray(np.cosh(x), m1), np.cosh(xm))
        assert_masked_equal(MaskedArray(np.sin(x), m1), np.sin(xm))
        assert_masked_equal(MaskedArray(np.sinh(x), m1), np.sinh(xm))
        assert_masked_equal(MaskedArray(np.tan(x), m1), np.tan(xm))
        assert_masked_equal(MaskedArray(np.tanh(x), m1), np.tanh(xm))
        assert_masked_equal(MaskedArray(np.sqrt(np.abs(x)), m1),
                                        np.sqrt(np.abs(xm)))
        assert_masked_equal(MaskedArray(np.log(np.abs(x)), m1),
                                        np.log(np.abs(xm)))
        assert_masked_equal(MaskedArray(np.log10(np.abs(x)), m1),
                                        np.log10(np.abs(xm)))
        assert_masked_equal(MaskedArray(np.exp(x), m1), np.exp(xm))
        mz = [0, 1, 0, 0]
        assert_masked_equal(MaskedArray(np.arcsin(z), mz), np.arcsin(zm))
        assert_masked_equal(MaskedArray(np.arccos(z), mz), np.arccos(zm))
        assert_masked_equal(MaskedArray(np.arctan(z), mz), np.arctan(zm))
        assert_masked_equal(MaskedArray(np.arctan2(x, y), m1 | m2),
                                        np.arctan2(xm, ym))
        assert_masked_equal(MaskedArray(np.absolute(x), m1),
                                        np.absolute(xm))
        assert_masked_equal(MaskedArray(np.angle(x + 1j*y), m1 | m2),
                                        np.angle(xm + 1j*ym))
        assert_masked_equal(MaskedArray(np.angle(x + 1j*y, deg=True), m1 | m2),
                                        np.angle(xm + 1j*ym, deg=True))
        assert_masked_equal(MaskedArray(np.equal(x, y), m1 | m2),
                                        np.equal(xm, ym))
        assert_masked_equal(MaskedArray(np.not_equal(x, y), m1 | m2),
                                        np.not_equal(xm, ym))
        assert_masked_equal(MaskedArray(np.less(x, y), m1 | m2),
                                        np.less(xm, ym))
        assert_masked_equal(MaskedArray(np.greater(x, y), m1 | m2),
                                        np.greater(xm, ym))
        assert_masked_equal(MaskedArray(np.less_equal(x, y), m1 | m2),
                                        np.less_equal(xm, ym))
        assert_masked_equal(MaskedArray(np.greater_equal(x, y), m1 | m2),
                                        np.greater_equal(xm, ym))
        assert_masked_equal(MaskedArray(np.conjugate(x), m1),
                                        np.conjugate(xm))

    def test_count_func(self):
        # Tests count
        ott = MaskedArray([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        res = ott.count()
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)

        ott = ott.reshape((2, 2))
        res = ott.count()
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)
        res = ott.count(0)
        assert_(isinstance(res, np.ndarray))
        assert_equal([1, 2], res)

        ott = MaskedArray([0., 1., 2., 3.])
        res = ott.count(0)
        assert_(res.dtype.type is np.intp)
        assert_raises(np.AxisError, ott.count, axis=1)

#    def test_count_on_python_builtins(self):
#        # Tests count works on python builtins (issue#8019)
#        assert_equal(3, count([1,2,3]))
#        assert_equal(2, count((1,2)))

    def test_minmax_func(self):
        # Tests minimum and maximum.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # max doesn't work if shaped
        xr = np.ravel(x)
        xmr = np.ravel(xm)
        # following are true because of careful selection of data
        assert_equal(np.max(xr), np.maximum.reduce(xmr).filled())
        assert_equal(np.min(xr), np.minimum.reduce(xmr).filled())

        MA = MaskedArray
        ma, mb = MaskedArray([1, 2, 3]), MaskedArray([4, 0, 9])
        assert_equal(np.minimum(ma, mb).filled(), [1, 0, 3])
        assert_equal(np.maximum(ma, mb).filled(), [4, 2, 9])
        x = MaskedArray(np.arange(5))
        y = MaskedArray(np.arange(5)) - 2
        x[3] = X
        y[0] = X
        assert_equal(np.minimum(x, y), np.where(np.less(x, y), x, y))
        assert_equal(np.maximum(x, y), np.where(np.greater(x, y), x, y))
        assert_(np.minimum.reduce(x) == 0)
        assert_(np.maximum.reduce(x) == 4)

        x = MaskedArray(np.arange(4)).reshape((2, 2))
        x[-1, -1] = X
        assert_equal(np.maximum.reduce(x, axis=None).filled(), 2)

    def test_minimummaximum_func(self):
        a = np.ones((2, 2))
        ma = MaskedArray(a)
        aminimum = np.minimum(ma, ma)
        assert_(isinstance(aminimum, MaskedArray))
        assert_equal(aminimum, np.minimum(a, a))

        aminimum = np.minimum.outer(ma, ma)
        assert_(isinstance(aminimum, MaskedArray))
        assert_equal(aminimum, np.minimum.outer(a, a))

        amaximum = np.maximum(ma, ma)
        assert_(isinstance(amaximum, MaskedArray))
        assert_equal(amaximum, np.maximum(a, a))

        amaximum = np.maximum.outer(ma, ma)
        assert_(isinstance(amaximum, MaskedArray))
        assert_equal(amaximum, np.maximum.outer(a, a))

        #XXX shouldn't this be tested with a masked value?

    def test_minmax_reduce(self):
        # Test np.min/maximum.reduce on array w/ full False mask
        a = MaskedArray([1, 2, 3], mask=[False, False, False])
        b = np.maximum.reduce(a)
        assert_equal(b.filled(), 3)

    def test_minmax_funcs_with_output(self):
        # Tests the min/max functions with explicit outputs
        mask = np.random.rand(12).round()
        data = np.random.uniform(0, 10, 12)
        xm = MaskedArray(data, mask)
        xm.shape = (3, 4)
        for func in (np.min, np.max):
            nout = MaskedArray(np.empty((4,), dtype=float))
            result = func(xm, axis=0, out=nout)
            assert_(result is nout)

    def test_minmax_methods(self):
        # Additional tests on max/min
        (_, _, _, _, _, xm, _, _, _, _) = self.d
        xm.shape = (xm.size,)
        print(repr(xm))
        assert_equal(xm.max().filled(), 10)
        assert_(xm[0].max().mask)
        assert_(xm[0].max(0).mask)
        assert_(xm[0].max(-1).mask)
        assert_equal(xm.min().filled(), -10.)
        assert_(xm[0].min().mask)
        assert_(xm[0].min(0).mask)
        assert_(xm[0].min(-1).mask)
        assert_equal(np.ptp(xm).filled(), 20.)
        assert_(xm[0].ptp().mask)
        assert_(xm[0].ptp(0).mask)
        assert_(xm[0].ptp(-1).mask)

        x = MaskedArray([1, 2, 3], mask=True)
        assert_(x.min().mask)
        assert_(x.max().mask)
        assert_(np.ptp(x).mask)

    def test_addsumprod(self):
        # Tests add, sum, product.
        (x, y, _, _, _, xm, ym, _, _, _) = self.d
        mx = MaskedArray(x)
        my = MaskedArray(y)

        assert_equal(np.add.reduce(x), np.add.reduce(mx).filled())
        assert_equal(np.add.accumulate(x), np.add.accumulate(mx).filled())
        assert_equal(4, np.sum(MaskedArray(4), axis=0).filled())
        assert_equal(4, np.sum(MaskedArray(4), axis=0).filled())
        assert_equal(np.sum(x, axis=0), np.sum(mx, axis=0).filled())
        assert_equal(np.sum(xm.filled(0), axis=0), np.sum(xm, axis=0).filled(0))
        assert_equal(np.sum(x, 0), np.sum(mx, 0).filled())
        assert_equal(np.product(x, axis=0), np.product(mx, axis=0).filled())
        assert_equal(np.product(x, 0), np.product(mx, 0).filled())
        assert_equal(np.product(xm.filled(1), axis=0),
                     np.product(xm, axis=0).filled(1))
        s = (3, 4)
        x.shape = y.shape = mx.shape = my.shape = s
        if len(s) > 1:
            assert_equal(np.concatenate((x, y), 1),
                         np.concatenate((mx, my), 1).filled())
            assert_equal(np.add.reduce(x, 1), np.add.reduce(mx, 1).filled())
            assert_equal(np.sum(x, 1), np.sum(mx, 1).filled())
            assert_equal(np.product(x, 1), np.product(mx, 1).filled())

    def test_binops_d2D(self):
        # Test binary operations on 2D data
        a = MaskedArray([[1.], [2.], [3.]], mask=[[False], [True], [True]])
        b = MaskedArray([[2., 3.], [4., 5.], [6., 7.]])

        test = a * b
        control = MaskedArray([[2., 3.], [2., 2.], [3., 3.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.filled(-1), control.filled(-1))
        assert_equal(test.mask, control.mask)

        test = b * a
        control = MaskedArray([[2., 3.], [4., 5.], [6., 7.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.filled(-1), control.filled(-1))
        assert_equal(test.mask, control.mask)

        a = MaskedArray([[1.], [2.], [3.]])
        b = MaskedArray([[2., 3.], [4., 5.], [6., 7.]],
                  mask=[[0, 0], [0, 0], [0, 1]])
        test = a * b
        control = MaskedArray([[2, 3], [8, 10], [18, 3]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.filled(-1), control.filled(-1))
        assert_equal(test.mask, control.mask)

        test = b * a
        control = MaskedArray([[2, 3], [8, 10], [18, 7]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.filled(-1), control.filled(-1))
        assert_equal(test.mask, control.mask)

    def test_domained_binops_d2D(self):
        # Test domained binary operations on 2D data
        a = MaskedArray([[1.], [2.], [3.]], mask=[[False], [True], [True]])
        b = MaskedArray([[2., 3.], [4., 5.], [6., 7.]])

        test = a / b
        control = MaskedArray([[1. / 2., 1. / 3.], [2., 2.], [3., 3.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test.filled(), control.filled())
        assert_equal(test.mask, control.mask)

        test = b / a
        control = MaskedArray([[2. / 1., 3. / 1.], [4., 5.], [6., 7.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test.filled(), control.filled())
        assert_equal(test.mask, control.mask)

        a = MaskedArray([[1.], [2.], [3.]])
        b = MaskedArray([[2., 3.], [4., 5.], [6., 7.]],
                  mask=[[0, 0], [0, 0], [0, 1]])
        test = a / b
        control = MaskedArray([[1. / 2, 1. / 3], [2. / 4, 2. / 5], [3. / 6, 3]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test.filled(), control.filled())
        assert_equal(test.mask, control.mask)

        test = b / a
        control = MaskedArray([[2 / 1., 3 / 1.], [4 / 2., 5 / 2.], [6 / 3., 7]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test.filled(), control.filled())
        assert_equal(test.mask, control.mask)

    def test_mod(self):
        # Tests mod
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        mx = MaskedArray(x)
        my = MaskedArray(y)
        assert_equal(np.mod(x, y), np.mod(mx, my).filled(np.nan))

        test = np.mod(ym, xm)
        assert_equal(test.mask, xm.mask | ym.mask)
        test = np.mod(xm, ym)
        assert_equal(test.mask, xm.mask | ym.mask)

    def test_TakeTransposeInnerOuter(self):
        # Test of take, transpose, inner, outer products
        x = MaskedArray(np.arange(24))
        y = np.arange(24)
        x[5:6] = X
        y[5:6] = -1
        x = x.reshape(2, 3, 4)
        y = y.reshape(2, 3, 4)
        assert_equal(np.transpose(y, (2, 0, 1)),
                     np.transpose(x, (2, 0, 1)).filled(-1))
        assert_equal(np.take(y, (2, 0, 1), 1),
                     np.take(x, (2, 0, 1), 1).filled(-1))
        assert_equal(np.inner(x.filled(0), y),
                     np.inner(x, y).filled())
        assert_equal(np.outer(x.filled(0), y),
                     np.outer(x, y).filled())
        y = MaskedArray(['abc', 1, 'def', 2, 3], dtype=object)
        y[2] = X
        t = np.take(y, [0, 3, 4]).filled()
        assert_(t[0] == 'abc')
        assert_(t[1] == 2)
        assert_(t[2] == 3)

    def test_imag_real(self):
        # Check complex
        xx = MaskedArray([1 + 10j, 20 + 2j], mask=[1, 0])
        assert_equal(xx.imag.filled(0), [0, 2])
        assert_equal(xx.imag.dtype, xx._data.imag.dtype)
        assert_equal(xx.real.filled(0), [0, 20])
        assert_equal(xx.real.dtype, xx._data.real.dtype)

    def test_methods_with_output(self):
        xm = MaskedArray(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = X

        funclist = ('sum', 'prod', 'var', 'std', 'max', 'min', 'ptp', 'mean',)

        for funcname in funclist:
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)
            # A ndarray as explicit input
            output = MaskedArray(np.empty(4, dtype=float))
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            # ... the result should be the given output
            assert_(result is output)
            assert_(output[0].mask)
            assert_equal(result, xmmeth(axis=0, out=output))

    def test_eq_on_structured(self):
        # Test the equality of structured arrays
        ndtype = [('A', int), ('B', int)]
        a = MaskedArray([(1, 1), (2, 2)], mask=[0, 1], dtype=ndtype)

        test = (a == a)
        assert_equal(test.filled(False), [True, False])
        assert_equal(test.mask, [False, True])

        test = (a == a[0])
        assert_equal(test.filled(False), [True, False])
        assert_equal(test.mask, [False, True])

        b = MaskedArray([(1, 1), (2, 2)], mask=[0, 0], dtype=ndtype)
        test = (a == b)
        assert_equal(test.filled(False), [True, False])
        assert_equal(test.mask, [False, True])

        test = (a[0] == b)
        assert_equal(test.filled(False), [True, False])
        assert_equal(test.mask, [False, False])

        b = MaskedArray([(1, 1), (2, 2)], mask=[1, 0], dtype=ndtype)
        test = (a == b)
        assert_equal(test.filled(False), [False, False])
        assert_equal(test.mask, [True, True])

        ## complicated dtype, 2-dimensional array.
        #ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        #a = MaskedArray([[(1, (1, 1)), (2, (2, 2))],
        #           [(3, (3, 3)), (4, (4, 4))]],
        #          mask=[[(0, (1, 0)), (0, (0, 1))],
        #                [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
        #test = (a[0, 0] == a)
        #assert_equal(test.data, [[True, False], [False, False]])
        #assert_equal(test.mask, [[False, False], [False, True]])
        #assert_(test.fill_value == True)

#    def test_ne_on_structured(self):
#        # Test the equality of structured arrays
#        ndtype = [('A', int), ('B', int)]
#        a = MaskedArray([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)

#        test = (a != a)
#        assert_equal(test.data, [False, False])
#        assert_equal(test.mask, [False, False])
#        assert_(test.fill_value == True)

#        test = (a != a[0])
#        assert_equal(test.data, [False, True])
#        assert_equal(test.mask, [False, False])
#        assert_(test.fill_value == True)

#        b = MaskedArray([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
#        test = (a != b)
#        assert_equal(test.data, [True, False])
#        assert_equal(test.mask, [True, False])
#        assert_(test.fill_value == True)

#        test = (a[0] != b)
#        assert_equal(test.data, [True, True])
#        assert_equal(test.mask, [True, False])
#        assert_(test.fill_value == True)

#        b = MaskedArray([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
#        test = (a != b)
#        assert_equal(test.data, [False, False])
#        assert_equal(test.mask, [False, False])
#        assert_(test.fill_value == True)

#        # complicated dtype, 2-dimensional array.
#        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
#        a = MaskedArray([[(1, (1, 1)), (2, (2, 2))],
#                   [(3, (3, 3)), (4, (4, 4))]],
#                  mask=[[(0, (1, 0)), (0, (0, 1))],
#                        [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
#        test = (a[0, 0] != a)
#        assert_equal(test.data, [[False, True], [True, True]])
#        assert_equal(test.mask, [[False, False], [False, True]])
#        assert_(test.fill_value == True)

#    def test_eq_ne_structured_extra(self):
#        # ensure simple examples are symmetric and make sense.
#        # from https://github.com/numpy/numpy/pull/8590#discussion_r101126465
#        dt = np.dtype('i4,i4')
#        for m1 in (mvoid((1, 2), mask=(0, 0), dtype=dt),
#                   mvoid((1, 2), mask=(0, 1), dtype=dt),
#                   mvoid((1, 2), mask=(1, 0), dtype=dt),
#                   mvoid((1, 2), mask=(1, 1), dtype=dt)):
#            ma1 = m1.view(MaskedArray)
#            r1 = ma1.view('2i4')
#            for m2 in (np.array((1, 1), dtype=dt),
#                       mvoid((1, 1), dtype=dt),
#                       mvoid((1, 0), mask=(0, 1), dtype=dt),
#                       mvoid((3, 2), mask=(0, 1), dtype=dt)):
#                ma2 = m2.view(MaskedArray)
#                r2 = ma2.view('2i4')
#                eq_expected = (r1 == r2).all()
#                assert_equal(m1 == m2, eq_expected)
#                assert_equal(m2 == m1, eq_expected)
#                assert_equal(ma1 == m2, eq_expected)
#                assert_equal(m1 == ma2, eq_expected)
#                assert_equal(ma1 == ma2, eq_expected)
#                # Also check it is the same if we do it element by element.
#                el_by_el = [m1[name] == m2[name] for name in dt.names]
#                assert_equal(array(el_by_el, dtype=bool).all(), eq_expected)
#                ne_expected = (r1 != r2).any()
#                assert_equal(m1 != m2, ne_expected)
#                assert_equal(m2 != m1, ne_expected)
#                assert_equal(ma1 != m2, ne_expected)
#                assert_equal(m1 != ma2, ne_expected)
#                assert_equal(ma1 != ma2, ne_expected)
#                el_by_el = [m1[name] != m2[name] for name in dt.names]
#                assert_equal(array(el_by_el, dtype=bool).any(), ne_expected)

    @pytest.mark.parametrize('dt', ['S', 'U'])
    def test_eq_for_strings(self, dt):
        # Test the equality of structured arrays
        a = MaskedArray(['a', 'b'], dtype=dt, mask=[0, 1])

        test = (a == a)
        assert_equal(test.filled(), [True, False])
        assert_equal(test.mask, [False, True])

        test = (a == a[0])
        assert_equal(test.filled(), [True, False])
        assert_equal(test.mask, [False, True])

        b = MaskedArray(['a', 'b'], dtype=dt, mask=[1, 0])
        test = (a == b)
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [True, True])

        # test = (a[0] == b)  # doesn't work in Python2
        test = (b == a[0])
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [True, False])

    @pytest.mark.parametrize('dt', ['S', 'U'])
    def test_ne_for_strings(self, dt):
        # Test the equality of structured arrays
        a = MaskedArray(['a', 'b'], dtype=dt, mask=[0, 1])

        test = (a != a)
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [False, True])

        test = (a != a[0])
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [False, True])

        b = MaskedArray(['a', 'b'], dtype=dt, mask=[1, 0])
        test = (a != b)
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [True, True])

        # test = (a[0] != b)  # doesn't work in Python2
        test = (b != a[0])
        assert_equal(test.filled(), [False, True])
        assert_equal(test.mask, [True, False])

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    def test_eq_for_numeric(self, dt1, dt2):
        # Test the equality of structured arrays
        a = MaskedArray([0, 1], dtype=dt1, mask=[0, 1])

        test = (a == a)
        assert_equal(test.filled(), [True, False])
        assert_equal(test.mask, [False, True])

        test = (a == a[0])
        assert_equal(test.filled(), [True, False])
        assert_equal(test.mask, [False, True])

        b = MaskedArray([0, 1], dtype=dt2, mask=[1, 0])
        test = (a == b)
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [True, True])

        # test = (a[0] == b)  # doesn't work in Python2
        test = (b == a[0])
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [True, False])

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    def test_ne_for_numeric(self, dt1, dt2):
        # Test the equality of structured arrays
        a = MaskedArray([0, 1], dtype=dt1, mask=[0, 1])

        test = (a != a)
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [False, True])

        test = (a != a[0])
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [False, True])

        b = MaskedArray([0, 1], dtype=dt2, mask=[1, 0])
        test = (a != b)
        assert_equal(test.filled(), [False, False])
        assert_equal(test.mask, [True, True])

        # test = (a[0] != b)  # doesn't work in Python2
        test = (b != a[0])
        assert_equal(test.filled(), [False, True])
        assert_equal(test.mask, [True, False])

    def test_eq_with_None(self):
        # Really, comparisons with None should not be done, but check them
        # anyway. Note that pep8 will flag these tests.
        # Deprecation is in place for arrays, and when it happens this
        # test will fail (and have to be changed accordingly).

        # With partial mask
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "Comparison to `None`")
            a = MaskedArray([None, 1], mask=[0, 1])
            assert_equal(a == None, MaskedArray([True, False], mask=[0, 1]))
            assert_equal(a.filled() == None, [True, False])
            assert_equal(a != None, MaskedArray([False, True], mask=[0, 1]))
            # With nomask
            a = MaskedArray([None, 1], mask=False)
            assert_equal(a == None, [True, False])
            assert_equal(a != None, [False, True])
            # With complete mask
            a = MaskedArray([None, 2], mask=True)
            assert_equal(a == None, MaskedArray([False, True], mask=True))
            assert_equal(a != None, MaskedArray([True, False], mask=True))
            # Fully masked, even comparison to None should return "masked"
            a[:] = X
            assert_((a == None).mask.all())

    def test_eq_with_scalar(self):
        a = MaskedArray(1)
        assert_equal((a == 1).filled(), True)
        assert_equal((a == 0).filled(), False)
        assert_equal((a != 1).filled(), False)
        assert_equal((a != 0).filled(), True)
        b = MaskedArray(1, mask=True)
        assert_((b == 0).mask.all())
        assert_((b == 1).mask.all())
        assert_((b != 0).mask.all())
        assert_((b != 1).mask.all())

    def test_eq_different_dimensions(self):
        m1 = MaskedArray([1, 1], mask=[0, 1])
        # test comparison with both masked and regular arrays.
        for m2 in (MaskedArray([[0, 1], [1, 2]]),
                   np.array([[0, 1], [1, 2]])):
            test = (m1 == m2)
            assert_equal(test.filled(), [[False, False],
                                         [True, False]])
            assert_equal(test.mask, [[False, True],
                                     [False, True]])

    def test_numpyarithmetics(self):
        # Check that the mask is not back-propagated when using numpy functions
        a = MaskedArray([-1, 0, 1, 2, X])
        control = MaskedArray([np.nan, -np.inf, 0, np.log(2), X])
        assert_masked_equal(np.log(a), control)
        assert_equal(a.mask, [0, 0, 0, 0, 1])


class TestMaskedArrayAttributes:
    #def test_flat(self):
    #    # Test that flat can return all types of items [#4585, #4615]
    #    # test 2-D record array
    #    # ... on structured array w/ masked records
    #    x = MaskedArray([[(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'thr')],
    #                     [(4, 4.4, 'fou'), (5, 5.5, 'fiv'), (6, 6.6, 'six')]],
    #                    dtype=[('a', int), ('b', float), ('c', '|S8')])
    #    x['a'][0, 1] = X
    #    x['b'][1, 0] = X
    #    x['c'][0, 2] = X
    #    x[-1, -1] = X
    #    xflat = x.flat
    #    assert_equal(xflat[0], x[0, 0])
    #    assert_equal(xflat[1], x[0, 1])
    #    assert_equal(xflat[2], x[0, 2])
    #    assert_equal(xflat[:3], x[0])
    #    assert_equal(xflat[3], x[1, 0])
    #    assert_equal(xflat[4], x[1, 1])
    #    assert_equal(xflat[5], x[1, 2])
    #    assert_equal(xflat[3:], x[1])
    #    assert_equal(xflat[-1], x[-1, -1])
    #    i = 0
    #    j = 0
    #    for xf in xflat:
    #        assert_equal(xf, x[j, i])
    #        i += 1
    #        if i >= x.shape[-1]:
    #            i = 0
    #            j += 1

    def test_assign_dtype(self):
        # check that the mask's dtype is updated when dtype is changed
        a = np.zeros(4, dtype='f4,i4')

        m = MaskedArray(a)
        m.dtype = np.dtype('f8')
        repr(m)  # raises?
        assert_equal(m.dtype, np.dtype('f8'))

        # check that dtype changes that change shape of mask are not allowed
        def assign():
            m = MaskedArray(a)
            m.dtype = np.dtype('f4')
        assert_raises(ValueError, assign)

        # check that nomask is preserved
        a = MaskedArray(np.zeros(4, dtype='f8'))
        b = a.copy()
        b.dtype = np.dtype('f4,i4')
        assert_equal(b.dtype, np.dtype('f4,i4'))
        assert_equal(b._mask, a._mask)


class TestFillingValues:
    def test_extremum_fill_value(self):
        # Tests extremum fill values for flexible type.
        a = MaskedArray([(1, (2, 3)), (4, (5, 6))],
                        mask=[0,1],
                        dtype=[('A', int), ('B', [('BA', int), ('BB', int)])])
        assert_raises(ValueError, a.filled, minmax='min')

class TestUfuncs:
    # Test class for the application of ufuncs on MaskedArrays.

    def setup(self):
        # Base data definition.
        self.d = (MaskedArray([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6),
                  MaskedArray([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6),)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown(self):
        np.seterr(**self.err_status)

    def test_testUfuncRegression(self):
        # Tests new ufuncs on MaskedArrays.
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  'floor', 'ceil',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor',
                  ]:
            uf = getattr(np, f)
            args = self.d[:uf.nin]
            mr = uf(*args) # test no fail

    def test_reduce(self):
        # Tests reduce on MaskedArrays.
        a = self.d[0]
        assert_(not np.alltrue(a, axis=0))
        assert_(np.sometrue(a, axis=0))
        assert_equal(np.sum(a[:3], axis=0).filled(), 0)
        assert_equal(np.product(a, axis=0).filled(), 0)
        assert_equal(np.add.reduce(a).filled(), pi)

    def test_minmax(self):
        # Tests extrema on MaskedArrays.
        a = np.arange(1, 13).reshape(3, 4)
        amask = MaskedArray(a, a < 5)
        assert_equal(amask.max().filled(), a.max())
        assert_equal(amask.min().filled(), 5)
        assert_equal(amask.max(0).filled(), a.max(0))
        assert_equal(amask.min(0).filled(), [5, 6, 7, 8])
        assert_(amask.max(1)[0].mask)
        assert_(amask.min(1)[0].mask)

    def test_ndarray_mask(self):
        # Check that the mask of the result is a ndarray (not a MaskedArray...)
        a = MaskedArray([-1, 0, 1, 2, X])
        test = np.sqrt(a)
        control = MaskedArray([np.nan, 0, 1, np.sqrt(2), X])
        assert_masked_equal(test, control)
        assert_(not isinstance(test.mask, MaskedArray))

    def test_treatment_of_NotImplemented(self):
        # Check that NotImplemented is returned at appropriate places

        a = MaskedArray([1., 2.], mask=[1, 0])
        assert_raises(TypeError, operator.mul, a, "abc")
        assert_raises(TypeError, operator.truediv, a, "abc")

    def test_no_masked_nan_warnings(self):
        # check that a nan in masked position does not
        # cause ufunc warnings

        m = MaskedArray([0.5, np.nan], mask=[0,1])

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            # test unary and binary ufuncs
            np.exp(m)
            np.add(m, 1)
            m > 0

            # test different unary domains
            np.sqrt(m)
            np.log(m)
            np.tan(m)
            np.arcsin(m)
            np.arccos(m)
            np.arccosh(m)

            # test binary domains
            np.divide(m, 2)

            # also check that allclose uses ma ufuncs, to avoid warning
            np.allclose(m, 0.5)

class TestMaskedArrayInPlaceArithmetics:
    # Test MaskedArray Arithmetics

    def setup(self):
        x = MaskedArray(np.arange(10))
        y = MaskedArray(np.arange(10))
        xm = MaskedArray(np.arange(10))
        xm[2] = X
        self.intdata = (x, y, xm)
        self.floatdata = (x.astype(float), y.astype(float), xm.astype(float))
        self.othertypes = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        self.othertypes = [np.dtype(_).type for _ in self.othertypes]
        self.uint8data = (
            x.astype(np.uint8),
            y.astype(np.uint8),
            xm.astype(np.uint8)
        )

    def test_inplace_addition_scalar(self):
        # Test of inplace additions
        (x, y, xm) = self.intdata
        x = x.copy()
        xm = xm.copy()

        xm[2] = X
        x += 1
        assert_equal(x, y + 1)
        xm += 1
        assert_equal(xm, y + 1)

        (x, _, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        id1 = x._data.ctypes.data
        x += 1.
        assert_(id1 == x._data.ctypes.data)
        assert_equal(x, y + 1.)

    def test_inplace_addition_array(self):
        # Test of inplace additions
        (x, y, xm) = self.intdata
        x = x.copy()
        xm = xm.copy()

        m = xm.mask
        a = MaskedArray(np.arange(10, dtype=np.int16))
        a[-1] = X
        x += a
        xm += a
        assert_equal(x, y + a)
        assert_equal(xm, y + a)
        assert_equal(xm.mask, m | a.mask)

    def test_inplace_subtraction_scalar(self):
        # Test of inplace subtractions
        (x, y, xm) = self.intdata
        x = x.copy()
        xm = xm.copy()

        x -= 1
        assert_equal(x, y - 1)
        xm -= 1
        assert_equal(xm, y - 1)

    def test_inplace_subtraction_array(self):
        # Test of inplace subtractions
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        m = xm.mask
        a = MaskedArray(np.arange(10, dtype=float))
        a[-1] = X
        x -= a
        xm -= a
        assert_equal(x, y - a)
        assert_equal(xm, y - a)
        assert_equal(xm.mask, m | a.mask)

    def test_inplace_multiplication_scalar(self):
        # Test of inplace multiplication
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        x *= 2.0
        assert_equal(x, y * 2)
        xm *= 2.0
        assert_equal(xm, y * 2)

    def test_inplace_multiplication_array(self):
        # Test of inplace multiplication
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        m = xm.mask
        a = MaskedArray(np.arange(10, dtype=float))
        a[-1] = X
        x *= a
        xm *= a
        assert_equal(x, y * a)
        assert_equal(xm, y * a)
        assert_equal(xm.mask, m | a.mask)

    def test_inplace_division_scalar_int(self):
        # Test of inplace division
        (x, y, xm) = self.intdata
        x = x.copy()
        xm = xm.copy()

        x = MaskedArray(np.arange(10) * 2)
        xm = MaskedArray(np.arange(10) * 2)
        xm[2] = X
        x //= 2
        assert_equal(x, y)
        xm //= 2
        assert_equal(xm, y)

    def test_inplace_division_scalar_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        x /= 2.0
        assert_equal(x, y / 2.0)
        with pytest.warns(RuntimeWarning, match='invalid value'):
            xm /= np.arange(10)
        assert_masked_equal(xm, MaskedArray([np.nan, 1, X, 1, 1, 1, 1, 1, 1,1]))

    def test_inplace_division_array_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        x = x.copy()
        xm = xm.copy()

        m = xm.mask
        a = MaskedArray(np.arange(10, dtype=float))
        a[-1] = X
        with pytest.warns(RuntimeWarning) as record:
            x /= a
            xm /= a
            assert_equal(x, y / a)
            assert_equal(xm, y / a)
            assert_equal(xm.mask, m)
        assert(len(record) == 4) # div by zero

    def test_inplace_division_misc(self):
        x = MaskedArray([X, 1, 1, -2, pi / 2., 4, X, -10., 10., 1., 2., 3.])
        y = MaskedArray([5, 0, X,  2,     -1., X, X, -10., 10., 1., 0., X])
        control = MaskedArray([X, np.inf, X, -1,-pi/2, X, X, 1, 1, 1, np.inf,X])

        with pytest.warns(RuntimeWarning) as record:
            z = x / y
        assert(len(record) == 1) # div by zero
        assert_masked_equal(z, control)

        x = x.copy()
        with pytest.warns(RuntimeWarning) as record:
            x /= y
        assert(len(record) == 1) # div by zero
        assert_masked_equal(x, control)

    def test_add(self):
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        # Test add w/ scalar
        xx = x + 1
        assert_equal(xx.filled(9), [2, 3, 9])
        assert_equal(xx.mask, [0, 0, 1])
        # Test iadd w/ scalar
        x += 1
        assert_equal(x.filled(9), [2, 3, 9])
        assert_equal(x.mask, [0, 0, 1])
        # Test add w/ array
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        xx = x + MaskedArray([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.filled(9), [9, 4, 9])
        assert_equal(xx.mask, [1, 0, 1])
        # Test iadd w/ array
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        x += MaskedArray([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.filled(9), [9, 4, 9])
        assert_equal(x.mask, [1, 0, 1])

    def test_sub(self):
        # Test sub w/ scalar
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        xx = x - 1
        assert_equal(xx.filled(9), [0, 1, 9])
        assert_equal(xx.mask, [0, 0, 1])
        # Test isub w/ scalar
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        x -= 1
        assert_equal(x.filled(9), [0, 1, 9])
        assert_equal(x.mask, [0, 0, 1])
        # Test sub w/ array
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        xx = x - MaskedArray([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.filled(9), [9, 0, 9])
        assert_equal(xx.mask, [1, 0, 1])
        # Test isub w/ array
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        x -= MaskedArray([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.filled(9), [9, 0, 9])
        assert_equal(x.mask, [1, 0, 1])

    def test_mul(self):
        # Test mul w/ scalar
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        xx = x * 2
        assert_equal(xx.filled(9), [2, 4, 9])
        assert_equal(xx.mask, [0, 0, 1])
        # Test imul w/ scalar
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        x *= 2
        assert_equal(x.filled(9), [2, 4, 9])
        assert_equal(x.mask, [0, 0, 1])
        # Test mul w/ array
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        xx = x * MaskedArray([10, 20, 30], mask=[1, 0, 0])
        assert_equal(xx.filled(9), [9, 40, 9])
        assert_equal(xx.mask, [1, 0, 1])
        # Test imul w/ array
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        x *= MaskedArray([10, 20, 30], mask=[1, 0, 0])
        assert_equal(x.filled(9), [9, 40, 9])
        assert_equal(x.mask, [1, 0, 1])

    def test_div(self):
        # Test div on scalar
        x = MaskedArray([1, 2, 3], mask=[0, 0, 1])
        xx = x / 2.
        assert_equal(xx.filled(9), [1 / 2., 2 / 2., 9])
        assert_equal(xx.mask, [0, 0, 1])
        # Test idiv on scalar
        x = MaskedArray([1., 2., 3.], mask=[0, 0, 1])
        x /= 2.
        assert_equal(x.filled(9), [1 / 2., 2 / 2., 9])
        assert_equal(x.mask, [0, 0, 1])
        # Test div on array
        x = MaskedArray([1., 2., 3.], mask=[0, 0, 1])
        xx = x / MaskedArray([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(xx.filled(9), [9., 2. / 20., 9.])
        assert_equal(xx.mask, [1, 0, 1])
        # Test idiv on array
        x = MaskedArray([1., 2., 3.], mask=[0, 0, 1])
        x /= MaskedArray([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(x.filled(9), [9., 2 / 20., 9.])
        assert_equal(x.mask, [1, 0, 1])

    def test_pow(self):
        # Test keeping filled(9) w/ (inplace) power
        # Test pow on scalar
        x = MaskedArray([1., 2., 3.], mask=[0, 0, 1])
        xx = x ** 2.5
        assert_equal(xx.filled(9), [1., 2. ** 2.5, 9.])
        assert_equal(xx.mask, [0, 0, 1])
        # Test ipow on scalar
        x **= 2.5
        assert_equal(x.filled(9), [1., 2. ** 2.5, 9])
        assert_equal(x.mask, [0, 0, 1])

    def test_add_arrays(self):
        a = MaskedArray([[1, 1], [3, 3]])
        b = MaskedArray([1, 1], mask=[0, 0])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        assert_equal(a.mask, [[0, 0], [0, 0]])

        a = MaskedArray([[1, 1], [3, 3]])
        b = MaskedArray([1, 1], mask=[0, 1])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_sub_arrays(self):
        a = MaskedArray([[1, 1], [3, 3]])
        b = MaskedArray([1, 1], mask=[0, 0])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        assert_equal(a.mask, [[0, 0], [0, 0]])

        a = MaskedArray([[1, 1], [3, 3]])
        b = MaskedArray([1, 1], mask=[0, 1])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_mul_arrays(self):
        a = MaskedArray([[1, 1], [3, 3]])
        b = MaskedArray([1, 1], mask=[0, 0])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        assert_equal(a.mask, [[0, 0], [0, 0]])

        a = MaskedArray([[1, 1], [3, 3]])
        b = MaskedArray([1, 1], mask=[0, 1])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_inplace_addition_scalar_type(self):
        # Test of inplace additions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                xm[2] = X
                x += t(1)
                assert_equal(x, y + t(1))
                xm += t(1)
                assert_equal(xm, y + t(1))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_addition_array_type(self):
        # Test of inplace additions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X
                x += a
                xm += a
                assert_equal(x, y + a)
                assert_equal(xm, y + a)
                assert_equal(xm.mask, m | a.mask)

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_subtraction_scalar_type(self):
        # Test of inplace subtractions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x -= t(1)
                assert_equal(x, y - t(1))
                xm -= t(1)
                assert_equal(xm, y - t(1))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_subtraction_array_type(self):
        # Test of inplace subtractions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X
                x -= a
                xm -= a
                assert_equal(x, y - a)
                assert_equal(xm, y - a)
                assert_equal(xm.mask, m | a.mask)

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_multiplication_scalar_type(self):
        # Test of inplace multiplication
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x *= t(2)
                assert_equal(x, y * t(2))
                xm *= t(2)
                assert_equal(xm, y * t(2))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_multiplication_array_type(self):
        # Test of inplace multiplication
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X
                x *= a
                xm *= a
                assert_equal(x, y * a)
                assert_equal(xm, y * a)
                assert_equal(xm.mask, m | a.mask)

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_floor_division_scalar_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = MaskedArray(np.arange(10, dtype=t)) * t(2)
                xm = MaskedArray(np.arange(10, dtype=t)) * t(2)
                xm[2] = X
                x //= t(2)
                xm //= t(2)
                assert_equal(x, y)
                assert_equal(xm, y)

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_floor_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(10, dtype=t))
                a[-1] = X
                x //= a
                xm //= a
                assert_equal(x, y // a)
                assert_equal(xm, y // a)
                assert_equal(xm.mask, m)

                assert_equal(len(w), 4, "Failed on type=%s." % t)

    def test_inplace_division_scalar_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)

                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = MaskedArray(np.arange(10, dtype=t)) * t(2)
                xm = MaskedArray(np.arange(10, dtype=t)) * t(2)
                xm[2] = X

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= t(2)
                    assert_equal(x, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= t(2)
                    assert_equal(xm, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, "Failed on type=%s." % t)
                else:
                    assert_equal(len(sup.log), 0, "Failed on type=%s." % t)

    def test_inplace_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = MaskedArray(np.arange(1, 11, dtype=t))
                a[-1] = X

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= a
                    assert_equal(x, y / a)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= a
                    assert_equal(xm, y / a)
                    assert_equal(xm.mask, m)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, "Failed on type=%s." % t)
                else:
                    assert_equal(len(sup.log), 0, "Failed on type=%s." % t)

    def test_inplace_pow_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                # Test pow on scalar
                x = MaskedArray([1, 2, 3], mask=[0, 0, 1], dtype=t)
                xx = x ** t(2)
                xx_r = MaskedArray([1, 2 ** 2, 3], mask=[0, 0, 1], dtype=t)
                assert_equal(xx.filled(9), xx_r.filled(9))
                assert_equal(xx.mask, xx_r.mask)
                # Test ipow on scalar
                x **= t(2)
                assert_equal(x.filled(9), xx_r.filled(9))
                assert_equal(x.mask, xx_r.mask)

                assert_equal(len(w), 0, "Failed on type=%s." % t)


class TestMaskedArrayMethods:
    # Test class for miscellaneous MaskedArrays methods.
    def setup(self):
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        x2 = x.reshape(6, 6)
        x4 = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        mx = MaskedArray(x, m)
        mx2 = MaskedArray(x2, m.reshape(x2.shape))
        mx4 = MaskedArray(x4, m.reshape(x4.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                       1, 1, 1, 1, 0, 1,
                       0, 0, 1, 1, 0, 1,
                       0, 0, 0, 1, 1, 1,
                       1, 0, 0, 1, 1, 0,
                       0, 0, 1, 0, 1, 1])
        m2x = MaskedArray(x.copy(), m2)
        m2x2 = MaskedArray(x2.copy(), m2.reshape(x2.shape))
        m2x4 = MaskedArray(x4.copy(), m2.reshape(x4.shape))
        self.d = (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4)

    def test_generic_methods(self):
        # Tests some MaskedArray methods.
        a = MaskedArray([1, 3, 2])
        assert_equal(a.any(), a.filled().any())
        assert_equal(a.all(), a.filled().all())
        assert_equal(a.argmax(), a.filled().argmax())
        assert_equal(a.argmin(), a.filled().argmin())
        #XXX choose doesn't support masked indices
        #assert_equal(a.choose([0, 1, 2, 3, 4]).filled(),
        #             a.filled().choose([0, 1, 2, 3, 4]))
        assert_equal(a.compress([1, 0, 1]).filled(),
                     a.filled().compress([1, 0, 1]))
        assert_equal(a.conj().filled(), a.filled().conj())
        assert_equal(a.conjugate().filled(), a.filled().conjugate())

        m = MaskedArray([[1, 2], [3, 4]])
        assert_equal(m.diagonal(), m.filled().diagonal())
        assert_equal(a.sum().filled(), a.filled().sum())
        assert_equal(a.take([1, 2]).filled(), a.filled().take([1, 2]))
        assert_equal(m.transpose().filled(), m.filled().transpose())

    def test_allclose(self):
        # Tests allclose on arrays
        a = MaskedArray(np.random.rand(10))
        b = a + np.random.rand(10) * 1e-8
        assert_(np.allclose(a, b))
        # Test allclose w/ infs
        a[0] = np.inf
        assert_(not np.allclose(a, b))
        b[0] = np.inf
        assert_(np.allclose(a, b))
        # Test allclose w/ masked
        a[-1] = X
        assert_(np.allclose(a, b))

        #XXX add a masked_equal arg to allclose somehow?
        #assert_(np.allclose(a, b, masked_equal=True))
        #assert_(not allclose(a, b, masked_equal=False))

        # Test comparison w/ scalar
        a *= 1e-8
        a[0] = 0
        assert_(np.allclose(a, 0))

        # Test that the function works for MIN_INT integer typed arrays
        a = MaskedArray([np.iinfo(np.int_).min], dtype=np.int_)
        assert_(np.allclose(a, a))

    def test_allany(self):
        # Checks the any/all methods/functions.
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool_)
        mx = MaskedArray(x, mask=m)
        mxbig = (mx > 0.5)
        mxsmall = (mx < 0.5)

        assert_(not mxbig.all())
        assert_(mxbig.any())
        assert_equal(mxbig.all(0), [False, False, True])
        assert_equal(mxbig.all(1), [False, False, True])
        assert_equal(mxbig.any(0), [False, False, True])
        assert_equal(mxbig.any(1), [True, True, True])

        assert_(not mxsmall.all())
        assert_(mxsmall.any())
        assert_equal(mxsmall.all(0), [True, True, False])
        assert_equal(mxsmall.all(1), [False, False, False])
        assert_equal(mxsmall.any(0), [True, True, False])
        assert_equal(mxsmall.any(1), [True, True, False])

    def test_allany_oddities(self):
        # Some fun with all and any
        store = MaskedArray(np.empty((), dtype=bool))
        full = MaskedArray([1, 2, 3], mask=True)

        assert_(full.all())
        full.all(out=store)
        assert_(store.filled())
        assert_equal(store.mask, False)

        store = np.empty((), dtype=bool)
        assert_(not full.any())
        full.any(out=store)
        assert_(not store)

    def test_argmax_argmin(self):
        # Tests argmin & argmax on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d

        assert_equal(mx.argmin(), 35)
        assert_equal(mx2.argmin(), 35)
        assert_equal(m2x.argmin(), 4)
        assert_equal(m2x2.argmin(), 4)
        assert_equal(mx.argmax(), 28)
        assert_equal(mx2.argmax(), 28)
        assert_equal(m2x.argmax(), 31)
        assert_equal(m2x2.argmax(), 31)

        assert_equal(mx2.argmin(0), [2, 2, 2, 5, 0, 5])
        assert_equal(m2x2.argmin(0), [2, 2, 4, 5, 0, 4])
        assert_equal(mx2.argmax(0), [0, 5, 0, 5, 4, 0])
        assert_equal(m2x2.argmax(0), [5, 5, 0, 5, 1, 0])

        assert_equal(mx2.argmin(1), [4, 1, 0, 0, 5, 5, ])
        assert_equal(m2x2.argmin(1), [4, 4, 0, 0, 5, 3])
        assert_equal(mx2.argmax(1), [2, 4, 1, 1, 4, 1])
        assert_equal(m2x2.argmax(1), [2, 4, 1, 1, 1, 1])

        # test all masked
        a = MaskedArray([X, X, X], dtype='f')
        assert_(a[np.argmax(a)].mask)
        assert_(a[np.argmin(a)].mask)

    def test_clip(self):
        # Tests clip on MaskedArrays.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=np.bool_)
        mx = MaskedArray(x, mask=m)
        clipped = mx.clip(2, 8)
        assert_equal(clipped.mask, mx.mask)
        assert_equal(clipped[~m].filled(), x.clip(2, 8)[~m])
        assert_equal(clipped.filled(2), mx.filled(0).clip(2, 8))

    def test_compress(self):
        # test compress
        a = MaskedArray([1., 2., 3., 4., 5.])
        condition = (a > 1.5) & (a < 3.5)
        assert_equal(a.compress(condition).filled(), [2., 3.])

        a[[2, 3]] = X
        b = a.compress(condition)
        assert_equal(b.filled(9), [2., 9.])
        assert_equal(b.mask, [0, 1])
        assert_equal(b.filled(9), a[condition].filled(9))

        condition = (a < 4.)
        b = a.compress(condition)
        assert_equal(b.filled(9), [1., 2.])
        assert_equal(b.mask, [0, 0])
        assert_equal(b.filled(9), a[condition].filled(9))

        a = MaskedArray([[10, 20, 30], [40, 50, 60]],
                         mask=[[0, 0, 1], [1, 0, 0]])
        b = a.compress(a.ravel() >= 22)
        assert_equal(b.filled(9), [50, 60])
        assert_equal(b.mask, [0, 0])

        x = np.array([3, 1, 2])
        b = a.compress(x >= 2, axis=1)
        assert_equal(b.filled(9), [[10, 9], [9, 60]])
        assert_equal(b.mask, [[0, 1], [1, 0]])

    def test_empty(self):
        # Tests empty/like
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = MaskedArray([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        b = np.empty_like(a)
        assert_equal(b.shape, a.shape)

        b = np.empty(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)

        # check empty_like mask handling
        a = MaskedArray([1, 2, 3], mask=[False, True, False])
        b = np.empty_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        #b = a.view(MaskedArray) #XXX views to be dealt with later
        #assert_(np.may_share_memory(a.mask, b.mask))

#    def test_put(self):
#        # Tests put.
#        d = arange(5)
#        n = [0, 0, 0, 1, 1]
#        m = make_mask(n)
#        x = MaskedArray(d, mask=m)
#        assert_(x[3] is masked)
#        assert_(x[4] is masked)
#        x[[1, 4]] = [10, 40]
#        assert_(x[3] is masked)
#        assert_(x[4] is not masked)
#        assert_equal(x, [0, 10, 2, -1, 40])

#        x = MaskedArray(arange(10), mask=[1, 0, 0, 0, 0] * 2)
#        i = [0, 2, 4, 6]
#        x.put(i, [6, 4, 2, 0])
#        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
#        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
#        x.put(i, MaskedArray([0, 2, 4, 6], [1, 0, 1, 0]))
#        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
#        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

#        x = MaskedArray(arange(10), mask=[1, 0, 0, 0, 0] * 2)
#        put(x, i, [6, 4, 2, 0])
#        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
#        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
#        put(x, i, MaskedArray([0, 2, 4, 6], [1, 0, 1, 0]))
#        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
#        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_put_nomask(self):
        # GitHub issue 6425
        x = MaskedArray(np.zeros(10))
        z = MaskedArray([3., -1.], mask=[False, True])

        x.put([1, 2], z)
        assert_(not x[0].mask)
        assert_equal(x[0].filled(9), 0)
        assert_(not x[1].mask)
        assert_equal(x[1].filled(9), 3)
        assert_(x[2].mask)
        assert_(not x[3].mask)
        assert_equal(x[3].filled(9), 0)

    def test_ravel(self):
        # Tests ravel
        a = MaskedArray([[1, 2, 3, 4, 5]], mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel.mask.shape, aravel.shape)
        a = MaskedArray([0, 0], mask=[1, 1])
        aravel = a.ravel()
        assert_equal(aravel.mask.shape, a.shape)

        a = MaskedArray(np.array([1, 2, 3, 4]), mask=[0, 0, 0, 0])
        assert_equal(a.ravel().mask, [0, 0, 0, 0])
        # Test that the fill_value is preserved
        a.shape = (2, 2)
        ar = a.ravel()
        assert_equal(ar.mask, [0, 0, 0, 0])
        assert_equal(ar.filled(), [1, 2, 3, 4])
        # Test index ordering
        assert_equal(a.ravel(order='C').filled(), [1, 2, 3, 4])
        assert_equal(a.ravel(order='F').filled(), [1, 3, 2, 4])

    def test_reshape(self):
        # Tests reshape
        x = MaskedArray(np.arange(4))
        x[0] = X
        y = x.reshape(2, 2)
        assert_equal(y.shape, (2, 2,))
        assert_equal(y.mask.shape, (2, 2,))
        assert_equal(x.shape, (4,))
        assert_equal(x.mask.shape, (4,))

    def test_sort(self):
        # Test sort
        x = MaskedArray([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        sortedx = np.sort(x)
        assert_equal(sortedx.filled(9), [1, 2, 3, 9])
        assert_equal(sortedx.mask, [0, 0, 0, 1])

        x.sort()
        assert_equal(x.filled(9), [1, 2, 3, 9])
        assert_equal(x.mask, [0, 0, 0, 1])

    def test_stable_sort(self):
        x = MaskedArray([1, 2, 3, 1, 2, 3], dtype=np.uint8)
        expected = MaskedArray([0, 3, 1, 4, 2, 5])
        computed = np.argsort(x, kind='stable')
        assert_equal(computed, expected)

    def test_argsort_matches_sort(self):
        x = MaskedArray([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        sortedx = np.sort(x)
        argsortedx = x[np.argsort(x)]
        assert_equal(sortedx.filled(), argsortedx.filled())
        assert_equal(sortedx.mask, argsortedx.mask)

    def test_sort_2d(self):
        # Check sort of 2D array.
        # 2D array w/o mask
        a = MaskedArray([[8, 4, 1], [2, 0, 9]])
        a.sort(0)
        assert_equal(a.filled(), [[2, 0, 1], [8, 4, 9]])
        a = MaskedArray([[8, 4, 1], [2, 0, 9]])
        a.sort(1)
        assert_equal(a.filled(), [[1, 4, 8], [0, 2, 9]])
        # 2D array w/mask
        a = MaskedArray([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(0)
        assert_equal(a.filled(3), [[2, 0, 1], [3, 4, 3]])
        assert_equal(a.mask, [[0, 0, 0], [1, 0, 1]])
        a = MaskedArray([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(1)
        assert_equal(a.filled(3), [[1, 4, 3], [0, 2, 3]])
        assert_equal(a.mask, [[0, 0, 1], [0, 0, 1]])
        # 3D
        a = MaskedArray([[[7, 8, 9], [4, 5, 6], [1, 2, 3]],
                          [[1, 2, 3], [7, 8, 9], [4, 5, 6]],
                          [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
                          [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
        a[a % 4 == 0] = X
        am = a.copy()
        an = a.filled(99).copy()
        am.sort(0)
        an.sort(0)
        assert_equal(am.filled(99), an)
        am = a.copy()
        an = a.filled(99).copy()
        am.sort(1)
        an.sort(1)
        assert_equal(am.filled(99), an)
        am = a.copy()
        an = a.filled(99).copy()
        am.sort(2)
        an.sort(2)
        assert_equal(am.filled(99), an)

    def test_mask_sorting(self):
        # -1 gets converted to unsigned max val
        a = MaskedArray([[1,X,3], [X,-1,X], [1,X,-1]], dtype='u4')
        b = np.take_along_axis(a, np.argsort(a, axis=1), axis=1)
        ctrl = MaskedArray([[1, 3, X], [-1, X, X], [1, -1, X]], dtype='u4')
        assert_masked_equal(b, ctrl)
        c = a.copy()
        c.sort(axis=1)
        assert_masked_equal(b, c)
        assert_equal(np.lexsort((a,), axis=1), np.argsort(a, axis=1))


#    def test_sort_flexible(self):
#        # Test sort on structured dtype.
#        a = MaskedArray(
#            data=[(3, 3), (3, 2), (2, 2), (2, 1), (1, 0), (1, 1), (1, 2)],
#            mask=[(0, 0), (0, 1), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
#            dtype=[('A', int), ('B', int)])
#        mask_last = MaskedArray(
#            data=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (1, 0)],
#            mask=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 0)],
#            dtype=[('A', int), ('B', int)])
#        mask_first = MaskedArray(
#            data=[(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3)],
#            mask=[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0)],
#            dtype=[('A', int), ('B', int)])

#        test = sort(a)
#        assert_equal(test, mask_last)
#        assert_equal(test.mask, mask_last.mask)

#        test = sort(a, endwith=False)
#        assert_equal(test, mask_first)
#        assert_equal(test.mask, mask_first.mask)

#        # Test sort on dtype with subarray (gh-8069)
#        # Just check that the sort does not error, structured array subarrays
#        # are treated as byte strings and that leads to differing behavior
#        # depending on endianess and `endwith`.
#        dt = np.dtype([('v', int, 2)])
#        a = a.view(dt)
#        test = sort(a)
#        test = sort(a, endwith=False)

    def test_squeeze(self):
        # Check squeeze
        data = MaskedArray([[1, 2, 3]])
        assert_equal(data.squeeze().filled(), [1, 2, 3])
        data = MaskedArray([[1, 2, 3]], mask=[[1, 1, 1]])
        assert_equal(data.squeeze().filled(), [0, 0, 0])
        assert_equal(data.squeeze().mask, [1, 1, 1])

        # normal ndarrays return a view
        arr = np.array([[1]])
        arr_sq = arr.squeeze()
        assert_equal(arr_sq, 1)
        arr_sq[...] = 2
        assert_equal(arr[0,0], 2)

        # so maskedarrays should too
        m_arr = MaskedArray([[1]], mask=True)
        m_arr_sq = m_arr.squeeze()
        assert_(m_arr_sq.mask)
        m_arr_sq[...] = 2
        assert_equal(m_arr[0,0].filled(), 2)

    def test_swapaxes(self):
        # Tests swapaxes on MaskedArrays.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        mX = MaskedArray(x, mask=m).reshape(6, 6)
        mXX = mX.reshape(3, 2, 2, 3)

        mXswapped = mX.swapaxes(0, 1)
        assert_equal(mXswapped[-1], mX[:, -1])

        mXXswapped = mXX.swapaxes(0, 2)
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    def test_take(self):
        # Tests take
        x = MaskedArray([10, 20, 30, 40], [0, 1, 0, 1])
        assert_masked_equal(x.take([0, 0, 3]),
                            MaskedArray([10, 10, 40], [0, 0, 1]))
        assert_masked_equal(x.take([0, 0, 3]), x[[0, 0, 3]])
        assert_masked_equal(x.take([[0, 1], [0, 1]]),
                            MaskedArray([[10, 20], [10, 20]], [[0, 1], [0, 1]]))

        # assert_equal crashes when passed np.ma.mask
        assert_(x[1].mask)
        assert_(x.take(1).mask)

        x = MaskedArray([[10, 20, X], [X, 50, 60]])
        assert_masked_equal(x.take([0, 2], axis=1),
                            MaskedArray([[10,  X], [ X, 60]]))
        assert_masked_equal(np.take(x, [0, 2], axis=1),
                            MaskedArray([[10, X], [X, 60]]))

    #def test_take_masked_indices(self):
    #    # Test take w/ masked indices
    #    a = MaskedArray([40, 18, 37, 9, 22])
    #    indices = np.arange(3)[None,:] + np.arange(5)[:, None]
    #    # No mask
    #    test = np.take(a, indices, mode='clip')
    #    ctrl = MaskedArray([[40, 18, 37],
    #                  [18, 37, 9],
    #                  [37, 9, 22],
    #                  [9, 22, 22],
    #                  [22, 22, 22]])
    #    assert_equal(test, ctrl)

    #    # Masked indices raise
    #    mindices = MaskedArray(indices, mask=(indices >= len(a)))
    #    assert_raises(ValueError, np.take, a, mindices)

    #    # Masked values
    #    a = MaskedArray([40, 18, 37, 9, 22], mask=[0, 1, 0, 0, 0])
    #    test = np.take(a, indices)
    #    ctrl[0, 1] = ctrl[1, 0] = masked
    #    assert_equal(test, ctrl)
    #    assert_equal(test.mask, ctrl.mask)

    def test_tolist(self):
        # Tests to list
        # ... on 1D
        x = MaskedArray(np.arange(12))
        x[[1, -2]] = X
        xlist = x.tolist()
        assert_(isinstance(xlist[1], MaskedScalar))
        assert_(isinstance(xlist[-2], MaskedScalar))
        # ... on 2D
        x.shape = (3, 4)
        xlist = x.tolist()
        Xi = X(np.int64)
        ctrl = [[0, Xi, 2, 3], [4, 5, 6, 7], [8, 9, Xi, 11]]
        assert_masked_equal(xlist[0], [0, Xi, 2, 3])
        assert_masked_equal(xlist[1], [4, 5, 6, 7])
        assert_masked_equal(xlist[2], [8, 9, Xi, 11])
        assert_masked_equal(xlist, ctrl)

    def test_arraymethod(self):
        # Test a _arraymethod w/ n argument
        marray = MaskedArray([[1, 2, 3, 4, 5]], mask=[0, 0, 1, 0, 0])
        control = MaskedArray([[1], [2], [3], [4], [5]],
                               mask=[[0], [0], [1], [0], [0]])
        assert_equal(marray.T, control)
        assert_equal(marray.transpose(), control)

        assert_equal(MaskedArray.cumsum(marray.T, 0), control.cumsum(0))

    def test_arraymethod_0d(self):
        # gh-9430
        x = MaskedArray(42, mask=True)
        assert_equal(x.T.mask, x.mask)
        assert_equal(x.T.filled(), x.filled())

    def test_transpose_view(self):
        x = MaskedArray([[1, 2, 3], [4, 5, 6]])
        x[0,1] = X
        xt = x.T

        xt[1,0] = 10
        xt[0,1] = X

        assert_equal(x.filled(), xt.T.filled())
        assert_equal(x.mask, xt.T.mask)

    def test_diagonal_view(self):
        x = MaskedArray(np.zeros((3,3)))
        x[0,0] = 10
        x[1,1] = X
        x[2,2] = 20
        xd = x.diagonal()
        x[1,1] = 15
        assert_equal(xd.mask, x.diagonal().mask)
        assert_equal(xd.filled(), x.diagonal().filled())


class TestMaskedArrayMathMethods:

    def setup(self):
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        x2 = x.reshape(6, 6)
        x4 = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = MaskedArray(data=x, mask=m)
        mx2 = MaskedArray(data=x2, mask=m.reshape(x2.shape))
        mx4 = MaskedArray(data=x4, mask=m.reshape(x4.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = MaskedArray(data=x, mask=m2)
        m2x2 = MaskedArray(data=x2, mask=m2.reshape(x2.shape))
        m2x4 = MaskedArray(data=x4, mask=m2.reshape(x4.shape))
        self.d = (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4)

    def test_cumsumprod(self):
        # Tests cumsum & cumprod on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d
        mx2cp = mx2.cumsum(0)
        assert_equal(mx2cp.filled(), mx2.filled(0).cumsum(0))
        mx2cp = mx2.cumsum(1)
        assert_equal(mx2cp.filled(), mx2.filled(0).cumsum(1))

        mx2cp = mx2.cumprod(0)
        assert_equal(mx2cp.filled(1), mx2.filled(1).cumprod(0))
        mx2cp = mx2.cumprod(1)
        assert_equal(mx2cp.filled(1), mx2.filled(1).cumprod(1))

    def test_cumsumprod_with_output(self):
        # Tests cumsum/cumprod w/ output
        xm = MaskedArray(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = X

        for funcname in ('cumsum', 'cumprod'):
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)

            # A ndarray as explicit input
            output = MaskedArray(np.empty((3, 4), dtype=float))
            result = npfunc(xm, axis=0, out=output)
            # ... the result should be the given output
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))

            output = MaskedArray(np.empty((3, 4), dtype=int))
            result = xmmeth(axis=0, out=output)
            assert_(result is output)

    def test_ptp(self):
        # Tests ptp on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d
        (n, m) = x2.shape
        assert_equal(mx.ptp(), mx[~mx.mask].ptp())
        rows = np.zeros(n, float)
        cols = np.zeros(m, float)
        for k in range(m):
            a = mx2[:, k]
            cols[k] = a[~a.mask].ptp().filled()
        for k in range(n):
            a = mx2[k]
            rows[k] = a[~a.mask].ptp().filled()
        assert_equal(mx2.ptp(0), cols)
        assert_equal(mx2.ptp(1), rows)

    def test_add_object(self):
        x = MaskedArray(['a', 'b'], mask=[1, 0], dtype=object)
        y = x + 'x'
        assert_equal(y[1].filled(), 'bx')
        assert_(y.mask[0])

    def test_sum_object(self):
        # Test sum on object dtype
        a = MaskedArray([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.sum().filled(), 5)
        a = MaskedArray([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.sum(axis=0).filled(), [5, 7, 9])

    def test_prod_object(self):
        # Test prod on object dtype
        a = MaskedArray([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.prod().filled(), 2 * 3)
        a = MaskedArray([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.prod(axis=0).filled(), [4, 10, 18])

    def test_meananom_object(self):
        # Test mean/anom on object dtype
        a = MaskedArray([1, 2, 3], dtype=object)
        assert_equal(a.mean().filled(), 2)

    def test_trace(self):
        # Tests trace on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d
        mx2diag = mx2.diagonal()
        assert_equal(mx2.trace(), mx2diag[~mx2diag.mask].sum())
        assert_almost_equal(mx2.trace().filled(),
                            x2.trace() - np.sum(mx2diag.mask * x2.diagonal(),
                                            axis=0))
        assert_equal(np.trace(mx2), mx2.trace())

        # gh-5560
        arr = np.arange(2*4*4).reshape(2,4,4)
        m_arr = MaskedArray(arr, False)
        assert_equal(arr.trace(axis1=1, axis2=2),
                     m_arr.trace(axis1=1, axis2=2).filled())

    def test_dot(self):
        # Tests dot on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d
        fx = mx.filled(0)
        r = mx.dot(mx)
        assert_almost_equal(r.filled(0), fx.dot(fx))
        assert_(not r.mask.any())

        fx2 = mx2.filled(0)
        r = mx2.dot(mx2)
        assert_almost_equal(r.filled(0), fx2.dot(fx2))
        assert_(r.mask[1,3])
        r1 = np.empty_like(r)
        mx2.dot(mx2, out=r1)
        assert_almost_equal(r.filled(), r1.filled())

        mYY = mx4.swapaxes(-1, -2)
        fx4, fYY = mx4.filled(0), mYY.filled(0)
        r = mx4.dot(mYY)
        assert_almost_equal(r.filled(0), fx4.dot(fYY))
        r1 = np.empty_like(r)
        mx4.dot(mYY, out=r1)
        assert_almost_equal(r.filled(), r1.filled())

    def test_dot_shape_mismatch(self):
        # regression test
        x = MaskedArray([[1,2],[3,4]], mask=[[0,1],[0,0]])
        y = MaskedArray([[1,2],[3,4]], mask=[[0,1],[0,0]])
        z = MaskedArray([[0,1],[3,3]])
        x.dot(y, out=z)
        assert_almost_equal(z.filled(0), [[1, 0], [15, 16]])
        assert_almost_equal(z.mask, [[0, 1], [0, 0]])

    def test_varmean_nomask(self):
        # gh-5769
        foo = MaskedArray([1,2,3,4], dtype='f8')
        bar = MaskedArray([1,2,3,4], dtype='f8')
        assert_equal(type(foo.mean().filled()), np.float64)
        assert_equal(type(foo.var().filled()), np.float64)
        assert((foo.mean() == bar.mean()).filled() is np.bool_(True))

        # check array type is preserved and out works
        foo = MaskedArray(np.arange(16).reshape((4,4)), dtype='f8')
        bar = MaskedArray(np.empty(4, dtype='f4'))
        assert_equal(type(foo.mean(axis=1)), MaskedArray)
        assert_equal(type(foo.var(axis=1)), MaskedArray)
        assert_(foo.mean(axis=1, out=bar) is bar)
        assert_(foo.var(axis=1, out=bar) is bar)

    def test_varstd(self):
        # Tests var & std on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d
        assert_almost_equal(mx2.var(axis=None).filled(),
                            mx2[~mx2.mask].filled().var())
        assert_almost_equal(mx2.std(axis=None).filled(),
                            mx2[~mx2.mask].filled().std())
        assert_almost_equal(mx2.std(axis=None, ddof=1).filled(),
                            mx2[~mx2.mask].filled().std(ddof=1))
        assert_almost_equal(mx2.var(axis=None, ddof=1).filled(),
                            mx2[~mx2.mask].filled().var(ddof=1))
        assert_equal(mx4.var(axis=3).shape, x4.var(axis=3).shape)
        assert_equal(mx2.var().shape, x2.var().shape)
        (mx2var0, mx2var1) = (mx2.var(axis=0), mx2.var(axis=1))
        assert_almost_equal(mx2.var(axis=None, ddof=2).filled(),
                            mx2[~mx2.mask].filled().var(ddof=2))
        assert_almost_equal(mx2.std(axis=None, ddof=2).filled(),
                            mx2[~mx2.mask].filled().std(ddof=2))
        for k in range(6):
            assert_almost_equal(mx2var1[k].filled(),
                                mx2[k][~mx2[k].mask].filled().var())
            assert_almost_equal(mx2var0[k].filled(),
                                mx2[:,k][~mx2[:,k].mask].filled().var())
            assert_almost_equal(np.sqrt(mx2var0[k]).filled(),
                                mx2[:,k][~mx2[:,k].mask].filled().std())

    @pytest.mark.skipif(sys.platform=='win32' and sys.version_info < (3, 6),
                        reason='Fails on Python < 3.6 on Windows, gh-9671')
    def test_varstd_specialcases(self):
        # Test a special case for var
        nout = np.array(-1, dtype=float)
        mout = MaskedArray(-1, dtype=float)

        x = MaskedArray(np.arange(10), mask=True)
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            assert_(method().mask)
            assert_(method(0).mask)
            assert_(method(-1).mask)
            # Using a masked array as explicit output
            method(out=mout)
            assert_(mout.mask)

        x = MaskedArray(np.arange(10), mask=True)
        x[-1] = 9
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            assert_(method(ddof=1).mask)
            assert_(method(0, ddof=1).mask)
            assert_(method(-1, ddof=1).mask)
            # Using a masked array as explicit output
            method(out=mout, ddof=1)
            assert_(mout.mask)

    def test_varstd_ddof(self):
        a = MaskedArray([[1, 1, 0], [1, 1, 0]], mask=[[0, 0, 1], [0, 0, 1]])
        test = a.std(axis=0, ddof=0)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=1)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=2)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [1, 1, 1])

    def test_diag(self):
        # Test diag
        x = MaskedArray(np.arange(9).reshape((3, 3)))
        x[1, 1] = X
        out = np.diag(x)
        assert_equal(out, [0, 4, 8])
        out = np.diag(x)
        assert_equal(out, [0, 4, 8])
        assert_equal(out.mask, [0, 1, 0])
        out = np.diag(out)
        control = MaskedArray([[0, 0, 0], [0, 4, 0], [0, 0, 8]],
                        mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(out, control)

    def test_axis_methods_nomask(self):
        # Test the combination nomask & methods w/ axis
        a = MaskedArray([[1, 2, 3], [4, 5, 6]])

        assert_equal(a.sum(0), [5, 7, 9])
        assert_equal(a.sum(-1), [6, 15])
        assert_equal(a.sum(1), [6, 15])

        assert_equal(a.prod(0), [4, 10, 18])
        assert_equal(a.prod(-1), [6, 120])
        assert_equal(a.prod(1), [6, 120])

        assert_equal(a.min(0), [1, 2, 3])
        assert_equal(a.min(-1), [1, 4])
        assert_equal(a.min(1), [1, 4])

        assert_equal(a.max(0), [4, 5, 6])
        assert_equal(a.max(-1), [3, 6])
        assert_equal(a.max(1), [3, 6])


class TestMaskedArrayMathMethodsComplex:
    # Test class for miscellaneous MaskedArrays methods.
    def setup(self):
        # Base data definition.
        x = np.array([8.375j, 7.545j, 8.828j, 8.5j, 1.757j, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479j,
                      7.189j, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993j])
        x2 = x.reshape(6, 6)
        x4 = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = MaskedArray(data=x, mask=m)
        mx2 = MaskedArray(data=x2, mask=m.reshape(x2.shape))
        mx4 = MaskedArray(data=x4, mask=m.reshape(x4.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = MaskedArray(data=x, mask=m2)
        m2x2 = MaskedArray(data=x2, mask=m2.reshape(x2.shape))
        m2x4 = MaskedArray(data=x4, mask=m2.reshape(x4.shape))
        self.d = (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4)

    def test_varstd(self):
        # Tests var & std on MaskedArrays.
        (x, x2, x4, m, mx, mx2, mx4, m2x, m2x2, m2x4) = self.d
        assert_almost_equal(mx2.var(axis=None), mx2[~mx2.mask].var())
        assert_almost_equal(mx2.std(axis=None), mx2[~mx2.mask].std())
        assert_equal(mx4.var(axis=3).shape, x4.var(axis=3).shape)
        assert_equal(mx2.var().shape, x2.var().shape)
        (mx2var0, mx2var1) = (mx2.var(axis=0), mx2.var(axis=1))
        assert_almost_equal(mx2.var(axis=None, ddof=2),
                            mx2[~mx2.mask].var(ddof=2))
        assert_almost_equal(mx2.std(axis=None, ddof=2),
                            mx2[~mx2.mask].std(ddof=2))
        for k in range(6):
            assert_almost_equal(mx2var1[k], mx2[k][~mx2[k].mask].var())
            assert_almost_equal(mx2var0[k], mx2[:,k][~mx2[:,k].mask].var())
            assert_almost_equal(np.sqrt(mx2var0[k]),
                                mx2[:,k][~mx2[:,k].mask].std())


class TestMaskedArrayFunctions:
    # Test class for miscellaneous functions.

    def setup(self):
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = MaskedArray(x.copy(), mask=m1)
        ym = MaskedArray(y.copy(), mask=m2)
        self.info = (xm, ym)

    def test_round(self):
        a = MaskedArray([1.23456, 2.34567, 3.45678, 4.56789, 5.67890],
                  mask=[0, 1, 0, 0, 0])
        MA = MaskedArray
        assert_masked_equal(a.round(), MA([1., X , 3., 5., 6.]))
        assert_masked_equal(a.round(1), MA([1.2, X , 3.5, 4.6, 5.7]))
        assert_masked_equal(a.round(3), MA([1.235, X , 3.457, 4.568, 5.679]))
        b = np.empty_like(a)
        a.round(out=b)
        assert_masked_equal(b, MaskedArray([1., X, 3., 5., 6.]))

        x = MaskedArray([1., 2., 3., 4., 5.])
        c = MaskedArray([1, 1, 1, 0, 0])
        x[2] = X
        z = np.where(c, x, -x)
        assert_masked_equal(z, MA([1., 2., X , -4., -5]))
        c[0] = X
        z = np.where(c, x, -x)
        assert_masked_equal(z, MA([-1., 2., X , -4., -5]))
        assert_(not z[0].mask)
        assert_(not z[1].mask)
        assert_(z[2].mask)

    def test_round_with_output(self):
        # Testing round with an explicit output

        xm = MaskedArray(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = X

        output = MaskedArray(np.empty((3, 4), dtype=float))
        result = xm.round(decimals=2, out=output)
        assert_(result is output)

    def test_round_with_scalar(self):
        # Testing round with scalar/zero dimension input
        # GH issue 2244
        a = MaskedArray(1.1, mask=False)
        assert_masked_equal(a.round(), 1)

        a = MaskedArray(1.1, mask=True)
        assert_(a.round().mask)

        a = MaskedArray(1.1, mask=False)
        output = MaskedArray(-9999., mask=True)
        a.round(out=output)
        assert_equal(output[()].filled(), 1)

        a = MaskedArray(1.1, mask=True)
        output = MaskedArray(-9999., mask=False)
        a.round(out=output)
        assert_(output[()].mask)

    def test_power(self):
        x = MaskedScalar(-1.1)
        assert_almost_equal(np.power(x, 2.).filled(), 1.21)
        assert_(np.power(x, X).mask)
        x = MaskedArray([-1.1, -1.1, 1.1, 1.1, 0.,   X])
        b = MaskedArray([ 0.5,   2., 0.5,  2.,  X, 1.0])
        with pytest.warns(RuntimeWarning, match='invalid value'):
            y = np.power(x, b)
        assert_almost_masked_equal(y,
                              MaskedArray([np.nan, 1.21, 1.1**0.5, 1.21, X, X]))
        b = MaskedArray([0.5, 2., 0.5, 2., -1., -1])
        with pytest.warns(RuntimeWarning) as record:
            y = np.power(x, b)
        assert(len(record) == 2) # invalid val, div by zero
        assert_almost_masked_equal(y,
                         MaskedArray([np.nan, 1.21, 1.1**0.5, 1.21, np.inf, X]))
        with pytest.warns(RuntimeWarning) as record:
            z = x ** b
        assert(len(record) == 2) # invalid val, div by zero
        assert_masked_equal(z, y)
        with pytest.warns(RuntimeWarning) as record:
            x **= b
        assert(len(record) == 2) # invalid val, div by zero
        assert_masked_equal(x, y)

    def test_power_with_broadcasting(self):
        # Test power w/ broadcasting
        a2 = np.array([[1., 2., 3.], [4., 5., 6.]])
        a2m = MaskedArray(a2, mask=[[1, 0, 0], [0, 0, 1]])
        b1 = np.array([2, 4, 3])
        b2 = np.array([b1, b1])
        b2m = MaskedArray(b2, mask=[[0, 1, 0], [0, 1, 0]])

        ctrl_dat = [[1 ** 2, 2 ** 4, 3 ** 3], [4 ** 2, 5 ** 4, 6 ** 3]]
        # No broadcasting, base & exp w/ mask
        test = a2m ** b2m
        ctrl = MaskedArray(ctrl_dat, mask=[[1, 1, 0], [0, 1, 1]])
        assert_masked_equal(test, ctrl)
        # No broadcasting, base w/ mask, exp w/o mask
        test = a2m ** b2
        ctrl = MaskedArray(ctrl_dat, mask=[[1, 0, 0], [0, 0, 1]])
        assert_masked_equal(test, ctrl)
        # No broadcasting, base w/o mask, exp w/ mask
        test = a2 ** b2m
        ctrl = MaskedArray(ctrl_dat, mask=[[0, 1, 0], [0, 1, 0]])
        assert_masked_equal(test, ctrl)

        ctrl = MaskedArray([[2 ** 2, 4 ** 4, 3 ** 3], [2 ** 2, 4 ** 4, 3 ** 3]],
                     mask=[[0, 1, 0], [0, 1, 0]])
        test = b1 ** b2m
        assert_masked_equal(test, ctrl)
        test = b2m ** b1
        assert_masked_equal(test, ctrl)

    def test_where(self):
        # Test the where function
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = MaskedArray(x, mask=m1)
        ym = MaskedArray(y, mask=m2)

        d = np.where(xm > 2, xm, -9)
        assert_masked_equal(d, [-9., -9., -9., -9., -9., 4.,
                                -9., -9., 10., -9., -9., 3.])
        d = np.where(xm > 2, -9, ym)
        assert_masked_equal(d, MaskedArray(
                [5., 0., X , 2., -1., -9.,  X , -10., -9., 1., 0., -9.]))
        d = np.where(xm > 2, xm, X)
        assert_masked_equal(d, MaskedArray([ X ,  X ,  X ,  X ,  X , 4.,
                                             X ,  X , 10.,  X ,  X , 3.]))
        tmp = xm._mask.copy()
        tmp[(xm <= 2).filled(True)] = True
        assert_equal(d._mask, tmp)

        ixm = xm.astype(int)
        d = np.where(ixm > 2, ixm, X)
        assert_masked_equal(d, MaskedArray([ X,  X,  X,  X,  X, 4,
                                             X,  X, 10,  X,  X, 3]))
        assert_equal(d.dtype, ixm.dtype)

    def test_where_object(self):
        a = np.array(None)
        b = MaskedArray(None)
        r = b.copy()
        assert_masked_equal(np.where(True, a, a), r)
        assert_masked_equal(np.where(True, b, b), r)

    def test_where_with_masked_choice(self):
        x = MaskedArray(np.arange(10))
        x[3] = X
        c = x >= 8
        # Set False to masked
        z = np.where(c, x, X)
        assert_(z.dtype is x.dtype)
        assert_(z[3].mask)
        assert_(z[4].mask)
        assert_(z[7].mask)
        assert_(not z[8].mask)
        assert_(not z[9].mask)
        assert_equal(x, z)
        # Set True to masked
        z = np.where(c, X, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3].mask)
        assert_(not z[4].mask)
        assert_(not z[7].mask)
        assert_(z[8].mask)
        assert_(z[9].mask)

#    def test_where_with_masked_condition(self):
#        x = MaskedArray([1., 2., 3., 4., 5.])
#        c = MaskedArray([1, 1, 1, 0, 0])
#        x[2] = masked
#        z = where(c, x, -x)
#        assert_equal(z, [1., 2., 0., -4., -5])
#        c[0] = masked
#        z = where(c, x, -x)
#        assert_equal(z, [1., 2., 0., -4., -5])
#        assert_(z[0] is masked)
#        assert_(z[1] is not masked)
#        assert_(z[2] is masked)

#        x = arange(1, 6)
#        x[-1] = masked
#        y = arange(1, 6) * 10
#        y[2] = masked
#        c = MaskedArray([1, 1, 1, 0, 0], mask=[1, 0, 0, 0, 0])
#        cm = c.filled(1)
#        z = where(c, x, y)
#        zm = where(cm, x, y)
#        assert_equal(z, zm)
#        assert_(getmask(zm) is nomask)
#        assert_equal(zm, [1, 2, 3, 40, 50])
#        z = where(c, masked, 1)
#        assert_equal(z, [99, 99, 99, 1, 1])
#        z = where(c, 1, masked)
#        assert_equal(z, [99, 1, 1, 99, 99])

    def test_where_type(self):
        # Test the type conservation with where
        x = MaskedArray(np.arange(4, dtype=np.int32))
        y = MaskedArray(np.arange(4, dtype=np.float32)) * 2.2
        test = np.where(x > 1.5, y, x).dtype
        control = np.find_common_type([np.int32, np.float32], [])
        assert_equal(test, control)

    def test_where_broadcast(self):
        # Issue 8599
        x = np.arange(9).reshape(3, 3)
        y = np.zeros(3)
        core = np.where([1, 0, 1], x, y)
        ma = np.where([1, 0, 1], MaskedArray(x), MaskedArray(y))

        assert_equal(core, ma.filled())
        assert_equal(core.dtype, ma.dtype)

    def test_where_structured(self):
        # Issue 8600
        dt = np.dtype([('a', int), ('b', int)])
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)
        y = np.array((10, 20), dtype=dt)
        core = np.where([0, 1, 1], x, y)
        ma = np.where([0, 1, 1], MaskedArray(x), MaskedArray(y))

        assert_equal(core, ma.filled())
        assert_equal(core.dtype, ma.dtype)

    def test_choose(self):
        # Test choose
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        chosen = np.choose([2, 3, 1, 0], choices)
        assert_equal(chosen, MaskedArray([20, 31, 12, 3]))
        chosen = np.choose([2, 4, 1, 0], choices, mode='clip')
        assert_equal(chosen, MaskedArray([20, 31, 12, 3]))
        chosen = np.choose([2, 4, 1, 0], choices, mode='wrap')
        assert_equal(chosen, MaskedArray([20, 1, 12, 3]))
        # XXX masked indices not supported
        ## Check with some masked indices
        #indices_ = MaskedArray([2, 4, 1, 0], mask=[1, 0, 0, 1])
        #chosen = np.choose(indices_, choices, mode='wrap')
        #assert_equal(chosen, MaskedArray([99, 1, 12, 99]))
        #assert_equal(chosen.mask, [1, 0, 0, 1])
        # Check with some masked choices
        choices = MaskedArray(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        chosen = np.choose(indices_, choices, mode='wrap')
        assert_equal(chosen, MaskedArray([20, 31, 12, 3]))
        assert_equal(chosen.mask, [1, 0, 0, 1])

    def test_choose_with_out(self):
        # Test choose with an explicit out keyword
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        #store = np.empty(4, dtype=int)
        #chosen = np.choose([2, 3, 1, 0], choices, out=store)
        #assert_equal(store, MaskedArray([20, 31, 12, 3]))
        #assert_(store is chosen)
        # XXX
        ## Check with some masked indices + out
        #store = empty(4, dtype=int)
        #indices_ = MaskedArray([2, 3, 1, 0], mask=[1, 0, 0, 1])
        #chosen = choose(indices_, choices, mode='wrap', out=store)
        #assert_equal(store, MaskedArray([99, 31, 12, 99]))
        #assert_equal(store.mask, [1, 0, 0, 1])
        # Check with some masked choices + out ina ndarray !
        choices = MaskedArray(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        store = MaskedArray(np.empty(4, dtype=int))
        chosen = np.choose(indices_, choices, mode='wrap', out=store)
        assert_masked_equal(store, MaskedArray([X, 31, 12, X]))

    def test_reshape(self):
        a = MaskedArray(np.arange(10))
        a[0] = X
        # Try the default
        b = a.reshape((5, 2))
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        # Try w/ arguments as list instead of tuple
        b = a.reshape(5, 2)
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        # Try w/ order
        b = a.reshape((5, 2), order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])
        # Try w/ order
        b = a.reshape(5, 2, order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])

        c = np.reshape(a, (2, 5))
        assert_(isinstance(c, MaskedArray))
        assert_equal(c.shape, (2, 5))
        assert_(c[0, 0].mask)
        assert_(c.flags['C'])

    def test_compress(self):
        # Test compress function on ndarray and masked array
        # Address Github #2495.
        arr = MaskedArray(np.arange(8))
        arr.shape = 4, 2
        cond = np.array([True, False, True, True])
        control = arr[[0, 2, 3]]
        test = np.compress(cond, arr, axis=0)
        assert_masked_equal(test, control)

    def test_convolve(self):
        a = MaskedArray(np.arange(5))
        a[2] = X
        b = np.array([1, 1])
        test = np.convolve(a, b)
        assert_masked_equal(test, MaskedArray([0, 1, 1, 3, 7, 4]))

        test = np.convolve(MaskedArray([1, 1]), MaskedArray([1, 1, 1]))
        assert_equal(test, np.array([1, 2, 2, 1]))

        # XXX implement propagate_mask?
        #a = [1, 1]
        #b = MaskedArray([1, X, X, 1])
        #test = np.ma.convolve(a, b, propagate_mask=False)
        #assert_equal(test, masked_equal([1, 1, -1, 1, 1], -1))
        #test = np.ma.convolve(a, b, propagate_mask=True)
        #assert_equal(test, masked_equal([-1, -1, -1, -1, -1], -1))

    def test_select(self):
        x = MaskedArray([0, 1, X, 2, X, 4, 5])
        choicelist = [x, x**2]
        condlist = [(x<3).filled(False), (x>5).filled(True)]

        assert_masked_equal(np.select(condlist, choicelist),
                            MaskedArray([0, 1, X, 2, X, 0, 0]))
        assert_masked_equal(np.select(condlist, choicelist, default=-9),
                            MaskedArray([0, 1, X, 2, X, -9, -9]))
        assert_masked_equal(np.select(condlist, choicelist, default=X),
                            MaskedArray([0, 1, X, 2, X, X, X]))

    def test_unique(self):
        a = MaskedArray([2,X,0,X])
        u, i, c = np.unique(a, return_inverse=True, return_counts=True)
        assert_equal(u, MaskedArray([0, 2, X]))
        assert_equal(i, np.array([1, 2, 0, 2]))
        assert_equal(c, np.array([1, 1, 2]))

        b = MaskedArray([X,X], dtype='f')
        u, i, c = np.unique(b, return_inverse=True, return_counts=True)
        assert_equal(u, MaskedArray([X], dtype=u.dtype))
        assert_equal(i, np.array([0, 0]))
        assert_equal(c, np.array([2]))

    def test_broadcast_to(self):
        # make sure scalars work
        assert_equal(np.broadcast_to(X('f'), (3,4)).filled(1), np.ones((3,4)))

    def test_interp(self):
        v = np.interp(np.linspace(0,12,10), [1,2,4,7,10],
                   MaskedArray([1, 2, X, 2, 10]), left=4, right=X)
        assert_almost_masked_equal(v, MaskedArray([4., 1.33333333, X, X, X, X,
                                            4.66666667, 8.22222222, X, X ]))

    def test_quntile(self):
        data = np.array([[[0.25515513, 0.81984257, 0.26292477],
                          [0.98868172, 0.13840085, 0.92959184],
                          [0.61194388, 0.80812464, 0.58633275]],
                         [[0.82612799, 0.02958672, 0.64152733],
                          [0.12933993, 0.71415531, 0.46514879],
                          [0.70051523, 0.76018618, 0.0172671 ]],
                         [[0.84570913, 0.62145495, 0.01087859],
                          [0.52438135, 0.82609156, 0.21855111],
                          [0.82448953, 0.77798911, 0.3643823 ]]])
        x = MaskedArray(data)
        x[:,:,-1] = X
        d = data[:,:,:-1]

        # test scalar case
        res = np.quantile(x, 0.2)
        assert_(isinstance(res, MaskedScalar))
        assert_equal(res.filled(), np.quantile(d, 0.2))

        # test different combinations of axis and q
        assert_equal(np.quantile(x, [0.2, 0.5]).filled(),
                     np.quantile(d, [0.2, 0.5]))
        assert_equal(np.quantile(x, [0.2, 0.5], axis=2).filled(),
                     np.quantile(d, [0.2, 0.5], axis=2))
        assert_equal(np.quantile(x, 0.2, axis=(1,2)).filled(),
                     np.quantile(d, 0.2, axis=(1,2)))
        assert_equal(np.quantile(x, [0.2, 0.5], axis=(1,2)).filled(),
                     np.quantile(d, [0.2, 0.5], axis=(1,2)))
        res = MaskedArray(np.empty((2,3,3)), mask=True)
        res[:,:,:-1] = np.quantile(d, [0.2, 0.5], axis=1)
        assert_masked_equal(np.quantile(x, [0.2, 0.5], axis=1), res)

        # test keepdims
        assert_equal(np.quantile(x, 0.2, axis=1, keepdims=True).shape, (3,1,3))
        assert_equal(np.quantile(x, [0.2,0.5], axis=1, keepdims=True).shape,
                     (2, 3, 1, 3))
        assert_equal(np.quantile(x, [0.2,0.5], axis=(1,2), keepdims=True).shape,
                     (2, 3, 1, 1))
        assert_equal(np.quantile(x, 0.2, axis=(1,2), keepdims=True).shape,
                     (3, 1, 1))


class TestMaskedObjectArray:

    def test_getitem(self):
        arr = MaskedArray([None, None])
        for dt in [float, object]:
            a0 = np.eye(2).astype(dt)
            a1 = np.eye(3).astype(dt)
            arr[0] = a0
            arr[1] = a1

            assert_(arr[0].filled() is a0)
            assert_(arr[1].filled() is a1)
            assert_(isinstance(arr[0,...], MaskedArray))
            assert_(isinstance(arr[1,...], MaskedArray))
            assert_(arr[0,...][()].filled() is a0)
            assert_(arr[1,...][()].filled() is a1)

            arr[0] = X

            assert_(arr[1].filled() is a1)
            assert_(isinstance(arr[0,...], MaskedArray))
            assert_(isinstance(arr[1,...], MaskedArray))
            assert_equal(arr[0,...].mask, True)
            assert_(arr[1,...][()].filled() is a1)

            # gh-5962 - object arrays of arrays do something special
            assert_equal(arr[0].filled(), 0)
            assert_equal(arr[0].mask, True)
            assert_equal(arr[0,...][()].filled(), 0)
            assert_equal(arr[0,...][()].mask, True)

    # XXX do we want to support this weird case?
    #def test_nested_ma(self):

    #    arr = MaskedArray([None, None])
    #    # set the first object to be an unmasked masked constant. A little fiddly
    #    arr[0,...] = np.array([X], object)[0,...]

    #    # check the above line did what we were aiming for
    #    assert_(arr.filled(None)[0] is None)

    #    # test that getitem returned the value by identity
    #    assert_(arr[0].mask)

    #    # now mask the masked value!
    #    arr[0] = X
    #    assert_(arr[0].mask)


#class TestMaskedView:

#    def setup(self):
#        iterator = list(zip(np.arange(10), np.random.rand(10)))
#        data = np.array(iterator)
#        a = MaskedArray(iterator, dtype=[('a', float), ('b', float)])
#        a.mask[0] = (1, 0)
#        controlmask = np.array([1] + 19 * [0], dtype=bool)
#        self.data = (data, a, controlmask)

#    def test_view_to_nothing(self):
#        (data, a, controlmask) = self.data
#        test = a.view()
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test._data, a._data)
#        assert_equal(test._mask, a._mask)

#    def test_view_to_type(self):
#        (data, a, controlmask) = self.data
#        test = a.view(np.ndarray)
#        assert_(not isinstance(test, MaskedArray))
#        assert_equal(test, a._data)
#        assert_equal_records(test, data.view(a.dtype).squeeze())

#    def test_view_to_simple_dtype(self):
#        (data, a, controlmask) = self.data
#        # View globally
#        test = a.view(float)
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test, data.ravel())
#        assert_equal(test.mask, controlmask)

#    def test_view_to_flexible_dtype(self):
#        (data, a, controlmask) = self.data

#        test = a.view([('A', float), ('B', float)])
#        assert_equal(test.mask.dtype.names, ('A', 'B'))
#        assert_equal(test['A'], a['a'])
#        assert_equal(test['B'], a['b'])

#        test = a[0].view([('A', float), ('B', float)])
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test.mask.dtype.names, ('A', 'B'))
#        assert_equal(test['A'], a['a'][0])
#        assert_equal(test['B'], a['b'][0])

#        test = a[-1].view([('A', float), ('B', float)])
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test.dtype.names, ('A', 'B'))
#        assert_equal(test['A'], a['a'][-1])
#        assert_equal(test['B'], a['b'][-1])

#    def test_view_to_subdtype(self):
#        (data, a, controlmask) = self.data
#        # View globally
#        test = a.view((float, 2))
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test, data)
#        assert_equal(test.mask, controlmask.reshape(-1, 2))
#        # View on 1 masked element
#        test = a[0].view((float, 2))
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test, data[0])
#        assert_equal(test.mask, (1, 0))
#        # View on 1 unmasked element
#        test = a[-1].view((float, 2))
#        assert_(isinstance(test, MaskedArray))
#        assert_equal(test, data[-1])

#    def test_view_to_dtype_and_type(self):
#        (data, a, controlmask) = self.data

#        test = a.view((float, 2), np.recarray)
#        assert_equal(test, data)
#        assert_(isinstance(test, np.recarray))
#        assert_(not isinstance(test, MaskedArray))


class TestOptionalArgs:
    def test_ndarrayfuncs(self):
        # test axis arg behaves the same as ndarray (including multiple axes)

        d = np.arange(24.0).reshape((2,3,4))
        m = np.zeros(24, dtype=bool).reshape((2,3,4))
        # mask out last element of last dimension
        a = MaskedArray(d, mask=m)

        def testaxis(f, a, d):
            f = numpy.__getattribute__(f)

            # test axis arg
            assert_masked_equal(f(a[...,:-1], axis=1), f(d[...,:-1], axis=1))
            assert_masked_equal(f(a[...,:-1], axis=(0,1)),
                                f(d[...,:-1], axis=(0,1)))

        def testkeepdims(f, a, d):
            f = numpy.__getattribute__(f)

            # test keepdims arg
            assert_equal(f(a, keepdims=True).shape,
                         f(d, keepdims=True).shape)
            assert_equal(f(a, keepdims=False).shape,
                         f(d, keepdims=False).shape)

            # test both at once
            assert_masked_equal(f(a[...,:-1], axis=1, keepdims=True),
                                f(d[...,:-1], axis=1, keepdims=True))
            assert_masked_equal(f(a[...,:-1], axis=(0,1), keepdims=True),
                                f(d[...,:-1], axis=(0,1), keepdims=True))

        for f in ['sum', 'prod', 'mean', 'var', 'std']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

        for f in ['min', 'max']:
            testaxis(f, a, d)

        d = (np.arange(24).reshape((2,3,4))%2 == 0)
        a = MaskedArray(d, mask=m)
        for f in ['all', 'any']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

#    def test_count(self):
#        # test np.ma.count specially

#        d = np.arange(24.0).reshape((2,3,4))
#        m = np.zeros(24, dtype=bool).reshape((2,3,4))
#        m[:,0,:] = True
#        a = np.ma.array(d, mask=m)

#        assert_equal(count(a), 16)
#        assert_equal(count(a, axis=1), 2*ones((2,4)))
#        assert_equal(count(a, axis=(0,1)), 4*ones((4,)))
#        assert_equal(count(a, keepdims=True), 16*ones((1,1,1)))
#        assert_equal(count(a, axis=1, keepdims=True), 2*ones((2,1,4)))
#        assert_equal(count(a, axis=(0,1), keepdims=True), 4*ones((1,1,4)))
#        assert_equal(count(a, axis=-2), 2*ones((2,4)))
#        assert_raises(ValueError, count, a, axis=(1,1))
#        assert_raises(np.AxisError, count, a, axis=3)

#        # check the 'nomask' path
#        a = np.ma.array(d, mask=nomask)

#        assert_equal(count(a), 24)
#        assert_equal(count(a, axis=1), 3*ones((2,4)))
#        assert_equal(count(a, axis=(0,1)), 6*ones((4,)))
#        assert_equal(count(a, keepdims=True), 24*ones((1,1,1)))
#        assert_equal(np.ndim(count(a, keepdims=True)), 3)
#        assert_equal(count(a, axis=1, keepdims=True), 3*ones((2,1,4)))
#        assert_equal(count(a, axis=(0,1), keepdims=True), 6*ones((1,1,4)))
#        assert_equal(count(a, axis=-2), 3*ones((2,4)))
#        assert_raises(ValueError, count, a, axis=(1,1))
#        assert_raises(np.AxisError, count, a, axis=3)

#        # check the 'masked' singleton
#        assert_equal(count(np.ma.masked), 0)

#        # check 0-d arrays do not allow axis > 0
#        assert_raises(np.AxisError, count, np.ma.array(1), axis=1)


#class TestMaskedConstant:
#    def _do_add_test(self, add):
#        # sanity check
#        assert_(add(np.ma.masked, 1) is np.ma.masked)

#        # now try with a vector
#        vector = np.array([1, 2, 3])
#        result = add(np.ma.masked, vector)

#        # lots of things could go wrong here
#        assert_(result is not np.ma.masked)
#        assert_(not isinstance(result, np.ma.core.MaskedConstant))
#        assert_equal(result.shape, vector.shape)
#        assert_equal(np.ma.getmask(result), np.ones(vector.shape, dtype=bool))

#    def test_ufunc(self):
#        self._do_add_test(np.add)

#    def test_operator(self):
#        self._do_add_test(lambda a, b: a + b)

#    def test_ctor(self):
#        m = np.ma.array(np.ma.masked)

#        # most importantly, we do not want to create a new MaskedConstant
#        # instance
#        assert_(not isinstance(m, np.ma.core.MaskedConstant))
#        assert_(m is not np.ma.masked)

#    def test_repr(self):
#        # copies should not exist, but if they do, it should be obvious that
#        # something is wrong
#        assert_equal(repr(np.ma.masked), 'masked')

#        # create a new instance in a weird way
#        masked2 = np.ma.MaskedArray.__new__(np.ma.core.MaskedConstant)
#        assert_not_equal(repr(masked2), 'masked')

#    def test_pickle(self):
#        from io import BytesIO

#        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
#            with BytesIO() as f:
#                pickle.dump(np.ma.masked, f, protocol=proto)
#                f.seek(0)
#                res = pickle.load(f)
#            assert_(res is np.ma.masked)

#    def test_copy(self):
#        # gh-9328
#        # copy is a no-op, like it is with np.True_
#        assert_equal(
#            np.ma.masked.copy() is np.ma.masked,
#            np.True_.copy() is np.True_)

#    def test__copy(self):
#        import copy
#        assert_(
#            copy.copy(np.ma.masked) is np.ma.masked)

#    def test_deepcopy(self):
#        import copy
#        assert_(
#            copy.deepcopy(np.ma.masked) is np.ma.masked)

#    def test_immutable(self):
#        orig = np.ma.masked
#        assert_raises(np.ma.core.MaskError, operator.setitem, orig, (), 1)
#        assert_raises(ValueError,operator.setitem, orig.data, (), 1)
#        assert_raises(ValueError, operator.setitem, orig.mask, (), False)

#        view = np.ma.masked.view(np.ma.MaskedArray)
#        assert_raises(ValueError, operator.setitem, view, (), 1)
#        assert_raises(ValueError, operator.setitem, view.data, (), 1)
#        assert_raises(ValueError, operator.setitem, view.mask, (), False)

#    def test_coercion_int(self):
#        a_i = np.zeros((), int)
#        assert_raises(MaskError, operator.setitem, a_i, (), np.ma.masked)
#        assert_raises(MaskError, int, np.ma.masked)

#    @pytest.mark.skipif(sys.version_info.major == 3,
#                        reason="long doesn't exist in Python 3")
#    def test_coercion_long(self):
#        assert_raises(MaskError, long, np.ma.masked)

#    def test_coercion_float(self):
#        a_f = np.zeros((), float)
#        assert_warns(UserWarning, operator.setitem, a_f, (), np.ma.masked)
#        assert_(np.isnan(a_f[()]))

#    @pytest.mark.xfail(reason="See gh-9750")
#    def test_coercion_unicode(self):
#        a_u = np.zeros((), 'U10')
#        a_u[()] = np.ma.masked
#        assert_equal(a_u[()], u'--')

#    @pytest.mark.xfail(reason="See gh-9750")
#    def test_coercion_bytes(self):
#        a_b = np.zeros((), 'S10')
#        a_b[()] = np.ma.masked
#        assert_equal(a_b[()], b'--')

#    def test_subclass(self):
#        # https://github.com/astropy/astropy/issues/6645
#        class Sub(type(np.ma.masked)): pass

#        a = Sub()
#        assert_(a is Sub())
#        assert_(a is not np.ma.masked)
#        assert_not_equal(repr(a), 'masked')

#    def test_attributes_readonly(self):
#        assert_raises(AttributeError, setattr, np.ma.masked, 'shape', (1,))
#        assert_raises(AttributeError, setattr, np.ma.masked, 'dtype', np.int64)


#class TestMaskedWhereAliases:

#    # TODO: Test masked_object, masked_equal, ...

#    def test_masked_values(self):
#        res = masked_values(np.array([-32768.0]), np.int16(-32768))
#        assert_equal(res.mask, [True])

#        res = masked_values(np.inf, np.inf)
#        assert_equal(res.mask, True)

#        res = np.ma.masked_values(np.inf, -np.inf)
#        assert_equal(res.mask, False)

#        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=True)
#        assert_(res.mask is np.ma.nomask)

#        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=False)
#        assert_equal(res.mask, [False] * 4)


def test_MaskedArray():
    a = MaskedArray([0, 1, 2, 3], mask=[0, 0, 1, 0])
    assert_equal(np.argwhere(a), [[1], [3]])

def test_append_MaskedArray():
    a = MaskedArray([1,X,3])
    b = MaskedArray([4,3,X])

    result = np.append(a, b)
    expected_data = [1, 2, 3, 4, 3, 2]
    expected_mask = [False, True, False, False, False, True]
    assert_masked_equal(result, MaskedArray(expected_data, expected_mask))

    a = MaskedArray([[X,X],[X,X]], dtype='f')
    b = MaskedArray(np.ones((3,1)))

    result = np.append(a, b)
    assert_masked_equal(result, MaskedArray([X, X, X, X, 1, 1, 1]))

    result = np.append(a, b, axis=None)
    assert_masked_equal(result, MaskedArray([X, X, X, X, 1, 1, 1]))


def test_append_MaskedArray_along_axis():
    a = MaskedArray([1,X,3])
    b = MaskedArray([[4, 5, 6], [X, 8, 9]])

    # When `axis` is specified, `values` must have the correct shape.
    assert_raises(ValueError, np.append, a, b, axis=0)

    result = np.append(a[np.newaxis,:], b, axis=0)
    expected = MaskedArray(np.arange(1, 10))
    expected[[1, 6]] = X
    expected = expected.reshape((3,3))
    assert_masked_equal(result, expected)


def test_ufunc_with_output():
    # check that giving an output argument always returns that output.
    # Regression test for gh-8416.
    x = MaskedArray([1., 2., 3.], mask=[0, 0, 1])
    y = np.add(x, 1., out=x)
    assert_(y is x)


def test_ufunc_with_out_varied():
    """ Test that masked arrays are immune to gh-10459 """
    # the mask of the output should not affect the result, however it is passed
    a        = MaskedArray([ 1,  2,  3], mask=[1, 0, 0])
    b        = MaskedArray([10, 20, 30], mask=[1, 0, 0])
    out      = MaskedArray([ 0,  0,  0], mask=[0, 0, 1])
    expected = MaskedArray([11, 22, 33], mask=[1, 0, 0])

    out_pos = out.copy()
    res_pos = np.add(a, b, out_pos)

    out_kw = out.copy()
    res_kw = np.add(a, b, out=out_kw)

    out_tup = out.copy()
    res_tup = np.add(a, b, out=(out_tup,))

    assert_masked_equal(res_kw, expected)
    assert_masked_equal(res_tup, expected)
    assert_masked_equal(res_pos, expected)


#def test_astype_mask_ordering():
#    descr = [('v', int, 3), ('x', [('y', float)])]
#    x = MaskedArray([
#        [([1, 2, 3], (1.0,)),  ([1, 2, 3], (2.0,))],
#        [([1, 2, 3], (3.0,)),  ([1, 2, 3], (4.0,))]], dtype=descr)
#    x[0]['v'][0] = np.ma.masked

#    x_a = x.astype(descr)
#    assert x_a.dtype.names == np.dtype(descr).names
#    assert x_a.mask.dtype.names == np.dtype(descr).names
#    assert_equal(x, x_a)

#    assert_(x is x.astype(x.dtype, copy=False))
#    assert_equal(type(x.astype(x.dtype, subok=False)), np.ndarray)

#    x_f = x.astype(x.dtype, order='F')
#    assert_(x_f.flags.f_contiguous)
#    assert_(x_f.mask.flags.f_contiguous)

#    # Also test the same indirectly, via np.array
#    x_a2 = np.array(x, dtype=descr, subok=True)
#    assert x_a2.dtype.names == np.dtype(descr).names
#    assert x_a2.mask.dtype.names == np.dtype(descr).names
#    assert_equal(x, x_a2)

#    assert_(x is np.array(x, dtype=descr, copy=False, subok=True))

#    x_f2 = np.array(x, dtype=x.dtype, order='F', subok=True)
#    assert_(x_f2.flags.f_contiguous)
#    assert_(x_f2.mask.flags.f_contiguous)


@pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
@pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
def test_astype_basic(dt1, dt2):
    # See gh-12070
    src = MaskedArray(np.ones(3, dt1))
    dst = src.astype(dt2)

    assert_(src.dtype == dt1)
    assert_(dst.dtype == dt2)
    assert_masked_equal(src, dst)


def test_fieldless_void():
    dt = np.dtype([])  # a void dtype with no fields
    x = np.empty(4, dt)

    # these arrays contain no values, so there's little to test - but this
    # shouldn't crash
    mx = MaskedArray(x)
    assert_equal(mx.dtype, x.dtype)
    assert_equal(mx.shape, x.shape)

    mx = MaskedArray(x, mask=[1,1,0,0])
    assert_equal(mx.dtype, x.dtype)
    assert_equal(mx.shape, x.shape)


################################################################################
#                    New Tests of ndarray_ducktypes.MaskedArray
################################################################################

class MA_Subclass(MaskedArray):
    pass
class MAscalar_Subclass(MaskedScalar):
    pass
ducktype_link(MA_Subclass, MAscalar_Subclass, known_types=(type(X),))

class dummyarr:
    def __init__(self, data, dtype=None, copy=False, order=None, ndmin=0,
                 **options):
        self.dtype = np.dtype('f8')
        self.shape = (3,)
        self.ndim = 1
    def __getitem__(self, ind):
        return dummyscalar()
    def __array_function__(self, func, types, arg, kwarg):
        pass
class dummyscalar:
    def __init__(self, data, dtype=None, copy=False, order=None, ndmin=0,
                 **options):
        self.dtype = np.dtype('f8')
        self.shape = ()
        self.ndim = 0
    def __getitem__(self, ind):
        return self
    def __array_function__(self, func, types, arg, kwarg):
        pass
ducktype_link(dummyarr, dummyscalar)


class Test_ReplaceX:
    def test_allX(self):
        assert_raises(ValueError, MaskedArray, [[X, X, X], [X, X, X]])
        arru1 = MaskedArray([[X, X, X], [X, X, X]], dtype='u1')
        assert_equal(repr(arru1), textwrap.dedent('''\
                        MaskedArray([[X, X, X],
                                     [X, X, X]], dtype=uint8)'''))
        arr = MaskedArray([[X, X, X], [X, X, X]], dtype=float)

        # dtype gets shown, even for dtypes usually omitted
        assert_equal(repr(arr), textwrap.dedent('''\
                        MaskedArray([[X, X, X],
                                     [X, X, X]], dtype=float64)'''))
        # also test case where one sub-list is all X
        arr2 = MaskedArray([[X, X, X], [X, X, X('f8')]])
        assert_masked_equal(arr, arr2)

        data, mask, _ = replace_X([[X, X, X], [X, X, X]], dtype='u1')
        assert_equal(arru1, MaskedArray(data, mask))

    def test_mixedndarray(self):
        arr = MaskedArray([np.array([1,2,3], dtype='i1'), [3, X, 5]])
        assert_masked_equal(arr, MaskedArray([[1,2,3],[3,X,5]], dtype='i1'))

    #XXX many more situations to test here


class Test_MA_construction:
    def test(self):
        MaskedScalar(1.0)
        MaskedScalar(np.array(1.0))
        MaskedScalar(X('f8'))
        MaskedScalar(np.array(X('f8')))

        MaskedArray(1.0)
        MaskedScalar(1.0)
        MaskedArray(X('f8'))
        MaskedArray(np.array(X('f8')))

        MaskedArray(MA_Subclass([1.0]))
        MaskedArray(MAscalar_Subclass(1.0))

        MaskedArray(MaskedArray([1,X,2]))
        MaskedArray(MaskedArray([1,X,2]), mask=[1,0,0])

        MaskedArray(dummyarr(0))
        MaskedArray(dummyscalar(0))


class Test_API:
    # tests for each ndarray-api implementation

    def test_all(self):
        # test some masked
        # test all masked
        assert_masked_equal(np.all(MaskedArray([True, X, True])), True)
        assert_masked_equal(np.all(MaskedArray([True, X, False])), False)
        assert_masked_equal(np.all(MaskedArray([X, X, X], dtype='?')), True)
        assert_masked_equal(np.all(MaskedArray([X, X, False])), False)
        ret = np.array([True, False, True])
        data = [[True, True, True],
                [True, X, False],
                [X, X, X]]
        assert_equal(np.all(MaskedArray(data), axis=1), ret)

        # test out argument
        out = np.array([True, True, True])
        o = np.all(MaskedArray(data), axis=1, out=out)
        assert_equal(out, ret)
        assert_(out is o)
        out = MaskedArray([True, True, True])
        o = np.all(MaskedArray(data), axis=1, out=out)
        assert_masked_equal(out, ret)
        assert_(out is o)

        # test subclasses
        assert_equal(np.all(MA_Subclass(data), axis=1), ret)

        # test kwds
        ret = np.array([[True], [False], [True]])
        out = MaskedArray([[True], [True], [True]])
        o = np.all(MaskedArray(data), axis=1, keepdims=True, out=out)
        assert_masked_equal(out, ret)
        assert_(out is o)

        # test scalar
        assert_equal(np.all(X(np.float64)), True)

    def test_any(self):
        assert_masked_equal(np.any(MaskedArray([False, X, False])), False)
        assert_masked_equal(np.any(MaskedArray([True, X, False])), True)
        assert_masked_equal(np.any(MaskedArray([X, X, X], dtype='?')), False)
        assert_masked_equal(np.any(MaskedArray([X, X, False])), False)
        assert_masked_equal(np.any(MaskedArray([X, X, True])), True)
        # assume rest works since same impl as np.all

    def test_max_min(self):
        # scalar
        ret = np.max(MaskedArray([1., X, 2.]))
        assert_(isinstance(ret, MaskedScalar))
        assert_masked_equal(ret, 2.)

        ret = np.min(MaskedArray([1., X, 2.]))
        assert_(isinstance(ret, MaskedScalar))
        assert_masked_equal(ret, 1.)

        assert_masked_equal(np.max(MaskedArray([X, X], dtype='f8')), X('f8'))
        assert_masked_equal(np.min(MaskedArray([X, X], dtype='f8')), X('f8'))
        assert_(type(np.max(MaskedArray([X, X], dtype='f8'))) is MaskedScalar)

        data = [[1., 2., 3.],
                [4., X , 6.],
                [X,  X , X ]]
        m = MaskedArray(data)

        # keepdims, initial, where
        assert_masked_equal(np.max(m, axis=1), MaskedArray([3., 6., X]))
        assert_masked_equal(np.max(m, keepdims=True, axis=1),
                            MaskedArray([[3.], [6.], [X]]))
        assert_masked_equal(np.max(m, axis=1, initial=5),
                            MaskedArray([5., 6., 5.]))
        assert_raises(ValueError, np.max, m, axis=1, initial=X)
        assert_raises(ValueError, np.max, m, axis=1, initial=X('f8'))
        assert_masked_equal(np.max(m, axis=1, initial=MaskedScalar(5)),
                            MaskedArray([5., 6., 5.]))
        assert_masked_equal(np.max(m, axis=1, where=[1,0,0], initial=2.5),
                            MaskedArray([2.5, 4., 2.5]))
        assert_masked_equal(np.min(m, axis=1, where=[1,0,0], initial=5),
                            MaskedArray([1., 4., 5.]))

        # out
        out = MaskedArray(np.zeros(3))
        o = np.max(m, axis=1, out=out)
        assert_masked_equal(out, MaskedArray([3., 6., X]))
        assert_(out is o)

        # subclass
        ret = np.max(MA_Subclass(m), axis=1)
        assert_masked_equal(ret, MA_Subclass([3., 6., X]))
        assert_(type(ret) == MA_Subclass)
        assert_(type(np.max(MA_Subclass([X, X('f8')]))) is MAscalar_Subclass)

    def test_argmax_argmin(self):
        minval = _minvals[np.dtype('int')]

        d = MaskedArray([[X, minval],
                         [minval, X],
                         [1, minval],
                         [minval, 1],
                         [X,      X]])
        assert_equal(np.argmax(d, axis=1), [1, 0, 0, 1, 0])

        maxval = _maxvals[np.dtype('int')]
        d = MaskedArray([[X, maxval],
                         [maxval, X],
                         [1, maxval],
                         [maxval, 1],
                         [X,      X]])
        assert_equal(np.argmin(d, axis=1), [1, 0, 0, 1, 0])

        out = np.empty(5, dtype='p')
        o = np.argmin(d, axis=1, out=out)
        assert_equal(out, [1, 0, 0, 1, 0])
        assert_(out is o)

        # test without axis
        d1 = d[0,:]
        assert_equal(np.argmin(d1), 1)
        d1 = d.ravel()
        assert_equal(np.argmin(d1), 4)

        # subclass
        d = MA_Subclass(d)
        assert_equal(np.argmin(d, axis=1), [1, 0, 0, 1, 0])
        assert_equal(type(np.argmin(d, axis=1)), np.ndarray)
        d1 = d[0,:]
        assert_equal(type(np.argmin(d1)), np.dtype('p'))

    def test_sort_argsort_partition_argpartition(self):
        maxval = _maxvals[np.dtype('int')]
        d = MaskedArray([2, maxval, X, 1, X, 3, maxval, 0])
        ret = MaskedArray([0, 1, 2, 3, maxval, maxval, X, X])
        assert_masked_equal(np.sort(d), ret)
        assert_masked_equal(d[np.argsort(d)], ret)

        def compare_partitions(x, y, k):
            xa, ya = set(x[:k]), set(y[:k])
            xb, yb = set(x[k:]), set(y[k:])
            assert_(len(xa.difference(ya)) == 0)
            assert_(len(xb.difference(yb)) == 0)
        rf = ret.filled(123)
        compare_partitions(np.partition(d, 3).filled(123), rf, 3)
        compare_partitions(d[np.argpartition(d, 3)].filled(123), rf, 3)

        inf, nan = np.inf, np.nan
        d = MaskedArray([2, inf, X, 1, -inf, nan, X, 3, nan, inf, 0])
        ret = MaskedArray([-inf, 0, 1, 2, 3, inf, inf, nan, nan, X, X])
        assert_masked_equal(np.sort(d), ret)
        assert_masked_equal(d[np.argsort(d)], ret)

        d   = MaskedArray([[2, inf, X],
                           [X,   X, X],
                           [1, -inf, nan],
                           [X, 3, nan],
                           [X, inf, nan],
                           [X, inf, 0]])
        ret = MaskedArray([[2, inf, X],
                           [X,   X, X],
                           [-inf, 1, nan],
                           [3, nan, X],
                           [inf, nan, X],
                           [0, inf, X]])
        assert_masked_equal(np.sort(d, axis=1), ret)
        assert_masked_equal(d[np.arange(6)[:,None], np.argsort(d, axis=1)], ret)
        # partition of 3 elements around middle elem is same as sort
        assert_masked_equal(np.partition(d, 1, axis=1), ret)

    def test_searchsorted(self):
        inf, nan = np.inf, np.nan
        d = MaskedArray([-inf, 0, 1, 2, 3, inf, inf, nan, nan, X, X])
        ind = MaskedArray([inf, nan, 2.5, X, -1])

        assert_equal(np.searchsorted(d, ind), [5, 7, 4, 9, 1])
        assert_equal(np.searchsorted(d, ind, side='right'), [7, 9, 4, 11, 1])

        # no need to test digitize extensively - uses searchsorted
        bins = np.arange(10.)
        d = MaskedArray([-inf, 0, 1, 2, 3, inf, nan, X])
        assert_equal(np.digitize(d, bins), [ 0,  1,  2,  3,  4, 10, 10, 10])

    def test_lexsort(self):
        a = MaskedArray([1,5,1,4,3,X,4,1])
        b = MaskedArray([9,4,0,4,0,0,1,X])
        assert_equal(np.lexsort((b, a)), [2, 0, 7, 4, 6, 3, 1, 5])

    def test_mean(self):
        a = MaskedArray([1,2,3,X,4])
        assert_masked_equal(np.mean(a), 2.5)
        assert_masked_equal(np.mean(MaskedArray([X, X('f8')])), X('f8'))

        # axis, dtype, keepdims, out
        a = MaskedArray([[2,4],[3,X],[X,X]])
        assert_masked_equal(np.mean(a, axis=1), MaskedArray([3, 3, X]))
        assert_masked_equal(np.mean(a, axis=1, keepdims=True),
                            MaskedArray([[3], [3], [X]]))
        out = MaskedArray([1,1,1])
        o = np.mean(a, axis=1, out=out)
        assert_masked_equal(out, MaskedArray([3, 3, X]))
        assert_(out is o)

        # scalar, 0d
        assert_masked_equal(np.mean(MaskedScalar(3)), 3)
        assert_masked_equal(np.mean(X('f8')), X('f8'))

        # subclass
        a = MA_Subclass([1,2,3,X,4])
        assert_masked_equal(np.mean(a), 2.5)
        assert_equal(type(np.mean(a)), MAscalar_Subclass)

    def test_var_std(self):
        a = MaskedArray([1,2,3,X,4])
        assert_masked_equal(np.var(a), 1.25)
        assert_masked_equal(np.var(MaskedArray([X, X('f8')])), X('f8'))

        # axis, dtype, keepdims, out, ddof
        a = MaskedArray([[2,4],[3,X],[X,X]])
        assert_masked_equal(np.var(a, axis=1), MaskedArray([1, 0, X]))
        assert_masked_equal(np.var(a, axis=1, keepdims=True),
                            MaskedArray([[1], [0], [X]]))
        assert_masked_equal(np.std(a, axis=1, keepdims=True),
                            MaskedArray([[1], [0], [X]]))
        # question: should ddof < 0 give a nan+warning, or X?
        assert_masked_equal(np.var(a, axis=1, ddof=1), MaskedArray([2, X, X]))
        out = MaskedArray([1,1,1])
        o = np.var(a, axis=1, out=out)
        assert_masked_equal(out, MaskedArray([1, 0, X]))
        assert_(out is o)

        # scalar, 0d
        assert_masked_equal(np.var(MaskedScalar(3)), 0)
        assert_masked_equal(np.var(MaskedArray(3)), 0)
        assert_equal(type(np.var(MaskedArray(3))), MaskedScalar)
        assert_masked_equal(np.var(X('f8')), X('f8'))

        # subclass
        a = MA_Subclass([1,2,3,X,4])
        assert_masked_equal(np.var(a), 1.25)
        assert_equal(type(np.var(a)), MAscalar_Subclass)

        # std is implemented in terms of var, just check it works
        assert_masked_equal(np.std(a), np.sqrt(1.25))

    def test_average(self):
        a = MaskedArray([1,2,3,X,4])
        assert_masked_equal(np.average(a), 2.5)
        avg, c = np.average(a, returned=True)
        assert_masked_equal(avg, 2.5)
        assert_masked_equal(c, 4)
        avg, c = np.average(MaskedArray([X,X,X('f8')]), returned=True)
        assert_masked_equal(avg, X('f8'))
        assert_masked_equal(c, 0)

        wgt = [0.1, 1, 0.1, 1, 0.1]
        avg, c = np.average(a, weights=wgt, returned=True)
        ret = (0.1*1+2+0.1*3+0.1*4)/(0.1*3+1)
        assert_almost_equal(c, 1.3)
        assert_almost_equal(avg, ret)
        assert_almost_masked_equal(np.average(a, weights=wgt), ret)

        # subclasses, axis, weights
        a = MA_Subclass([[1,2,X], [X,X,X], [X,X,1]])
        wgt = [0.1, 0.2, 0.3]
        avg, c = np.average(a, axis=1, weights=wgt, returned=True)
        assert_almost_masked_equal(avg, MA_Subclass([0.5/0.3, X, 1]))
        assert_equal(c, [0.3, 0, 0.3])

    def test_quantile(self):
        # median, percentile implemented in terms of quantile
        a = MaskedArray([2,1,4,X,3])
        assert_masked_equal(np.quantile(a, 0.5), 2.5)
        assert_masked_equal(np.quantile(a, 0.5, interpolation='lower'), 2)
        assert_masked_equal(np.quantile(a, 0.5, interpolation='higher'), 3)
        assert_masked_equal(np.percentile(a, 50), 2.5)
        assert_masked_equal(np.median(a), 2.5)
        assert_masked_equal(np.quantile(MaskedArray([X,X('f8')]), 0.5), X('f8'))

        assert_almost_masked_equal(np.quantile(a, [0.5,0.8]), [2.5, 3.4])

        a = MaskedArray([[2,1,4,X,3],
                         [5,6,X,X,1],
                         [X,X,X,X,X]])
        ret = MaskedArray([2.5, 5., X])
        assert_almost_masked_equal(np.quantile(a, 0.5, axis=1), ret)
        assert_almost_masked_equal(np.quantile(a, 0.5, axis=0),
                                   MaskedArray([7/2, 7/2, 4, X, 2]))
        assert_almost_masked_equal(np.quantile(a, 0.5, axis=1, keepdims=True),
                                   MaskedArray([[2.5], [5.], [X]]))
        out = MaskedArray([0.,0,0])
        o = np.quantile(a, 0.5, axis=1, out=out)
        assert_almost_masked_equal(out, ret)
        assert_(out is o)
        out = MaskedArray(np.zeros((2,5)))
        o = np.quantile(a, np.array([0.1, 0.9]), axis=0, out=out)
        ret2 = MaskedArray([[2.3, 1.5, 4., X, 1.2],
                            [4.7, 5.5, 4., X, 2.8]])
        assert_almost_masked_equal(out, ret2)
        assert_(out is o)

        # subclasses
        b = MA_Subclass(a)
        val = np.quantile(b, 0.5, axis=1)
        assert_almost_masked_equal(val, ret)
        assert_(type(val) == MA_Subclass)

        # scalars
        val = np.quantile(MaskedArray(4), 0.5)
        assert_masked_equal(val, 4.)
        assert_(type(val) == MaskedScalar)
        assert_masked_equal(np.quantile(MaskedArray(X('f8')), 0.5), X('f8'))
        val = np.quantile(MaskedScalar(4), 0.5)
        assert_masked_equal(val, 4.)
        assert_(type(val) == MaskedScalar)
        assert_masked_equal(np.quantile(X('f8'), 0.5), X('f8'))

    def test_cov_corrcoef(self):
        x = MaskedArray([-2.1, -1,  4.3, X, 1.2, 2, X])
        y = MaskedArray([3,  1.1,  0.12, 1, 0.5, X, X])
        xy = np.stack((x, y), axis=0)

        # compute expected result with ndarrays:
        dx = x._data - np.mean(x._data[~x.mask])
        dy = y._data - np.mean(y._data[~y.mask])
        mx, my = ~(x.mask), ~(y.mask)
        mxy = mx & my
        c = np.array([[np.sum((dx*dx)[mx]), np.sum((dx*dy)[mxy])],
                      [np.sum((dy*dx)[mxy]), np.sum((dy*dy)[my])]])
        d = np.array([[np.sum(mx), np.sum(mxy)],
                      [np.sum(mxy), np.sum(my)]])
        ret = c/(d-1)
        stddev = np.sqrt(np.diag(ret))
        cc = (ret/stddev[:,None]/stddev[None,:]).clip(-1,1)

        assert_almost_masked_equal(np.cov(xy), ret)
        assert_almost_masked_equal(np.cov(xy, rowvar=True), ret)
        assert_almost_masked_equal(np.cov(xy.T, rowvar=False), ret)
        assert_almost_masked_equal(np.cov(x, y), ret)
        assert_almost_masked_equal(np.corrcoef(xy), cc)

        # result values below not checked carefully for correctness... this
        # test is for consistency after code changes.

        ret2 = [[ 8.4893333, -4.57816  ],
                [-4.57816  ,  1.6435733]]
        assert_almost_masked_equal(np.cov(xy, ddof=2), ret2)

        fweights = [1,2,3,4,5,6,7]
        aweights = np.arange(7)*0.1
        ret = np.array([[ 1.2378581, -0.3981544],
                        [-0.3981544,  0.118843 ]])
        val = np.cov(xy, ddof=2, fweights=fweights, aweights=aweights)
        assert_almost_masked_equal(val, ret)

    def test_clip(self):
        a = MaskedArray([[2.1, X, 3.1], [0.5, 1.0, 6.0]])
        ret = MaskedArray([[2.1, X, 3.1], [2., 2.0, 5.0]])
        assert_masked_equal(np.clip(a, 2, 5), ret)
        out = a[::-1].copy()
        o = np.clip(a, 2, 5, out=out)
        assert_masked_equal(out, ret)
        assert_(out is o)
        assert_masked_equal(np.clip(MA_Subclass(a), 2, 5), ret)
        assert_equal(type(np.clip(MA_Subclass(a), 2, 5)), MA_Subclass)
        assert_masked_equal(np.clip(X('f8'), 2, 5), X('f8'))

    def test_compress(self):
        a = MaskedArray([[2.1, X, 3.1], [0.5, 1.0, 6.0]])
        ret = MaskedArray([[X, 3.1], [1.0, 6.0]])
        assert_masked_equal(np.compress([0,1,1], a, axis=1), ret)
        ret = MaskedArray([[X], [1.0]])
        assert_masked_equal(np.compress([0,1,X], a, axis=1), ret)

    def test_copy(self):
        a = MaskedArray([[2.1, X, 3.1], [0.5, 1.0, 6.0]])
        b = np.copy(a)
        assert_masked_equal(a, b)
        assert_(a is not b)
        assert_equal(type(np.copy(MA_Subclass(a))), MA_Subclass)
        assert_equal(type(np.copy(X('f8'))), MaskedArray)

    def test_prod_cumprod(self):
        a = MaskedArray([[X, 2.1, 3.1], [0.5, 1.0, 6.0], [X, X, X]])
        assert_almost_masked_equal(np.prod(a), MaskedArray(19.53))
        assert_almost_masked_equal(np.prod(a, axis=1),
                                   MaskedArray([6.51, 3.0, X]))
        assert_almost_masked_equal(np.prod(a, axis=1, keepdims=True),
                                   MaskedArray([[6.51], [3.0], [X]]))
        assert_almost_masked_equal(
            np.prod(a, axis=1, keepdims=True, dtype=np.complex128),
            MaskedArray([[6.51], [3.0], [X]], dtype=np.complex128))

        assert_masked_equal(np.prod(X('f8')), X('f8'))
        assert_masked_equal(np.product(X('f8')), X('f8'))

        out = MaskedArray([1., 1, 1])
        o = np.prod(a, axis=1, out=out)
        assert_almost_masked_equal(out, MaskedArray([6.51, 3.0, X]))
        assert_(out is o)

        assert_almost_masked_equal(
            np.cumprod(a, axis=1, dtype=np.complex128),
            MaskedArray([[X  , 2.1, 6.51],
                         [0.5, 0.5, 3. ],
                         [X  , X  , X  ]], dtype=np.complex128))

    def test_sum_cumsum(self):
        a = MaskedArray([[X, 2.1, 3.1], [0.5, 1.0, 6.0], [X, X, X]])
        assert_almost_masked_equal(np.sum(a), MaskedArray(12.7))
        assert_almost_masked_equal(np.sum(a, axis=1),
                                   MaskedArray([5.2, 7.5, X]))
        assert_almost_masked_equal(np.sum(a, axis=1, keepdims=True),
                                   MaskedArray([[5.2], [7.5], [X]]))
        assert_almost_masked_equal(
            np.sum(a, axis=1, keepdims=True, dtype=np.complex128),
            MaskedArray([[5.2], [7.5], [X]], dtype=np.complex128))

        assert_masked_equal(np.sum(X('f8')), X('f8'))

        out = MaskedArray([1., 1, 1])
        o = np.sum(a, axis=1, out=out)
        assert_almost_masked_equal(out, MaskedArray([5.2, 7.5, X]))
        assert_(out is o)

        assert_almost_masked_equal(
            np.cumsum(a, axis=1, dtype=np.complex128),
            MaskedArray([[X  , 2.1, 5.2],
                         [0.5, 1.5, 7.5],
                         [X  , X  , X  ]], dtype=np.complex128))

    def test_diagonal(self):
        a = MaskedArray([[X,1,2], [X,X,X], [1, X, 2]])
        assert_masked_equal(np.diagonal(a), MaskedArray([X, X, 2]))
        assert_equal(np.diagonal(a).filled(), [0, 0, 2])
        assert_masked_equal(np.diagonal(a, offset=1), MaskedArray([1, X]))

    def test_diag(self):
        a = MaskedArray([[X,1,2], [X,X,X], [1, X, 2]])
        assert_masked_equal(np.diag(a), MaskedArray([X, X, 2]))
        assert_masked_equal(np.diag(np.diag(a)),
                            MaskedArray([[X,0,0], [0,X,0], [0, 0, 2]]))
        assert_masked_equal(np.diag(np.diag(a, k=1), k=1),
                            MaskedArray([[0,1,0], [0,0,X], [0, 0, 0]]))

    def test_diagflat(self):
        a = MaskedArray([[X,1], [X,X], [2, X]])
        assert_masked_equal(np.diagflat(a), MaskedArray([[X, 0, 0, 0, 0, 0],
                                                         [0, 1, 0, 0, 0, 0],
                                                         [0, 0, X, 0, 0, 0],
                                                         [0, 0, 0, X, 0, 0],
                                                         [0, 0, 0, 0, 2, 0],
                                                         [0, 0, 0, 0, 0, X]]))
    def test_tril_triu(self):
        a = MaskedArray([[X, 2.1, 3.1], [0.5, 1.0, 6.0], [X, X, X]])
        r = MaskedArray([[X, 2.1, 3.1], [0  , 1.0, 6.0], [0, 0, X]])
        assert_masked_equal(np.triu(a), r)
        r = MaskedArray([[X, 2.1, 3.1], [0.5, 1.0, 6.0], [0, X, X]])
        assert_masked_equal(np.triu(a, k=-1), r)
        assert_equal(type(np.triu(MA_Subclass(a), k=1)), MA_Subclass)

    def test_trace(self):
        a = MaskedArray(np.arange(27).reshape((3,3,3)))
        a[2,2,1] = X
        a[0,0,2] = a[1,1,2] = a[2,2,2] = X

        assert_masked_equal(np.trace(a), MaskedArray([36, 14, X]))
        out = MaskedArray([0,0,0])
        o = np.trace(a, out=out)
        assert_masked_equal(out, MaskedArray([36, 14, X]))
        assert_(out is o)
