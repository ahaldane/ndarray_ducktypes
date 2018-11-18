#!/usr/bin/env python
import numpy as np
from duckprint import (duck_str, duck_repr, default_duckprint_options,
                       default_duckprint_formatters, FormatDispatcher)
import builtins
import numpy.core.umath as umath
from numpy.lib.mixins import NDArrayOperatorsMixin
import numpy.core.numerictypes as ntypes

class MaskedArray(NDArrayOperatorsMixin):
    def __init__(self, data=None, mask=None, dtype=None, copy=False,
                order=None, subok=True, ndmin=0, **options):
        self.data = np.array(data, dtype, copy=copy, order=order, subok=subok,
                             ndmin=ndmin)

        # Note: We do not try to mask individual fields structured types.
        # Instead you get one mask value per struct element. Use an
        # ArrayCollection of MaskedArrays if you want to mask individual
        # fields.

        if isinstance(data, MaskedArray):
            self.data = data.data
            self.mask = data.mask
            if mask is not None:
                raise Exception("don't use mask if passing a maskedarray")
                #XXX should this override the mask?

        elif mask is None:
            self.mask = np.zeros(data.shape, dtype='bool', order=order)
        else:
            self.mask = np.empty(data.shape, dtype='bool')
            self.mask[:] = np.broadcast_to(mask, self.data.shape)

        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MaskedArray objects
        if not all(issubclass(t, MaskedArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in masked_ufuncs:
            return NotImplemented

        return getattr(masked_ufuncs[ufunc], method)(*inputs, **kwargs)

    @classmethod
    def __nd_duckprint_dispatcher__(cls):
        return masked_dispatcher

    def __str__(self):
        return duck_str(self)

    def __repr__(self):
        return duck_repr(self)

    def reshape(self, shape, order='C'):
        return MaskedArray(self.data.reshape(shape, order=order),
                           self.mask.reshape(shape, order=order))

    def __getitem__(self, ind):
        data = self.data[ind]
        mask = self.mask[ind]
        if data.shape == ():
            return MaskedScalar(data, mask)
        return MaskedArray(data, mask)

    def __setitem__(self, ind, val):
        if isinstance(val, (MaskedArray, MaskedScalar)):
            self.data[ind] = val.data
            self.mask[ind] = val.mask
        else:
            self.data[ind] = val
            self.mask[ind] = False


    def filled(self, fill_value=0):
        result = self.data.copy()
        fill_value = self.dtype.type(fill_value)
        np.copyto(result, fill_value, where=self.mask)
        return result

# Ndarrays return scalars when "fully indexed" (integer at each axis). Ducktype
# implementors need to mimic this. However, they often want the scalars to
# behave specially - eg be masked for MaskedArray. I see a few different
# scalar strategies:
# 1. Make a MaskedScalar class which wraps all scalars. This is implemented
#    below. Problem: It fails code which uses "isisntance(np.int32)". But maybe
#    that is good to force people to use `filled` before using this way.
# 2. Subclass each numpy scalar type individually to keep parent methods and
#    use super(), but modify the repr, add a "filled" method and fillvalue.
#    Problem: 1. currently subclassing numpy scalars does not work properly. 2.
#    Other code is not aware of the mask and ignores it.
# 3. return normal numpy scalars for unmasked values, and return separate masked
#    values when masked. How to implement the masked values? As in #2?
# 4. Like #3 but return a "masked" singleton for masked values like in the old
#    MaskedArray. Problem: it has a fixed dtype of float64 causing lots of
#    casting bugs, and unintentionally modifying the singleton (not too hard to
#    do) leads to bugs. Also fails the isinstance checks, and more.
#
# Question: What should happen when you fancy-index using a masked integer
# array? Probably it should fail - you should use filled first.
#
class MaskedScalar:
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask
        self.dtype = data.dtype
        self.shape = ()
        self.size = 1
        self.ndim = 0

    def __str__(self):
        if self.mask:
            return MASK_STR
        return str(self.data)

    def __repr__(self):
        if self.mask:
            return MASK_STR  # or "masked" ?
        return repr(self.data)

    def __format__(self, format_spec):
        if self.mask:
            return MASK_STR
        return format(self.data, format_spec)

    def filled(self, fill_value=0):
        if self.mask:
            return self.data.dtype.type(fill_value)
        return self.data

################################################################################
#                               Printing setup
################################################################################

def as_masked_fmt(formattercls):
    # we subclass the original formatter class, and wrap the result of
    # `get_format_func` to take care of masked values.

    class MaskedFormatter(formattercls):
        def get_format_func(self, elem, **options):

            if not elem.mask.any():
                default_fmt = super().get_format_func(elem.data, **options)
                return lambda x: default_fmt(x.data)

            # only get fmt_func based on non-masked values
            # (we take care of masked elements ourselves)
            unmasked = elem.data[~elem.mask]
            default_fmt = super().get_format_func(unmasked, **options)

            # default_fmt should always give back same str length.
            # Figure out what this is with a test call. XXX flatten before index
            example_str = default_fmt(unmasked[0]) if len(unmasked) > 0 else ''
            masked_str = options['masked_str']
            reslen = builtins.max(len(example_str), len(masked_str))

            # pad the columns to align when including the masked string
            masked_str = masked_str.rjust(reslen)

            def fmt(x):
                if x.mask:
                    return masked_str
                return default_fmt(x.data).rjust(reslen)

            return fmt

    return MaskedFormatter

MASK_STR = '_'  # make this more configurable later

masked_formatters = [as_masked_fmt(f) for f in default_duckprint_formatters]
default_options = default_duckprint_options.copy()
default_options['masked_str'] = MASK_STR
masked_dispatcher = FormatDispatcher(masked_formatters, default_options)


################################################################################
#                               Ufunc setup
################################################################################

masked_ufuncs = {}

class _Masked_UFunc:
    def __init__(self, ufunc):
        self.f = ufunc
        self.__doc__ = ufunc.__doc__
        self.__name__ = ufunc.__name__

    def __str__(self):
        return "Masked version of {}".format(self.f)

def getdata(a):
    if isinstance(a, MaskedArray):
        return a.data
    return a

def getmask(a):
    if isinstance(a, MaskedArray):
        return a.mask
    return False

class _Masked_UniOp(_Masked_UFunc):
    """
    Masked version of unary ufunc. Assumes 1 output.

    Parameters
    ----------
    ufunc : ufunc
        The ufunc for which to define a masked version.
    maskdomain : function
        Function which returns true for inputs whose output should be masked.
    """

    def __init__(self, ufunc, maskdomain=None):
        super().__init__(ufunc)
        self.domain = maskdomain

    def __call__(self, a, *args, **kwargs):
        out = kwargs.get('out', ())
        if out and isinstance(out[0], MaskedArray):
            kwargs['out'] = (out[0].data,)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(getdata(a), *args, **kwargs)

        if self.domain is None:
            m = getmask(a)
        else:
            m = self.domain(d) | getmask(a)

        if out is not ():
            out[0].mask[:] = m
            return out[0]

        if np.isscalar(result):
            return MaskedScalar(result, m)

        return MaskedArray(result, m)

class _Masked_BinOp(_Masked_UFunc):
    """
    Masked version of binary ufunc. Assumes 1 output.

    Parameters
    ----------
    ufunc : ufunc
        The ufunc for which to define a masked version.
    maskdomain : funcion
        Function which returns true for inputs whose output should be masked.
    """

    def __init__(self, ufunc, maskdomain=None):
        super().__init__(ufunc)
        self.domain = maskdomain

    def __call__(self, a, b, **kwargs):
        da, db = getdata(a), getdata(b)

        out = kwargs.get('out', ())
        if out and isinstance(out[0], MaskedArray):
            kwargs['out'] = (out[0].data,)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, **kwargs)

        m = getmask(a) | getmask(b)
        if self.domain is not None:
            m |= self.domain(da, db)

        if out is not ():
            out[0].mask[:] = m
            return out[0]

        if np.isscalar(result):
            return MaskedScalar(result, m)

        return MaskedArray(result, m)

    #def reduce(self, target, axis=0, dtype=None):
    #def outer(self, a, b):
    #def accumulate(self, target, axis=0):

def maskdom_divide(a, b):
    out_dtype = np.result_type(a, b)

    # if floating, use finfo to determine domain
    if isinstance(out_dtype, np.inexact):
        tolerance = np.finfo(out_dtype).tiny
        with np.errstate(invalid='ignore'):
            return umath.absolute(a) * tolerance >= umath.absolute(b)

    # otherwise, for integer types, only 0 is a problem
    return b == 0

def maskdom_greater_equal(x):
    def maskdom_interval(a):
        with np.errstate(invalid='ignore'):
            return umath.less(a, x)
    return maskdom_interval

def maskdom_greater(x):
    def maskdom_interval(a):
        with np.errstate(invalid='ignore'):
            return umath.less_equal(a, x)
    return maskdom_interval

def maskdom_tan(a):
    with np.errstate(invalid='ignore'):
        return umath.less(umath.absolute(umath.cos(x)), 1e-35) #XXX use finfo?

def make_maskdom_interval(lo, hi):
    def maskdom(a):
        with np.errstate(invalid='ignore'):
            return umath.logical_or(umath.greater(x, hi),
                                    umath.less(x, lo))
    return maskdom

def setup_ufuncs():
    # unary funcs
    for ufunc in [umath.exp, umath.conjugate, umath.sin, umath.cos, umath.tan,
                  umath.arctan, umath.arcsinh, umath.sinh, umath.cosh,
                  umath.tanh, umath.absolute, umath.fabs, umath.negative,
                  umath.floor, umath.ceil, umath.logical_not, umath.isfinite,
                  umath.isinf, umath.isnan]:
        masked_ufuncs[ufunc] = _Masked_UniOp(ufunc)

    # domained unary funcs
    masked_ufuncs[umath.sqrt] = _Masked_UniOp(umath.sqrt,
                                              maskdom_greater_equal(0.))
    masked_ufuncs[umath.log] = _Masked_UniOp(umath.log, maskdom_greater(0.))
    masked_ufuncs[umath.log2] = _Masked_UniOp(umath.log2, maskdom_greater(0.))
    masked_ufuncs[umath.log10] = _Masked_UniOp(umath.log10, maskdom_greater(0.))
    masked_ufuncs[umath.tan] = _Masked_UniOp(umath.tan, maskdom_tan)
    maskdom_11 = make_maskdom_interval(-1., 1.)
    masked_ufuncs[umath.arcsin] = _Masked_UniOp(umath.arcsin, maskdom_11)
    masked_ufuncs[umath.arccos] = _Masked_UniOp(umath.arccos, maskdom_11)
    masked_ufuncs[umath.arccosh] = _Masked_UniOp(umath.arccos,
                                                 maskdom_greater_equal(1.))
    masked_ufuncs[umath.arctanh] = _Masked_UniOp(umath.arctanh,
                                       make_maskdom_interval(-1+1e-15, 1+1e-15))
                                       # XXX use finfo?
    # XXX should these all be customized for the float size?

    # binary ufuncs
    for ufunc in [umath.add, umath.subtract, umath.multiply, umath.arctan2,
                  umath.equal, umath.not_equal, umath.less_equal,
                  umath.greater_equal, umath.less, umath.greater,
                  umath.logical_and, umath.logical_or, umath.logical_xor,
                  umath.bitwise_and, umath.bitwise_or, umath.bitwise_xor,
                  umath.hypot]:
        masked_ufuncs[ufunc] = _Masked_BinOp(ufunc)

    # domained binary ufuncs
    for ufunc in [umath.true_divide, umath.floor_divide, umath.remainder,
                  umath.fmod, umath.mod]:
        masked_ufuncs[ufunc] = _Masked_BinOp(ufunc, maskdom_divide)

setup_ufuncs()

################################################################################
#                         __array_function__ setup
################################################################################

HANDLED_FUNCTIONS = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for MaskedArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

def setup_ducktype():

    max_filler = ntypes._minvals
    max_filler.update([(k, -np.inf) for k in [np.float32, np.float64]])
    min_filler = ntypes._maxvals
    min_filler.update([(k, +np.inf) for k in [np.float32, np.float64]])
    if 'float128' in ntypes.typeDict:
        max_filler.update([(np.float128, -np.inf)])
        min_filler.update([(np.float128, +np.inf)])

    @implements(np.max)
    def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        filled = a.filled(max_filler[a.dtype])

        # XXX handle out properly
        result = np.max(filled, axis, out, keepdims, initial)
        result_mask = np.all(a.mask, axis, out, keepdims)

        if np.isscalar(result):
            return MaskedScalar(result, result_mask)
        return MaskedArray(result, result_mask)

    @implements(np.min)
    def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        filled = a.filled(min_filler[a.dtype])

        result = np.min(filled, axis, out, keepdims, initial)
        result_mask = np.all(a.mask, axis, out, keepdims)

        if np.isscalar(result):
            return MaskedScalar(result, result_mask)
        return MaskedArray(result, result_mask)

    #@implements(np.concatenate)
    #def concatenate(arrays, axis=0, out=None):
    #    ...  # implementation of concatenate for MyArray objects

    #@implements(np.broadcast_to)
    #def broadcast_to(array, shape):
    #    ...  # implementation of broadcast_to for MyArray objects

setup_ducktype()

################################################################################
#                               testing code
################################################################################

# See discussion here:
# https://docs.scipy.org/doc/numpy-1.13.0/neps/missing-data.html
#
# Main features of MaskedArray we want to have here:
#  1. We use "Ignore/skip" mask propagation
#  1. invalid domain ufunc calls (div by 0 etc) get converted to masks
#  1. MaskedArray has freedom to set the data at masked elem (for optimization)
#
# Far-out ideas for making these choice configurable: Since the mask is stored
# as a byte anyway, maybe we could have two kinds of masked values: Sticky and
# nonsticky masks? So the mask would be stored as 'u1', 0=unmasked, 1=unsticky,
# 2=sticky. For the invalid domain conversions, someone might also want for
# this not to happen. Maybe instead we should implement these choices as
# subclasses, so we would have a subclass without invalid domin conversion.

if __name__ == '__main__':
    A = MaskedArray(np.arange(12), np.arange(12)%2).reshape((4,3))
    print(hasattr(A, '__array_function__'))
    print(A)
    print(repr(A))
    print(A[2:4])
    print("")
    A = MaskedArray(np.arange(12)).reshape((4,3))
    B = MaskedArray(np.arange(12) % 2).reshape((4,3))
    print(A)
    print(B)
    C = A/B
    print(C)
    print(np.max(C, axis=1))
    print(np.sin(C))
    print(np.sin(C)*np.full(3, 100))
