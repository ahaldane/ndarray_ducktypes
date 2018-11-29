#!/usr/bin/env python
import numpy as np
from duckprint import (duck_str, duck_repr, default_duckprint_options,
                       default_duckprint_formatters, FormatDispatcher)
import builtins
import numpy.core.umath as umath
from numpy.lib.mixins import NDArrayOperatorsMixin
import numpy.core.numerictypes as ntypes
from ndarray_api_mixin import NDArrayAPIMixin

class MaskedArray(NDArrayOperatorsMixin, NDArrayAPIMixin):
    def __init__(self, data=None, mask=None, dtype=None, copy=False,
                order=None, subok=True, ndmin=0, **options):

        self._base = None
        if isinstance(data, MaskedArray):
            self._data = np.array(data._data, copy=copy, order=order,
                                              sobok=subok, ndmin=ndmin)
            self._mask = np.array(data._mask, copy=copy, order=order,
                                              sobok=subok, ndmin=ndmin)

            if mask is not None:
                #XXX should this override the mask? Or be OR'd in?
                raise ValueError("don't use mask if passing a maskedarray")

            self._base = data if data._base is None else data._base
        else:
            if mask is None:
                # if mask is None, user can put masked values in the data.
                # Otherwise, we will get some kind of failure in the line after.
                data, mask = replace_X(data)

            self._data = np.array(data, dtype, copy=copy, order=order,
                                  subok=subok, ndmin=ndmin)
            if mask is None:
                self._mask = np.zeros(self._data.shape, dtype='bool',
                                      order=order)
            elif (is_ndducktype(mask) and mask.shape == self._data.shape and
                    isinstance(mask.dtype.type, np.bool_)):
                self._mask = np.array(mask, dtype, copy=copy, order=order,
                                      subok=subok, ndmin=ndmin)
            else:
                self._mask = np.empty(self._data.shape, dtype='bool')
                self._mask[:] = np.broadcast_to(mask, self._data.shape)

        #XXX make into property
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
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

    def __getitem__(self, ind):
        data = self._data[ind]
        mask = self._mask[ind]
        if data.shape == ():
            return MaskedScalar(data, mask)
        return MaskedArray(data, mask)

    def __setitem__(self, ind, val):
        if isinstance(val, Masked):
            self._mask[ind] = True
        elif isinstance(val, (MaskedArray, MaskedScalar)):
            self._data[ind] = val._data
            self._mask[ind] = val._mask
        else:
            self._data[ind] = val
            self._mask[ind] = False

    @property
    def base(self):
        return self._base

    def _set_base(self, base):
        # private method allowing base to be set by code in this module
        self.base = base


    def filled(self, fill_value=0):
        result = self._data.copy()
        fill_value = self.dtype.type(fill_value)
        np.copyto(result, fill_value, where=self._mask)
        return result

    def count(self, axis=None, keepdims=False):
        """
        Count the non-masked elements of the array along the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the count is performed.
            The default (`axis` = `None`) is perform the count sum over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.

            If this is a tuple of ints, the count is performed on multiple
            axes, instead of a single axis or all the axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        result : ndarray or scalar
            An array with the same shape as self, with the specified
            axis removed. If self is a 0-d array, or if `axis` is None, a scalar
            is returned.

        See Also
        --------
        count_masked : Count masked elements in array or along a given axis.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.arange(6).reshape((2, 3))
        >>> a[1, :] = ma.X
        >>> a
        masked_array(data =
         [[0 1 2]
         [-- -- --]],
                     mask =
         [[False False False]
         [ True  True  True]],
               fill_value = 999999)
        >>> a.count()
        3

        When the `axis` keyword is specified an array of appropriate size is
        returned.

        >>> a.count(axis=0)
        array([1, 1, 1])
        >>> a.count(axis=1)
        array([3, 0])

        """
        return (~self._mask).sum(axis=axis, dtype=np.intp, keepdims=keepdims)

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
        self._data = data
        self._mask = mask
        self.dtype = data.dtype
        self.shape = ()
        self.size = 1
        self.ndim = 0

    def __str__(self):
        if self._mask:
            return MASK_STR
        return str(self._data)

    def __repr__(self):
        if self._mask:
            return 'masked_{}'.format(self.dtype.name)
        return repr(self._data)

    def __format__(self, format_spec):
        if self._mask:
            return 'masked_{}'.format(self.dtype.name)
        return format(self._data, format_spec)

    def filled(self, fill_value=0):
        if self._mask:
            return self._data.dtype.type(fill_value)
        return self._data

# create a special dummy object which signifies "masked", which users can put
# in lists to pass to MaskedArray constructur, or can assign to elements of
# a MaskedArray, to set the mask.
class Masked:
    def __repr__(self):
        return 'masked'
    def __str__(self):
        return 'masked'
masked = X = Masked()

# takes array input, replaces masked value by another element in the array
# This is more-or-less a reimplementation of PyArray_DTypeFromObject to
# account for masked values
def replace_X(data, fill=None):

    # we do two passes: First we figure out the output dtype, then we replace
    # all masked values by the filler "type(0)".

    def get_dtype(data, cur_dtype=None):
        if isinstance(data, list):
            out_dtype = np.result_type(*(get_dtype(d) for d in data))

            if cur_dtype is None:
                return out_dtype
            else:
                return np.promote_types(out_dtype, cur_dtype)

        if is_ndducktype(data):
            return data.dtype

        # otherwise try to coerce it to an ndarray (accounts for __array__,
        # __array_interface__ implementors)
        return np.array(data).dtype

    dt = get_dtype(data)
    fill = dt.type(0)

    def replace(data):
        if data is masked:
            return fill, True
        if isinstance(data, (MaskedScalar, MaskedArray)):
            return data._data, data._mask
        if isinstance(data, list):
            return tuple(zip(*(replace_X(d) for d in data)))
        if is_ndducktype(data):
            return data, np.broadcast_to(False, data.shape)
        # otherwise assume it is some kind of scalar
        return data, False

    return replace(data)

################################################################################
#                               Printing setup
################################################################################

def as_masked_fmt(formattercls):
    # we subclass the original formatter class, and wrap the result of
    # `get_format_func` to take care of masked values.

    class MaskedFormatter(formattercls):
        def get_format_func(self, elem, **options):

            if not elem._mask.any():
                default_fmt = super().get_format_func(elem._data, **options)
                return lambda x: default_fmt(x._data)

            # only get fmt_func based on non-masked values
            # (we take care of masked elements ourselves)
            unmasked = elem._data[~elem._mask]
            default_fmt = super().get_format_func(unmasked, **options)

            # default_fmt should always give back same str length.
            # Figure out what this is with a test call.
            example_str = default_fmt(unmasked[0]) if len(unmasked) > 0 else ''
            masked_str = options['masked_str']
            reslen = builtins.max(len(example_str), len(masked_str))

            # pad the columns to align when including the masked string
            if issubclass(elem.dtype.type, np.floating) and example_str != '':
                # for floats, try to align with decimal point if present
                frac = example_str.partition('.')
                nfrac = len(frac[1]) + len(frac[2])
                masked_str = (masked_str + ' '*nfrac).rjust(reslen)
            else:
                masked_str = masked_str.rjust(reslen)

            def fmt(x):
                if x._mask:
                    return masked_str
                return default_fmt(x._data).rjust(reslen)

            return fmt

    return MaskedFormatter

MASK_STR = 'X'  # make this more configurable later

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
        return a._data
    return a

def getmask(a):
    if isinstance(a, MaskedArray):
        return a._mask
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
            kwargs['out'] = (out[0]._data,)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(getdata(a), *args, **kwargs)

        if self.domain is None:
            m = getmask(a)
        else:
            m = self.domain(d) | getmask(a)

        if out is not ():
            out[0]._mask[:] = m
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
            kwargs['out'] = (out[0]._data,)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, **kwargs)

        m = getmask(a) | getmask(b)
        if self.domain is not None:
            m |= self.domain(da, db)

        if out is not ():
            out[0]._mask[:] = m
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

def is_ndducktype(val):
    return hasattr(val, '__array_function__')

def maskedarray_or_scalar(data, mask, out=None):
    if out is not None:
        return out
    if np.isscalar(data):
        return MaskedScalar(data, mask)
    return MaskedArray(data, mask)

def get_maskedout(out):
    if out is not None:
        if isinstance(out, MaskedArray):
            return out._data, out._mask
        raise Exception("out must be a masked array")
    return None, None

def _copy_mask(mask, outmask=None):
    if outmask is not None:
        result_mask = out_mask
        result_mask[:] = mask
    else:
        result_mask = mask.copy()
    return result_mask

def setup_ducktype():

    @implements(np.all)
    def all(a, axis=None, out=None, keepdims=np._NoValue):
        outdata, outmask = get_maskedout(out)
        result_data = np.all(a.filled(1), axis, outdata, keepdims)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.any)
    def any(a, axis=None, out=None, keepdims=np._NoValue):
        outdata, outmask = get_maskedout(out)
        result_data = np.any(a.filled(0), axis, outdata, keepdims)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    max_filler = ntypes._minvals
    max_filler.update([(k, -np.inf) for k in [np.float32, np.float64]])
    min_filler = ntypes._maxvals
    min_filler.update([(k, +np.inf) for k in [np.float32, np.float64]])
    if 'float128' in ntypes.typeDict:
        max_filler.update([(np.float128, -np.inf)])
        min_filler.update([(np.float128, +np.inf)])

    @implements(np.max)
    def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(max_filler[a.dtype])
        result_data = np.max(filled, axis, outdata, keepdims, initial)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.argmax)
    def argmax(a, axis=None, out=None):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(max_filler[a.dtype])
        result_data = np.argmax(filled, axis, outdata)
        result_mask = np.all(a._mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.min)
    def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(min_filler[a.dtype])
        result_data = np.min(filled, axis, outdata, keepdims, initial)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.argmin)
    def argmin(a, axis=None, out=None):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(min_filler[a.dtype])
        result_data = np.argmin(filled, axis, outdata)
        result_mask = np.all(a._mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.argsort)
    def argsort(a, axis=-1, kind='quicksort', order=None):
        # masked values get masked indices, and are sorted to min
        filled = a.filled(min_filler[a.dtype])
        # XXX where does the min_filler get put when in v?
        # Probably need to extract on the unmasked values, do argsort,
        # then reconstruct indice with masked elements present.
        result_data = np.argsort(filled, axis, kind, order)
        result_mask = a._mask[result]
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.argpartition)
    def argpartition(a, kth, axis=-1, kind='introselect', order=None):
        filled = a.filled(min_filler[a.dtype])
        result_data = np.argpartition(filled, kth, axis, kind, order)
        result_mask = a._mask[result]
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.searchsorted)
    def searchsorted(a, v, side='left', sorter=None):
        # XXX account for mask in v?
        # XXX return a MaskedArray?
        # XXX where does the min_filler get put when in v?
        # Probably the "correct" way to do this is do searchsorted
        # only on the unmasked values,
        filled = a.filled(min_filler[a.dtype])
        return np.searchsorted(filled, v, side, sorter)

    @implements(np.sort)
    def sort(a, axis=-1, kind='quicksort', order=None):
        # XXX where does the min_filler get put when in v?
        # masked values get masked indices, and are sorted to min
        filled = a.filled(min_filler[a.dtype])
        inds = np.argsort(filled, axis, kind, order)
        result_data = a._data[inds]
        result_mask = a._mask[inds]
        return maskedarray_or_scalar(result_data, result_mask)


    @implements(np.mean)
    def mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Returns the average of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        ndarray.mean : corresponding function for ndarrays
        numpy.mean : Equivalent function
        numpy.ma.average: Weighted average.

        Examples
        --------
        >>> a = np.ma.array([1,2,3], mask=[False, False, True])
        >>> a
        masked_array(data = [1 2 --],
                     mask = [False False  True],
               fill_value = 999999)
        >>> a.mean()
        1.5

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        outdata, outmask = get_maskedout(out)

        # code partly copied from _mean in numpy/core/_methods.py

        is_float16_result = False
        rcount = a.count(axis=axis, **kwargs)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None:
            if issubclass(a.dtype.type, (np.integer, np.bool_)):
                dtype = np.dtype('f8')
            elif issubclass(a.dtype.type, np.float16):
                dtype = np.dtype('f4')
                is_float16_result = True

        ret = np.sum(a.filled(0), axis=axis, out=outdata, dtype=dtype, **kwargs)
        retmask = np.all(a._mask, axis=axis, out=outmask, **kwargs)

        with np.errstate(divide='ignore', invalid='ignore'):
            if is_ndducktype(ret):
                ret = np.true_divide(
                        ret, rcount, out=ret, casting='unsafe', subok=False)
                if is_float16_result and out is None:
                    ret = arr.dtype.type(ret)
            elif hasattr(ret, 'dtype'):
                if is_float16_result:
                    ret = arr.dtype.type(ret / rcount)
                else:
                    ret = ret.dtype.type(ret / rcount)
            else:
                ret = ret / rcount

        return maskedarray_or_scalar(ret, retmask, out)

    @implements(np.var)
    def var(a, axis=None, dtype=None, out=None, ddof=0,
            keepdims=np._NoValue):
        """
        Returns the variance of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        ndarray.var : corresponding function for ndarrays
        numpy.var : Equivalent function
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        outdata, outmask = get_maskedout(out)

        # code largely copied from _methods.var

        # Make this warning show up on top.
        if ddof >= rcount:
            warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                          stacklevel=2)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None and issubclass(a.dtype.type, (np.integer, np.bool_)):
            dtype = np.dtype('f8')

        # Compute the mean, keeping same dims. Note that if dtype is not of
        # inexact type then arraymean will not be either.
        rcount = a.count(axis=axis, keepdims=True)
        arrmean = a.filled(0).sum(axis=axis, dtype=dtype, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            if (not is_ndducktype(arrmean) and hasattr(arrmean, 'dtype')):
                arrmean = np.true_divide(arrmean, rcount, out-arrmean,
                                         casting='unsafe', subok=False)
            else:
                arrmean = arrmean.dtype.type(arrmean / rcount)

        # Compute sum of squared deviations from mean
        x = a - arrmean
        if issubclass(a.dtype.type, np.complexfloating):
            x = np.multiply(x, np.conjugate(x), out=x).real
        else:
            x = np.multiply(x, x, out=x)
        ret = x.sum(axis, dtype, out=outdata, **kwargs)

        # Compute degrees of freedom and make sure it is not negative.
        rcount = self.count(axis=axis, **kwargs)
        rcount = np.maximum(rcount - ddof, 0)

        # divide by degrees of freedom
        with np.errstate(divide='ignore', invalid='ignore'):
            if is_ndducktype(ret):
                ret = np.true_divide(
                        ret, rcount, out=ret, casting='unsafe', subok=False)
            elif hasattr(ret, 'dtype'):
                ret = ret.dtype.type(ret / rcount)
            else:
                ret = ret / rcount

        return maskedarray_or_scalar(ret, rcount == 0, out)

    @implements(np.std)
    def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        ret = np.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                     keepdims=keepdims)

        if is_ndducktype(ret):
            ret = np.sqrt(ret, out=ret)
        elif hasattr(ret, 'dtype'):
            ret = ret.dtype.type(np.sqrt(ret))
        else:
            ret = np.sqrt(ret)
        return ret

    @implements(np.choose)
    def choose(a, choices, out=None, mode='raise'):
        outdata, outmask = get_maskedout(out)
        result_data = np.choose(a._data, choices, outdata, mode)
        result_mask = np.choose(a._mask, choices, outmask, mode)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.clip)
    def clip(a, a_min, a_max, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.clip(a._data, a_min, a_max, outdata)
        result_mask = _copy_mask(a._mask, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.compress)
    def compress(condition, a, axis=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.compress(condition, a._data, axis, outdata)
        result_mask = np.compress(condition, a._data, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.copy)
    def copy(a, order='K'):
        result_data = np.copy(a._data, order=order)
        result_mask = np.copy(a._mask, order=order)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.prod)
    def prod(a, axis=None, dtype=None, out=None, keepdims=False):
        outdata, outmask = get_maskedout(out)
        result_data = np.prod(a.filled(1), axis=axis, dtype=dtype, out=outdata,
                                           keepdims=keepdims)
        result_mask = np.all(a._mask, axis=axis, out=outmask, keepdims=keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.cumprod)
    def cumprod(a, axis=None, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.cumprod(a.filled(1), axis, dtype=dtype, out=outdata)
        result_mask = np.all(a._mask, axis, out=outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.sum)
    def sum(a, axis=None, dtype=None, out=None, keepdims=False):
        outdata, outmask = get_maskedout(out)
        result_data = np.sum(a.filled(0), axis, dtype=dtype, out=outdata)
        result_mask = np.all(a._mask, axis, out=outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.cumsum)
    def cumsum(self, axis=None, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.cumsum(a.filled(0), axis, dtype=dtype, out=outdata)
        result_mask = np.all(a._mask, axis, out=outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.diagonal)
    def diagonal(a, offset=0, axis1=0, axis2=1):
        result = np.diagonal(a._data, offset=offset, axis1=axis1, axis2=axis2)
        rmask = np.diagonal(a._mask, offset=offset, axis1=axis1, axis2=axis2)
        return maskedarray_or_scalar(result, rmask)

    @implements(np.trace)
    def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result = np.trace(a.filled(0), offset=offset, axis1=axis1, axis2=axis2,
                                       dtype=dtype, out=outdata)
        mask_trace = np.trace(~a._mask, offset=offset, axis1=axis1, axis2=axis2,
                                        dtype=dtype, out=outdata)
        result_mask = mask_trace == 0
        return maskedarray_or_scalar(result, result_mask)

    @implements(np.dot)
    def dot(a, b, out=None):
        outdata, outmask = get_maskedout(out)
        a, b = MaskedArray(a), MaskedArray(b)
        result_data = np.dot(a.filled(0), b.filled(0), out=outdata)
        result_mask = np.logical_not(np.dot(~a._mask, ~b._mask), out=outmask)
        return maskedarray_or_scalar(result_data, rmask, out)

    @implements(np.real)
    def real(self):
        # returns a view of the data only. mask is a copy.
        result_data = np.real(a._data)
        result_mask = a._mask.copy()
        return maskedarray_or_scalar(result_data, rmask)
        # XXX not clear if mask should be copied or viewed: If we unmask
        # the imag part, should be real part be unmasked too?

    @implements(np.imag)
    def imag(a):
        # returns a view of the data only. mask is a copy.
        result_data = np.imag(a._data)
        result_mask = a._mask.copy()
        return maskedarray_or_scalar(result_data, rmask)
        # XXX not clear if mask should be copied or viewed: If we unmask
        # the imag part, should be real part be unmasked too?

    @implements(np.nonzero)
    def nonzero(a):
        #XXX should this return a MaskedArray?
        return np.nonzero(a.filled(0))

    @implements(np.partition)
    def partition(a, kth, axis=-1, kind='introselect', order=None):
        # We use argparition to construt a fancy index.
        # For fancy index: np.arange at all axes except the selected axis
        # (Note: This is the Xth time I have wanted a function to do this kind
        # of thing for me in numpy)
        ndim = a.ndim
        fancy_ind = []
        for ax in range(ndim):
            if ax == axis:
                filled = a.filled(min_filler[a.dtype])
                inds = np.argpartition(filled, kth, axis, kind, order)
                fancy_ind.append(inds)
            else:
                newdim = (None,)*(ax-1) + (slice(None),) + (None,)*(ndim-ax)
                fancy_ind.append(np.arange(a.shape[ax])[newdim])

        return a[tuple(fancy_ind)]

    @implements(np.ptp)
    def ptp(a, axis=None, out=None, keepdims=False):
        # Original numpy function is fine.
        return np.ptp.__wrapped__(self, axis, out, keepdims)

    @implements(np.take)
    def take(self, indices, axis=None, out=None, mode='raise'):
        outdata, outmask = get_maskedout(out)
        result_data = np.take(a._data, indices, axis, outdata, mode)
        result_mask = np.take(a._mask, indices, axis, outmask, mode)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.put)
    def put(a, indices, values, mode='raise'):
        data, mask = replace_X(data)
        np.put(a._data, indices, data, mode)
        np.put(a._mask, indices, mask, mode)
        return None

    @implements(np.ravel)
    def ravel(a, order='C'):
        return MaskedArray(a._data.ravel(order=order),
                           a._mask.ravel(order=order))

    @implements(np.repeat)
    def repeat(a, repeats, axis=None):
        return MaskedArray(a._data.repeat(repeats, axis),
                           a._mask.repeat(repeats, axis))

    @implements(np.reshape)
    def reshape(a, shape, order='C'):
        # XXX set base
        return MaskedArray(a._data.reshape(shape, order=order),
                           a._mask.reshape(shape, order=order))


    @implements(np.resize)
    def resize(a, new_shape, refcheck=True): #XXX what is this refcheck?
        return MaskedArray(a._data.resize(new_shape, refcheck),
                           a._mask.resize(new_shape, refcheck))

    @implements(np.round)
    def round(a, decimals=0, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.round(a._data, decimals, outdata)
        result_mask = _copy_mask(a._mask, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.squeeze)
    def squeeze(a, axis=None):
        return MaskedArray(a._data.squeeze(new_shape, refcheck),
                           a._mask.squeeze(new_shape, refcheck))
    @implements(np.swapaxes)
    def swapaxes(a, axis1, axis2):
        return MaskedArray(a._data.swapaxes(axis1, axis2),
                           a._mask.swapaxes(axis1, axis2))

    @implements(np.transpose)
    def transpose(a, *axes):
        return MaskedArray(a._data.transpose(*axes),
                           a._mask.transpose(*axes))

    @implements(np.concatenate)
    def concatenate(arrays, axis=0, out=None):
        outdata, outmask = get_maskedout(out)
        arrays = [MaskedArray(a) for a in arrays]
        result_data = np.concatenate([a._data for a in arrays], axis, outdata)
        result_mask = np.concatenate([a._mask for a in arrays], axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask)

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
    print(A[1:3])
    print("")
    A = MaskedArray(np.arange(12)).reshape((4,3))
    B = MaskedArray(np.arange(12) % 2).reshape((4,3))
    print(A)
    print(B)
    C = A/B
    print(C)
    print(C.T)
    print(np.max(C, axis=1))
    print(np.sin(C))
    print(np.sin(C)*np.full(3, 100))
    print(repr(MaskedArray([[X, X, 3], [1, X, 1]])))
    print(repr(MaskedArray([[X, X, X], [X, X, X]])))
