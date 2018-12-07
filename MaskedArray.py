#!/usr/bin/env python
import numpy as np
from duckprint import (duck_str, duck_repr, duck_array2string,
    default_duckprint_options, default_duckprint_formatters, FormatDispatcher)
import builtins
import numpy.core.umath as umath
from numpy.lib.mixins import NDArrayOperatorsMixin
from ndarray_api_mixin import NDArrayAPIMixin
import numpy.core.numerictypes as ntypes
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import _broadcast_shape
import operator

class MaskedArray(NDArrayOperatorsMixin, NDArrayAPIMixin):
    def __init__(self, data, mask=None, dtype=None, copy=False,
                order=None, subok=True, ndmin=0, **options):

        self._base = None
        if isinstance(data, (MaskedArray, MaskedScalar)):
            self._data = np.array(data._data, copy=copy, order=order,
                                              subok=subok, ndmin=ndmin)
            self._mask = np.array(data._mask, copy=copy, order=order,
                                              subok=subok, ndmin=ndmin)

            if mask is not None:
                #XXX should this override the mask? Or be OR'd in?
                raise ValueError("don't use mask if passing a maskedarray")

            self._base = data if data._base is None else data._base
        elif data is X and mask is None:
            # 0d masked array
            if dtype is None:
                raise ValueError("must supply dtype if all elements are masked")
            self._data = np.array(dtype.type(0))
            self._mask = np.array(True)
        else:
            if mask is None:
                # if mask is None, user can put X in the data.
                # Otherwise, X will cause some kind of error in np.array below
                data, mask = replace_X(data, dtype=dtype)

                # replace_X sometimes uses broadcast_to, which returns a
                # readonly array with funny strides. Make writeable if so.
                if (isinstance(mask, np.ndarray) and
                        mask.flags['WRITEABLE'] == False):
                    mask = mask.copy()

            self._data = np.array(data, dtype=dtype, copy=copy, order=order,
                                  subok=subok, ndmin=ndmin)

            if mask is None:
                self._mask = np.zeros(self._data.shape, dtype='bool',
                                      order=order)
            elif (is_ndducktype(mask) and mask.shape == self._data.shape and
                    issubclass(mask.dtype.type, np.bool_)):
                self._mask = np.array(mask, dtype, copy=copy, order=order,
                                      subok=subok, ndmin=ndmin)
            else:
                self._mask = np.empty(self._data.shape, dtype='bool')
                self._mask[...] = np.broadcast_to(mask, self._data.shape)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        impl, checked_args = HANDLED_FUNCTIONS[func]

        if checked_args is not None:
            types = (type(a) for n,a in enumerate(args) if n in checked_args)

        #types are allowed to be Masked* or plain ndarrays
        def allowed(t):
            return (issubclass(t, (MaskedArray, MaskedScalar)) or
                    t is np.ndarray)
        if not all(allowed(t) for t in types):
            return NotImplemented

        return impl(*args, **kwargs)

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
        if not isinstance(ind, tuple):
            ind = (ind,)

        # If a boolean MaskedArray is provided as an ind, treat masked vals as
        # False. Allows code like "a[a>0]", which is then the same as
        # "a[np.nonzero(a>0)]"
        ind = tuple(i.filled(False) if
                (isinstance(i, MaskedArray) and i.dtype.type is np.bool_)
                else i for i in ind)

        data = self._data[ind]
        mask = self._mask[ind]

        if np.isscalar(mask): # use mask, not data, to account for obj arrays
            return MaskedScalar(data, mask, dtype=self.dtype)
        return MaskedArray(data, mask)

    def __setitem__(self, ind, val):
        if not isinstance(ind, tuple):
            ind = (ind,)

        # If a boolean MaskedArray is provided as an ind, treat masked vals as
        # False. Allows code like "a[a>0] = X"
        ind = tuple(i.filled(False) if
                (isinstance(i, MaskedArray) and i.dtype.type is np.bool_)
                else i for i in ind)

        if val is X:
            self._mask[ind] = True
        elif isinstance(val, (MaskedArray, MaskedScalar)):
            self._data[ind] = val._data
            self._mask[ind] = val._mask
        else:
            self._data[ind] = val
            self._mask[ind] = False

    # override the NDArrayOperatorsMixin implementations for cmp ops, as
    # currently those ufuncs don't work for flexible types
    def _cmp_op(self, other, op):
        if other is X:
            db, mb = self._data.dtype.type(0), np.bool_(True)
        else:
            db, mb = getdata(other), getmask(other)

        result = op(self._data, db)
        m = self._mask | mb

        if np.isscalar(result):
            return MaskedScalar(result, m)
        return MaskedArray(result, m)

    def __lt__(self, other):
        return self._cmp_op(other, operator.lt)

    def __le__(self, other):
        return self._cmp_op(other, operator.le)

    def __eq__(self, other):
        return self._cmp_op(other, operator.eq)

    def __ne__(self, other):
        return self._cmp_op(other, operator.ne)

    def __gt__(self, other):
        return self._cmp_op(other, operator.gt)

    def __ge__(self, other):
        return self._cmp_op(other, operator.ge)

    @property
    def shape(self):
        return self._data.shape

    @shape.setter
    def shape(self, shp):
        self._data.shape = shp
        self._mask.shape = shp

    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, dt):
        self._data.dtype = dt

    # XXX decisions about how to treat base still to be made
    @property
    def base(self):
        return self._base

    def _set_base(self, base):
        # private method allowing base to be set by code in this module
        self.base = base

    @property
    def mask(self):
        # return a readonly view of mask
        m = np.ndarray(self._mask.shape, np.bool_, self._mask)
        m.flags['WRITEABLE'] = False
        return m

    def view(self, dtype=None, type=None):
        if type is not None:
            raise ValueError("subclasses not yet supported")

        if dtype is None:
            dtype = self.dtype
        else:
            try:
                dtype = np.dtype(dtype)
            except ValueError:
                raise ValueError("dtype must be a dtype, not subclass")

        if dtype.itemsize != self.itemsize:
            raise ValueError("views of MaskedArrays cannot change the "
                             "datatype's itemsize")

        return MaskedArray(self._data.view(dtype), self._mask)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        result_data = self._data.astype(dtype, order, casting, subok, copy)
        result_mask = self._mask.astype(bool, order, casting, subok, copy)
        return MaskedArray(result_data, result_mask)

    # rename this to "X" to be shorter, since it is heavily used?
    #      arr.X()     arr.X(0)   arr.filled()   arr.filled(0)
    def filled(self, fill_value=0, minmax=None):

        if minmax is not None:
            if fill_value != 0:
                raise Exception("Do not give fill_value if providing minmax")
            if minmax == 'max':
                fill_value = _max_filler[self.dtype]
            elif minmax == 'min':
                fill_value = _min_filler[self.dtype]
            else:
                raise ValueError("minmax should be 'min' or 'max'")

        result = self._data.copy()
        # next line is more complicated that it should be due to struct types
        result[self._mask] = np.array(fill_value, dtype=self.dtype)[()]
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

    # This works inplace, unlike np.sort
    def sort(self, axis=-1, kind='quicksort', order=None):
        # Note: See comment in np.sort impl below for trick used here.
        # This is the inplace version
        self._data[self._mask] = _min_filler[self.dtype]
        self._data.sort(axis, kind, order)
        self._mask.sort(axis, kind)

    # This works inplace, unlike np.resize, and fills with repeat instead of 0
    def resize(self, new_shape, refcheck=True):
        self._data.resize(new_shape, refcheck)
        self._mask.resize(new_shape, refcheck)

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
class MaskedScalar(NDArrayOperatorsMixin, NDArrayAPIMixin):
    def __init__(self, data, mask=None, dtype=None):
        if isinstance(data, MaskedScalar):
            self._data = data._data
            self._mask = data._mask
            if mask is not None:
                raise ValueError("don't use mask if passing a maskedscalar")
            self._dtype = self._data.dtype
        elif data is X:
            if dtype is None:
                raise ValueError("Must supply dtype when data is X")
            if mask is not None:
                raise ValueError("don't supply mask when data is X")
            self._data = dtype.type(0)
            self._mask = np.bool_(True)
            self._dtype = self._data.dtype
        else:
            if dtype is None or dtype.type is not np.object_:
                self._data = np.array(data, dtype=dtype)[()]
                self._mask = np.bool_(mask)
                if not np.isscalar(self._data) or not np.isscalar(self._mask):
                    raise ValueError("MaskedScalar must be called with scalars")
                self._dtype = self._data.dtype
            else:
                # object dtype treated specially
                self._data = data
                self._mask = np.bool_(mask)
                self._dtype = dtype

    @property
    def shape(self):
        return ()

    @property
    def dtype(self):
        return self._dtype

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        impl, checked_args = HANDLED_FUNCTIONS[func]

        if checked_args is not None:
            types = (t for n,t in enumerate(types) if n in checked_args)
        if not all(issubclass(t, (MaskedArray, MaskedScalar)) for t in types):
            return NotImplemented

        return impl(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in masked_ufuncs:
            return NotImplemented

        return getattr(masked_ufuncs[ufunc], method)(*inputs, **kwargs)

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

    @property
    def mask(self):
        return self._mask

    def filled(self, fill_value=0, minmax=None):
        if minmax is not None:
            if fill_value != 0:
                raise Exception("Do not give fill_value if providing minmax")
            if minmax == 'max':
                fill_value = _max_filler[self.dtype]
            elif minmax == 'min':
                fill_value = _min_filler[self.dtype]
            else:
                raise ValueError("minmax should be 'min' or 'max'")

        if self._mask:
            return self._data.dtype.type(fill_value)
        return self._data

# create a special dummy object which signifies "masked", which users can put
# in lists to pass to MaskedArray constructor, or can assign to elements of
# a MaskedArray, to set the mask.
class Masked:
    def __repr__(self):
        return 'input_mask'
    def __str__(self):
        return 'input_mask'

    # prevent X from being used as an element in np.array, to avoid
    # confusing the user. X should only be used in MaskedArrays
    def __array__(self):
        # hack: the only Exception that numpy doesn't clear here is MemoryError
        raise MemoryError("Masked X should only be used in "
                          "MaskedArray assignment or construction")

masked = X = Masked()

# takes array input, replaces masked value by 0 and return filled data & mask.
# This is more-or-less a reimplementation of PyArray_DTypeFromObject to
# account for masked values
def replace_X(data, dtype=None):

    # we do two passes: First we figure out the output dtype, then we replace
    # all masked values by the filler "type(0)".

    def get_dtype(data, cur_dtype=X):
        if isinstance(data, (list, tuple)):
            dtypes = (get_dtype(d, cur_dtype) for d in data)
            dtypes = [dt for dt in dtypes  if dt is not X]
            if not dtypes:
                return cur_dtype

            out_dtype = np.result_type(*dtypes)
            if cur_dtype is X:
                return out_dtype
            else:
                return np.promote_types(out_dtype, cur_dtype)

        if data is X:
            return X

        if is_ndducktype(data):
            return data.dtype

        # otherwise try to coerce it to an ndarray (accounts for __array__,
        # __array_interface__ implementors)
        return np.array(data).dtype

    if dtype is None:
        dtype = get_dtype(data)
        if dtype is X:
            raise ValueError("must supply dtype if all elements are masked")
    else:
        dtype = np.dtype(dtype)

    fill = dtype.type(0)

    def replace(data):
        if data is X:
            return fill, True
        if isinstance(data, (MaskedScalar, MaskedArray)):
            return data._data, data._mask
        if isinstance(data, list):
            return (list(x) for x in zip(*(replace(d) for d in data)))
        if is_ndducktype(data):
            return data, np.broadcast_to(False, data.shape)
        # otherwise assume it is some kind of scalar
        return data, False

    return replace(data)

# carried over from numpy's MaskedArray, but naming is somewhat confusing
# as the max_filler is actually the minimum value. Change?
_max_filler = ntypes._minvals
_max_filler.update([(k, -np.inf) for k in [np.float32, np.float64]])
_min_filler = ntypes._maxvals
_min_filler.update([(k, +np.inf) for k in [np.float32, np.float64]])
if 'float128' in ntypes.typeDict:
    _max_filler.update([(np.float128, -np.inf)])
    _min_filler.update([(np.float128, +np.inf)])

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
                # XXX safer/better: simply center the X?
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
    if isinstance(a, (MaskedArray, MaskedScalar)):
        return a._data
    return a

def getmask(a):
    if isinstance(a, (MaskedArray, MaskedScalar)):
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
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
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
        ma, mb = getmask(a), getmask(b)

        # treat X as a masked value of the other array's dtype
        if da is X:
            da, ma = db.dtype.type(0), np.bool_(True)
        if db is X:
            db, mb = da.dtype.type(0), np.bool_(True)

        out = kwargs.get('out', ())
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, **kwargs)

        m = ma | mb
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

def implements(numpy_function, checked_args=None):
    """Register an __array_function__ implementation for MaskedArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = (func, checked_args)
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
    # Some design principles (subject to revision):
    #
    # * out arguments must be maskedarrays. This forces users to use safe code
    #   which doesn't lose the mask or risk using uninitialized data under mask.
    # * methods which return `int` arrays meant for indexing (eg, argsort,
    #   nonzero) will return plain ndarrays. Otherwise return maskedarrays.
    # * mask behaves as "skipna" style (see NEP)
    # * masks sort as greater than all other values
    #
    # XXX some methods below may need special-casing in case 'X' was supplied
    # as an argument. Maybe?

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

    @implements(np.max)
    def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(minmax='max')
        result_data = np.max(filled, axis, outdata, keepdims, initial)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.argmax)
    def argmax(a, axis=None, out=None):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(minmax='max')
        result_data = np.argmax(filled, axis, outdata)
        result_mask = np.all(a._mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.min)
    def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(minmax='min')
        result_data = np.min(filled, axis, outdata, keepdims, initial)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.argmin)
    def argmin(a, axis=None, out=None):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(minmax='min')
        result_data = np.argmin(filled, axis, outdata)
        result_mask = np.all(a._mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.sort)
    def sort(a, axis=-1, kind='quicksort', order=None):
        # Note: This is trickier than it looks. The first line sorts the mask
        # together with any min_vals which may be present, so there appears to
        # be a problem ordering mask vs min_val elements.
        # But, since we know all the masked elements have to end up at the end
        # of the axis, we can sort the mask too and everything works out. The
        # mask-sort only swaps the mask between min_val and masked positions
        # which have the same underlying data.
        result_data = np.sort(a.filled(minmax='min'), axis, kind, order)
        result_mask = np.sort(a._mask, axis, kind)  #or partition for speed?
        return maskedarray_or_scalar(result_data, result_mask)
        # Note: lexsort may be faster, but doesn't provide kind or order kwd

    @implements(np.argsort)
    def argsort(a, axis=-1, kind='quicksort', order=None):
        # Similar to mask-sort trick in sort above, here after sorting data we
        # re-sort based on mask. Use the property that if you argsort the index
        # array produced by argsort you get the element rank, which can be
        # argsorted again to get back the sort indices. However, here we
        # modify the rank based on the mask before inverting back to indices.
        # Uses two argsorts plus a temp array. Further speedups?
        inds = np.argsort(a.filled(minmax='min'), axis, kind, order)
        # next two lines "reverse" the argsort (same as double-argsort)
        ranks = np.empty(inds.shape, dtype=inds.dtype)
        np.put_along_axis(ranks, inds, np.arange(a.shape[axis]), axis)
        # prepare to resort but make masked elem highest rank
        ranks[a._mask] = _min_filler[ranks.dtype]
        return np.argsort(ranks, axis, kind)

    @implements(np.argpartition)
    def argpartition(a, kth, axis=-1, kind='introselect', order=None):
        # see argsort for explanation
        filled = a.filled(minmax='min')
        inds = np.argpartition(filled, kth, axis, kind, order)
        ranks = np.empty(inds.shape, dtype=inds.dtype)
        np.put_along_axis(ranks, inds, np.arange(a.shape[axis]), axis)
        ranks[a._mask] = _min_filler[ranks.dtype]
        return np.argpartition(ranks, kth, axis, kind)

    @implements(np.searchsorted)
    def searchsorted(a, v, side='left', sorter=None):
        inds = np.searchsorted(a.filled(minmax='min'), v.filled(minmax='min'),
                               side, sorter)

        # Line above treats mask and minval as the same, we need to fix it up
        maskleft = len(a) - np.sum(a._mask)
        if side == 'left':
            # masked vals in v need to be moved right to the left end of the
            # masked vals in a (which have to be to the right end of a).
            inds[v._mask] = maskleft
        else:
            # minvals in v meed to be moved left to the left end of the
            # masked vals in a.
            minval = _min_filler[v.dtype]
            inds[(v._data == minval) & ~v._mask] = maskleft

        return inds

    @implements(np.digitize)
    def digitize(x, bins, right=False):
        # here for compatibility, searchsorted below is happy to take this
        # XXX comment above copied from orig. What does it mean?
        if np.issubdtype(x.dtype, _nx.complexfloating):
            raise TypeError("x may not be complex")

        #XXX implement this part
        #mono = _monotonicity(bins)
        #if mono == 0:
        #    raise ValueError("bins must be monotonically "
        #                     "increasing or decreasing")

        # this is backwards because the arguments below are swapped
        side = 'left' if right else 'right'
        if mono == -1:
            # reverse the bins, and invert the results
            return len(bins) - np.searchsorted(bins[::-1], x, side=side)
        else:
            return np.searchsorted(bins, x, side=side)

    @implements(np.lexsort)
    def lexsort(keys, axis=-1):
        if not isinstance(keys, tuple):
            keys = tuple(keys)

        # strategy: for each key, split into a mask and data key.
        # So, we end up sorting twice as many keys. Mask is primary key (last).
        keys = tuple(x for k in keys for x in (k._data, k._mask))
        return np.lexsort(keys, axis)

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
        rcount = a.count(axis=axis, **kwargs)
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

    @implements(np.average, checked_args=(0,))
    def average(a, axis=None, weights=None, returned=False):
        if weights is None:
            avg = a.mean(axis)
            if returned:
                return avg, avg.dtype.dtype(a.count(axis)/avg.count(axis))
            return avg

        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)
            # Note: No float16 special case, since ndarray.average skips it

        wgt = weights

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        if isinstance(wgt, MaskedArray):
            mask = a._mask | wgt._mask
            wgt = MaskedArray(wgt._data, mask)
            a = MaskedArray(a._data, mask)
        else:
            wgt = MaskedArray(wgt, a._mask)


        scl = wgt.sum(axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)/scl

        if returned:
            return avg, scl
        return avg

    #@implements(np.median)
    #def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    #    return np.percentile(a, q=50., axis=axis, out=out,
    #                      overwrite_input=overwrite_input,
    #                      interpolation="linear", keepdims=keepdims)

    #@implements(np.percentile)
    #@implements(np.quantile)
    #@implements(np.cov)
    #@implements(np.corrcoef)

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

    @implements(np.product)
    def product(*args, **kwargs):
        return prod(*args, **kwargs)

    @implements(np.cumprod)
    def cumprod(a, axis=None, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.cumprod(a.filled(1), axis, dtype=dtype, out=outdata)
        result_mask = np.all(a._mask, axis, out=outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.cumproduct)
    def cumproduct(*args, **kwargs):
        return cumprod(*args, **kwargs)

    @implements(np.sum)
    def sum(a, axis=None, dtype=None, out=None, keepdims=False):
        outdata, outmask = get_maskedout(out)
        result_data = np.sum(a.filled(0), axis, dtype=dtype, out=outdata)
        result_mask = np.all(a._mask, axis, out=outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.cumsum)
    def cumsum(a, axis=None, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.cumsum(a.filled(0), axis, dtype=dtype, out=outdata)
        result_mask = np.all(a._mask, axis, out=outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.diagonal)
    def diagonal(a, offset=0, axis1=0, axis2=1):
        result = np.diagonal(a._data, offset=offset, axis1=axis1, axis2=axis2)
        rmask = np.diagonal(a._mask, offset=offset, axis1=axis1, axis2=axis2)
        return maskedarray_or_scalar(result, rmask)

    @implements(np.diag)
    def diag(v, k=0):
        s = v.shape
        if len(s) == 1:
            n = s[0]+abs(k)
            res = MaskedArray(np.zeros((n, n), v.dtype))
            if k >= 0:
                i = k
            else:
                i = (-k) * n
            res[:n-k].flat[i::n+1] = v
            return res
        elif len(s) == 2:
            return np.diagonal(v, k)
        else:
            raise ValueError("Input must be 1- or 2-d.")

    @implements(np.diagflat)
    def diagflat(v, k=0):
        return np.diag(v.ravel(), k)

    @implements(np.tril)
    def tril(m, k=0):
        mask = np.tri(*m.shape[-2:], k=k, dtype=bool)
        return np.where(mask, m, zeros(1, m.dtype))

    @implements(np.triu)
    def triu(m, k=0):
        mask = np.tri(*m.shape[-2:], k=k-1, dtype=bool)
        return np.where(mask, zeros(1, m.dtype), m)

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
        result_mask = np.dot(~a._mask, ~b._mask, out=outmask)
        result_mask = np.logical_not(result_mask, out=result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.vdot)
    def vdot(a, b):
        #a, b = MaskedArray(a), MaskedArray(b)
        result_data = np.vdot(a.filled(0), b.filled(0))
        result_mask = np.vdot(~a._mask, ~b._mask)
        result_mask = np.logical_not(result_mask, out=result_mask)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.cross)
    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        #a, b = MaskedArray(a), MaskedArray(b)
        result_data = np.cross(a.filled(0), b.filled(0), axisa, axisb, axisc,
                               axis)
        result_mask = np.cross(~a._mask, ~b._mask, axisa, axisb, axisc, axis)
        result_mask = np.logical_not(result_mask, out=result_mask)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.inner)
    def inner(a, b):
        #a, b = MaskedArray(a), MaskedArray(b)
        result_data = np.inner(a.filled(0), b.filled(0))
        result_mask = np.inner(~a._mask, ~b._mask)
        result_mask = np.logical_not(result_mask, out=result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.outer)
    def outer(a, b, out=None):
        outdata, outmask = get_maskedout(out)
        #a, b = MaskedArray(a), MaskedArray(b)
        result_data = np.outer(a.filled(0), b.filled(0), out=outdata)
        result_mask = np.outer(~a._mask, ~b._mask, out=outmask)
        result_mask = np.logical_not(result_mask, out=result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.kron)
    def kron(a, b):
        a = MaskedArray(a, copy=False, subok=True, ndmin=b.ndim)
        if (a.ndim == 0 or b.ndim == 0):
            return np.multiply(a, b)
        a_shape = a.shape
        b_shape = b.shape
        nd = ndb
        if b.ndim > a.ndim:
            a_shape = (1,)*(ndb-nda) + a_shape
        elif b.ndim < a.ndim:
            b_shape = (1,)*(nda-ndb) + b_shape
            nd = nda

        result = np.outer(a, b).reshape(a_shape + b_shape)
        axis = nd-1
        for _ in range(nd):
            result = np.concatenate(result, axis=axis)
        return result

    @implements(np.tensordot)
    def tensordot(a, b, axes=2):
        try:
            iter(axes)
        except Exception:
            axes_a = list(range(-axes, 0))
            axes_b = list(range(0, axes))
        else:
            axes_a, axes_b = axes

        def nax(ax):
            try:
                return len(ax), list(ax)
            except TypeError:
                return 1, [ax]

        na, axes_a = nax(axes_a)
        nb, axes_b = nax(axes_b)

        ashape, bshape = a.shape, b.shape
        nda, ndb = a.ndim, b.ndim
        equal = True
        if na != nb:
            equal = False
        else:
            for k in range(na):
                if ashape[axes_a[k]] != bshape[axes_b[k]]:
                    equal = False
                    break
                if axes_a[k] < 0:
                    axes_a[k] += nda
                if axes_b[k] < 0:
                    axes_b[k] += ndb
        if not equal:
            raise ValueError("shape-mismatch for sum")

        # Move the axes to sum over to the end of "a"
        # and to the front of "b"
        notin = [k for k in range(nda) if k not in axes_a]
        newaxes_a = notin + axes_a
        N2 = 1
        for axis in axes_a:
            N2 *= ashape[axis]
        newshape_a = (int(multiply.reduce([ashape[ax] for ax in notin])), N2)
        olda = [ashape[axis] for axis in notin]

        notin = [k for k in range(ndb) if k not in axes_b]
        newaxes_b = axes_b + notin
        N2 = 1
        for axis in axes_b:
            N2 *= bshape[axis]
        newshape_b = (N2, int(np.multiply.reduce([bshape[ax] for ax in notin])))
        oldb = [bshape[axis] for axis in notin]

        at = a.transpose(newaxes_a).reshape(newshape_a)
        bt = b.transpose(newaxes_b).reshape(newshape_b)
        res = np.dot(at, bt)
        return res.reshape(olda + oldb)

    @implements(np.einsum)
    def einsum(*operands, **kwargs):
        out = None
        if 'out' in kwargs:
            out = kwargs.pop('out')
            outdata, outmask = get_maskedout(out)

        data, nmask = zip(*((x._data, ~x._mask) for x in operands))

        result_data = np.einsum(data, out=outdata, **kwargs)
        result_mask = np.einsum(nmask, out=outmask, **kwargs)
        result_mask = np.logical_not(result_mask, out=result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    #@implements(np.einsum_path)

    @implements(np.correlate)
    def correlate(a, v, mode='valid'):
        result_data = np.correlate(a.filled(0), v.filled(0), mode)
        result_mask = ~np.correlate(~a._mask, v._mask, mode)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.convolve)
    def convolve(a, v, mode='full'):
        result_data = np.convolve(a.filled(0), v.filled(0), mode)
        result_mask = ~np.convolve(~a._mask, v._mask, mode)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.real)
    def real(a):
        # returns a view of the data only. mask is a copy.
        result_data = np.real(a._data)
        result_mask = a._mask.copy()
        return maskedarray_or_scalar(result_data, result_mask)
        # XXX not clear if mask should be copied or viewed: If we unmask
        # the imag part, should be real part be unmasked too?

    @implements(np.imag)
    def imag(a):
        # returns a view of the data only. mask is a copy.
        result_data = np.imag(a._data)
        result_mask = a._mask.copy()
        return maskedarray_or_scalar(result_data, result_mask)
        # XXX not clear if mask should be copied or viewed: If we unmask
        # the imag part, should be real part be unmasked too?

    @implements(np.partition)
    def partition(a, kth, axis=-1, kind='introselect', order=None):
        inds = np.argpartition(a, kth, axis, kind, order)
        return np.take_along_axis(a, inds, axis=axis)

    @implements(np.ptp)
    def ptp(a, axis=None, out=None, keepdims=False):
        # Original numpy function is fine.
        return np.ptp.__wrapped__(a, axis, out, keepdims)

    #XXX in the functions below, indices can be a plain ndarray
    @implements(np.take)
    def take(a, indices, axis=None, out=None, mode='raise'):
        outdata, outmask = get_maskedout(out)
        result_data = np.take(a._data, indices, axis, outdata, mode)
        result_mask = np.take(a._mask, indices, axis, outmask, mode)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.put)
    def put(a, indices, values, mode='raise'):
        data, mask = replace_X(values, dtype=a.dtype)
        np.put(a._data, indices, data, mode)
        np.put(a._mask, indices, mask, mode)
        return None

    @implements(np.take_along_axis, checked_args=(0,))
    def take_along_axis(arr, indices, axis):
        result_data = np.take_along_axis(arr._data, indices, axis)
        result_mask = np.take_along_axis(arr._mask, indices, axis)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.put_along_axis, checked_args=(0,))
    def put_along_axis(arr, indices, values, axis):
        if isinstance(values, (MaskedArray, MaskedScalar)):
            np.put_along_axis(arr._mask, indices, values._mask, axis)
            values = values._data
        np.put_along_axis(arr._data, indices, values, axis)

    #@implements(np.apply_along_axis)
    #def apply_along_axis(func1d, axis, arr, *args, **kwargs)

    #@implements(np.apply_over_axes)

    @implements(np.ravel)
    def ravel(a, order='C'):
        return MaskedArray(np.ravel(a._data, order=order),
                           np.ravel(a._mask, order=order))

    @implements(np.repeat)
    def repeat(a, repeats, axis=None):
        return MaskedArray(np.repeat(a._data, repeats, axis),
                           np.repeat(a._mask, repeats, axis))

    @implements(np.reshape)
    def reshape(a, shape, order='C'):
        # XXX set base
        return MaskedArray(np.reshape(a._data, shape, order=order),
                           np.reshape(a._mask, shape, order=order))


    @implements(np.resize)
    def resize(a, new_shape):
        return MaskedArray(np.resize(a._data, new_shape),
                           np.resize(a._mask, new_shape))

    @implements(np.meshgrid)
    def meshgrid(*xi, **kwargs):
        data, mask = zip(*((x._data, x._mask) for x in xi))
        result_data = np.meshgrid(data, **kwargs)
        result_mask = np.meshgrid(mask, **kwargs)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.around)
    def around(a, decimals=0, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.round(a._data, decimals, outdata)
        result_mask = _copy_mask(a._mask, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out)

    @implements(np.around)
    def round(a, decimals=0, out=None):
        return np.around(a, decimals, out)

    @implements(np.fix)
    def fix(x, out=None):
        outdata, outmask = get_maskedout(out)
        res = np.ceil(x, out=out)
        res = np.floor(x, out=res, where=np.greater_equal(x, 0))
        return out or res

    @implements(np.squeeze)
    def squeeze(a, axis=None):
        return MaskedArray(np.squeeze(a._data, axis),
                           np.squeeze(a._mask, axis))

    @implements(np.swapaxes)
    def swapaxes(a, axis1, axis2):
        return MaskedArray(np.swapaxes(a._data, axis1, axis2),
                           np.swapaxes(a._mask, axis1, axis2))

    @implements(np.transpose)
    def transpose(a, *axes):
        return MaskedArray(np.transpose(a._data, *axes),
                           np.transpose(a._mask, *axes))

    @implements(np.roll)
    def roll(a, shift, axis=None):
        return MaskedArray(np.roll(a._data, shift, axis),
                           np.roll(a._mask, shift, axis))

    @implements(np.rollaxis)
    def rollaxis(a, axis, start=0):
        return MaskedArray(np.rollaxis(a._data, axis, start),
                           np.rollaxis(a._mask, axis, start))

    @implements(np.moveaxis)
    def moveaxis(a, source, destination):
        return MaskedArray(np.moveaxis(a._data, source, destination),
                           np.moveaxis(a._mask, source, destination))

    @implements(np.flip)
    def flip(m, axis=None):
        return MaskedArray(np.flip(m._data, axis),
                           np.flip(m._mask, axis))

    #@implements(np.rot90)
    #def rot90(m, k=1, axes=(0,1)):
    #    # XXX copy code from np.rot90 but remove asarray

    #@implements(np.fliplr)
    #@implements(np.flipud)

    @implements(np.expand_dims)
    def expand_dims(a, axis):
        return MaskedArray(np.expand_dims(a._data, axis),
                           np.expand_dims(a._mask, axis))

    @implements(np.concatenate)
    def concatenate(arrays, axis=0, out=None):
        outdata, outmask = get_maskedout(out)
        arrays = [MaskedArray(a) for a in arrays] # XXX may need tweaking
        data, mask = zip(*((x._data, x._mask) for x in arrays))
        result_data = np.concatenate(data, axis, outdata)
        result_mask = np.concatenate(mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.block)
    def block(arrays):
        data, mask = replace_X(arrays)
        result_data = np.block(data)
        result_mask = np.block(mask)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.column_stack)
    def column_stack(tup):
        arrays = []
        for v in tup:
            arr = MaskedArray(v, copy=False, subok=True)
            if arr.ndim < 2:
                arr = MaskedArray(arr, copy=False, subok=True, ndmin=2).T
            arrays.append(arr)
        return np.concatenate(arrays, 1)

    @implements(np.dstack)
    def dstack(tup):
        return np.dstack.__wrapped__(tup)

    @implements(np.vstack)
    def vstack(tup):
        return np.vstack.__wrapped__(tup)

    @implements(np.hstack)
    def hstack(tup):
        return np.hstack.__wrapped__(tup)

    @implements(np.array_split, checked_args=(0,))
    def array_split(ary, indices_or_sections, axis=0):
        return np.array_split.__wrapped__(ary, indices_or_sections, axis)

    @implements(np.split, checked_args=(0,))
    def split(ary, indices_or_sections, axis=0):
        return np.split.__wrapped__(ary, indices_or_sections, axis)

    @implements(np.hsplit)
    def hsplit(ary, indices_or_sections):
        return np.hsplit.__wrapped__(ary, indices_or_sections)

    @implements(np.vsplit)
    def vsplit(ary, indices_or_sections):
        return np.vsplit.__wrapped__(ary, indices_or_sections)

    @implements(np.dsplit)
    def dsplit(ary, indices_or_sections):
        return np.dsplit.__wrapped__(ary, indices_or_sections)

    @implements(np.tile)
    def tile(A, reps):
        try:
            tup = tuple(reps)
        except TypeError:
            tup = (reps,)
        d = len(tup)

        if all(x == 1 for x in tup):
            return MaskedArray(A, copy=True, subok=True, ndmin=d)
        else:
            c = MaskedArray(A, copy=False, subok=True, ndmin=d)

        if (d < c.ndim):
            tup = (1,)*(c.ndim-d) + tup
        shape_out = tuple(s*t for s, t in zip(c.shape, tup))
        n = c.size
        if n > 0:
            for dim_in, nrep in zip(c.shape, tup):
                if nrep != 1:
                    c = c.reshape(-1, n).repeat(nrep, 0)
                n //= dim_in
        return c.reshape(shape_out)

    @implements(np.atleast_1d)
    def atleast_1d(*arys):
        res = []
        for ary in arys:
            #ary = asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1)
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res

    @implements(np.atleast_2d)
    def atleast_2d(*arys):
        res = []
        for ary in arys:
            #ary = asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1)
            elif ary.ndim == 1:
                result = ary[newaxis,:]
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res

    @implements(np.atleast_3d)
    def atleast_3d(*arys):
        res = []
        for ary in arys:
            #ary = asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1, 1)
            elif ary.ndim == 1:
                result = ary[newaxis,:, newaxis]
            elif ary.ndim == 2:
                result = ary[:,:, newaxis]
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res

    @implements(np.stack)
    def stack(arrays, axis=0, out=None):
        #arrays = [asanyarray(arr) for arr in arrays]
        if not arrays:
            raise ValueError('need at least one array to stack')

        shapes = set(arr.shape for arr in arrays)
        if len(shapes) != 1:
            raise ValueError('all input arrays must have the same shape')

        result_ndim = arrays[0].ndim + 1
        axis = normalize_axis_index(axis, result_ndim)

        sl = (slice(None),) * axis + (np.newaxis,)
        expanded_arrays = [arr[sl] for arr in arrays]
        return np.concatenate(expanded_arrays, axis=axis, out=out)

    @implements(np.delete)
    def delete(arr, obj, axis=None):
        return MaskedArray(np.delete(a._data, obj, axis),
                           np.delete(a._mask, obj, axis))

    @implements(np.insert)
    def insert(arr, obj, values, axis=None):
        return MaskedArray(np.insert(a._data, obj, values, axis),
                           np.insert(a._mask, obj, values, axis))

    @implements(np.append)
    def append(arr, values, axis=None):
        return MaskedArray(np.append(a._data, values, axis),
                           np.append(a._mask, values, axis))

    @implements(np.extract)
    def extract(condition, arr):
        return np.extract.__wrapped__(condition, arr)

    @implements(np.place)
    def place(arr, mask, vals):
        return np.insert(arr, mask, vals)

    #@implements(np.pad)
    #def pad(array, pad_width, mode, **kwargs):
    # XXX need identical code to np.pad but wihout call to asarray

    @implements(np.broadcast_to)
    def broadcast_to(array, shape):
        return MaskedArray(np.broadcast_to(a._data, shape),
                           np.broadcast_to(a._mask, shape))

    @implements(np.broadcast_arrays)
    def broadcast_arrays(*args, **kwargs):
        if kwargs:
            raise TypeError('broadcast_arrays() got an unexpected keyword '
                            'argument {!r}'.format(list(kwargs.keys())[0]))
        shape = _broadcast_shape(*args)

        if all(array.shape == shape for array in args):
            return args

        return [np.broadcast_to(array, shape, subok=subok, readonly=False)
                for array in args]

    @implements(np.empty_like)
    def empty_like(prototype, dtype=None, order='K', subok=True):
        return MaskedArray(np.empty_like(prototype._data, dtype, order, subok))

    @implements(np.ones_like)
    def ones_like(prototype, dtype=None, order='K', subok=True):
        return MaskedArray(np.ones_like(prototype._data, dtype, order, subok))

    @implements(np.zeros_like)
    def zeros_like(prototype, dtype=None, order='K', subok=True):
        return MaskedArray(np.zeros_like(prototype._data, dtype, order, subok))

    @implements(np.full_like)
    def full_like(a, fill_value, dtype=None, order='K', subok=True):
        return MaskedArray(np.full_like(a._data, fill_value, dtype, order,
                           subok))

    @implements(np.where)
    def where(condition, x=np._NoValue, y=np._NoValue):
        if x is np._NoValue and y is np._NoValue:
            return np.nonzero(condition)

        # condition should not be masked?
        data_args = tuple(a._data for a in (x, y) if a is not np._NoValue)
        result_data = np.where(condition, *data_args)

        mask_args = tuple(a._mask for a in (x, y) if a is not np._NoValue)
        result_mask = np.where(condition, *mask_args)

        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.argwhere)
    def argwhere(a):
        return np.transpose(np.nonzero(a))

    @implements(np.choose)
    def choose(a, choices, out=None, mode='raise'):
        outdata, outmask = get_maskedout(out)
        result_data = np.choose(a._data, choices, outdata, mode)
        result_mask = np.choose(a._mask, choices, outmask, mode)
        return maskedarray_or_scalar(result_data, result_mask, out)

    #@implements(np.piecewise)
    #def piecewise(x, condlist, funclist, *args, **kw):

    ##@implements(np.select)
    #def select(condlist, choicelist, default=0):
    #    # XXX same code as np.select but without asarray

    #@implements(np.unique)

    @implements(np.can_cast, checked_args=())
    def can_cast(from_, to, casting='safe'):
        if isinstance(from_, (MaskedArray, MaskedScalar)):
            from_ = from_._data
        if isinstance(to, (MaskedArray, MaskedScalar)):
            to = to._data
        return np.can_cast(from_, to, casting)

    @implements(np.min_scalar_type)
    def min_scalar_type(a):
        return a.dtype

    @implements(np.result_type, checked_args=())
    def result_type(*arrays_and_dtypes):
        dat = [a._data if isinstance(a, (MaskedArray, MaskedScalar)) else a
               for a in arrays_and_dtypes]
        return np.result_type(dat)

    @implements(np.common_type, checked_args=())
    def common_type(*arrays_and_dtypes):
        dat = [a._data if isinstance(a, (MaskedArray, MaskedScalar)) else a
               for a in arrays_and_dtypes]
        return np.common_type(dat)

    @implements(np.bincount)
    def bincount(x, weights=None, minlength=0):
        return np.bincount(x._data[~x._mask], weights, minlength)

    @implements(np.count_nonzero)
    def count_nonzero(a, axis=None):
        return np.count_nonzero(a.filled(0), axis)

    @implements(np.nonzero)
    def nonzero(a):
        return np.nonzero(a.filled(0))  # not MaskedArray since is for indexing

    @implements(np.flatnonzero)
    def flatnonzero(a):
        return np.nonzero(np.ravel(a))[0]

    @implements(np.histogram, checked_args=(0,))
    def histogram(a, bins=10, range=None, normed=None, weights=None,
                  density=None):
        a = a.ravel()
        keep = ~a._mask
        dat = a._data[keep]
        if weights is not None:
            weights = weights.ravel()[keep]

        return np.histogram(dat, bins, range, normed, weights, density)

    @implements(np.histogram2d, checked_args=(0,1))
    def histogram2d(x, y, bins=10, range=None, normed=None, weights=None,
                    density=None):
        return np.histogram2d.__wrapped__(x, y, bins, range, normed, weights,
                                          density)

    @implements(np.histogramdd)
    def histogramdd(sample, bins=10, range=None, normed=None, weights=None,
                    density=None):
        try:
            # Sample is an ND-array.
            N, D = sample.shape
        except (AttributeError, ValueError):
            # Sample is a sequence of 1D arrays.
            sample = np.atleast_2d(sample).T
            N, D = sample.shape

        keep = ~np.any(sample._mask, axis=0)
        sample = sample._data[...,keep]
        if weights is not None:
            weights = weights[keep]

        return histogramdd(sample, bins, range, normed, weights, density)

    @implements(np.histogram_bin_edges)
    def histogram_bin_edges(a, bins=10, range=None, weights=None):
        a = a.ravel()
        keep = ~a._mask
        dat = a._data[keep]
        if weights is not None:
            weights = weights.ravel()[keep]
        return np.histogram_bin_edges(dat, bins, range, weights)

    #@implements(np.diff)
    #@implements(np.interp)
    #@implements(np.ediff1d)
    #@implements(np.gradient)

    @implements(np.array2string)
    def array2string(a, max_line_width=None, precision=None,
            suppress_small=None, separator=' ', prefix='', style=np._NoValue,
            formatter=None, threshold=None, edgeitems=None, sign=None,
            floatmode=None, suffix='', **kwarg):
        return duck_array2string(a, max_line_width, precision, suppress_small,
            separator, prefix, style, formatter, threshold, edgeitems, sign,
            floatmode, suffix, **kwarg)

    @implements(np.array_repr)
    def array_repr(arr, max_line_width=None, precision=None,
                   suppress_small=None):
        return duck_repr(arr, max_line_width=None, precision=None,
                         suppress_small=None)

    @implements(np.array_str)
    def array_str(a, max_line_width=None, precision=None, suppress_small=None):
        return duck_str(a, max_line_width, precision, suppress_small)

    @implements(np.shape)
    def shape(a):
        return a.shape

    @implements(np.alen)
    def alen(a):
        return len(MaskedArray(a, ndmin=1))

    @implements(np.ndim)
    def ndim(a):
        return a.ndim

    @implements(np.size)
    def size(a):
        return a.size

    @implements(np.copyto, checked_args=(0))
    def copyto(dst, src, casting='same_kind', where=True):
        np.copyto(dst._data, src._data, casting, where)
        np.copyto(dst._mask, src._mask, casting, where)

    @implements(np.putmask)
    def putmask(a, mask, values):
        np.putmask(a._data, mask, values._data)
        np.putmask(a._mask, mask, values._mask)

    @implements(np.packbits)
    def packbits(myarray, axis=None):
        result_data = np.packbits(myarray._data, axis)
        result_mask = np.packbits(myarray._mask, axis) != 0
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.unpackbits)
    def unpackbits(myarray, axis=None):
        result_data = np.unpackbits(myarray._data, axis)
        result_mask = np.unpackbits(myarray._mask*np.uint8(255), axis)
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.isposinf)
    def isposinf(x, out=None):
        return MaskedArray(np.isposinf(x._data), x._mask.copy())

    @implements(np.isneginf)
    def isneginf(x, out=None):
        return MaskedArray(np.isneginf(x._data), x._mask.copy())

    @implements(np.iscomplex)
    def iscomplex(x):
        return MaskedArray(np.iscomplex(x._data), x._mask.copy())

    @implements(np.isreal)
    def isreal(x):
        return MaskedArray(np.isreal(x._data), x._mask.copy())

    @implements(np.iscomplexobj)
    def iscomplexobj(x):
        return MaskedArray(np.iscomplexobj(x._data), x._mask.copy())

    @implements(np.isrealobj)
    def isrealobj(x):
        return MaskedArray(np.isrealobj(x._data), x._mask.copy())

    @implements(np.nan_to_num)
    def nan_to_num(x, copy=True):
        return MaskedArray(np.nan_to_num(x._data, copy),
                           x._mask.copy() if copy else x._mask)

    @implements(np.real_if_close)
    def real_if_close(a, tol=100):
        return MaskedArray(np.real_if_close(x._data, tol), x._mask.copy())

    @implements(np.isclose)
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        result_data = np.isclose(a._data, b._data, rtol, atol, equal_nan)
        result_mask = a._mask | b._mask
        return maskedarray_or_scalar(result_data, result_mask)

    @implements(np.allclose)
    def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return np.all(np.isclose(a, b, rtol, aton, equal_nan))
        # Note: unlike original func, this doesn't return a bool

    @implements(np.array_equal)
    def array_equal(a1, a2):
        return np.all(a1 == a2)
        # Note: unlike original func, this doesn't return a bool

    @implements(np.array_equiv)
    def array_equal(a1, a2):
        try:
            np.broadcast(a1, a2)
        except:
            return MaskedScalar(np.bool_(False), False)
        return np.all(a1 == a2)
        # Note: unlike original func, this doesn't return a bool

    @implements(np.sometrue)
    def sometrue(*args, **kwargs):
        return np.any(*args, **kwargs)

    @implements(np.alltrue)
    def alltrue(*args, **kwargs):
        return np.all(*args, **kwargs)

    @implements(np.angle)
    def angle(z, deg=False):
        if issubclass(z.dtype.type, np.complexfloating):
            zimag = z.imag
            zreal = z.real
        else:
            zimag = 0
            zreal = z

        a = arctan2(zimag, zreal)
        if deg:
            a *= 180/pi
        return a

    @implements(np.sinc)
    def sinc(x):
        y = np.pi * np.where((x == 0).filled(False), 1.0e-20, x)
        return np.sin(y)/y

    #@implements(np.unwrap)



    # Deprecated, don't implement
    #@implements(np.rank)
    #@implements(np.asscalar)

    # these won't be implemented since they apply to index arrays, which should
    # not be masked.
    #@implements(np.ravel_multi_index)
    #@implements(np.unravel_index)

    # unclear how to implement
    #@implements(np.shares_memory)
    #@implements(np.may_share_memory)

    # XXX not yet implemented:

    #@implements(np.is_busday)
    #@implements(np.busday_offset)
    #@implements(np.busday_count)
    #@implements(np.datetime_as_string)

    #@implements(np.asfarray)
    #@implements(np.vander)
    #@implements(np.tril_indices_from)
    #@implements(np.triu_indices_from)

    #@implements(np.sort_complex)
    #@implements(np.trim_zeros)
    #@implements(np.i0)
    #@implements(np.msort)
    #@implements(np.trapz)

    #@implements(np.ix_)
    #@implements(np.fill_diagonal)
    #@implements(np.diag_indices_from)

    #@implements(np.lib.scimath.sqrt)
    #@implements(np.lib.scimath.log)
    #@implements(np.lib.scimath.log10)
    #@implements(np.lib.scimath.logn)
    #@implements(np.lib.scimath.log2)
    #@implements(np.lib.scimath.power)
    #@implements(np.lib.scimath.arccos)
    #@implements(np.lib.scimath.arcsin)
    #@implements(np.lib.scimath.arctanh)

    #@implements(np.poly)
    #@implements(np.roots)
    #@implements(np.polyint)
    #@implements(np.polyder)
    #@implements(np.polyfit)
    #@implements(np.polyval)
    #@implements(np.polyadd)
    #@implements(np.polysub)
    #@implements(np.polymul)
    #@implements(np.polydiv)
    #@implements(np.intersect1d)
    #@implements(np.setxor1d)
    #@implements(np.in1d)
    #@implements(np.isin)
    #@implements(np.union1d)
    #@implements(np.setdiff1d)
    #@implements(np.fv)
    #@implements(np.pmt)
    #@implements(np.nper)
    #@implements(np.ipmt)
    #@implements(np.ppmt)
    #@implements(np.pv)
    #@implements(np.rate)
    #@implements(np.irr)
    #@implements(np.npv)
    #@implements(np.mirr)

    #@implements(np.save)
    #@implements(np.savez)
    #@implements(np.savez_compressed)
    #@implements(np.savetxt)


setup_ducktype()


# temporary code to figure out our api coverage
api = ['np.empty_like', 'np.concatenate', 'np.inner', 'np.where', 'np.lexsort',
'np.can_cast', 'np.min_scalar_type', 'np.result_type', 'np.dot', 'np.vdot',
'np.bincount', 'np.ravel_multi_index', 'np.unravel_index', 'np.copyto',
'np.putmask', 'np.packbits', 'np.unpackbits', 'np.shares_memory',
'np.may_share_memory', 'np.is_busday', 'np.busday_offset', 'np.busday_count',
'np.datetime_as_string', 'np.zeros_like', 'np.ones_like', 'np.full_like',
'np.count_nonzero', 'np.argwhere', 'np.flatnonzero', 'np.correlate',
'np.convolve', 'np.outer', 'np.tensordot', 'np.roll', 'np.rollaxis',
'np.moveaxis', 'np.cross', 'np.allclose', 'np.isclose', 'np.array_equal',
'np.array_equiv', 'np.take', 'np.reshape', 'np.choose', 'np.repeat', 'np.put',
'np.swapaxes', 'np.transpose', 'np.partition', 'np.argpartition', 'np.sort',
'np.argsort', 'np.argmax', 'np.argmin', 'np.searchsorted', 'np.resize',
'np.squeeze', 'np.diagonal', 'np.trace', 'np.ravel', 'np.nonzero', 'np.shape',
'np.compress', 'np.clip', 'np.sum', 'np.any', 'np.all', 'np.cumsum', 'np.ptp',
'np.amax', 'np.amin', 'np.alen', 'np.prod', 'np.cumprod', 'np.ndim', 'np.size',
'np.around', 'np.mean', 'np.std', 'np.var', 'np.round_', 'np.product',
'np.cumproduct', 'np.sometrue', 'np.alltrue', 'np.rank', 'np.array2string',
'np.array_repr', 'np.array_str', 'np.char.equal', 'np.char.not_equal',
'np.char.greater_equal', 'np.char.less_equal', 'np.char.greater',
'np.char.less', 'np.char.str_len', 'np.char.add', 'np.char.multiply',
'np.char.mod', 'np.char.capitalize', 'np.char.center', 'np.char.count',
'np.char.decode', 'np.char.encode', 'np.char.endswith', 'np.char.expandtabs',
'np.char.find', 'np.char.index', 'np.char.isalnum', 'np.char.isalpha',
'np.char.isdigit', 'np.char.islower', 'np.char.isspace', 'np.char.istitle',
'np.char.isupper', 'np.char.join', 'np.char.ljust', 'np.char.lower',
'np.char.lstrip', 'np.char.partition', 'np.char.replace', 'np.char.rfind',
'np.char.rindex', 'np.char.rjust', 'np.char.rpartition', 'np.char.rsplit',
'np.char.rstrip', 'np.char.split', 'np.char.splitlines', 'np.char.startswith',
'np.char.strip', 'np.char.swapcase', 'np.char.title', 'np.char.translate',
'np.char.upper', 'np.char.zfill', 'np.char.isnumeric', 'np.char.isdecimal',
'np.atleast_1d', 'np.atleast_2d', 'np.atleast_3d', 'np.vstack', 'np.hstack',
'np.stack', 'np.block', 'np.einsum_path', 'np.einsum', 'np.fix', 'np.isposinf',
'np.isneginf', 'np.asfarray', 'np.real', 'np.imag', 'np.iscomplex',
'np.isreal', 'np.iscomplexobj', 'np.isrealobj', 'np.nan_to_num',
'np.real_if_close', 'np.asscalar', 'np.common_type', 'np.fliplr', 'np.flipud',
'np.diag', 'np.diagflat', 'np.tril', 'np.triu', 'np.vander', 'np.histogram2d',
'np.tril_indices_from', 'np.triu_indices_from', 'np.linalg.tensorsolve',
'np.linalg.solve', 'np.linalg.tensorinv', 'np.linalg.inv',
'np.linalg.matrix_power', 'np.linalg.cholesky', 'np.linalg.qr',
'np.linalg.eigvals', 'np.linalg.eigvalsh', 'np.linalg.eig', 'np.linalg.eigh',
'np.linalg.svd', 'np.linalg.cond', 'np.linalg.matrix_rank', 'np.linalg.pinv',
'np.linalg.slogdet', 'np.linalg.det', 'np.linalg.lstsq', 'np.linalg.norm',
'np.linalg.multi_dot', 'np.histogram_bin_edges', 'np.histogram',
'np.histogramdd', 'np.rot90', 'np.flip', 'np.average', 'np.piecewise',
'np.select', 'np.copy', 'np.gradient', 'np.diff', 'np.interp', 'np.angle',
'np.unwrap', 'np.sort_complex', 'np.trim_zeros', 'np.extract', 'np.place',
'np.cov', 'np.corrcoef', 'np.i0', 'np.sinc', 'np.msort', 'np.median',
'np.percentile', 'np.quantile', 'np.trapz', 'np.meshgrid', 'np.delete',
'np.insert', 'np.append', 'np.digitize', 'np.broadcast_to',
'np.broadcast_arrays', 'np.ix_', 'np.fill_diagonal', 'np.diag_indices_from',
'np.nanmin', 'np.nanmax', 'np.nanargmin', 'np.nanargmax', 'np.nansum',
'np.nanprod', 'np.nancumsum', 'np.nancumprod', 'np.nanmean', 'np.nanmedian',
'np.nanpercentile', 'np.nanquantile', 'np.nanvar', 'np.nanstd',
'np.take_along_axis', 'np.put_along_axis', 'np.apply_along_axis',
'np.apply_over_axes', 'np.expand_dims', 'np.column_stack', 'np.dstack',
'np.array_split', 'np.split', 'np.hsplit', 'np.vsplit', 'np.dsplit', 'np.kron',
'np.tile', 'np.lib.scimath.sqrt', 'np.lib.scimath.log', 'np.lib.scimath.log10',
'np.lib.scimath.logn', 'np.lib.scimath.log2', 'np.lib.scimath.power',
'np.lib.scimath.arccos', 'np.lib.scimath.arcsin', 'np.lib.scimath.arctanh',
'np.poly', 'np.roots', 'np.polyint', 'np.polyder', 'np.polyfit', 'np.polyval',
'np.polyadd', 'np.polysub', 'np.polymul', 'np.polydiv', 'np.ediff1d',
'np.unique', 'np.intersect1d', 'np.setxor1d', 'np.in1d', 'np.isin',
'np.union1d', 'np.setdiff1d', 'np.save', 'np.savez', 'np.savez_compressed',
'np.savetxt', 'np.fv', 'np.pmt', 'np.nper', 'np.ipmt', 'np.ppmt', 'np.pv',
'np.rate', 'np.irr', 'np.npv', 'np.mirr', 'np.pad', 'np.fft.fftshift',
'np.fft.ifftshift', 'np.fft.fft', 'np.fft.ifft', 'np.fft.rfft', 'np.fft.irfft',
'np.fft.hfft', 'np.fft.ihfft', 'np.fft.fftn', 'np.fft.ifftn', 'np.fft.fft2',
'np.fft.ifft2', 'np.fft.rfftn', 'np.fft.rfft2', 'np.fft.irfftn',
'np.fft.irfft2']

n_implemented, n_skipped, n_missing = 0, 0, 0
for a in api:
    if a.startswith('np.char.'):
        n_skipped += 1
        continue
    if a.startswith('np.fft.'):
        n_skipped += 1
        continue

    parts = a.split('.')[1:]
    f = np
    while parts and f:
        f = getattr(f, parts.pop(0), None)
    if f is None:
        print("Missing", a)
        continue
    if f not in HANDLED_FUNCTIONS:
        n_missing += 1
        #print(a)
        pass
    else:
        n_implemented += 1
    #    print("Have", a)
print("Total api:   ", len(api))
print("Skipped:     ", n_skipped)
print("Implemented: ", n_implemented)
print("Missing:     ", n_missing)


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
    try:
        print(repr(MaskedArray([[X, X, X], [X, X, X]])))
    except ValueError as v:
        print("Got Exception: ", v)
    print(repr(MaskedArray([[X, X, X], [X, X, X]], dtype='u1')))

    a = MaskedArray([[1,X,3], [X,-1,X], [1,X,-1]], dtype='u4')
    b = np.take_along_axis(a, np.argsort(a, axis=1), axis=1)
    print(repr(b))
    c = a.copy()
    c.sort(axis=1)
    print(repr(c))

    a = MaskedArray([[1,X,3], [X,4,X], [1,X,6]], dtype='u4')
    print(np.lexsort((a,), axis=1))
    print(np.argsort(a, axis=1))
    print(repr(np.block([[a,a],[a,a]])))
    print(repr(a == a))
    print(repr(a == X))

    m = a.mask
    try:
        m[0,0] = 1
    except ValueError:
        pass
    a[0,0] = X
    print(m[0,0] == True)

