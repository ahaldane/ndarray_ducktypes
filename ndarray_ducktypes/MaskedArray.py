#!/usr/bin/env python
import builtins
import operator
import warnings

from .duckprint import (duck_str, duck_repr, duck_array2string, typelessdata,
    default_duckprint_options, default_duckprint_formatters, FormatDispatcher)
from .common import (is_ndducktype, is_ndscalar, is_ndarr, is_ndtype,
    new_ducktype_implementation, ducktype_link, get_duck_cls, as_duck_cls)
from .ndarray_api_mixin import NDArrayAPIMixin

import numpy as np
from numpy import newaxis
import numpy.core.umath as umath
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.lib.function_base import _quantile_is_valid
import numpy.core.numerictypes as ntypes
from numpy.core.multiarray import (normalize_axis_index,
    interp as compiled_interp, interp_complex as compiled_interp_complex)
from numpy.lib.stride_tricks import _broadcast_shape
from numpy.core.numeric import normalize_axis_tuple

class MaskedOperatorMixin(NDArrayOperatorsMixin):
    # shared implementations for MaskedArray, MaskedScalar

    # override the NDArrayOperatorsMixin implementations for cmp ops, as
    # currently those don't work for flexible types.
    def _cmp_op(self, other, op):
        if other is X:
            db, mb = self._data.dtype.type(0), np.bool_(True)
        else:
            db, mb = getdata(other), getmask(other)

        cls = get_duck_cls(self, other)

        data = op(self._data, db)
        mask = self._mask | mb
        return maskedarray_or_scalar(data, mask, cls=cls)

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

    def __complex__(self):
        raise TypeError("Use .filled() before converting to non-masked scalar")

    def __int__(self):
        raise TypeError("Use .filled() before converting to non-masked scalar")

    def __float__(self):
        raise TypeError("Use .filled() before converting to non-masked scalar")

    def __index__(self):
        raise TypeError("Use .filled() before converting to non-masked scalar")

    def __array_function__(self, func, types, arg, kwarg):
        impl, check_args = implements.handled_functions.get(func, (None, None))
        if impl is None or not check_args(arg, kwarg, types, self.known_types):
            return NotImplemented

        return impl(*arg, **kwarg)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in _masked_ufuncs:
            return NotImplemented

        return getattr(_masked_ufuncs[ufunc], method)(*inputs, **kwargs)

    def _get_fill_value(self, fill_value, minmax):
        if minmax is not None:
            if fill_value != np._NoValue:
                raise Exception("Do not give fill_value if providing minmax")
            if minmax == 'max':
                fill_value = _maxvals[self.dtype]
            elif minmax == 'maxnan':
                if issubclass(self.dtype.type, np.inexact):
                    # some functions, eg np.sort, treat nan as largest
                    fill_value = np.nan
                else:
                    fill_value = _maxvals[self.dtype]
            elif minmax == 'min':
                fill_value = _minvals[self.dtype]
            else:
                raise ValueError("minmax should be 'min' or 'max'")

            if fill_value is None:
                raise ValueError("minmax not supported for dtype {}".format(
                                  self.dtype))
        elif fill_value is np._NoValue:
            # default is 0 for all types (*not* np.nan for inexact)
            fill_value = 0

        return fill_value

    @property
    def flat(self):
        return MaskedIterator(self)

def duck_require(data, dtype=None, ndmin=0, copy=True, order='K'):
    """
    Return an ndarray-like that satisfies requirements.

    Returns a view if possible.

    Parameters
    ----------
    data : array-like
        Must be an ndarray or ndarray ducktype.
    dtype : numpy datatype
        Datatype to convert to
    ndmin : integer
        Same as 'ndmin' argument of np.array
    copy : bool
        Whether to guarantee a copy is made
    order : one of 'K', 'F', 'C', 'A'
        Same as 'order' argument of np.array
    """

    # we must use only properties that work for ndarray ducktypes.
    # This rules out using np.require

    newdtype = dtype if dtype is not None else data.dtype
    if copy or (newdtype != data.dtype):
        data = data.astype(newdtype, order=order)

    if order != 'K' and order is not None:
        warnings.warn('order parameter of MaskedArray is ignored')

    if ndmin != 0 and data.ndim < ndmin:
        nd = ndmin - data.ndim
        data = data[(None,)*nd + (Ellipsis,)]

    return data

def asarr(v, **kwarg):
    if is_ndarr(v):
        return duck_require(v, **kwarg)
    else: # must be ndscalar
        if is_ndducktype(v):
            # convert to duck-array class using our ducktype conventions
            return get_duck_cls(v)(v, **kwarg)
        else: # usually, np.generic type
            return np.array(v, **kwarg)

class MaskedArray(MaskedOperatorMixin, NDArrayAPIMixin):
    "An ndarray ducktype allowing array elements to be masked"

    def __init__(self, data, mask=None, dtype=None, copy=False,
                order=None, subok=False, ndmin=0):
        """
        Constructs a MaskedArray given data and optional mask.

        Parameters
        ----------
        data : array-like
            Any object following the numpy ducktype api or convertible to an
            ndarray, but also allowing the masked signifier `X` to mark masked
            elements.  See Notes below.
        mask : array-like
            Any object convertible to a boolean `ndarray` of the same
            shape as data, where true elements are masked. If omitted, defaults
            to all `False`. See Notes below.
        dtype : data-type, optional
            The desired data-type for the array. See `np.array` argument.
        copy : bool, optional
            If false (default), the MaskedArray will view the data and mask
            if they are ndarrays with the right properties. Otherwise
            a they will be copied.
        order : {'K', 'A', 'C', 'F'}, optional
            Memory layout of the array. See `np.array` argument. This affects
            both the data and mask.
        ndmin : int, optional
            Specifies the minimum number of dimensions the resulting array
            should have. See `np.array` argument.

        Returns
        -------
        out : MaskedArray
            The resulting MaskedArray.

        Notes
        -----
        This MaskedArray constructor supports a few different ways to mark
        masked elements, which are sometimes exclusive.

        First, `data` may be a MaskedArray, in which case `mask` should not
        be supplied.

        If `mask` is not supplied, then masked elements may be marked in the
        `data` using the masked input element `X`. That is, `data` can be a
        list-of-lists containing numerical scalars and `ndarray`s,
        similar to that accepted by `np.array`, but additionally allowing
        some elements to be replaced with `X`. The dtype will be inferred
        based on the converted dtype of the non-masked elements. If all
        elements are `X`, the `dtype` argument of `MaskedArray` must be
        supplied:

            >>> a = MaskedArray([[1, X, 3], np.arange(3)])
            >>> b = MaskedArray([X, X, X], dtype='f8')

        If `mask` is supplied, `X` should not be used in the `data. `mask`
        should be any object convertible to bool datatype and broadcastable
        to the shape of the data. If `mask` is already a bool ndarray
        of the same shape as `data`, it will be viewed, otherwise it will
        be copied.

        """

        if isinstance(data, MaskedScalar):
            self.__init__(data._data, data._mask, dtype=data.dtype,
                          order=order, ndmin=ndmin)
            return
        elif isinstance(data, MaskedArray):
            self._mask = duck_require(data._mask, copy=copy, order=order,
                                      ndmin=ndmin)

            if mask is not None:
                self._data = duck_require(data._data, copy=True, order=order,
                                          ndmin=ndmin)
                mask = np.array(mask, dtype=bool, copy=False)
                self._mask |= np.broadcast_to(mask, self._data.shape)

            else:
                self._data = duck_require(data._data, copy=copy, order=order,
                                          ndmin=ndmin)
            return
        elif data is X and mask is None:
            # 0d masked array
            if dtype is None:
                raise ValueError("must supply dtype if all elements are X")
            self._data = np.array(dtype.type(0))
            self._mask = np.array(True)
            return

        # Otherwise got non-masked type, we convert data/mask to MaskedArray:

        if mask is None:
            # if mask is None, user can put X in the data.
            # Otherwise, X will cause some kind of error in np.array below
            data, mask, _ = replace_X(data, dtype=dtype)

            # replace_X sometimes uses broadcast_to, which returns a
            # readonly array with funny strides. Make writeable if so,
            # since we will end up in the is_ndducktype code-path below.
            if (isinstance(mask, np.ndarray) and
                    mask.flags['WRITEABLE'] == False):
                mask = mask.copy()

        self._data = asarr(data, dtype=dtype, copy=copy,order=order,ndmin=ndmin)

        if mask is None:
            self._mask = np.zeros(self._data.shape, dtype='bool', order=order)
        elif is_ndtype(mask):
            self._mask = asarr(mask, dtype=np.bool_, copy=copy, order=order)
            if self._mask.shape != self._data.shape:
                self._mask = np.broadcast_to(self._mask,self._data.shape).copy()
        else:
            self._mask = np.empty(self._data.shape, dtype='bool')
            self._mask[...] = np.broadcast_to(mask, self._data.shape)

    @classmethod
    def __nd_duckprint_dispatcher__(cls):
        return masked_dispatcher

    def __str__(self):
        return duck_str(self)

    def __repr__(self):
        return duck_repr(self, showdtype=self._mask.all())

    def __getitem__(self, ind):
        if is_string_or_list_of_strings(ind):
            # for viewing fields of structured arrays, return readonly view.
            # (see .real/.imag discussion in user guide)
            ret = self._data[ind]
            ret.flags['WRITEABLE'] = False
            return type(self)(ret, self._mask)

        if not isinstance(ind, tuple):
            ind = (ind,)

        # If a boolean MaskedArray is provided as an ind, treat masked vals as
        # False. Allows code like "a[a>0]", which is then the same as
        # "a[np.nonzero(a>0)]"
        ind = tuple(i.filled(False, view=1) if
                (isinstance(i, MaskedArray) and i.dtype.type is np.bool_)
                else i for i in ind)

        # TODO: Possible future improvement would be to support masked
        # integer arrays as indices. Then marr[boolmask] should behave
        # the same as marr[where(boolmask)], i.e. masked indices are
        # ignored.

        data = self._data[ind]
        mask = self._mask[ind]

        if is_ndscalar(mask): # test mask not data, to account for obj arrays
            return type(self)._scalartype(data, mask, dtype=self.dtype)
        return type(self)(data, mask, dtype=self.dtype)

    def __setitem__(self, ind, val):
        if not self.flags.writeable:
            raise ValueError("assignment destination is read-only")

        if self.dtype.names and is_string_or_list_of_strings(ind):
            raise ValueError("Cannot assign to fields of a Masked structured "
                             "array")

        if not isinstance(ind, tuple):
            ind = (ind,)

        # If a boolean MaskedArray is provided as an ind, treat masked vals as
        # False. Allows code like "a[a>0] = X"
        ind = tuple(i.filled(False, view=1) if
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

    def __len__(self):
        return len(self._data)

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
        dt = np.dtype(dt)

        if self._data.dtype.itemsize != dt.itemsize:
            raise ValueError("views of MaskedArrays cannot change the "
                             "datatype's itemsize")
        self._data.dtype = dt

    @property
    def flags(self):
        return self._data.flags

    @property
    def strides(self):
        return self._data.strides

    @property
    def mask(self):
        # return a readonly view of mask
        m = self._mask.view()
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

        return type(self)(self._data.view(dtype), self._mask)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        result_data = self._data.astype(dtype, order, casting, subok, copy)
        # force a copy of mask if data was copied
        if copy == False and result_data is not self:
            copy = True
        result_mask = self._mask.astype(bool, order, casting, subok, copy)
        return type(self)(result_data, result_mask)

    def tolist(self):
        return [x.tolist() for x in self]

    def filled(self, fill_value=np._NoValue, minmax=None, view=False):
        """
        Parameters
        ==========
        fill_value : scalar, optional
            value to put in masked positions of the array. Defaults to 0
            if minmax is not provided.
        minmax : string 'min', 'max' or 'maxnan', optional
            If 'min', fill masked elements with the minimum value for this
            array's datatype. If 'max', fill with maximum value for this
            datatype. If 'maxnan', fill with nan if a floating type, otherwise
            same as 'max'.
        view : boolean, optional
            If True, then the returned array is a view of the underlying data
            array rather than a copy (optimization). Be careful, as subsequent
            actions on the maskedarray can put nonsense data in the view.
            If the array is writeonly, this option is ignored and a copy is
            always returned.

        Returns
        =======
        data : ndarray
            Returns a copy of this MaskedArray with masked elements replaced
            by the fill value. (or a view of view=True).
        """
        if view and self._data.flags['WRITEABLE']:
            d = self._data.view()
            d[self._mask] = self._get_fill_value(fill_value, minmax)
            d.flags['WRITEABLE'] = False
            return d

        d = self._data.copy(order='K')
        d[self._mask] = self._get_fill_value(fill_value, minmax)
        return d

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
        self._data[self._mask] = _maxvals[self.dtype]
        self._data.sort(axis, kind, order)
        self._mask.sort(axis, kind)

    # This works inplace, unlike np.resize, and fills with repeat instead of 0
    def resize(self, new_shape, refcheck=True):
        self._data.resize(new_shape, refcheck)
        self._mask.resize(new_shape, refcheck)


class MaskedScalar(MaskedOperatorMixin, NDArrayAPIMixin):
    "An ndarray scalar ducktype allowing the value to be masked"

    def __init__(self, data, mask=None, dtype=None):
        """
        Construct  masked scalar given a data value and mask value.

        Parameters
        ----------
        data : numpy scalar, MaskedScalar, or X
            The value of the scalar. If `X` is given, `dtype` must be supplied.
        mask : bool
            If true, the scalar is masked. Default is false.
        dtype : numpy dtype
            dtype to convert to the data to

        Notes
        -----
        To construct a masked MaskedScalar of a certain dtype, it may be
        preferrable to use ``X(dtype)``.

        If `data` is a MaskedScalar, do not supply a `mask`.

        """
        if isinstance(data, MaskedScalar):
            self._data = data._data
            self._mask = data._mask
            if mask is not None:
                raise ValueError("don't use mask if passing a maskedscalar")
            self._dtype = self._data.dtype
            return
        elif data is X:
            if dtype is None:
                raise ValueError("Must supply dtype when data is X")
            if mask is not None:
                raise ValueError("don't supply mask when data is X")
            self._data = np.dtype(dtype).type(0)
            self._mask = np.bool_(True)
            self._dtype = self._data.dtype
            return

        # Otherwise, convert data/mask to MaskedScalar:

        if dtype is not None:
            dtype = np.dtype(dtype)

        if dtype is None or dtype.type is not np.object_:
            if is_ndtype(data):
                if dtype is not None and data.dtype != dtype:
                    data = data.astype(dtype, copy=False)[()]
                if not is_ndscalar(data):
                    data = data[()]
                self._data = data
            else:
                # next line is more complicated than desired due to struct
                # types, which numpy does not have a constructor for
                # convert to scalar
                self._data = np.array(data, dtype=dtype)[()]

            self._mask = np.bool_(mask)
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

    def __getitem__(self, ind):
        if (self.dtype.names and is_string_or_list_of_strings(ind) or
                isinstance(ind, int)):
            # like structured scalars, support string indexing and int indexing
            data = self._data[ind]
            mask = self._mask
            return type(self)(data, mask)

        if ind == ():
            return self

        if ind == Ellipsis or ind == (Ellipsis,):
            return MaskedArray(self)

        raise IndexError("invalid index to scalar variable.")

    def __setitem__(self, ind, val):
        # non-masked structured scalars normally allow assignment (eg, to
        # individual fields), but here we disallow *all* assignment, because of
        # ambiguity about what to do with mask. See discussion of .real/.imag
        raise ValueError("assignment destination is read-only")

    def __str__(self):
        if self._mask:
            return MASK_STR
        return str(self._data)

    def __repr__(self):
        if self._mask:
            return "X({})".format(str(self.dtype))

        if self.dtype.type in typelessdata and self.dtype.names is None:
            dtstr = ''
        else:
            dtstr = ', dtype={}'.format(str(self.dtype))

        return "MaskedScalar({}{})".format(repr(self._data), dtstr)

    def __format__(self, format_spec):
        if self._mask:
            return 'X'
        return format(self._data, format_spec)

    def __bool__(self):
        if self._mask:
            return False
        return bool(self._data)

    def __hash__(self):
        if self._mask:
            return 0
        return hash(self._data)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        result_data = self._data.astype(dtype, order, casting, subok, copy)
        return MaskedScalar(result_data, self._mask)

    def tolist(self):
        if self._mask:
            return self
        return self._data.item()

    @property
    def mask(self):
        return self._mask

    def filled(self, fill_value=np._NoValue, minmax=None, view=False):
        # view is ignored
        fill_value = self._get_fill_value(fill_value, minmax)

        if self._mask:
            if self.dtype.names:
                # next line is more complicated than desired due to struct
                # types, which numpy does not have a constructor for
                return np.array(fill_value, dtype=self.dtype)[()]
            return type(self._data)(fill_value)
        return self._data

    def count(self, axis=None, keepdims=False):
        return 0 if self._mask else 1

# create a special dummy object which signifies "masked", which users can put
# in lists to pass to MaskedArray constructor, or can assign to elements of
# a MaskedArray, to set the mask.
class MaskedX:
    def __repr__(self):
        return 'masked_input_X'
    def __str__(self):
        return 'masked_input_X'

    # as a convenience, can make this typed by calling with a dtype
    def __call__(self, dtype):
        return MaskedScalar(0, True, dtype=dtype)

    # prevent X from being used as an element in np.array, to avoid
    # confusing the user. X should only be used in MaskedArrays
    def __array__(self):
        # hack: the only Exception that numpy doesn't clear here is MemoryError
        raise MemoryError("Masked X should only be used in "
                          "MaskedArray assignment or construction")

masked = X = MaskedX()

ducktype_link(MaskedArray, MaskedScalar, (MaskedX,))

def replace_X(data, dtype=None):
    """
    takes array-like input, replaces masked value by 0 and return filled data &
    mask. This is more-or-less a reimplementation of PyArray_DTypeFromObject to
    account for masked values

    Parameters
    ==========
    data : nested tuple.list of ndarrays/MaskedArrays/X
    dtype : dtype to force for output

    Returns
    =======
    data : ndarray (or duck)
        The data array of the combined inputs
    mask : ndarray (or duck)
        The mask array of the combined inputs
    cls : type
        The most derived MaskedArray subtype seen in the inputs
    """

    if isinstance(data, (list, tuple)) and len(data) == 0:
        return data, [], MaskedArray

    # we do two passes: First we figure out the output dtype, then we replace
    # all masked values by the filler "type(0)".

    def get_dtype(data, cur_dtype=X):
        if isinstance(data, (list, tuple)):
            dtypes = (get_dtype(d, cur_dtype) for d in data)
            dtypes = [dt for dt in dtypes if dt is not X]
            if not dtypes:
                return cur_dtype

            out_dtype = np.result_type(*dtypes)
            if cur_dtype is X:
                return out_dtype
            else:
                return np.promote_types(out_dtype, cur_dtype)

        if data is X:
            return X

        if is_ndtype(data):
            return data.dtype

        # otherwise try to coerce it to an ndarray (accounts for __array__,
        # __array_interface__ implementors)
        return np.array(data).dtype

    if dtype is None:
        dtype = get_dtype(data)
        if dtype is X:
            raise ValueError("must supply dtype if all elements are X")
    else:
        dtype = np.dtype(dtype)

    fill = dtype.type(0)
    cls = MaskedArray

    def replace(data):
        nonlocal cls
        if data is X:
            return fill, True
        if isinstance(data, (MaskedScalar, MaskedArray)):
            # whenever we come across a Masked* subtype, update cls
            cls = get_duck_cls(cls, data)
            return data._data, data._mask
        if isinstance(data, list):
            return (list(x) for x in zip(*(replace(d) for d in data)))
        if is_ndtype(data):
            return data, np.broadcast_to(False, data.shape)
        # otherwise assume it is some kind of scalar
        return data, False

    out_dat, out_mask = replace(data)
    return out_dat, out_mask, cls

# used by marr.flat
class MaskedIterator:
    def __init__(self, ma):
        self.dataiter = ma._data.flat
        self.maskiter = ma._mask.flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        data = self.dataiter.__getitem__(indx)
        mask = self.maskiter.__getitem__(indx)
        return maskedarray_or_scalar(data, mask, cls=type(self))

    def __setitem__(self, index, value):
        if value is X or (isinstance(value, MaskedScalar) and value.mask):
            self.maskiter[index] = True
        else:
            self.dataiter[index] = getdata(value)
            self.maskiter[index] = getmask(value)

    def __next__(self):
        return maskedarray_or_scalar(next(self.dataiter), next(self.maskiter),
                                     cls=type(self))

    next = __next__

_minvals = ntypes._minvals
_minvals.update([(k, -np.inf) for k in [np.float16, np.float32, np.float64]])
_maxvals = ntypes._maxvals
_maxvals.update([(k, +np.inf) for k in [np.float16, np.float32, np.float64]])
if 'float128' in ntypes.typeDict:
    _minvals.update([(np.float128, -np.inf)])
    _maxvals.update([(np.float128, +np.inf)])

def is_string_or_list_of_strings(val):
    if isinstance(val, str):
        return True
    if not isinstance(val, list):
        return False
    for v in val:
        if not isinstance(v, str):
            return False
    return True

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

            masked_str = options['masked_str']

            # only get fmt_func based on non-masked values
            # (we take care of masked elements ourselves)
            unmasked = elem._data[~elem._mask]
            if unmasked.size == 0:
                default_fmt = lambda x: ''
                reslen = len(masked_str)
            else:
                default_fmt = super().get_format_func(unmasked, **options)

                # default_fmt should always give back same str length.
                # Figure out what this is with a test call.
                # This is a bit complicated to account for struct types.
                example_elem = elem._data.ravel()[0]
                example_str = default_fmt(example_elem)
                reslen = builtins.max(len(example_str), len(masked_str))

            # pad the columns to align when including the masked string
            if issubclass(elem.dtype.type, np.floating) and unmasked.size > 0:
                # for floats, try to align with decimal point if present
                frac = example_str.partition('.')
                nfrac = len(frac[1]) + len(frac[2])
                masked_str = (masked_str + ' '*nfrac).rjust(reslen)
                # Would it be safer/better to simply center the X?
            else:
                masked_str = masked_str.rjust(reslen)

            def fmt(x):
                if x._mask:
                    return masked_str
                return default_fmt(x._data).rjust(reslen)

            return fmt

    return MaskedFormatter

MASK_STR = 'X'

masked_formatters = [as_masked_fmt(f) for f in default_duckprint_formatters]
default_options = default_duckprint_options.copy()
default_options['masked_str'] = MASK_STR
masked_dispatcher = FormatDispatcher(masked_formatters, default_options)

################################################################################
#                               Ufunc setup
################################################################################

_masked_ufuncs = {}

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
    """

    def __init__(self, ufunc):
        super().__init__(ufunc)

    def __call__(self, a, *args, **kwargs):
        if a is X:
            raise ValueError("must supply dtype if all inputs are X")

        a = as_duck_cls(a, base=MaskedArray)

        out = kwargs.get('out', ())
        if not isinstance(out, tuple):
            out = (out,)
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)

        d, m = a._data, a._mask

        where = ~m
        kwhere = kwargs.get('where', None)
        if isinstance(kwhere, (MaskedArray, MaskedScalar)):
            if kwhere.dtype.type != np.bool_:
                raise ValueError("'where' only supports masks for boolean "
                                 "dtype")
            kwhere = kwhere.filled(False)
        if kwhere is not None:
            where &= kwhere
        kwargs['where'] = where

        result = self.f(d, *args, **kwargs)

        if out != ():
            out[0]._mask[...] = m
            return out[0]

        cls = get_duck_cls(a, base=MaskedArray)
        if is_ndscalar(result):
            return type(a)._scalartype(result, m)

        return type(a)(result, m)

class _Masked_BinOp(_Masked_UFunc):
    """
    Masked version of binary ufunc. Assumes 1 output.

    Parameters
    ----------
    ufunc : ufunc
        The ufunc for which to define a masked version.
    reduce_fill : function or scalar, optional
        Determines what fill_value is used during reductions. If a function is
        supplied, it shoud accept a dtype as argument and return a fill value
        with that dtype. A scalar value may also be supplied, which is used
        for all dtypes of the ufunc.
    """

    def __init__(self, ufunc, reduce_fill=None):
        super().__init__(ufunc)

        if reduce_fill is None:
            reduce_fill = ufunc.identity

        if (reduce_fill is not None and
                (is_ndscalar(reduce_fill) or not callable(reduce_fill))):
            self.reduce_fill = lambda dtype: reduce_fill
        else:
            self.reduce_fill = reduce_fill

    def __call__(self, a, b, **kwargs):
        # treat X as a masked value of the other array's dtype
        if a is X:
            a = X(b.dtype)
        if b is X:
            b = X(a.dtype)

        a, b = as_duck_cls(a, b, base=MaskedArray)
        da, db = a._data, b._data
        ma, mb = a._mask, b._mask

        mkwargs = {}
        for k in ['where', 'order']:
            if k in kwargs:
                mkwargs[k] = kwargs[k]

        out = kwargs.get('out', ())
        if not isinstance(out, tuple):
            out = (out,)
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)
            mkwargs['out'] = (out[0]._mask,)

        m = np.logical_or(ma, mb, **mkwargs)

        where = ~m
        kwhere = kwargs.get('where', None)
        if isinstance(kwhere, (MaskedArray, MaskedScalar)):
            if kwhere.dtype.type != np.bool_:
                raise ValueError("'where' only supports masks for boolean "
                                 "dtype")
            kwhere = kwhere.filled(False)
        if kwhere is not None:
            where &= kwhere
        kwargs['where'] = where

        result = self.f(da, db, **kwargs)

        if out:
            return out[0]

        if is_ndscalar(result):
            return type(a)._scalartype(result, m)
        return type(a)(result, m)

    def reduce(self, a, **kwargs):
        if self.reduce_fill is None:
            raise TypeError("reduce not supported for masked {}".format(self.f))

        da, ma = getdata(a), getmask(a)

        mkwargs = kwargs.copy()
        for k in ['initial', 'dtype']:
            if k in mkwargs:
                del mkwargs[k]

        out = kwargs.get('out', ())
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)
            mkwargs['out'] = (out[0]._mask,)

        initial = kwargs.get('initial', None)
        if isinstance(initial, (MaskedScalar, MaskedX)):
            raise ValueError("initial should not be masked")

        if 0: # two different implementations, investigate performance
            wheremask = ~ma
            if 'where' in kwargs:
                wheremask &= kwargs['where']
            kwargs['where'] = wheremask
            if 'initial' not in kwargs:
                kwargs['initial'] = self.reduce_fill(da.dtype)

            result = self.f.reduce(da, **kwargs)
            m = np.logical_and.reduce(ma, **mkwargs)
        else:
            if not is_ndscalar(da):
                da[ma] = self.reduce_fill(da.dtype)
                # if da is a scalar, we get correct result no matter fill

            result = self.f.reduce(da, **kwargs)
            m = np.logical_and.reduce(ma, **mkwargs)

        if out:
            return out[0]

        cls = get_duck_cls(a, base=MaskedArray)
        if is_ndscalar(result):
            return cls._scalartype(result, m)
        return cls(result, m)

    def accumulate(self, a, axis=0, dtype=None, out=None):
        if self.reduce_fill is None:
            raise TypeError("accumulate not supported for masked {}".format(
                            self.f))

        da, ma = getdata(a), getmask(a)

        dataout, maskout = None, None
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            dataout = out[0]._data
            maskout = out[0]._mask

        if not is_ndscalar(da):
            da[ma] = self.reduce_fill(da.dtype)
        result = self.f.accumulate(da, axis, dtype, dataout)
        m = np.logical_and.accumulate(ma, axis, out=maskout)

        if out:
            return out[0]
        if is_ndscalar(result):
            return MaskedScalar(result, m)
        return type(a)(result, m)

    def outer(self, a, b, **kwargs):
        if self.reduce_fill is None:
            raise TypeError("outer not supported for masked {}".format(self.f))

        da, db = getdata(a), getdata(b)
        ma, mb = getmask(a), getmask(b)

        # treat X as a masked value of the other array's dtype
        if da is X:
            da, ma = db.dtype.type(0), np.bool_(True)
        if db is X:
            db, mb = da.dtype.type(0), np.bool_(True)

        mkwargs = kwargs.copy()
        if 'dtype' in mkwargs:
            del mkwargs['dtype']

        out = kwargs.get('out', ())
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)
            mkwargs['out'] = (out[0]._mask,)

        if not is_ndscalar(da):
            da[ma] = self.reduce_fill(da.dtype)
        if not is_ndscalar(db):
            db[mb] = self.reduce_fill(db.dtype)

        result = self.f.outer(da, db, **kwargs)
        m = np.logical_or.outer(ma, mb, **mkwargs)

        if out:
            return out[0]
        if is_ndscalar(result):
            return MaskedScalar(result, m)
        return type(a)(result, m)

    def reduceat(self, a, indices, **kwargs):
        if self.reduce_fill is None:
            raise TypeError("reduce not supported for masked {}".format(self.f))

        da, ma = getdata(a), getmask(a)

        mkwargs = kwargs.copy()
        for k in ['initial', 'dtype']:
            if k in mkwargs:
                del mkwargs[k]

        out = kwargs.get('out', ())
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)
            mkwargs['out'] = (out[0]._mask,)

        initial = kwargs.get('initial', None)
        if isinstance(initial, (MaskedScalar, MaskedX)):
            raise ValueError("initial should not be masked")

        if not is_ndscalar(da):
            da[ma] = self.reduce_fill(da.dtype)
            # if da is a scalar, we get correct result no matter fill

        result = self.f.reduceat(da, indices, **kwargs)
        m = np.logical_and.reduceat(ma, indices, **mkwargs)

        if out:
            return out[0]
        if is_ndscalar(result):
            return MaskedScalar(result, m)
        return type(a)(result, m)

    def at(self, a, indices, b=None):
        if isinstance(indices, (MaskedArray, MaskedScalar)):
            raise ValueError("indices should not be masked. "
                             "Use .filled() first")

        da, ma = getdata(a), getmask(a)
        db, mb = None, None
        if b is not None:
            db, mb = getdata(b), getmask(b)

        self.f.at(da, indices, db)
        np.logical_or.at(ma, indices, mb)

def _add_ufunc(ufunc, uni=False, glob=globals(), **kwargs):
    if uni:
        impl = _Masked_UniOp(ufunc, **kwargs)
    else:
        impl = _Masked_BinOp(ufunc, **kwargs)
    _masked_ufuncs[ufunc] = impl
    glob[ufunc.__name__] = impl

# unary funcs
for ufunc in [umath.exp, umath.conjugate, umath.sin, umath.cos, umath.tan,
              umath.arctan, umath.arcsinh, umath.sinh, umath.cosh,
              umath.tanh, umath.absolute, umath.fabs, umath.negative,
              umath.floor, umath.ceil, umath.logical_not, umath.isfinite,
              umath.isinf, umath.isnan, umath.invert, umath.sqrt, umath.log,
              umath.log2, umath.log10, umath.tan, umath.arcsin,
              umath.arccos, umath.arccosh, umath.arctanh]:
    _add_ufunc(ufunc, uni=True)

# binary ufuncs
for ufunc in [umath.add, umath.subtract, umath.multiply,
              umath.arctan2, umath.hypot, umath.equal, umath.not_equal,
              umath.less_equal, umath.greater_equal, umath.less,
              umath.greater, umath.logical_and, umath.logical_or,
              umath.logical_xor, umath.bitwise_and, umath.bitwise_or,
              umath.bitwise_xor, umath.true_divide, umath.floor_divide,
              umath.remainder, umath.fmod, umath.mod, umath.power]:
    _add_ufunc(ufunc)

# fill value depends on dtype
_add_ufunc(umath.maximum, reduce_fill=lambda dt: _minvals[dt])
_add_ufunc(umath.minimum, reduce_fill=lambda dt: _maxvals[dt])


################################################################################
#                         __array_function__ setup
################################################################################

implements = new_ducktype_implementation()

def get_maskedout(out):
    if out is not None:
        if isinstance(out, MaskedArray):
            return out._data, out._mask
        raise Exception("out must be a masked array")
    return None, None

def maskedarray_or_scalar(data, mask, out=None, cls=MaskedArray):
    if out is not None:
        return out
    if is_ndscalar(data):
        return cls._scalartype(data, mask)
    return cls(data, mask)

def _copy_mask(mask, outmask=None):
    if outmask is not None:
        result_mask = outmask
        result_mask[...] = mask
    else:
        result_mask = mask.copy()
    return result_mask

def _inplace_not(v):
    if isinstance(v, np.ndarray):
        return np.logical_not(v, out=v)
    return np.logical_not(v)

################################################################################
#                        npy-api implementations
################################################################################

@implements(np.all)
def all(a, axis=None, out=None, keepdims=np._NoValue):
    a = as_duck_cls(a, base=MaskedArray)
    # out can be maskedarray or ndarray since we never return masked elements
    # (or.. should we only allow ndarray out?)
    if isinstance(out, MaskedArray):
        np.all(a.filled(True, view=1), axis, out._data, keepdims)
        out._mask[...] = False
        return out
    return np.all(a.filled(True, view=1), axis, out, keepdims)
    # Note: returns boolean, not MaskedArray. If case of fully masked,
    # return True, like np.all([]).

@implements(np.any)
def any(a, axis=None, out=None, keepdims=np._NoValue):
    a = as_duck_cls(a, base=MaskedArray)
    if isinstance(out, MaskedArray):
        np.any(a.filled(False, view=1), axis, out._data, keepdims)
        out._mask[...] = False
        return out
    return np.any(a.filled(False, view=1), axis, out, keepdims)
    # Note: returns boolean, not MaskedArray. If case of fully masked,
    # return False, like np.any([])

@implements(np.amax)
@implements(np.max)
def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
        where=True):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)

    kwarg = {}
    if keepdims is not np._NoValue:
        kwarg['keepdims'] = keepdims
    if where is not np._NoValue:
        kwarg['where'] = where

    initial_m = initial_d = np._NoValue
    if initial is not np._NoValue:
        ismasked = isinstance(initial, MaskedScalar)
        if initial is X or ismasked and initial._mask:
            raise ValueError("initial cannot be masked")
        initial_m = False
        initial_d = initial._data if ismasked else initial

    filled = a.filled(minmax='min', view=1)
    result_data = np.max(filled, axis, outdata, initial=initial_d, **kwarg)
    result_mask = np.logical_and.reduce(a._mask, axis, out=outmask,
                                        initial=initial_m, **kwarg)

    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.argmax)
def argmax(a, axis=None, out=None):
    if isinstance(out, MaskedArray):
        raise TypeError("out argument of argmax should be an ndarray")
    a = as_duck_cls(a, base=MaskedArray)

    # most of the time this is enough
    filled = a.filled(minmax='min', view=1)
    result_data = np.argmax(filled, axis, out)

    # except if the only unmasked elem is minval. Have to check and do carefully
    data_min = filled == _minvals[a.dtype]
    is_min = data_min & ~a._mask
    has_min = np.any(is_min, axis=axis)
    if np.any(has_min):
        has_no_other_data = np.all(data_min, axis=axis)
        has_lonely_min = has_min & has_no_other_data
        if np.any(has_lonely_min):
            min_ind = np.argmax(is_min, axis=axis)
            if is_ndscalar(result_data):
                return min_ind
            result_data[has_lonely_min] = min_ind[has_lonely_min]
    # one day, might speed up with numba/extension. Or with np.take?

    return result_data

@implements(np.amin)
@implements(np.min)
def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
        where=np._NoValue):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)

    kwarg = {}
    if keepdims is not np._NoValue:
        kwarg['keepdims'] = keepdims
    if where is not np._NoValue:
        kwarg['where'] = where

    initial_m = initial_d = np._NoValue
    if initial is not np._NoValue:
        ismasked = isinstance(initial, MaskedScalar)
        if initial is X or ismasked and initial._mask:
            raise ValueError("initial cannot be masked")
        initial_m = False
        initial_d = initial._data if ismasked else initial

    filled = a.filled(minmax='max', view=1)
    result_data = np.min(filled, axis, outdata, initial=initial_d, **kwarg)
    result_mask = np.logical_and.reduce(a._mask, axis, out=outmask,
                                        initial=initial_m, **kwarg)

    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.argmin)
def argmin(a, axis=None, out=None):
    if isinstance(out, MaskedArray):
        raise TypeError("out argument of argmax should be an ndarray")

    a = as_duck_cls(a, base=MaskedArray)

    # most of the time this is enough
    filled = a.filled(minmax='max', view=1)
    result_data = np.argmin(filled, axis, out)

    # except if the only unmasked elem is maxval. Have to check and do carefully
    data_max = filled == _maxvals[a.dtype]
    is_max = data_max & ~a._mask
    has_max = np.any(is_max, axis=axis)
    if np.any(has_max):
        has_no_other_data = np.all(data_max, axis=axis)
        has_lonely_max = has_max & has_no_other_data
        if np.any(has_lonely_max):
            max_ind = np.argmax(is_max, axis=axis)
            if is_ndscalar(result_data):
                return max_ind
            result_data[has_lonely_max] = max_ind[has_lonely_max]

    return result_data

@implements(np.sort)
def sort(a, axis=-1, kind='quicksort', order=None):
    a = as_duck_cls(a, base=MaskedArray)
    # Note: This is trickier than it looks. The first line sorts the mask
    # together with any min_vals which may be present, so there appears to
    # be a problem ordering mask vs min_val elements.
    # But, since we know all the masked elements have to end up at the end
    # of the axis, we can sort the mask too and everything works out. The
    # mask-sort only swaps the mask between min_val and masked positions
    # which have the same underlying data.

    # np.nan should sort higher than all others, so use it as fill if floating
    result_data = np.sort(a.filled(minmax='maxnan', view=1), axis, kind, order)
    result_mask = np.sort(a._mask, axis, kind)  #or partition for speed?
    return maskedarray_or_scalar(result_data, result_mask, cls=type(a))
    # Note: lexsort may be faster, but doesn't provide kind or order kwd

@implements(np.argsort)
def argsort(a, axis=-1, kind='quicksort', order=None):
    a = as_duck_cls(a, base=MaskedArray)
    # Similar to mask-sort trick in sort above, here after sorting data we
    # re-sort based on mask. Use the property that if you argsort the index
    # array produced by argsort you get the element rank, which can be
    # argsorted again to get back the sort indices. However, here we
    # modify the rank based on the mask before inverting back to indices.
    # Uses two argsorts plus a temp array.
    inds = np.argsort(a.filled(minmax='maxnan', view=1), axis, kind, order)
    # next two lines "reverse" the argsort (same as double-argsort)
    ranks = np.empty(inds.shape, dtype=inds.dtype)
    np.put_along_axis(ranks, inds, np.arange(a.shape[axis]), axis)
    # prepare to resort but make masked elem highest rank
    ranks[a._mask] = _maxvals[ranks.dtype]
    return np.argsort(ranks, axis, kind)

@implements(np.partition)
def partition(a, kth, axis=-1, kind='introselect', order=None):
    a = as_duck_cls(a, base=MaskedArray)
    inds = np.argpartition(a, kth, axis, kind, order)
    return np.take_along_axis(a, inds, axis=axis)

@implements(np.argpartition)
def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    # see argsort for explanation
    a = as_duck_cls(a, base=MaskedArray)
    filled = a.filled(minmax='maxnan', view=1)
    inds = np.argpartition(filled, kth, axis, kind, order)
    ranks = np.empty(inds.shape, dtype=inds.dtype)
    np.put_along_axis(ranks, inds, np.arange(a.shape[axis]), axis)
    ranks[a._mask] = _maxvals[ranks.dtype]
    return np.argpartition(ranks, kth, axis, kind)

@implements(np.searchsorted, checked_args=('v',))
def searchsorted(a, v, side='left', sorter=None):
    a = as_duck_cls(a, base=MaskedArray)
    maskleft = len(a) - np.sum(a._mask)
    aval = a.filled(minmax='maxnan', view=1)
    inds = np.searchsorted(aval, v.filled(minmax='maxnan', view=1),
                           side, sorter)

    # Line above treats mask and maxval as the same, we need to fix it up
    if side == 'left':
        # masked vals in v need to be moved right to the left end of the
        # masked vals in a (which have to be to the right end of a).
        inds[v._mask] = maskleft
    else:
        # maxvals in v meed to be moved left to the left end of the
        # masked vals in a.
        if issubclass(v.dtype.type, np.inexact):
            maxinds = np.isnan(v._data)
        else:
            maxinds = v._data == _maxvals[v.dtype]
        inds[maxinds & ~v._mask] = maskleft

    return inds

@implements(np.digitize)
def digitize(x, bins, right=False):
    x = as_duck_cls(x, base=MaskedArray)

    # Original comment:
    # here for compatibility, searchsorted below is happy to take this
    if np.issubdtype(x.dtype, np.complexfloating):
        raise TypeError("x may not be complex")

    if isinstance(bins, (MaskedArray, MaskedScalar)):
        raise ValueError("bins should not be masked. "
                         "Use .filled() first")

    mono = np.lib.function_base._monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically "
                         "increasing or decreasing")

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

    keys = as_duck_cls(*keys, base=MaskedArray, single=False)

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

    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)

    cls = get_duck_cls(a, base=MaskedArray)
    if type(a) is not cls:
        a = cls(a)

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

    ret = np.sum(a.filled(0, view=1), axis=axis, out=outdata, dtype=dtype,
                 **kwargs)
    retmask = np.all(a._mask, axis=axis, out=outmask, **kwargs)

    with np.errstate(divide='ignore', invalid='ignore'):
        if is_ndarr(ret):
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

    return maskedarray_or_scalar(ret, retmask, out, type(a))

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

    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)

    # code largely copied from _methods.var
    rcount = a.count(axis=axis, **kwargs)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(a.dtype.type, (np.integer, np.bool_)):
        dtype = np.dtype('f8')

    # Compute the mean, keeping same dims. Note that if dtype is not of
    # inexact type then arraymean will not be either.
    rcount = a.count(axis=axis, keepdims=True)
    arrmean = a.filled(0).sum(axis=axis, dtype=dtype, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        if not is_ndscalar(arrmean):
            arrmean = np.true_divide(arrmean, rcount, out=arrmean,
                                     casting='unsafe', subok=False)
        else:
            arrmean = arrmean.dtype.type(arrmean / rcount)

    # Compute sum of squared deviations from mean
    x = type(a)(a - arrmean)
    if issubclass(a.dtype.type, np.complexfloating):
        x = np.multiply(x, np.conjugate(x), out=x).real
    else:
        x = np.multiply(x, x, out=x)
    ret = x.filled(0, view=1).sum(axis, dtype, out=outdata, **kwargs)

    # Compute degrees of freedom and make sure it is not negative.
    rcount = a.count(axis=axis, **kwargs)
    rcount = np.maximum(rcount - ddof, 0)

    # divide by degrees of freedom
    with np.errstate(divide='ignore', invalid='ignore'):
        if is_ndarr(ret):
            ret = np.true_divide(
                    ret, rcount, out=ret, casting='unsafe', subok=False)
        elif hasattr(ret, 'dtype'):
            ret = ret.dtype.type(ret / rcount)
        else:
            ret = ret / rcount

    if out is not None:
        out[rcount == 0] = X
        return out
    return maskedarray_or_scalar(ret, rcount == 0, cls=type(a))

@implements(np.std)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    a = as_duck_cls(a, base=MaskedArray)
    ret = var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
              keepdims=keepdims)

    if isinstance(ret, MaskedArray):
        ret = np.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = np.sqrt(ret).astype(ret.dtype)
    else:
        ret = np.sqrt(ret)
    return ret

@implements(np.average, checked_args=('a',))
def average(a, axis=None, weights=None, returned=False):
    a = as_duck_cls(a, base=MaskedArray)

    if weights is None:
        avg = a.mean(axis)
        if returned:
            return avg, avg.dtype.type(a.count(axis))
        return avg

    wgt = weights if is_ndtype(weights) else np.array(weights)

    if isinstance(wgt, MaskedArray):
        raise TypeError("weight must not be a MaskedArray")

    if issubclass(a.dtype.type, (np.integer, np.bool_)):
        result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
    else:
        result_dtype = np.result_type(a.dtype, wgt.dtype)
        # Note: No float16 special case, since ndarray.average skips it

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
        if wgt.shape != a.shape:
            wgt = np.broadcast_to(wgt, a.shape)

    wgt = type(a)(wgt, a._mask)
    scl = wgt.sum(axis=axis, dtype=result_dtype)
    if np.any(scl == 0.0):
        raise ZeroDivisionError(
            "Weights sum to zero, can't be normalized")

    avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)/scl

    if returned:
        return avg, scl
    return avg

def _move_reduction_axis_last(a, axis=None):
    """
    Modified from numpy.lib.function_base._ureduce.

    Reshape/transpose array so desired axes are grouped at the end.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or iterable of ints
        axes or axis to reduce

    Returns
    -------
    arr : ndarray
        Input ndarray with iteration axis/axes moved to be a single axis
        at the end.
    keepdims : tuple
        a.shape with axis dims set to 1 which can be used to reshape the
        result of a reduction to the same shape a ufunc with keepdims=True
        would produce.

    """
    if axis is not None:
        keepdim = list(a.shape)
        nd = a.ndim
        axis = normalize_axis_tuple(axis, nd)

        for ax in axis:
            keepdim[ax] = 1

        if len(axis) == 1:
            # arr, with the iteration axis at the end
            ax = axis[0]
            dims = list(range(a.ndim))
            a = np.transpose(a, dims[:ax] + dims[ax+1:] + [ax])
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))

        keepdim = tuple(keepdim)
    else:
        keepdim = (1,) * a.ndim
        a = a.ravel()

    return a, keepdim

@implements(np.median)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return np.quantile(a, 0.5, axis=axis, out=out,
                        overwrite_input=overwrite_input,
                        interpolation='midpoint', keepdims=keepdims)

@implements(np.percentile)
def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False):
    q = np.true_divide(q, 100)
    q = np.asanyarray(q)  # undo any decay the ufunc performed (gh-13105)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, interpolation, keepdims)

@implements(np.quantile)
def quantile(a, q, axis=None, out=None, overwrite_input=False,
             interpolation='linear', keepdims=False):
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, interpolation, keepdims)

def _quantile_unchecked(a, q, axis=None, out=None, overwrite_input=False,
                        interpolation='linear', keepdims=False):
    """Assumes that q is in [0, 1], and is an ndarray"""

    a = as_duck_cls(a, base=MaskedArray)
    a, kdim = _move_reduction_axis_last(a, axis)

    if len(q.shape) > 1:
        raise ValueError("q must be a scalar or 1d array")

    out_shape = (q.size,) + a.shape[:-1]

    if out is None:
        dt = np.promote_types(a.dtype, np.float64)
        outarr = get_duck_cls(a)(np.empty(out_shape, dtype=dt))
    else:
        if out.shape == out_shape:
            outarr = out
        elif q.size == 1 and (1,)+out.shape == out_shape:
            outarr = out[None,...]
        else:
            raise ValueError('out has wrong shape')

    inds = np.ndindex(a.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)
    for ind in inds:
        ai = a[ind]
        dat = ai._data[~ai.mask]
        oind = (slice(None),) + ind
        if dat.size == 0:
            outarr[oind] = X
        else:
            outarr[oind] = np.quantile(dat, q, interpolation=interpolation)

    if out is not None:
        return out

    # return a scalar in simple case
    if q.shape == () and axis is None:
        return outarr[0]

    out_dim = kdim if keepdims else a.shape[:-1]
    return outarr.reshape(q.shape + out_dim)

@implements(np.cov, checked_args=('m', 'y'))
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None):
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    # Handles complex arrays too
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    cls = get_duck_cls(m, base=MaskedArray)
    if type(m) is not cls:
        m = cls(m)

    if y is None:
        dtype = np.result_type(m, np.float64)
    else:
        if not is_ndtype(y):
            y = cls(y)
        else:
            cls = get_duck_cls(m, y, base=MaskedArray)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")
        dtype = np.result_type(m, y, np.float64)

    X = cls(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return cls([]).reshape(0, 0)
    if y is not None:
        y = cls(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if np.any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if np.any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg = np.average(X, axis=1, weights=w)

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    c = np.dot(X, X_T.conj())

    # Determine the normalization
    nomask = ~X.mask
    wnm = nomask.astype(dtype) if w is None else w*nomask
    w_sum = np.dot(wnm, nomask.T)
    if ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        a_sum = np.dot(w*aweights*nomask, nomask.T)
        fact = w_sum - ddof*a_sum/w_sum

    nonpos_fact = fact <= 0
    if np.any(nonpos_fact):
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact[nonpos_fact] = X

    c *= np.true_divide(1, fact)
    return c.squeeze()

@implements(np.corrcoef, checked_args=('x', 'y'))
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue):
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=3)
    c = np.cov(x, y, rowvar)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    cd = c._data
    with np.errstate(invalid='ignore'):
        np.clip(cd.real, -1, 1, out=cd.real)
        if np.iscomplexobj(cd):
            np.clip(cd.imag, -1, 1, out=cd.imag)

    return c

@implements(np.clip)
def clip(a, a_min, a_max, out=None):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.clip(a._data, a_min, a_max, outdata)
    result_mask = _copy_mask(a._mask, outmask)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.compress)
def compress(condition, a, axis=None, out=None):
    # Note: masked values in condition treated as False
    outdata, outmask = get_maskedout(out)
    cls = get_duck_cls(condition, a, base=MaskedArray)
    cond = cls(condition).filled(False, view=1)
    a = cls(a)
    result_data = np.compress(cond, a._data, axis, outdata)
    result_mask = np.compress(cond, a._mask, axis, outmask)
    return maskedarray_or_scalar(result_data, result_mask, out, cls)

@implements(np.copy)
def copy(a, order='K'):
    a = as_duck_cls(a, base=MaskedArray)
    result_data = np.copy(a._data, order=order)
    result_mask = np.copy(a._mask, order=order)
    return maskedarray_or_scalar(result_data, result_mask, cls=type(a))

@implements(np.product)
@implements(np.prod)
def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.prod(a.filled(1, view=1), axis=axis, dtype=dtype,
                          out=outdata, keepdims=keepdims)
    result_mask = np.all(a._mask, axis=axis, out=outmask, keepdims=keepdims)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.cumproduct)
@implements(np.cumprod)
def cumprod(a, axis=None, dtype=None, out=None):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.cumprod(a.filled(1, view=1), axis, dtype=dtype,
                             out=outdata)
    result_mask = np.logical_or.accumulate(~a._mask, axis, out=outmask)
    result_mask =_inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.sum)
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.sum(a.filled(0, view=1), axis, dtype=dtype,
                         out=outdata, keepdims=keepdims)
    result_mask = np.all(a._mask, axis, out=outmask, keepdims=keepdims)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.cumsum)
def cumsum(a, axis=None, dtype=None, out=None):
    a = as_duck_cls(a, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.cumsum(a.filled(0, view=1), axis, dtype=dtype,
                            out=outdata)
    result_mask = np.logical_or.accumulate(~a._mask, axis, out=outmask)
    result_mask =_inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1):
    a = as_duck_cls(a, base=MaskedArray)
    result = np.diagonal(a._data, offset=offset, axis1=axis1, axis2=axis2)
    rmask = np.diagonal(a._mask, offset=offset, axis1=axis1, axis2=axis2)
    return maskedarray_or_scalar(result, rmask, cls=type(a))

@implements(np.diag)
def diag(v, k=0):
    v = as_duck_cls(v, base=MaskedArray)
    s = v.shape
    if len(s) == 1:
        n = s[0]+abs(k)
        res = type(v)(np.zeros((n, n), v.dtype))
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
    v = as_duck_cls(v, base=MaskedArray)
    return np.diag(v.ravel(), k)

@implements(np.tril)
def tril(m, k=0):
    m = as_duck_cls(m, base=MaskedArray)
    mask = np.tri(*m.shape[-2:], k=k, dtype=bool)
    return np.where(mask, m, np.zeros(1, m.dtype))

@implements(np.triu)
def triu(m, k=0):
    m = as_duck_cls(m, base=MaskedArray)
    mask = np.tri(*m.shape[-2:], k=k-1, dtype=bool)
    return np.where(mask, np.zeros(1, m.dtype), m)

@implements(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    outdata, outmask = get_maskedout(out)
    a = as_duck_cls(a, base=MaskedArray)
    result_data = np.trace(a.filled(0, view=1), offset=offset, axis1=axis1,
                      axis2=axis2, dtype=dtype, out=outdata)
    result_mask = np.trace(~a._mask, offset=offset, axis1=axis1, axis2=axis2,
                          dtype=bool, out=outmask)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.dot)
def dot(a, b, out=None):
    outdata, outmask = get_maskedout(out)
    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)
    result_data = np.dot(a.filled(0, view=1), b.filled(0, view=1),
                         out=outdata)
    result_mask = np.dot(~a._mask, ~b._mask, out=outmask)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, cls)

@implements(np.vdot)
def vdot(a, b):
    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)
    result_data = np.vdot(a.filled(0, view=1), b.filled(0, view=1))
    result_mask = np.vdot(~a._mask, ~b._mask)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)

    # because of mask calculation, we don't support vectors of length 2.
    # convert them if present. First have to do axis manip as in np.cross

    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3
        axis = None
    # Check axisa and axisb are within bounds
    axisa = normalize_axis_index(axisa, a.ndim, msg_prefix='axisa')
    axisb = normalize_axis_index(axisb, b.ndim, msg_prefix='axisb')

    # Move working axis to the end of the shape
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    msg = ("incompatible dimensions for cross product\n"
           "(dimension must be 2 or 3)")
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)
    if a.shape[-1] == 2:
        a = np.append(a, np.broadcast_to(0, a.shape[:-1] + (1,)), axis=-1)
    if b.shape[-1] == 2:
        b = np.append(b, np.broadcast_to(0, b.shape[:-1] + (1,)), axis=-1)

    result_data = np.cross(a.filled(0, view=1), b.filled(0, view=1), axisa,
                           axisb, axisc, axis)
    # trick: use nan behavior to compute mask
    ma = np.where(a._mask, np.nan, 0)
    mb = np.where(b._mask, np.nan, 0)
    mab = np.cross(ma, mb, axisa, axisb, axisc, axis)
    result_mask = np.isnan(mab)
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.inner)
def inner(a, b):
    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)
    result_data = np.inner(a.filled(0, view=1), b.filled(0, view=1))
    result_mask = np.inner(~a._mask, ~b._mask)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.outer)
def outer(a, b, out=None):
    outdata, outmask = get_maskedout(out)
    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)
    result_data = np.outer(a.filled(0, view=1), b.filled(0, view=1),
                           out=outdata)
    result_mask = np.outer(~a._mask, ~b._mask, out=outmask)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, cls)

@implements(np.kron)
def kron(a, b):
    cls = get_duck_cls(a, b, base=MaskedArray)
    a = cls(a, copy=False, subok=True, ndmin=b.ndim)
    nda, ndb = a.ndim, b.ndim
    if (nda == 0 or ndb == 0):
        return np.multiply(a, b)
    a_shape = a.shape
    b_shape = b.shape
    nd = ndb
    if ndb > nda:
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

    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)
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
    newshape_a = (int(np.multiply.reduce([ashape[ax] for ax in notin])), N2)
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

def _process_einsum_operands(operands):
    # operands can either start with a strong, followed by op arrays,
    # or can alternate op arrays and axes
    if isinstance(operands[0], str):
        arrs = operands[1:]
        cls = get_duck_cls(*arrs, base=MaskedArray)
        arrs = tuple(cls(x) for x in arrs)
        data_ops = (operands[0],) + tuple(a.filled(0) for a in arrs)
        imask_ops = (operands[0],) + tuple(~a._mask for a in arrs)
    else:
        cls = get_duck_cls(*operands[0::2], base=MaskedArray)
        ops = tuple(o if n%2 else cls(o) for n,o in enumerate(operands))
        data_ops = tuple(o if n%2 else o.filled(0) for n,o in enumerate(ops))
        imask_ops = tuple(o if n%2 else ~o._mask for n,o in enumerate(ops))
    return data_ops, imask_ops, cls

@implements(np.einsum)
def einsum(*operands, **kwargs):
    out = kwargs.pop('out', None)
    outdata, outmask = get_maskedout(out)
    dtype = kwargs.pop('dtype', None)

    data_ops, imask_ops, cls = _process_einsum_operands(operands)
    result_data = np.einsum(*data_ops, out=outdata, dtype=dtype, **kwargs)
    result_mask = np.einsum(*imask_ops, out=outmask, **kwargs)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, cls)

@implements(np.einsum_path)
def einsum_path(*operands, **kwargs):
    out = kwargs.pop('out', None)
    outdata, outmask = get_maskedout(out)
    dtype = kwargs.pop('dtype', None)

    data_ops, imask_ops, cls = _process_einsum_operands(operands)
    result_data = np.einsum_path(*data_ops, out=outdata, dtype=dtyle, **kwargs)
    result_mask = np.einsum_path(*imask_ops, out=outmask, **kwargs)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, out, cls)

@implements(np.correlate)
def correlate(a, v, mode='valid'):
    cls = get_duck_cls(a, v)
    result_data = np.correlate(a.filled(view=1), v.filled(view=1), mode)
    result_mask = np.correlate(~a._mask, v._mask, mode)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.convolve)
def convolve(a, v, mode='full'):
    cls = get_duck_cls(a, v)
    a, v = cls(a), cls(v)
    result_data = np.convolve(a.filled(view=1), v.filled(view=1), mode)
    result_mask = np.convolve(~a._mask, ~v._mask, mode)
    result_mask = _inplace_not(result_mask)
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.real)
def real(a):
    result_data = np.real(a._data)
    result_data.flags['WRITEABLE'] = False
    result_mask = a._mask.copy()
    return maskedarray_or_scalar(result_data, result_mask, cls=type(a))

@implements(np.imag)
def imag(a):
    result_data = np.imag(a._data)
    result_data.flags['WRITEABLE'] = False
    result_mask = a._mask
    return maskedarray_or_scalar(result_data, result_mask, cls=type(a))

@implements(np.ptp)
def ptp(a, axis=None, out=None, keepdims=False):
    return np.subtract(
        np.maximum.reduce(a, axis, None, out, keepdims),
        np.minimum.reduce(a, axis, None, None, keepdims), out)

@implements(np.take)
def take(a, indices, axis=None, out=None, mode='raise'):
    outdata, outmask = get_maskedout(out)

    if isinstance(indices, (MaskedArray, MaskedScalar)):
        raise ValueError("indices should not be masked. "
                         "Use .filled() first")

    result_data = np.take(a._data, indices, axis, outdata, mode)
    result_mask = np.take(a._mask, indices, axis, outmask, mode)
    return maskedarray_or_scalar(result_data, result_mask, out, cls=type(a))

@implements(np.put)
def put(a, indices, values, mode='raise'):
    data, mask, _ = replace_X(values, dtype=a.dtype)
    np.put(a._data, indices, data, mode)
    np.put(a._mask, indices, mask, mode)
    return None

@implements(np.take_along_axis, checked_args=('arr',))
def take_along_axis(arr, indices, axis):
    result_data = np.take_along_axis(arr._data, indices, axis)
    result_mask = np.take_along_axis(arr._mask, indices, axis)
    return maskedarray_or_scalar(result_data, result_mask, cls=type(arr))

@implements(np.put_along_axis, checked_args=('arr',))
def put_along_axis(arr, indices, values, axis):
    data, mask, _ = replace_X(values, dtype=arr.dtype)
    np.put_along_axis(arr._data, indices, data, axis)
    np.put_along_axis(arr._mask, indices, mask, axis)

@implements(np.apply_along_axis)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    # handle negative axes
    cls = get_duck_cls(arr)
    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_view = np.transpose(arr, in_dims[:axis] + in_dims[axis+1:] + [axis])

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = np.ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError('Cannot apply_along_axis when any '
                         'iteration dimensions are 0')
    res = func1d(inarr_view[ind0], *args, **kwargs)

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = cls(np.empty(inarr_view.shape[:-1] + res.shape, res.dtype), False)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0 : axis] +
        buff_dims[buff.ndim-res.ndim : buff.ndim] +
        buff_dims[axis : buff.ndim-res.ndim]
    )

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = func1d(inarr_view[ind], *args, **kwargs)

    # finally, rotate the inserted axes back to where they belong
    return transpose(buff, buff_permute)

@implements(np.apply_over_axes)
def apply_over_axes(func, a, axes):
    val = a
    N = a.ndim
    if np.array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = np.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                                 "an array of the correct shape")
    return val

@implements(np.ravel)
def ravel(a, order='C'):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.ravel(a._data, order=order),
                   np.ravel(a._mask, order=order))

@implements(np.repeat)
def repeat(a, repeats, axis=None):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.repeat(a._data, repeats, axis),
                   np.repeat(a._mask, repeats, axis))

@implements(np.reshape)
def reshape(a, shape, order='C'):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.reshape(a._data, shape, order=order),
                   np.reshape(a._mask, shape, order=order))

@implements(np.resize)
def resize(a, new_shape):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.resize(a._data, new_shape),
                   np.resize(a._mask, new_shape))

@implements(np.meshgrid)
def meshgrid(*xi, **kwargs):
    cls = get_duck_cls(*xi)
    xi = (cls(x) for x in xi)
    data, mask = zip(*((x._data, x._mask) for x in xi))
    result_data = np.meshgrid(*data, **kwargs)
    result_mask = np.meshgrid(*mask, **kwargs)
    return [maskedarray_or_scalar(d, m, cls=cls)
            for d, m in zip(result_data, result_mask)]

@implements(np.around)
def around(a, decimals=0, out=None):
    outdata, outmask = get_maskedout(out)
    result_data = np.round(a._data, decimals, outdata)
    result_mask = _copy_mask(a._mask, outmask)
    return maskedarray_or_scalar(result_data, result_mask, out, type(a))

@implements(np.round)
def round(a, decimals=0, out=None):
    return np.around(a, decimals, out)

@implements(np.fix)
def fix(x, out=None):
    res = np.ceil(x, out=out)
    res = np.floor(x, out=res, where=np.greater_equal(x, 0))
    return out or res

@implements(np.squeeze)
def squeeze(a, axis=None):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.squeeze(a._data, axis),
                   np.squeeze(a._mask, axis))

@implements(np.swapaxes)
def swapaxes(a, axis1, axis2):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.swapaxes(a._data, axis1, axis2),
                   np.swapaxes(a._mask, axis1, axis2))

@implements(np.transpose)
def transpose(a, *axes):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.transpose(a._data, *axes),
                   np.transpose(a._mask, *axes))

@implements(np.roll)
def roll(a, shift, axis=None):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.roll(a._data, shift, axis),
                   np.roll(a._mask, shift, axis))

@implements(np.rollaxis)
def rollaxis(a, axis, start=0):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.rollaxis(a._data, axis, start),
                   np.rollaxis(a._mask, axis, start))

@implements(np.moveaxis)
def moveaxis(a, source, destination):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.moveaxis(a._data, source, destination),
                   np.moveaxis(a._mask, source, destination))

@implements(np.flip)
def flip(m, axis=None):
    m = as_duck_cls(m, base=MaskedArray)
    return type(m)(np.flip(m._data, axis),
                   np.flip(m._mask, axis))

@implements(np.rot90)
def rot90(m, k=1, axes=(0,1)):
    m = as_duck_cls(m, base=MaskedArray)

    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if (axes[0] >= m.ndim or axes[0] < -m.ndim
        or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError("Axes={} out of range for array of ndim={}."
            .format(axes, m.ndim))

    k %= 4

    if k == 0:
        return m[:]
    if k == 2:
        return np.flip(np.flip(m, axes[0]), axes[1])

    axes_list = np.arange(0, m.ndim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])

    if k == 1:
        return np.transpose(np.flip(m,axes[1]), axes_list)
    else:
        # k == 3
        return np.flip(np.transpose(m, axes_list), axes[1])

@implements(np.fliplr)
def fliplr(m):
    m = as_duck_cls(m, base=MaskedArray)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]

@implements(np.flipud)
def flipud(m):
    m = as_duck_cls(m, base=MaskedArray)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]

@implements(np.expand_dims)
def expand_dims(a, axis):
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.expand_dims(a._data, axis),
                   np.expand_dims(a._mask, axis))

@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    outdata, outmask = get_maskedout(out)
    arrays = as_duck_cls(*arrays, base=MaskedArray, single=False)
    data, mask = zip(*((x._data, x._mask) for x in arrays))
    result_data = np.concatenate(data, axis, outdata)
    result_mask = np.concatenate(mask, axis, outmask)
    cls = type(arrays[0])
    return maskedarray_or_scalar(result_data, result_mask, out, cls=cls)

@implements(np.block)
def block(arrays):
    data, mask, cls = replace_X(arrays)
    result_data = np.block(data)
    result_mask = np.block(mask)
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.column_stack)
def column_stack(tup):
    cls = get_duck_cls(tup, base=MaskedArray)
    arrays = []
    for v in tup:
        arr = cls(v, copy=False, subok=True)
        if arr.ndim < 2:
            arr = cls(arr, copy=False, subok=True, ndmin=2).T
        arrays.append(arr)
    return np.concatenate(arrays, 1)

@implements(np.dstack)
def dstack(tup):
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 2)

@implements(np.vstack)
def vstack(tup):
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 0)

@implements(np.hstack)
def hstack(tup):
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return concatenate(arrs, 0)
    else:
        return concatenate(arrs, 1)

@implements(np.array_split, checked_args=('ary',))
def array_split(ary, indices_or_sections, axis=0):
    # array_split is the only *split function that accepts list input
    # as first arg if indices is an integer
    if not is_ndtype(ary):
        ary = MaskedArray(ary)
    return np.array_split.__wrapped__(ary, indices_or_sections, axis)

@implements(np.split, checked_args=('ary',))
def split(ary, indices_or_sections, axis=0):
    # ary can be list if indices_or_sections is not an integer
    if not isinstance(indices_or_sections, int) and not is_ndtype(ary):
        ary = MaskedArray(ary)
    return np.split.__wrapped__(ary, indices_or_sections, axis)

@implements(np.hsplit)
def hsplit(ary, indices_or_sections):
    # note: do not support list input
    return np.hsplit.__wrapped__(ary, indices_or_sections)

@implements(np.vsplit)
def vsplit(ary, indices_or_sections):
    # note: do not support list input
    return np.vsplit.__wrapped__(ary, indices_or_sections)

@implements(np.dsplit)
def dsplit(ary, indices_or_sections):
    # note: do not support list input
    return np.dsplit.__wrapped__(ary, indices_or_sections)

@implements(np.tile)
def tile(A, reps):
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)

    A = as_duck_cls(A, base=MaskedArray)
    if builtins.all(x == 1 for x in tup):
        return type(A)(A, copy=True, subok=True, ndmin=d)
    else:
        c = type(A)(A, copy=False, subok=True, ndmin=d)

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

# if these atleast_*d functions only accepted a single argument our life would
# be easier since we could just drop the asarray.
# But since multiple args are allowed, should we allow inuts like:
# (MaskedArray([1,2,3]), [4, X, 6], np.array([1,2,3])), i.e mixtures od
# maskedarrays and lists and ndarrays to be converted to maskedarrays?
# Note we may also want the user to be able to do
# ndarray_ducktypes.MaskedArray.atleast1d([1, X, 3]), i.e. allow
# user to use X-aware conversion of plain-lists.
# I think least-bad approach right now is to convert lists to MaskedArray,
# otherwise leave ducktype alone.


@implements(np.atleast_1d)
def atleast_1d(*arys):
    res = []
    for ary in arys:
        ary = ary if is_ndtype(ary) else MaskedArray(ary)
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
        ary = ary if is_ndtype(ary) else MaskedArray(ary)
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
        ary = ary if is_ndtype(ary) else MaskedArray(ary)
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
    arrays = [ary if is_ndtype(ary) else MaskedArray(ary) for ary in arrays]
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
    arr = as_duck_cls(arr, base=MaskedArray)
    return type(arr)(np.delete(arr._data, obj, axis),
                     np.delete(arr._mask, obj, axis))

@implements(np.insert)
def insert(arr, obj, values, axis=None):
    arr = as_duck_cls(arr, base=MaskedArray)
    values, vmask, _ = replace_X(values, dtype=arr.dtype)
    return type(arr)(np.insert(arr._data, obj, values, axis),
                     np.insert(arr._mask, obj, vmask, axis))

@implements(np.append)
def append(arr, values, axis=None):
    cls = get_duck_cls(arr, values, base=MaskedArray)
    arr, values = cls(arr), cls(values)
    return cls(np.append(arr._data, values._data, axis),
               np.append(arr._mask, values._mask, axis))

@implements(np.extract)
def extract(condition, arr):
    arr = as_duck_cls(arr, base=MaskedArray)
    return np.extract.__wrapped__(condition, arr)

@implements(np.place)
def place(arr, mask, vals):
    arr = as_duck_cls(arr, base=MaskedArray)
    vals, vmask, _ = replace_X(vals, dtype=arr.dtype)
    np.place(arr._data, mask, vals)
    np.place(arr._mask, mask, vmask)

#@implements(np.pad)
#def pad(array, pad_width, mode, **kwargs):
# XXX takes too much effort to implement

@implements(np.broadcast_to)
def broadcast_to(array, shape, subok=False):
    array = as_duck_cls(array, base=MaskedArray)
    return type(array)(np.broadcast_to(array._data, shape, subok),
                       np.broadcast_to(array._mask, shape))

@implements(np.broadcast_arrays)
def broadcast_arrays(*args, **kwargs):
    if kwargs:
        raise TypeError('broadcast_arrays() got an unexpected keyword '
                        'argument {!r}'.format(list(kwargs.keys())[0]))
    args = as_duck_cls(*args, base=MaskedArray, single=False)
    shape = _broadcast_shape(*args)

    if builtins.all(array.shape == shape for array in args):
        return args

    return [broadcast_to(array, shape, **kwargs) for array in args]

@implements(np.empty_like)
def empty_like(prototype, dtype=None, order='K', subok=True):
    p = as_duck_cls(prototype, base=MaskedArray)
    return type(p)(np.empty_like(p._data, dtype, order, subok))

@implements(np.ones_like)
def ones_like(prototype, dtype=None, order='K', subok=True):
    p = as_duck_cls(prototype, base=MaskedArray)
    return type(p)(np.ones_like(p._data, dtype, order, subok))

@implements(np.zeros_like)
def zeros_like(prototype, dtype=None, order='K', subok=True):
    p = as_duck_cls(prototype, base=MaskedArray)
    return type(p)(np.zeros_like(p._data, dtype, order, subok))

@implements(np.full_like)
def full_like(a, fill_value, dtype=None, order='K', subok=True):
    p = as_duck_cls(a, base=MaskedArray)
    return type(p)(np.full_like(p._data, fill_value, dtype, order, subok))

@implements(np.where)
def where(condition, x=None, y=None):
    if x is None and y is None:
        return np.nonzero(condition)

    cls = get_duck_cls(condition, x, y, base=MaskedArray)

    # convert x, y to MaskedArrays, using the other's dtype if one is X
    if x is X:
        if y is X:
            # why would anyone do this? But it is in unit tests, so...
            raise ValueError("must supply dtype if x and y are both X, "
                             "eg using X(dtype)")
        x, y = cls(X, dtype=y.dtype), cls(y)
    elif y is X:
        x, y = cls(x), cls(X, dtype=x.dtype)
    else:
        x, y = cls(x), cls(y)
    # Note: we do not help the user if they supply something like [X,X,X], they
    # have to supply a dtype to one of the Xs then.

    if isinstance(condition, (MaskedArray, MaskedScalar)):
        condition = condition.filled(False, view=1)

    result_data = np.where(condition, *(a._data for a in (x, y)))
    result_mask = np.where(condition, *(a._mask for a in (x, y)))

    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.argwhere)
def argwhere(a):
    a = as_duck_cls(a, base=MaskedArray)
    # nonzero does not behave well on 0d, so promote to 1d
    if ndim(a) == 0:
        a = atleast_1d(a)
        # then remove the added dimension
        return argwhere(a)[:,:0]
    return transpose(nonzero(a))

@implements(np.choose, checked_args=lambda a,k,t,n: [type(x) for x in a[1]])
def choose(a, choices, out=None, mode='raise'):
    if isinstance(a, (MaskedArray, MaskedScalar)):
        raise TypeError("choice indices should not be masked")

    cls = get_duck_cls(*choices, base=MaskedArray)
    choices = [cls(choice) for choice in choices]
    choices_data = [c._data for c in choices]
    choices_mask = [c._mask for c in choices]

    outdata, outmask = get_maskedout(out)
    result_data = np.choose(a, choices_data, outdata, mode)
    result_mask = np.choose(a, choices_mask, outmask, mode)
    return maskedarray_or_scalar(result_data, result_mask, out, cls)

@implements(np.piecewise)
def piecewise(x, condlist, funclist, *args, **kw):
    # condlist may be boolean maskedarrays, mask is treated as False
    # masked elements in x stay masked in result.
    x = as_duck_cls(x, base=MaskedArray)

    n2 = len(funclist)

    # undocumented: single condition is promoted to a list of one condition
    if is_ndscalar(condlist) or (
            not isinstance(condlist[0], (list, np.ndarray, MaskedArray))
            and x.ndim != 0):
        condlist = [condlist]

    condlist = [c.filled(False) if isinstance(c, (MaskedArray, MaskedScalar))
                else c for c in condlist]

    condlist = np.array(condlist, dtype=bool)
    n = len(condlist)

    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
            .format(n, n, n+1)
        )

    # initialize output to all masked
    y = type(x)(np.empty(x.shape, x.dtype), True)
    for k in range(n):
        item = funclist[k]
        if not callable(item):
            y[condlist[k]] = item
        else:
            vals = x[condlist[k]]
            if vals.size > 0:
                y[condlist[k]] = item(vals, *args, **kw)

    return y

@implements(np.select, checked_args=lambda a,k,t,n: [type(x) for x in a[1]])
def select(condlist, choicelist, default=0):
    # choicelist may contain maskedarrays. Condlist must be unmasked
    # boolean  arrays. Note default is 0, not X!

    # Check the size of condlist and choicelist are the same, or abort.
    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    # Now that the dtype is known, handle the deprecated select([], []) case
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is "
                         "not possible")

    condlist = list(as_duck_cls(*condlist, base=MaskedArray, single=False))
    condlist = [c.filled() for c in condlist]

    choicelist = list(as_duck_cls(*choicelist, base=MaskedArray, single=False))
    cls = type(choicelist[0])

    # need to get the result type before broadcasting for correct scalar
    # behaviour
    if default is X:
        dtype = np.result_type(*choicelist)
        default = cls._scalartype(X, dtype=dtype)
        choicelist.append(default)
    else:
        choicelist.append(cls._scalartype(default))
        dtype = np.result_type(*choicelist)

    # Convert conditions to arrays and broadcast conditions and choices
    # as the shape is needed for the result. Doing it separately optimizes
    # for example when all choices are scalars.
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)

    # If cond array is not an ndarray in bool format or scalar bool, abort.
    deprecated_ints = False
    for i in range(len(condlist)):
        cond = condlist[i]
        if cond.dtype.type is not np.bool_:
            raise TypeError('invalid entry {} in condlist: '
                            'should be boolean ndarray'.format(i))

    if choicelist[0].ndim == 0:
        # This may be common, so avoid the call.
        result_shape = condlist[0].shape
    else:
        bcast = np.broadcast_arrays(condlist[0], choicelist[0]._data)
        result_shape = bcast[0].shape

    result = np.broadcast_to(choicelist[-1], result_shape).astype(dtype)

    # Use np.copyto to burn each choicelist array onto result, using the
    # corresponding condlist as a boolean mask. This is done in reverse
    # order since the first choice should take precedence.
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)

    return result

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = as_duck_cls(ar, base=MaskedArray)
    ar = ar.flatten()
    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    # argsort has put mask at end. As implementation hack, use the fact
    # that argsort/argsort used .filled(minval, view=True)
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    n_masked = np.sum(ar._mask)
    if n_masked > 0:
        # main change to account for mask: keep all but one mased elem
        mask[-n_masked:] = False
        mask[-n_masked] = True

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret

def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x

@implements(np.unique)
def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    # masked values are treated as a unique value

    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)

    # TODO: this might be implemented using lexsort
    raise NotImplementedError('axis argument to unique is not supported '
                              'for MaskedArray')

@implements(np.can_cast, checked_args=())
def can_cast(from_, to, casting='safe'):
    if isinstance(from_, (MaskedArray, MaskedScalar)):
        from_ = from_._data
    if isinstance(to, (MaskedArray, MaskedScalar)):
        to = to._data
    return np.can_cast(from_, to, casting)

@implements(np.min_scalar_type)
def min_scalar_type(a):
    # for masked scalars, just return the dtype
    if isinstance(a, MaskedScalar) and a.mask:
        return a.dtype
    if isinstance(a, (MaskedArray, MaskedScalar)):
        a = a._data
    return np.min_scalar_type(a)

@implements(np.result_type, checked_args=())
def result_type(*arrays_and_dtypes):
    dat = [a._data if isinstance(a, (MaskedArray, MaskedScalar)) else a
           for a in arrays_and_dtypes]
    return np.result_type(*dat)

@implements(np.common_type, checked_args=())
def common_type(*arrays_and_dtypes):
    dat = [a._data if isinstance(a, (MaskedArray, MaskedScalar)) else a
           for a in arrays_and_dtypes]
    return np.common_type(*dat)

@implements(np.bincount)
def bincount(x, weights=None, minlength=0):
    x = as_duck_cls(x, base=MaskedArray)
    return np.bincount(x._data[~x._mask], weights, minlength)

@implements(np.count_nonzero)
def count_nonzero(a, axis=None):
    a = as_duck_cls(a, base=MaskedArray)
    return np.count_nonzero(a.filled(0, view=1), axis)

@implements(np.nonzero)
def nonzero(a):
    a = as_duck_cls(a, base=MaskedArray)
    return np.nonzero(a.filled(0, view=1))

@implements(np.flatnonzero)
def flatnonzero(a):
    a = as_duck_cls(a, base=MaskedArray)
    return np.nonzero(np.ravel(a))[0]

@implements(np.histogram, checked_args=('a',))
def histogram(a, bins=10, range=None, normed=None, weights=None,
              density=None):
    a = as_duck_cls(a, base=MaskedArray)
    if isinstance(bins, (MaskedArray, MaskedScalar)):
        raise ValueError("bins must not be a MaskedArray")
    if isinstance(weights, (list, tuple)):
        weights = as_duck_cls(weights, base=MaskedArray)
    if isinstance(weights, (MaskedArray, MaskedScalar)):
        weights = weights.filled()

    a = a.ravel()
    keep = ~a._mask
    dat = a._data[keep]
    if weights is not None:
        weights = weights.ravel()[keep]

    return np.histogram(dat, bins, range, normed, weights, density)

@implements(np.histogram2d, checked_args=('x', 'y'))
def histogram2d(x, y, bins=10, range=None, normed=None, weights=None,
                density=None):
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        xedges = yedges = np.asarray(bins)  # bins should become ndarray
        bins = [xedges, yedges]
    hist, edges = histogramdd([x, y], bins, range, normed, weights, density)
    return hist, edges[0], edges[1]

@implements(np.histogramdd)
def histogramdd(sample, bins=10, range=None, normed=None, weights=None,
                density=None):
    if not np.isscalar(bins):
        for b in bins:
            if isinstance(b, (MaskedArray, MaskedScalar)):
                raise ValueError("bins must not be a MaskedArray")
    if isinstance(weights, (list, tuple)):
        weights = as_duck_cls(weights, base=MaskedArray)
    if isinstance(weights, (MaskedArray, MaskedScalar)):
        weights = weights.filled()

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = atleast_2d(sample).T
        N, D = sample.shape

    # drop any samples containing a masked value
    keep = ~np.any(sample._mask, axis=1)

    sample = sample._data[keep,...]
    if weights is not None:
        weights = weights[keep]

    return np.histogramdd(sample, bins, range, normed, weights, density)

@implements(np.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    a = a.ravel()
    keep = ~a._mask
    dat = a._data[keep]
    if weights is not None:
        weights = weights.ravel()[keep]
    return np.histogram_bin_edges(dat, bins, range, weights)

@implements(np.diff)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))

    inputs = [a, prepend, append]
    inputs = [i for i in inputs if is_ndtype(i)]
    cls = get_duck_cls(*inputs, base=MaskedArray)
    a = cls(a)

    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one "
                         "dimensional")
    axis = normalize_axis_index(axis, nd)

    combined = []
    if prepend is X:
        prepend = X(a.dtype)
    if prepend is not np._NoValue:
        prepend = cls(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is X:
        append = X(a.dtype)
    if append is not np._NoValue:
        append = cls(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = np.not_equal if a.dtype == np.bool_ else np.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a

def _interp_checkarg(args, kwds, types, known_types):
    if is_ndtype(args[1]) and not isinstance(args[1], np.ndarray):
        raise NotImplementedError
    a = [type(args[i]) for i in [0,2]]
    kw = [type(kwds[n]) for n in ['left', 'right'] if n in kwds]
    return [x for x in a+kw if is_ndtype(x)]

@implements(np.interp, checked_args=_interp_checkarg)
def interp(x, xp, fp, left=None, right=None, period=None):
    if isinstance(xp, (MaskedArray, MaskedScalar)):
        raise ValueError("xp may not be masked")

    # convert appropriate args to common masked class
    objs = [fp]
    if left is not None and left is not X:
        objs.append(left)
    if right is not None and right is not X:
        objs.append(right)
    cls = get_duck_cls(objs, base=MaskedArray)
    x = cls(x)
    fp = cls(fp)
    if left is X:
        left = cls._scalartype(X, dtype=fp.dtype)
    elif left is not None:
        left = cls._scalartype(left)
    if right is X:
        right = cls._scalartype(X, dtype=fp.dtype)
    elif right is not None:
        right = cls._scalartype(right)

    if np.iscomplexobj(fp):
        interp_func = compiled_interp_complex
        input_dtype = np.complex128
    else:
        interp_func = compiled_interp
        input_dtype = np.float64

    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None

        x = get_duck_cls(x, base=MaskedArray)(x, dtype=np.float64)
        xp = np.asarray(xp, dtype=np.float64)
        fp = fp.astype(input_dtype)

        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        # normalizing periodic boundaries
        x = x % period
        xp = xp % period
        asort_xp = np.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = np.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))

    leftd = None if left is None else left.filled(0)
    rightd = None if right is None else right.filled(0)
    xd = x.filled() if isinstance(x, (MaskedArray, MaskedScalar)) else x
    ret_data = interp_func(xd, xp, fp.filled(0, view=True), leftd, rightd)

    # we get interpolated mask using nan trick
    v = np.array([0, np.nan])
    leftm = None if left is None else v[left.mask.view('u1')]
    rightm = None if right is None else v[right.mask.view('u1')]
    xd[np.isnan(xd)] = 0
    ret_nanmask = interp_func(xd, xp, v[fp.mask.view('u1')], leftm, rightm)
    ret_mask = np.isnan(ret_nanmask)
    if isinstance(x, (MaskedArray, MaskedScalar)):
        ret_mask |= x.mask

    return cls(ret_data, ret_mask)

@implements(np.ediff1d)
def ediff1d(ary, to_end=None, to_begin=None):
    inputs = [ary]
    if to_end is not None:
        inputs.append(to_end)
    if to_begin is not None:
        inputs.append(to_begin)
    cls = get_duck_cls(*inputs, base=MaskedArray)

    # force a 1d array
    ary = cls(ary).ravel()

    # enforce propagation of the dtype of input
    # ary to returned result
    dtype_req = ary.dtype

    # fast track default case
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]

    if to_begin is None:
        l_begin = 0
    else:
        to_begin = cls(to_begin, dtype=dtype_req)
        if not np.can_cast(to_begin, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `ary` under the `same_kind` rule.")
        to_begin = to_begin.ravel()
        l_begin = len(to_begin)

    if to_end is None:
        l_end = 0
    else:
        to_end = cls(to_end, dtype=dtype_req)
        if not np.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `ary` under the `same_kind` rule.")
        to_end = to_end.ravel()
        l_end = len(to_end)

    # do the calculation in place and copy to_begin and to_end
    l_diff = builtins.max(len(ary) - 1, 0)
    result = np.empty(l_diff + l_begin + l_end, dtype=ary.dtype)
    result = cls(result)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    np.subtract(ary[1:], ary[:-1], result[l_begin:l_begin + l_diff])
    return result

@implements(np.gradient)
def gradient(f, *varargs, axis=None, edge_order=1):
    cls = get_duck_cls(*((f,) + varargs), base=MaskedArray)
    varargs = [cls(v) for v in varargs]
    f = cls(f)

    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = cls([1.0] * len_axes)
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = varargs
        for i, distances in enumerate(dx):
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).filled(False).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype
    if otype.type is np.datetime64:
        # the timedelta dtype with the same unit information
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        # view as timedelta to allow addition
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        # result allocation
        out = cls(np.empty_like(f, dtype=otype))

        # spacing for the current axis
        uniform_spacing = np.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = ((f[tuple(slice4)] - f[tuple(slice2)])
                                  / (2. * ax_dx))
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2)/(dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape

            D1 = a * f[tuple(slice2)]
            D2 = b * f[tuple(slice3)]
            D3 = c * f[tuple(slice4)]
            # if the middle prefactor is 0, we can ignore mask there.
            # This is the main change in this MaskedArray impl, so that uniform
            # scalar spacing gives same result as array of uniform spacing.
            D2[b == 0] = 0

            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = (D1 + D2 + D3)

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = - dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = (a * f[tuple(slice2)] +
                                  b * f[tuple(slice3)] +
                                  c * f[tuple(slice4)])

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = - (dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = (a * f[tuple(slice2)] +
                                  b * f[tuple(slice3)] +
                                  c * f[tuple(slice4)])

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


@implements(np.array2string)
def array2string(a, max_line_width=None, precision=None,
        suppress_small=None, separator=' ', prefix='', style=np._NoValue,
        formatter=None, threshold=None, edgeitems=None, sign=None,
        floatmode=None, suffix='', **kwarg):
    return duck_array2string(a, linewidth=max_line_width,
                                precision=precision,
                                suppress_small=suppress_small,
                                separator=separator,
                                prefix=prefix,
                                style=style,
                                formatter=formatter,
                                threshold=threshold,
                                edgeitems=edgeitems,
                                sign=sign,
                                floatmode=floatmode,
                                suffix=suffix,
                                **kwarg)

@implements(np.array_repr)
def array_repr(arr, max_line_width=None, precision=None,
               suppress_small=None):
    return duck_repr(arr, linewidth=max_line_width,
                          precision=precision,
                          suppress_small=suppress_small)

@implements(np.array_str)
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    return duck_str(a, linewidth=max_line_width,
                       precision=precision,
                       suppress_small=suppress_small)

@implements(np.shape)
def shape(a):
    a = as_duck_cls(a, base=MaskedArray)
    return a.shape

@implements(np.alen)
def alen(a):
    try:
        return len(a)
    except TypeError:
        return len(get_duck_cls(a, base=MaskedArray)(a, ndmin=1))

@implements(np.ndim)
def ndim(a):
    a = as_duck_cls(a, base=MaskedArray)
    return a.ndim

@implements(np.size)
def size(a):
    a = as_duck_cls(a, base=MaskedArray)
    return a.size

@implements(np.copyto, checked_args=('dst',))
def copyto(dst, src, casting='same_kind', where=True):
    if not is_ndtype(dst):
        raise TypeError('copyto() argument 1 must be ndarray, not list')
    dst, src = as_duck_cls(dst, src, base=MaskedArray, single=False)
    if isinstance(where, (MaskedArray, MaskedScalar)):
        where = where.filled(False)
    np.copyto(dst._data, src._data, casting, where)
    np.copyto(dst._mask, src._mask, casting, where)

@implements(np.putmask)
def putmask(a, mask, values):
    if not is_ndtype(a):
        raise TypeError('putmask() argument 1 must be ndarray, not list')
    a, values = as_duck_cls(a, values, base=MaskedArray, single=False)
    if isinstance(mask, MaskedArray):
        mask = mask.filled(False)
    np.putmask(a._data, mask, values._data)
    np.putmask(a._mask, mask, values._mask)

@implements(np.packbits)
def packbits(myarray, axis=None):
    myarray = as_duck_cls(myarray, base=MaskedArray)
    result_data = np.packbits(myarray._data, axis)
    result_mask = np.packbits(myarray._mask, axis) != 0
    return maskedarray_or_scalar(result_data, result_mask,cls=type(myarray))

@implements(np.unpackbits)
def unpackbits(myarray, axis=None):
    myarray = as_duck_cls(myarray, base=MaskedArray)
    result_data = np.unpackbits(myarray._data, axis)
    result_mask = np.unpackbits(myarray._mask*np.uint8(255), axis)
    return maskedarray_or_scalar(result_data, result_mask, cls=type(myarray))

@implements(np.isposinf)
def isposinf(x, out=None):
    x = as_duck_cls(x, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.isposinf(x._data, out=outdata)
    if outmask is None:
        outmask = x._mask.copy()
    else:
        outmask[...] = x._mask
    return maskedarray_or_scalar(result_data, outmask, out=out, cls=type(x))

@implements(np.isneginf)
def isneginf(x, out=None):
    x = as_duck_cls(x, base=MaskedArray)
    outdata, outmask = get_maskedout(out)
    result_data = np.isneginf(x._data, out=outdata)
    if outmask is None:
        outmask = x._mask.copy()
    else:
        outmask[...] = x._mask
    return maskedarray_or_scalar(result_data, outmask, out=out, cls=type(x))

@implements(np.iscomplex)
def iscomplex(x):
    if isinstance(x, (MaskedArray, MaskedScalar)):
        return type(x)(np.iscomplex(x._data), x._mask.copy())
    x = as_duck_cls(x, base=MaskedArray)
    return type(x)(np.iscomplex(x._data), x._mask)

@implements(np.isreal)
def isreal(x):
    if isinstance(x, (MaskedArray, MaskedScalar)):
        return type(x)(np.isreal(x._data), x._mask.copy())
    x = as_duck_cls(x, base=MaskedArray)
    return type(x)(np.isreal(x._data), x._mask.copy())

@implements(np.iscomplexobj)
def iscomplexobj(x):
    x = as_duck_cls(x, base=MaskedArray)
    return np.iscomplexobj(x._data)

@implements(np.isrealobj)
def isrealobj(x):
    x = as_duck_cls(x, base=MaskedArray)
    return np.isrealobj(x._data)

@implements(np.nan_to_num)
def nan_to_num(x, copy=True):
    if isinstance(x, (MaskedArray, MaskedScalar)):
        mask = x._mask.copy() if copy else x._mask
    else:
        x = as_duck_cls(x, base=MaskedArray)
        mask = x._mask
    return type(x)(np.nan_to_num(x._data, copy), mask)

@implements(np.real_if_close)
def real_if_close(a, tol=100):
    if isinstance(a, (MaskedArray, MaskedScalar)):
        return type(a)(np.real_if_close(a._data, tol), a._mask.copy())
    a = as_duck_cls(a, base=MaskedArray)
    return type(a)(np.real_if_close(a._data, tol), a._mask)

@implements(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    cls = get_duck_cls(a, b, base=MaskedArray)
    a, b = cls(a), cls(b)
    result_data = np.isclose(a._data, b._data, rtol, atol, equal_nan)
    result_mask = a._mask | b._mask
    return maskedarray_or_scalar(result_data, result_mask, cls=cls)

@implements(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return all(isclose(a, b, rtol, atol, equal_nan))

@implements(np.array_equal)
def array_equal(a1, a2, equal_nan=False):
    # interpret equal_nan to mean to also count masks as equal, so
    # np.array_equal(a,a) returns true no matter masks/nan. Otherwise
    # return False if there are masks (like with nan)
    a1, a2 = as_duck_cls(a1, a2, base=MaskedArray, single=False)
    if a1.shape != a2.shape:
        return False
    if equal_nan:
        ret = np.array_equal(a1.filled(view=1),
                             a2.filled(view=1), equal_nan=equal_nan)
        return ret and np.all(a1.mask == a2.mask)
    else:
        return np.all((a1 == a2).filled(False))

@implements(np.array_equiv)
def array_equiv(a1, a2):
    a1, a2 = as_duck_cls(a1, a2, base=MaskedArray, single=False)
    try:
        np.broadcast(a1._data, a2._data)
    except:
        return False
    return np.all((a1 == a2).filled(False))

@implements(np.sometrue)
def sometrue(*args, **kwargs):
    return any(*args, **kwargs)

@implements(np.alltrue)
def alltrue(*args, **kwargs):
    return all(*args, **kwargs)

@implements(np.angle)
def angle(z, deg=False):
    z = as_duck_cls(z, base=MaskedArray)
    if issubclass(z.dtype.type, np.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z

    a = np.arctan2(zimag, zreal)
    if deg:
        a *= 180/np.pi
    return a

@implements(np.sinc)
def sinc(x):
    x = as_duck_cls(x, base=MaskedArray)
    y = np.pi * np.where((x == 0).filled(False, view=1), 1.0e-20, x)
    return np.sin(y)/y

@implements(np.unwrap)
def unwrap(p, discont=np.pi, axis=-1):
    p = as_duck_cls(p, base=MaskedArray)
    pi = np.pi
    nd = p.ndim

    dd = diff(p, axis=axis)
    ddmod = np.mod(dd + pi, 2*pi) - pi
    copyto(ddmod, pi, where=(ddmod == -pi) & (dd > 0))
    ph_correct = ddmod - dd
    copyto(ph_correct, 0, where=abs(dd) < discont)
    up = type(p)(p, copy=True, dtype='d')

    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)

    return up



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

