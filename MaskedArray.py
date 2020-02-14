#!/usr/bin/env python
import numpy as np
from duckprint import (duck_str, duck_repr, duck_array2string, typelessdata,
    default_duckprint_options, default_duckprint_formatters, FormatDispatcher,
    is_ndducktype, is_duckscalar)
import builtins
import numpy.core.umath as umath
from numpy.lib.mixins import NDArrayOperatorsMixin
from ndarray_api_mixin import NDArrayAPIMixin
import numpy.core.numerictypes as ntypes
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import _broadcast_shape
import operator
import warnings

class MaskedOperatorMixin(NDArrayOperatorsMixin):
    # shared implementations for MaskedArray, MaskedScalar

    # override the NDArrayOperatorsMixin implementations for cmp ops, as
    # currently those don't work for flexible types.
    def _cmp_op(self, other, op):
        if other is X:
            db, mb = self._data.dtype.type(0), np.bool_(True)
        else:
            db, mb = getdata(other), getmask(other)

        cls = get_mask_cls(self, other)

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

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        impl, checked_args = HANDLED_FUNCTIONS[func]

        if checked_args is not None:
            types = (type(a) for n,a in enumerate(args) if n in checked_args)

        #types are allowed to be Masked* or plain ndarrays
        if not all((issubclass(t, self.known_types) or
                    t is np.ndarray or np.isscalar(t)) for t in types):
            return NotImplemented

        return impl(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in masked_ufuncs:
            return NotImplemented

        return getattr(masked_ufuncs[ufunc], method)(*inputs, **kwargs)

    ## This would be a nice way to convert to plain array, but seems
    ## to cause problems with __array_function__.
    #def __array__(self):
    #    return self.filled(view=True) # this returns a (readonly) view.

    def _get_fill_value(self, fill_value, minmax):
        if minmax is not None:
            if fill_value != np._NoValue:
                raise Exception("Do not give fill_value if providing minmax")
            if minmax == 'max':
                fill_value = _max_filler[self.dtype]
            elif minmax == 'min':
                fill_value = _min_filler[self.dtype]
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

def duck_require(data, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    # this function takes an nd-ducktype (data) and makes sure all the
    # other arguments normally supplied to np.array are satisfied.

    if copy or (dtype is not None and dtype != data.dtype):
        data = data.astype(dtype, order=order)
    # XXX deal with order and subok
    if ndmin is not None and data.ndim < ndmin:
        nd = ndmin - data.ndim
        data = data[(None,)*nd + (Ellipsis,)]

    return data

class MaskedArray(MaskedOperatorMixin, NDArrayAPIMixin):
    "An ndarray ducktype allowing array elements to be masked"

    def __init__(self, data, mask=None, dtype=None, copy=False,
                order=None, subok=True, ndmin=0, **options):
        """
        Constructs a MaskedArray given data and optional mask.

        Parameters
        ----------
        data : array-like
            Any object convertible to an `ndarray`, but also allowing
            the masked signifier `X` to mark masked elements. See Notes below.
        mask : array-like
            Any object convertible to a boolean `ndarray` of the same
            shape as data, where true elements are masked. If omitted, defaults
            to all `False`. See Notes below.
        dtype : data-type, optional
            The desired data-type for the array. See `np.array` argument.
        copy : cool, optional
            If false (default), the MaskedArray will view the data and mask
            if they are ndarrays with the right properties. Otherwise
            a they will be copied.
        order : {'K', 'A', 'C', 'F'}, optional
            Memory layout of the array. See `np.array` argument.
        subok : bool, optional
            Not yet fully supported.
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

        if isinstance(data, (MaskedArray, MaskedScalar)):
            self._mask = np.array(data._mask, copy=copy, order=order,
                                              subok=subok, ndmin=ndmin)

            if mask is not None:
                self._data = duck_require(data._data, copy=True, order=order,
                                                  subok=subok, ndmin=ndmin)
                self._mask |= np.array(mask, copy=False)

            else:
                self._data = duck_require(data._data, copy=copy, order=order,
                                                  subok=subok, ndmin=ndmin)
        elif data is X and mask is None:
            # 0d masked array
            if dtype is None:
                raise ValueError("must supply dtype if all elements are X")
            self._data = np.array(dtype.type(0))
            self._mask = np.array(True)
        else:
            if mask is None:
                # if mask is None, user can put X in the data.
                # Otherwise, X will cause some kind of error in np.array below
                data, mask = replace_X(data, dtype=dtype)

                # replace_X sometimes uses broadcast_to, which returns a
                # readonly array with funny strides. Make writeable if so,
                # since we will end up in the is_ndducktype code-path below.
                if (isinstance(mask, np.ndarray) and
                        mask.flags['WRITEABLE'] == False):
                    mask = mask.copy()

            if is_ndducktype(data):
                self._data = duck_require(data, copy=copy, order=order,
                                          subok=subok, ndmin=ndmin)
            else:
                self._data = np.array(data, dtype=dtype, copy=copy, order=order,
                                      subok=subok, ndmin=ndmin)

            # XXX need to make sure mask order is same as data?
            if mask is None:
                self._mask = np.zeros(self._data.shape, dtype='bool',
                                      order=order)
            elif (is_ndducktype(mask) and mask.shape == self._data.shape and
                    issubclass(mask.dtype.type, np.bool_)):
                self._mask = np.array(mask, copy=copy, order=order,
                                      subok=subok, ndmin=ndmin)
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
            return type(self)(self._data[ind], self._mask.copy())

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

        if np.isscalar(mask): # test mask, not data, to account for obj arrays
            return MaskedScalar(data, mask, dtype=self.dtype)
        return type(self)(data, mask)

    def __setitem__(self, ind, val):
        if is_string_or_list_of_strings(ind):
            view = type(self)(self._data[ind], self._mask.copy())
            view[...] = val
            return

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
        result_mask = self._mask.astype(bool, order, casting, subok, copy)
        # XXX compute copy in mask based on whether result_data is a copy?
        return type(self)(result_data, result_mask)

    def tolist(self):
        return [x.tolist() for x in self]

    def filled(self, fill_value=np._NoValue, minmax=None, view=False):
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
        elif data is X:
            if dtype is None:
                raise ValueError("Must supply dtype when data is X")
            if mask is not None:
                raise ValueError("don't supply mask when data is X")
            self._data = np.dtype(dtype).type(0)
            self._mask = np.bool_(True)
            self._dtype = self._data.dtype
        else:
            if dtype is not None:
                dtype = np.dtype(dtype)

            if dtype is None or dtype.type is not np.object_:
                if is_ndducktype(data) or is_duckscalar(data):
                    if dtype is not None and data.dtype != dtype:
                        data = data.astype(dtype, copy=False)[()]
                    self._data = data
                else:
                    self._data = np.array(data, dtype=dtype)[()]

                self._mask = np.bool_(mask)
                if not is_duckscalar(self._data) or not np.isscalar(self._mask):
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

    def __getitem__(self, ind):
        if (self.dtype.names and is_string_or_list_of_strings(ind) or
                isinstance(ind, int)):
            # like structured scalars, support string indexing and int indexing
            data = self._data[ind]
            mask = self._mask
            #XXX unclear if mask should be view or copy. See .real/.imag
            return type(self)(data, mask)

        if ind == ():
            return self

        raise IndexError("invalid index to scalar variable")

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
                # next line is more complicated than desired due to struct types,
                # which numpy does not have a constructor for
                return np.array(fill_value, dtype=self.dtype)[()]
                # XXX this will need to be modified to support structured duck-scalars.
                # In particular, need a way to construct arbitrary scalars "like"
                # the data scalar (same dtype and any other properties)
            return type(self._data)(fill_value)
        return self._data

MaskedOperatorMixin.ScalarType = MaskedScalar
MaskedOperatorMixin.ArrayType = MaskedArray
MaskedOperatorMixin.known_types = (MaskedArray, MaskedScalar)

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

# takes array-like input, replaces masked value by 0 and return filled data &
# mask. This is more-or-less a reimplementation of PyArray_DTypeFromObject to
# account for masked values
def replace_X(data, dtype=None):

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

        if is_ndducktype(data):
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

class MaskedIterator:
    def __init__(self, ma):
        self.dataiter = ma._data.flat
        self.maskiter = ma._mask.flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        data = self.dataiter.__getitem__(indx)
        mask = self.maskiter.__getitem__(indx)
        return maskedarray_or_scalar(data, mask, cls=type(data))

    def __setitem__(self, index, value):
        if value is X or (isinstance(value, MaskedScalar) and value.mask):
            self.maskiter[index] = True
        else:
            self.dataiter[index] = getdata(value)
            self.maskiter[index] = getmask(value)

    def __next__(self):
        return maskedarray_or_scalar(next(self.dataiter), next(self.maskiter),
                                     cls=type(seld))

    next = __next__

# carried over from numpy's MaskedArray, but naming is somewhat confusing
# as the max_filler is actually the minimum value. Change?
_max_filler = ntypes._minvals
_max_filler.update([(k, -np.inf) for k in [np.float16, np.float32, np.float64]])
_min_filler = ntypes._maxvals
_min_filler.update([(k, +np.inf) for k in [np.float16, np.float32, np.float64]])
if 'float128' in ntypes.typeDict:
    _max_filler.update([(np.float128, -np.inf)])
    _min_filler.update([(np.float128, +np.inf)])

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

        d = getdata(a)

        # mask computation before domain, in case domain is inplace
        if self.domain is None:
            m = getmask(a)
        else:
            m = self.domain(d) | getmask(a)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(d, *args, **kwargs)

        if out != ():
            out[0]._mask[...] = m
            return out[0]

        if is_duckscalar(result):
            return MaskedScalar(result, m)

        return type(a)(result, m)

class _Masked_BinOp(_Masked_UFunc):
    """
    Masked version of binary ufunc. Assumes 1 output.

    Parameters
    ----------
    ufunc : ufunc
        The ufunc for which to define a masked version.
    maskdomain : funcion, optional
        Function which returns true for inputs whose output should be masked.
    reduce_fill : function or scalar, optional
        Determines what fill_value is used during reductions. If a function is
        supplied, it shoud accept a dtype as argument and return a fill value
        with that dtype. A scalar value may also be supplied, which is used
        for all dtypes of the ufunc.
    """

    def __init__(self, ufunc, maskdomain=None, reduce_fill=None):
        super().__init__(ufunc)
        self.domain = maskdomain

        if reduce_fill is None:
            reduce_fill = ufunc.identity

        if (reduce_fill is not None and
                (is_duckscalar(reduce_fill) or not callable(reduce_fill))):
            self.reduce_fill = lambda dtype: reduce_fill
        else:
            self.reduce_fill = reduce_fill

    def __call__(self, a, b, **kwargs):
        da, db = getdata(a), getdata(b)
        ma, mb = getmask(a), getmask(b)

        # treat X as a masked value of the other array's dtype
        if da is X:
            da, ma = db.dtype.type(0), np.bool_(True)
        if db is X:
            db, mb = da.dtype.type(0), np.bool_(True)

        mkwargs = {}
        for k in ['where', 'order']:
            if k in kwargs:
                mkwargs[k] = kwargs[k]

        out = kwargs.get('out', ())
        if out:
            if not isinstance(out[0], MaskedArray):
                raise ValueError("out must be a MaskedArray")
            kwargs['out'] = (out[0]._data,)
            mkwargs['out'] = (out[0]._mask,)

        m = np.logical_or(ma, mb, **mkwargs)
        if self.domain is not None:
            if 'out' not in mkwargs and isinstance(m, np.ndarray):
                mkwargs['out'] = (m,)
            m = np.logical_or(m, self.domain(da, db), **mkwargs)
            #XXX add test that this gets where right

        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, **kwargs)

        if out:
            return out[0]

        cls = get_mask_cls(a, b)
        if is_duckscalar(result):
            return cls.ScalarType(result, m)
        return cls(result, m)

    def reduce(self, a, **kwargs):
        if self.domain is not None:
            raise TypeError("domained ufuncs do not support reduce")
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
            if not is_duckscalar(da):
                da[ma] = self.reduce_fill(da.dtype)
                # if da is a scalar, we get correct result no matter fill

            result = self.f.reduce(da, **kwargs)
            m = np.logical_and.reduce(ma, **mkwargs)

        ## Code that might be used to support domained ufuncs. WIP
        #with np.errstate(divide='ignore', invalid='ignore'):
        #    result = self.f.reduce(da, **kwargs)
        #    m = np.logical_and.reduce(ma, **mkwargs)
        #if self.domain is not None:
        #    m = np.logical_or(ma, dom, **mkwargs)

        if out:
            return out[0]

        cls = get_mask_cls(a)
        if is_duckscalar(result):
            return cls.ScalarType(result, m)
        return cls(result, m)

    def accumulate(self, a, axis=0, dtype=None, out=None):
        if self.domain is not None:
            raise RuntimeError("domained ufuncs do not support reduce")
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

        if not is_duckscalar(da):
            da[ma] = self.reduce_fill(da.dtype)
        result = self.f.accumulate(da, axis, dtype, dataout)
        m = np.logical_and.accumulate(ma, axis, out=maskout)

        ## Code that might be used to support domained ufuncs. WIP
        #with np.errstate(divide='ignore', invalid='ignore'):
        #    result = self.f.accumulate(da, axis, dtype, dataout)
        #    m = np.logical_and.accumulate(ma, axis, out=maskout)
        #if self.domain is not None:
        #    dom = self.domain(result[...,:-1], da[...,1:])
        #    m = np.logical_or(ma, dom, **mkwargs)

        if out:
            return out[0]
        if is_duckscalar(result):
            return MaskedScalar(result, m)
        return type(a)(result, m)

    def outer(self, a, b, **kwargs):
        if self.domain is not None:
            raise RuntimeError("domained ufuncs do not support reduce")
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

        if not is_duckscalar(da):
            da[ma] = self.reduce_fill(da.dtype)
        if not is_duckscalar(db):
            db[mb] = self.reduce_fill(db.dtype)

        result = self.f.outer(da, db, **kwargs)
        m = np.logical_or.outer(ma, mb, **mkwargs)

        if out:
            return out[0]
        if is_duckscalar(result):
            return MaskedScalar(result, m)
        return type(a)(result, m)

    def reduceat(self, a, indices, **kwargs):
        if self.domain is not None:
            raise TypeError("domained ufuncs do not support reduce")
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

        if not is_duckscalar(da):
            da[ma] = self.reduce_fill(da.dtype)
            # if da is a scalar, we get correct result no matter fill

        result = self.f.reduceat(da, indices, **kwargs)
        m = np.logical_and.reduceat(ma, indices, **mkwargs)

        if out:
            return out[0]
        if is_duckscalar(result):
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

        with np.errstate(divide='ignore', invalid='ignore'):
            self.f.at(da, indices, db)
            np.logical_or.at(ma, indices, mb)

def maskdom_divide(a, b):
    out_dtype = np.result_type(a, b)

    # if floating, use finfo to determine domain
    if isinstance(out_dtype, np.inexact):
        tolerance = np.finfo(out_dtype).tiny
        with np.errstate(invalid='ignore'):
            return umath.absolute(a) * tolerance >= umath.absolute(b)

    # otherwise, for integer types, only 0 is a problem
    return b == 0

def maskdom_power(a, b):
    out_dtype = np.result_type(a, b)
    if issubclass(out_dtype.type, np.floating):
        # non-integer powers of negative floats not allowed
        # negative powers of 0 not allowed
        return ((a < 0) & (b != np.rint(b))) | ((a == 0) & (b < 0))
    if issubclass(out_dtype.type, np.integer):
        # integers to negative powers not allowed
        # Note: the binop actually raises a ValueError, so this test is redundant
        return b < 0
    return np.broadcast_to(False, a.shape)

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

def maskdom_tan(x):
    #Use original MaskedArrays's strategy. But might be improved by using finfo
    with np.errstate(invalid='ignore'):
        return umath.less(umath.absolute(umath.cos(x)), 1e-35)

def make_maskdom_interval(lo, hi):
    def maskdom(x):
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
                  umath.isinf, umath.isnan, umath.invert]:
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
    # XXX The last lines use the original MaskedArrays's strategy of hardcoded
    # limits. But would be nice to improve by adding float-specific limits
    # (diff for float32 vs float64) using finfo.
    # XXX document these limits and behavior

    # binary ufuncs
    for ufunc in [umath.add, umath.subtract, umath.multiply,
                  umath.arctan2, umath.hypot, umath.equal, umath.not_equal,
                  umath.less_equal, umath.greater_equal, umath.less,
                  umath.greater, umath.logical_and, umath.logical_or,
                  umath.logical_xor, umath.bitwise_and, umath.bitwise_or,
                  umath.bitwise_xor]:
        masked_ufuncs[ufunc] = _Masked_BinOp(ufunc)

    # fill value depends on dtype
    masked_ufuncs[umath.maximum] = _Masked_BinOp(umath.maximum,
                                         reduce_fill=lambda dt: _max_filler[dt])
    masked_ufuncs[umath.minimum] = _Masked_BinOp(umath.minimum,
                                         reduce_fill=lambda dt: _min_filler[dt])

    # domained binary ufuncs
    for ufunc in [umath.true_divide, umath.floor_divide, umath.remainder,
                  umath.fmod, umath.mod]:
        masked_ufuncs[ufunc] = _Masked_BinOp(ufunc, maskdom_divide)
    masked_ufuncs[umath.power] = _Masked_BinOp(umath.power, maskdom_power)

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

def get_mask_cls(*args):
    """
    Helper to make MaskedArray Subclass-friendly.

    Finds the most derived class of MaskedArray/MaskedScalar.
    If given both an Array and a Scalar, convert the Scalar to an array first.
    In the case of two non-inheriting subclasses, raise TypeError.
    """
    cls = None
    for arg in args:
        if isinstance(arg, (MaskedArray, MaskedScalar)):
            acl = type(arg)
            if cls is None or issubclass(acl, cls):
                cls = acl
                continue
            elif issubclass(cls, MaskedScalar) and issubclass(acl, MaskedArray):
                cls = cls.ArrayType
            elif issubclass(acl, MaskedScalar) and issubclass(cls, MaskedArray):
                acl = acl.ArrayType

            if issubclass(acl, cls):
                cls = acl
            elif not issubclass(cls, acl):
                raise TypeError(("Ambiguous mix of MaskedArray subtypes {} and "
                                "{}").format(cls, acl))
        elif isinstance(arg, (list, tuple)):
            tmpcls = get_mask_cls(*arg)
            if tmpcls is not None and (cls is None or issubclass(cls, tmpcls)):
                cls = tmpcls

    if cls is None:
        return None
    return cls.ArrayType

def maskedarray_or_scalar(data, mask, out=None, cls=MaskedArray):
    if out is not None:
        return out
    if is_duckscalar(data):
        return cls.ScalarType(data, mask)
    return cls.ArrayType(data, mask)

def get_maskedout(out):
    if out is not None:
        if isinstance(out, MaskedArray):
            return out._data, out._mask
        raise Exception("out must be a masked array")
    return None, None

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

def setup_ducktype():
    @implements(np.all)
    def all(a, axis=None, out=None, keepdims=np._NoValue):
        if isinstance(out, MaskedArray):
            np.all(a.filled(True, view=1), axis, out._data, keepdims)
            out._mask[...] = False
            return out
        return np.all(a.filled(True, view=1), axis, out, keepdims)
        # Note: returns boolean, not MaskedArray. If case of fully masked,
        # return True, like np.all([]).

    @implements(np.any)
    def any(a, axis=None, out=None, keepdims=np._NoValue):
        if isinstance(out, MaskedArray):
            np.any(a.filled(False, view=1), axis, out._data, keepdims)
            out._mask[...] = False
            return out
        return np.any(a.filled(False, view=1), axis, out, keepdims)
        # Note: returns boolean, not MaskedArray. If case of fully masked,
        # return False, like np.any([])

    @implements(np.max)
    def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(minmax='max', view=1)
        result_data = np.max(filled, axis, outdata, keepdims, initial)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.argmax)
    def argmax(a, axis=None, out=None):
        if isinstance(out, MaskedArray):
            raise TypeError("out argument of argmax should be an ndarray")
        filled = a.filled(minmax='max', view=1)
        result_data = np.argmax(filled, axis, out)
        return result_data

    @implements(np.min)
    def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue):
        outdata, outmask = get_maskedout(out)
        filled = a.filled(minmax='min', view=1)
        result_data = np.min(filled, axis, outdata, keepdims, initial)
        result_mask = np.all(a._mask, axis, outmask, keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.argmin)
    def argmin(a, axis=None, out=None):
        if isinstance(out, MaskedArray):
            raise TypeError("out argument of argmax should be an ndarray")
        filled = a.filled(minmax='min', view=1)
        result_data = np.argmin(filled, axis, out)
        return result_data

    @implements(np.sort)
    def sort(a, axis=-1, kind='quicksort', order=None):
        # Note: This is trickier than it looks. The first line sorts the mask
        # together with any min_vals which may be present, so there appears to
        # be a problem ordering mask vs min_val elements.
        # But, since we know all the masked elements have to end up at the end
        # of the axis, we can sort the mask too and everything works out. The
        # mask-sort only swaps the mask between min_val and masked positions
        # which have the same underlying data.
        result_data = np.sort(a.filled(minmax='min', view=1), axis, kind, order)
        result_mask = np.sort(a._mask, axis, kind)  #or partition for speed?
        return maskedarray_or_scalar(result_data, result_mask, cls=type(a))
        # Note: lexsort may be faster, but doesn't provide kind or order kwd

    @implements(np.argsort)
    def argsort(a, axis=-1, kind='quicksort', order=None):
        # Similar to mask-sort trick in sort above, here after sorting data we
        # re-sort based on mask. Use the property that if you argsort the index
        # array produced by argsort you get the element rank, which can be
        # argsorted again to get back the sort indices. However, here we
        # modify the rank based on the mask before inverting back to indices.
        # Uses two argsorts plus a temp array.
        inds = np.argsort(a.filled(minmax='min', view=1), axis, kind, order)
        # next two lines "reverse" the argsort (same as double-argsort)
        ranks = np.empty(inds.shape, dtype=inds.dtype)
        np.put_along_axis(ranks, inds, np.arange(a.shape[axis]), axis)
        # prepare to resort but make masked elem highest rank
        ranks[a._mask] = _min_filler[ranks.dtype]
        return np.argsort(ranks, axis, kind)

    @implements(np.argpartition)
    def argpartition(a, kth, axis=-1, kind='introselect', order=None):
        # see argsort for explanation
        filled = a.filled(minmax='min', view=1)
        inds = np.argpartition(filled, kth, axis, kind, order)
        ranks = np.empty(inds.shape, dtype=inds.dtype)
        np.put_along_axis(ranks, inds, np.arange(a.shape[axis]), axis)
        ranks[a._mask] = _min_filler[ranks.dtype]
        return np.argpartition(ranks, kth, axis, kind)

    @implements(np.searchsorted, checked_args=(1,))
    def searchsorted(a, v, side='left', sorter=None):

        if isinstance(a, MaskedArray):
            maskleft = len(a) - np.sum(a._mask)
            aval = a.filled(minmax='min', view=1)
        else:  # plain ndarray
            maskleft = len(a)
            aval = a

        inds = np.searchsorted(aval, v.filled(minmax='min', view=1),
                               side, sorter)

        # Line above treats mask and minval as the same, we need to fix it up
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

        ret = np.sum(a.filled(0, view=1), axis=axis, out=outdata, dtype=dtype,
                     **kwargs)
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
        ret = x.filled(0, view=1).sum(axis, dtype, out=outdata, **kwargs)

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

        if out is not None:
            out[rcount == 0] = X
            return out
        return maskedarray_or_scalar(ret, rcount == 0, cls=type(a))

    @implements(np.std)
    def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        ret = np.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                     keepdims=keepdims)

        if isinstance(ret, MaskedArray):
            ret = np.sqrt(ret, out=ret)
        elif hasattr(ret, 'dtype'):
            ret = np.sqrt(ret).astype(ret.dtype)
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
            wgt = type(wgt)(wgt._data, mask)
            a = type(a)(a._data, mask)
        else:
            wgt = type(wgt)(wgt, a._mask)

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
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.compress)
    def compress(condition, a, axis=None, out=None):
        # Note: masked values in condition treated as False
        outdata, outmask = get_maskedout(out)
        cls = get_mask_cls(condition, a)
        cond = cls(condition).filled(False, view=1)
        a = cls(a)
        result_data = np.compress(cond, a._data, axis, outdata)
        result_mask = np.compress(cond, a._mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, out, cls)

    @implements(np.copy)
    def copy(a, order='K'):
        result_data = np.copy(a._data, order=order)
        result_mask = np.copy(a._mask, order=order)
        return maskedarray_or_scalar(result_data, result_mask, cls=type(a))

    @implements(np.prod)
    def prod(a, axis=None, dtype=None, out=None, keepdims=False):
        outdata, outmask = get_maskedout(out)
        result_data = np.prod(a.filled(1, view=1), axis=axis, dtype=dtype,
                              out=outdata, keepdims=keepdims)
        result_mask = np.all(a._mask, axis=axis, out=outmask, keepdims=keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.product)
    def product(*args, **kwargs):
        return prod(*args, **kwargs)

    @implements(np.cumprod)
    def cumprod(a, axis=None, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.cumprod(a.filled(1, view=1), axis, dtype=dtype,
                                 out=outdata)
        result_mask = np.logical_or.accumulate(~a._mask, axis, out=outmask)
        result_mask =_inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.cumproduct)
    def cumproduct(*args, **kwargs):
        return cumprod(*args, **kwargs)

    @implements(np.sum)
    def sum(a, axis=None, dtype=None, out=None, keepdims=False):
        outdata, outmask = get_maskedout(out)
        result_data = np.sum(a.filled(0, view=1), axis, dtype=dtype,
                             out=outdata, keepdims=keepdims)
        result_mask = np.all(a._mask, axis, out=outmask, keepdims=keepdims)
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.cumsum)
    def cumsum(a, axis=None, dtype=None, out=None):
        outdata, outmask = get_maskedout(out)
        result_data = np.cumsum(a.filled(0, view=1), axis, dtype=dtype,
                                out=outdata)
        result_mask = np.logical_or.accumulate(~a._mask, axis, out=outmask)
        result_mask =_inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out, type(a))

    @implements(np.diagonal)
    def diagonal(a, offset=0, axis1=0, axis2=1):
        result = np.diagonal(a._data, offset=offset, axis1=axis1, axis2=axis2)
        rmask = np.diagonal(a._mask, offset=offset, axis1=axis1, axis2=axis2)
        # Unlike np.diagonal, we make a copy to make it writeable, so "filled"
        # works on it.
        return maskedarray_or_scalar(result, rmask, cls=type(a))

    @implements(np.diag)
    def diag(v, k=0):
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
        result = np.trace(a.filled(0, view=1), offset=offset, axis1=axis1,
                          axis2=axis2, dtype=dtype, out=outdata)
        mask_trace = np.trace(~a._mask, offset=offset, axis1=axis1, axis2=axis2,
                                        dtype=dtype, out=outdata)
        result_mask = mask_trace == 0
        return maskedarray_or_scalar(result, result_mask, cls=type(a))

    @implements(np.dot)
    def dot(a, b, out=None):
        outdata, outmask = get_maskedout(out)
        cls = get_mask_cls(a, b)
        a, b = cls(a), cls(b)
        result_data = np.dot(a.filled(0, view=1), b.filled(0, view=1),
                             out=outdata)
        result_mask = np.dot(~a._mask, ~b._mask, out=outmask)
        result_mask = _inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out, cls)

    @implements(np.vdot)
    def vdot(a, b):
        cls = get_mask_cls(a, b)
        a, b = cls(a), cls(b)
        result_data = np.vdot(a.filled(0, view=1), b.filled(0, view=1))
        result_mask = np.vdot(~a._mask, ~b._mask)
        result_mask = _inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.cross)
    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        cls = get_mask_cls(a, b)
        a, b = cls(a), cls(b)
        result_data = np.cross(a.filled(0, view=1), b.filled(0, view=1), axisa,
                               axisb, axisc, axis)
        result_mask = np.cross(~a._mask, ~b._mask, axisa, axisb, axisc, axis)
        result_mask = _inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.inner)
    def inner(a, b):
        cls = get_mask_cls(a, b)
        a, b = cls(a), cls(b)
        result_data = np.inner(a.filled(0, view=1), b.filled(0, view=1))
        result_mask = np.inner(~a._mask, ~b._mask)
        result_mask = _inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.outer)
    def outer(a, b, out=None):
        outdata, outmask = get_maskedout(out)
        cls = get_mask_cls(a, b)
        a, b = cls(a), cls(b)
        result_data = np.outer(a.filled(0, view=1), b.filled(0, view=1),
                               out=outdata)
        result_mask = np.outer(~a._mask, ~b._mask, out=outmask)
        result_mask = _inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out, cls)

    @implements(np.kron)
    def kron(a, b):
        cls = get_mask_cls(a, b)
        a = cls(a, copy=False, subok=True, ndmin=b.ndim)
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
        cls = get_mask_cls(operands)

        result_data = np.einsum(data, out=outdata, **kwargs)
        result_mask = np.einsum(nmask, out=outmask, **kwargs)
        result_mask = _inplace_not(result_mask)
        return maskedarray_or_scalar(result_data, result_mask, out, cls)

    #@implements(np.einsum_path)

    @implements(np.correlate)
    def correlate(a, v, mode='valid'):
        cls = get_mask_cls(a, v)
        result_data = np.correlate(a.filled(view=1), v.filled(view=1), mode)
        result_mask = ~np.correlate(~a._mask, v._mask, mode)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.convolve)
    def convolve(a, v, mode='full'):
        cls = get_mask_cls(a, v)
        a, v = cls(a), cls(v)
        result_data = np.convolve(a.filled(view=1), v.filled(view=1), mode)
        result_mask = ~np.convolve(~a._mask, ~v._mask, mode)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.real)
    def real(a):
        result_data = np.real(a._data)
        result_mask = a._mask.copy()
        return maskedarray_or_scalar(result_data, result_mask, cls=type(a))

    @implements(np.imag)
    def imag(a):
        result_data = np.imag(a._data)
        result_mask = a._mask.copy()
        return maskedarray_or_scalar(result_data, result_mask, cls=type(a))

    @implements(np.partition)
    def partition(a, kth, axis=-1, kind='introselect', order=None):
        inds = np.argpartition(a, kth, axis, kind, order)
        return np.take_along_axis(a, inds, axis=axis)

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
        data, mask = replace_X(values, dtype=a.dtype)
        np.put(a._data, indices, data, mode)
        np.put(a._mask, indices, mask, mode)
        return None

    @implements(np.take_along_axis, checked_args=(0,))
    def take_along_axis(arr, indices, axis):
        result_data = np.take_along_axis(arr._data, indices, axis)
        result_mask = np.take_along_axis(arr._mask, indices, axis)
        return maskedarray_or_scalar(result_data, result_mask, cls=type(arr))

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
        return type(a)(np.ravel(a._data, order=order),
                       np.ravel(a._mask, order=order))

    @implements(np.repeat)
    def repeat(a, repeats, axis=None):
        return type(a)(np.repeat(a._data, repeats, axis),
                       np.repeat(a._mask, repeats, axis))

    @implements(np.reshape)
    def reshape(a, shape, order='C'):
        return type(a)(np.reshape(a._data, shape, order=order),
                       np.reshape(a._mask, shape, order=order))

    @implements(np.resize)
    def resize(a, new_shape):
        return type(a)(np.resize(a._data, new_shape),
                       np.resize(a._mask, new_shape))

    @implements(np.meshgrid)
    def meshgrid(*xi, **kwargs):
        cls = get_mask_cls(xi)
        data, mask = zip(*((x._data, x._mask) for x in xi))
        result_data = np.meshgrid(data, **kwargs)
        result_mask = np.meshgrid(mask, **kwargs)
        return maskedarray_or_scalar(result_data, result_mask, cls)

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
        outdata, outmask = get_maskedout(out)
        res = np.ceil(x, out=out)
        res = np.floor(x, out=res, where=np.greater_equal(x, 0))
        return out or res

    @implements(np.squeeze)
    def squeeze(a, axis=None):
        return type(a)(np.squeeze(a._data, axis),
                       np.squeeze(a._mask, axis))

    @implements(np.swapaxes)
    def swapaxes(a, axis1, axis2):
        return type(a)(np.swapaxes(a._data, axis1, axis2),
                       np.swapaxes(a._mask, axis1, axis2))

    @implements(np.transpose)
    def transpose(a, *axes):
        return type(a)(np.transpose(a._data, *axes),
                       np.transpose(a._mask, *axes))

    @implements(np.roll)
    def roll(a, shift, axis=None):
        return type(a)(np.roll(a._data, shift, axis),
                       np.roll(a._mask, shift, axis))

    @implements(np.rollaxis)
    def rollaxis(a, axis, start=0):
        return type(a)(np.rollaxis(a._data, axis, start),
                       np.rollaxis(a._mask, axis, start))

    @implements(np.moveaxis)
    def moveaxis(a, source, destination):
        return type(a)(np.moveaxis(a._data, source, destination),
                       np.moveaxis(a._mask, source, destination))

    @implements(np.flip)
    def flip(m, axis=None):
        return type(a)(np.flip(m._data, axis),
                       np.flip(m._mask, axis))

    #@implements(np.rot90)
    #def rot90(m, k=1, axes=(0,1)):
    #    # XXX copy code from np.rot90 but remove asarray

    #@implements(np.fliplr)
    #@implements(np.flipud)

    @implements(np.expand_dims)
    def expand_dims(a, axis):
        return type(a)(np.expand_dims(a._data, axis),
                           np.expand_dims(a._mask, axis))

    @implements(np.concatenate)
    def concatenate(arrays, axis=0, out=None):
        outdata, outmask = get_maskedout(out)
        cls = get_mask_cls(arrays)
        arrays = [cls(a) for a in arrays] # XXX may need tweaking
        data, mask = zip(*((x._data, x._mask) for x in arrays))
        result_data = np.concatenate(data, axis, outdata)
        result_mask = np.concatenate(mask, axis, outmask)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.block)
    def block(arrays):
        data, mask = replace_X(arrays)
        result_data = np.block(data)
        result_mask = np.block(mask)
        cls = get_mask_cls(arrays)
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.column_stack)
    def column_stack(tup):
        cls = get_mask_cls(tup)
        arrays = []
        for v in tup:
            arr = cls(v, copy=False, subok=True)
            if arr.ndim < 2:
                arr = cls(arr, copy=False, subok=True, ndmin=2).T
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

    @implements(np.atleast_1d)
    def atleast_1d(*arys):
        res = []
        for ary in arys:
            #removed: ary = asanyarray(ary)
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
            #removed: ary = asanyarray(ary)
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
            #removed: ary = asanyarray(ary)
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
        return type(arr)(np.delete(arr._data, obj, axis),
                         np.delete(arr._mask, obj, axis))

    @implements(np.insert)
    def insert(arr, obj, values, axis=None):
        return type(arr)(np.insert(arr._data, obj, values, axis),
                         np.insert(arr._mask, obj, values, axis))

    @implements(np.append)
    def append(arr, values, axis=None):
        cls = get_mask_cls(arr, values)
        arr, values = cls(arr), cls(values)
        return cls(np.append(arr._data, values._data, axis),
                   np.append(arr._mask, values._mask, axis))

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
        return type(array)(np.broadcast_to(array._data, shape),
                           np.broadcast_to(array._mask, shape))

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
        cls = type(prototype)
        return cls(np.empty_like(prototype._data, dtype, order, subok))

    @implements(np.ones_like)
    def ones_like(prototype, dtype=None, order='K', subok=True):
        cls = type(prototype)
        return cls(np.ones_like(prototype._data, dtype, order, subok))

    @implements(np.zeros_like)
    def zeros_like(prototype, dtype=None, order='K', subok=True):
        cls = type(prototype)
        return cls(np.zeros_like(prototype._data, dtype, order, subok))

    @implements(np.full_like)
    def full_like(a, fill_value, dtype=None, order='K', subok=True):
        return cls(np.full_like(a._data, fill_value, dtype, order,
                           subok))

    @implements(np.where)
    def where(condition, x=None, y=None):
        if x is None and y is None:
            return np.nonzero(condition)

        cls = get_mask_cls(condition, x, y)

        # convert x, y to MaskedArrays, using the other's dtype if one is X
        if x is X:
            if y is X:
                # why would anyone do this? But it is in unit tests, so...
                raise ValueError("must supply dtype if x and y are both X, "
                                 "eg using X(dtype)")
            y = cls(y)
            x = cls(X, dtype=y.dtype)
        elif y is X:
            x = cls(x)
            y = cls(X, dtype=x.dtype)
        else:
            y = cls(y)
            x = cls(x)

        if isinstance(condition, (MaskedArray, MaskedScalar)):
            condition = condition.filled(False, view=1)

        result_data = np.where(condition, *(a._data for a in (x, y)))
        result_mask = np.where(condition, *(a._mask for a in (x, y)))

        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.argwhere)
    def argwhere(a):
        return np.transpose(np.nonzero(a))

    @implements(np.choose, checked_args=(1,))
    def choose(a, choices, out=None, mode='raise'):
        if isinstance(a, (MaskedArray, MaskedScalar)):
            raise TypeError("choice indices should not be masked")

        outdata, outmask = get_maskedout(out)
        result_data = np.choose(a, choices._data, outdata, mode)
        result_mask = np.choose(a, choices._mask, outmask, mode)
        return maskedarray_or_scalar(result_data, result_mask, out, type(choices))

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
        return np.count_nonzero(a.filled(0, view=1), axis)

    @implements(np.nonzero)
    def nonzero(a):
        return np.nonzero(a.filled(0, view=1))

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
        return len(type(a)(a, ndmin=1))

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
        return maskedarray_or_scalar(result_data, result_mask, cls=type(myarray))

    @implements(np.unpackbits)
    def unpackbits(myarray, axis=None):
        result_data = np.unpackbits(myarray._data, axis)
        result_mask = np.unpackbits(myarray._mask*np.uint8(255), axis)
        return maskedarray_or_scalar(result_data, result_mask, cls=type(myarray))

    @implements(np.isposinf)
    def isposinf(x, out=None):
        return type(x)(np.isposinf(x._data), x._mask.copy())

    @implements(np.isneginf)
    def isneginf(x, out=None):
        return type(x)(np.isneginf(x._data), x._mask.copy())

    @implements(np.iscomplex)
    def iscomplex(x):
        return type(x)(np.iscomplex(x._data), x._mask.copy())

    @implements(np.isreal)
    def isreal(x):
        return type(x)(np.isreal(x._data), x._mask.copy())

    @implements(np.iscomplexobj)
    def iscomplexobj(x):
        return np.iscomplexobj(x._data)

    @implements(np.isrealobj)
    def isrealobj(x):
        return np.isrealobj(x._data)

    @implements(np.nan_to_num)
    def nan_to_num(x, copy=True):
        return type(x)(np.nan_to_num(x._data, copy),
                           x._mask.copy() if copy else x._mask)

    @implements(np.real_if_close)
    def real_if_close(a, tol=100):
        return type(a)(np.real_if_close(x._data, tol), x._mask.copy())

    @implements(np.isclose)
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        cls = get_mask_cls(a, b)
        a, b = cls(a), cls(b)
        result_data = np.isclose(a._data, b._data, rtol, atol, equal_nan)
        result_mask = a._mask | b._mask
        return maskedarray_or_scalar(result_data, result_mask, cls=cls)

    @implements(np.allclose)
    def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return np.all(np.isclose(a, b, rtol, atol, equal_nan))

    @implements(np.array_equal)
    def array_equal(a1, a2):
        return np.all(a1 == a2)

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

        a = np.arctan2(zimag, zreal)
        if deg:
            a *= 180/np.pi
        return a

    @implements(np.sinc)
    def sinc(x):
        y = np.pi * np.where((x == 0).filled(False, view=1), 1.0e-20, x)
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
