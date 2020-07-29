#!/usr/bin/env python
import numpy as np
import warnings
from .duckprint import duck_repr, duck_str
from .common import is_ndtype
from .ndarray_api_mixin import NDArrayAPIMixin
import sys
import operator
from functools import reduce

# we use python 3.7+ dict order guarantee
if sys.version_info < (3,7):
    raise RuntimeError("ArrayCollection requires Python 3.7+")

class CollectionMixin(NDArrayAPIMixin):
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        impl, checked_args = HANDLED_FUNCTIONS[func]

        if checked_args is not None:
            types = (type(a) for n,a in enumerate(args) if n in checked_args)

        #types are allowed to be Masked* or plain ndarrays
        if not all((issubclass(t, (ArrayCollection, CollectionScalar)) or
                    t is np.ndarray) for t in types):
            return NotImplemented

        return impl(*args, **kwargs)

    #def __array_ufunc__():
    #    # TODO: ArrayCollection supports only one ufunc: np.equal
    # XXX actually, why don;t we try to support more?

def _asarraylike(val, dtype=None):
    if is_ndtype(val):
        if dtype is not None:
            return val.astype(dtype)
        return val
    return np.array(val, dtype=dtype)


class ArrayCollection(CollectionMixin):
    """
    NDarray-like type which stores a set of named arrays with a common shape
    but which may be different dtypes. Allows numpy-style indexing of them as a
    group. An ArrayCollection looks and behaves very similarly to a structured
    ndarray, but will often have better performance because it has better
    memory layout.

    Can be thought of as a "lite" version of pandas dataframe, or Xarray
    dataset.  Look to those projects if more features are desired, such
    as named axes.

    Tip: One can use `np.broadcast_to` to broadcast the input arrays to the
    same shape in a memory-saving way, though this makes the arrays readonly.
    See the `np.broadcast_to` docstring for details.

    """
    def __init__(self, data, dtype=None, skip_validation=False):
        """
        Parameters
        ----------
        data : various inputs allowed
            Data to make a collection from. Accepts multiple forms of input:
            1. Dict of form "{name: arr}" where "name" is the fieldname and
               "arr" is a numpy array of data, which is viewed.
            2. List of tuples of form "(name: arr)", with the same meaning
               as for the dict input.
            3. An ndarray with structured dtype. The fieldnames are taken from
               the dtype, and the data arrays are copied. Nested structured
               fields will be converted to nested ArrayCollections.
            4. Another ArrayCollection. This produces a view of all the data
               arrays.
            5. A list of ndarray-like objects. The corresponding field names 
               must be supplied through the `dtype` argument. The arrays will
               be cast to the dtypes of the fields of the supplied dtype. 
        dtype : structured datatype or list of fieldnames
            Should only be supplied if the `data` argument is a list of
            ndarray-like objects. May be a structured datatype, in which case
            the data arrays will be cast to the corresponding field dtypes,
            or a list of fieldnames and no casting will occur.
        skip_validation : bool
            If False, check that the arrays have the same shape.
        """

        if dtype is not None:
            if isinstance(dtype, list):
                arrays = {str(name): _asarraylike(a) 
                          for name, a  in zip(dtype, data)}
            elif isinstance(dtype, np.dtype):
                dts = (dtype.fields[n][0] for n in dtype.names)
                arrays = {name: _asarraylike(ai, dtype=dt)
                          for name, ai, dt in zip(dtype.names, zip(*data), dts)}
            else:
                raise TypeError("dtype must be a np.dtype or a list of "
                                "fieldnames")
        # if data is a dict of {name: arr} key/values.
        elif isinstance(data, dict):
            arrays = {k: _asarraylike(v) for k, v in data.items()}
        # if data is a list of (name, arr) tuples:
        elif isinstance(data, list):
            arrays = {k: _asarraylike(v) for k, v in data}
            #raise TypeError("data provided as a list must contain tuples "
            #                "of (name, arr) pairs, or must be a list of "
            #                "arrs and dtype must be supplied")
        # if data is a structured array
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            # unpack a structured array
            arrays = {}
            for n in data.dtype.names:
                if data[n].dtype.names is not None:
                    # unpack nested types recursively
                    arrays[n] = type(self)(data[n])
                else:
                    arrays[n] = data[n].copy()
                    # note that this folds in dims of subarrays
        # if data is another ArrayCollection
        elif isinstance(data, ArrayCollection):
            arrays = data._arrays
        else:
            raise TypeError("invalid data format")

        # check all arrays have the same shape
        if not skip_validation:
            for name, arr in arrays.items():
                if not isinstance(name, str):
                    raise TypeError('field names must be strings')
                if not is_ndtype(arr):
                    raise TypeError('data was not arraylike')
            shapes = [a.shape for a in arrays.values()]
            if shapes.count(shapes[0]) != len(shapes):
                raise Exception('All arrays in a collection must have the '
                                'same shape')

        self._arrays = arrays
        # for now, hijack structured dtypes to represent our dtype
        self._dtype = np.dtype([(n, a.dtype) for n, a in arrays.items()])

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        try:
            return next(iter(self._arrays.values())).shape
        except StopIteration:
            return ()

    @shape.setter
    def shape(self, val):
        for a in self._arrays.values():
            a.shape = val

    @property
    def strides(self):
        return None

    def __getitem__(self, ind):
        # for a single field name, return the bare ndarray (view)
        if isinstance(ind, str):
            return self._arrays[ind]

        # for a list of field names return an arraycollection (view)
        if is_list_of_strings(ind):
            return type(self)({n: self._arrays[n] for n in ind},
                               skip_validation=True)

        # single integers get converted to tuple
        if isinstance(ind, (int, np.integer)):
            ind = (ind,)

        # fall through: Use ndarray indexing
        out = {n: a[ind] for n, a in self._arrays.items()}

        # scalars get returned as a CollectionScalar
        if next(iter(out.values())).shape == ():
            return CollectionScalar(tuple(out.values()), self._dtype)

        return type(self)(out)

    def __setitem__(self, ind, val):
        # for a single field name, assign to that array
        if isinstance(ind, str):
            self._arrays[ind][:] = val

        # for a list of field names, assign to the ArrayCollection view
        elif is_list_of_strings(ind):
            view = self[ind]
            view[:] = val

        # for a tuple, assign values to each array in order
        elif isinstance(val, tuple):
            if len(val) != len(self.arrays.keys()):
                raise ValueError("wrong number of values")
            for dst,v in zip(self.arrays.values(), val):
                dst[ind] = v

        # for a structured ndarray, fail
        elif isinstance(val, np.ndarray) and val.dtype.names is not None:
            raise Exception("Convert structured arrays to "
                            "ArrayCollection first")

        # for another arraycollection, assign successive arrays
        elif isinstance(val, ArrayCollection):
            if len(self._arrays) != len(val._arrays):
                raise ValueError("wrong number of values")
            for dst, src in zip(self._arrays.values(), val._arrays.values()):
                dst[ind] = src

        # otherwise, try to assign val to each array (includes scalars and
        # unstructured ndarrays)
        else:
            for dst in self._arrays.values():
                dst[ind] = val

    def __str__(self):
        return duck_str(self)

    def __repr__(self):
        return duck_repr(self)

    #def __len__(self):
    #    try:
    #        return len(next(iter(self._arrays.values())))
    #    except StopIteration:
    #        return 0

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        kwds = {'order': order, 'casting': casting, 'subok': subok,
                'copy': copy}

        dtype = np.dtype(dtype)
        types = [(n, dtype.fields[n][0]) for n in dtype.names]
        if len(types) != len(self._arrays):
            raise Exception("Cannot change number of fields")

        # force a copy if the names changed
        if list(self._arrays.keys()) != [n for n,dt in types]:
            copy = True

        mapping = [(n, arr, arr.astype(dt, *kwds))
                   for arr,(n,dt) in zip(self._arrays.values(), types)]

        if copy is False:
            # only return self if no array needed a copy
            if not any(ai is ao for (n, ai, ao) in mapping):
                return self
            # otherwise fall back to a copy for all arrays
            mapping = [((n, ai, ao.copy() if ai is ao else ao)
                       for (n, ai, ao) in mapping)]

        return ArrayCollection({n: ao for (n, ai, ao) in mapping})

    def view(dtype=None, type=None):
        # "type" kwd of ndarrays is not yet supported: How to mix
        # ducktyping and subclassing?
        if type is not None:
            raise ValueError("type argument is not supported")

        dtype = np.dtype(dtype)
        newtypes = [(n, dtype.fields[n][0]) for n in dtype.names]
        if len(newtypes) != len(self._arrays):
            raise Exception("Cannot change number of fields")

        out = {n: arr.view(dt)
               for arr,(n,dt) in zip(self._arrays.values(), types)}

        return ArrayCollection(out)

    def copy(self, order='C'):
        return ArrayCollection({n: a.copy(order=order)
                                for n, a in self._arrays.items()})

    # unlike np.reshape, allows shape to be passed as separate args
    def reshape(self, *shape, **kwargs):
        if len(shape) > 1:
            shape = (shape,)
        return np.reshape(self, *shape, **kwargs)

    # This works inplace, unlike np.resize, and fills with repeat instead of 0
    def resize(self, new_shape, refcheck=True):
        for a in self._arrays.values():
            a.resize(new_shape, refcheck)

class CollectionScalar(CollectionMixin):
    def __init__(self, vals, dtype=None):
        if isinstance(vals, tuple):
            self._data = vals
            self._dtype = np.dtype(dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return ()

    def __getitem__(self, ind):
        # for a single field name, return the bare ndarray
        if isinstance(ind, str):
            return self.data[self._dtype.names.index(ind)]

        # for a list of field names return a CollectionScalar
        if is_list_of_strings(ind):
            new_dtype = np.dtype([(n, self._dtype.fields[n][0]) for n in ind])
            new_data = (self._data[self._dtype.names.index(n)] for n in inds)
            return CollectionScalar(tuple(new_data), new_dtype)

        if ind == ():
            return self

        # integer index
        return self._data[ind]

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return "CollectionScalar({}, dtype={})".format(str(self._data),
                                                       str(self._dtype))

def is_list_of_strings(val):
    if not isinstance(val, list):
        return False
    for v in val:
        if not isinstance(v, str):
            return False
    return True

def empty_collection(shape, dtype, order='C'):
    dtype = np.dtype(dtype)

    arrays = {}
    for n in dtype.names:
        dt = dtype.fields[n][0]
        nshape = shape + dt.shape
        if dt.names:
            arrays[n] = empty_collection(nshape, dt, order=order)
        else:
            arrays[n] = np.empty(nshape, dt, order=order)

    return ArrayCollection(arrays)

HANDLED_FUNCTIONS = {}

def implements(numpy_function, checked_args=None):
    """Register an __array_function__ implementation for MaskedArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = (func, checked_args)
        return func
    return decorator

def setup_ducktype():

    def check_common_fields(arrays, dtype=None):
        # validate the dtypes are similar enough (same field names)
        dtype = None

        ac = []
        for a in arrays:
            if isinstance(a, list):
                a, dtype = check_common_fields(a, dtype)
            else:
                if a.dtype.names is None:
                    raise Exception("mismatched number of fields")
                a = ArrayCollection(a)

                if dtype is None:
                    dtype = a.dtype

                if a.dtype.names != dtype.names:
                    raise Exception("mismatched field names")

            ac.append(a)

        return ac, dtype

    def check_out(out, dtype):
        if out is not None:
            if not isinstance(out, ArrayCollection):
                raise Exception("out must be an arraycollection")

            if len(out.dtype.names) != len(dtype.names):
                raise Exception("out has mismatched field names")

            # construct view of out with same field names as dtype
            return out._arrays.values()
        else:
            return [None]*len(dtype.names)


    @implements(np.array_repr)
    def array_repr(arr, max_line_width=None, precision=None,
                   suppress_small=None):
        return array_repr_impl(arr, max_line_width, precision, suppress_small)

    @implements(np.array2string)
    def array2string(a, max_line_width=None, precision=None,
                     suppress_small=None, separator=' ', prefix="",
                     style=np._NoValue, formatter=None, threshold=None,
                     edgeitems=None, sign=None, floatmode=None, suffix="",
                     **kwarg):
        return array2string_impl(a, max_line_width, precision, suppress_small,
                          separator, prefix, style, formatter, threshold,
                          edgeitems, sign, floatmode, suffix, **kwarg)

    @implements(np.copy)
    def copy(a, order='K'):
        return ArrayCollection({name: np.copy(ai, order)
                                for name, ai in a._arrays.items()()})

    @implements(np.diagonal)
    def diagonal(a, offset=0, axis1=0, axis2=1):
        return ArrayCollection({name: np.diagonal(ai, offset, axis1, axis2)
                                for name, ai in a._arrays.items()})

    @implements(np.diag)
    def diag(v, k=0):
        return ArrayCollection({name: np.diag(ai, k)
                                for name, ai in v._arrays.items()})

    @implements(np.diagflat)
    def diagflat(v, k=0):
        return ArrayCollection({name: np.diagflat(ai, k)
                                for name, ai in v._arrays.items()})

    @implements(np.tril)
    def tril(m, k=0):
        return ArrayCollection({name: np.tril(ai, k)
                                for name, ai in m._arrays.items()})

    @implements(np.triu)
    def triu(m, k=0):
        return ArrayCollection({name: np.triu(ai, k)
                                for name, ai in m._arrays.items()})

    @implements(np.take, checked_args=(0,))
    def take(a, indices, axis=None, out=None, mode='raise'):
        arrs = {name: np.take(ai, indices, axis, o, mode) for (name, ai), o
                in zip(a._arrays.items(), check_out(out, a.dtype))}
        return out if out is not None else ArrayCollection(arrs)

    @implements(np.put, checked_args=(0,))
    def put(a, indices, values, mode='raise'):
        vals = ArrayCollection(values, dtype=a.dtype)
        for name, ai in a._arrays.items():
            np.put(ai, indices, vals[name], mode)

    @implements(np.take_along_axis, checked_args=(0,))
    def take_along_axis(arr, indices, axis):
        return ArrayCollection({name: np.take_along_axis(ai, indices, axis)
                                for name, ai in a._arrays.items()})

    @implements(np.put_along_axis, checked_args=(0,))
    def put_along_axis(arr, indices, values, axis):
        vals = ArrayCollection(values, dtype=arr.dtype)
        for name, ai in arr._arrays.items():
            np.put_along_axis(ai, indices, vals[name], axis)

    @implements(np.ravel)
    def ravel(a, order='C'):
        return ArrayCollection({name: np.ravel(ai, order)
                                for name, ai in a._arrays.items()})

    @implements(np.repeat)
    def repeat(a, repeats, axis=None):
        return ArrayCollection({name: np.repeat(ai, repeats, axis)
                                for name, ai in a._arrays.items()})

    @implements(np.reshape)
    def reshape(a, newshape, order='C'):
        return ArrayCollection({name: np.reshape(ai, newshape, order)
                                for name, ai in a._arrays.items()})

    @implements(np.resize)
    def resize(a, new_shape):
        return ArrayCollection({name: np.resize(ai, new_shape)
                                for name, ai in a._arrays.items()})

    @implements(np.squeeze)
    def squeeze(a, axis=None):
        return ArrayCollection({name: np.squeeze(ai, axis)
                                for name, ai in a._arrays.items()})

    @implements(np.swapaxes)
    def swapaxes(a, axis1, axis2):
        return ArrayCollection({name: np.swapaxes(ai, axis1, axis2)
                                for name, ai in a._arrays.items()})

    @implements(np.transpose)
    def transpose(a, axes=None):
        return ArrayCollection({name: np.transpose(ai, axes)
                                for name, ai in a._arrays.items()})

    @implements(np.roll)
    def roll(a, shift, axis=None):
        return ArrayCollection({name: np.roll(ai, shift, axis)
                                for name, ai in a._arrays.items()})

    @implements(np.rollaxis)
    def rollaxis(a, axis, start=0):
        return ArrayCollection({name: np.rollaxis(ai, axis, start)
                                for name, ai in a._arrays.items()})

    @implements(np.moveaxis)
    def moveaxis(a, source, destination):
        return ArrayCollection({name: np.moveaxis(ai, source, destination)
                                for name, ai in a._arrays.items()})

    @implements(np.flip)
    def flip(m, axis=None):
        return ArrayCollection({name: np.flip(ai, axis)
                                for name, ai in m._arrays.items()})

    @implements(np.expand_dims)
    def expand_dims(a, axis):
        return ArrayCollection({name: np.expand_dims(ai, axis)
                                for name, ai in a._arrays.items()})

    @implements(np.concatenate)
    def concatenate(arrays, axis=0, out=None):
        out_dt, out_subtype = promote_duckarrays(arrays)
        #arrays, dtype = check_common_fields(arrays)
        arrs = {name: np.concatenate([ai[name] for ai in arrays], axis, o)
                for name, o in zip(dtype.names, check_out(out, dtype))}
        return out if out is not None else ArrayCollection(arrs)

    @implements(np.block)
    def block(arrays):
        arrays, dtype = check_common_fields(arrays)
        #XXX

    @implements(np.column_stack)
    def column_stack(tup):
        arrays, dtype = check_common_fields(tup)
        fields = zip(*(a._arrays.values() for a in tup))
        return ArrayCollection({name: np.column_stack(ai)
                               for name,ai in zip(dtype.names, fields)})

    @implements(np.dstack)
    def dstack(tup):
        arrays, dtype = check_common_fields(tup)
        fields = zip(*(a._arrays.values() for a in tup))
        return ArrayCollection({name: np.dstack(ai)
                               for name,ai in zip(dtype.names, fields)})

    @implements(np.vstack)
    def vstack(tup):
        arrays, dtype = check_common_fields(tup)
        fields = zip(*(a._arrays.values() for a in tup))
        return ArrayCollection({name: np.vstack(ai)
                               for name,ai in zip(dtype.names, fields)})

    @implements(np.hstack)
    def hstack(tup):
        arrays, dtype = check_common_fields(tup)
        fields = zip(*(a._arrays.values() for a in tup))
        return ArrayCollection({name: np.hstack(ai)
                               for name,ai in zip(dtype.names, fields)})

    @implements(np.stack)
    def stack(arrays, axis=0, out=None):
        arrays, dtype = check_common_fields(tup)
        fields = zip(*(a._arrays.values() for a in arrays))
        arrs = {name: np.stack(ai, axis, o) for name, ai, o in
                zip(dtype.names, fields, check_out(out, dtype))}
        return out if out is not None else ArrayCollection(arrs)

    @implements(np.array_split, checked_args=(0,))
    def array_split(ary, indices_or_sections, axis=0):
        ios = indices_or_sections
        return ArrayCollection({name: np.array_split(ai, ios, axis)
                                for name, ai in ary._arrays.items()})

    @implements(np.split, checked_args=(0,))
    def split(ary, indices_or_sections, axis=0):
        return ArrayCollection({name: np.split(ai, indices_or_sections, axis)
                                for name, ai in ary._arrays.items()})

    @implements(np.hsplit)
    def hsplit(ary, indices_or_sections):
        return ArrayCollection({name: np.hsplit(ai, indices_or_sections)
                                for name, ai in ary._arrays.items()})

    @implements(np.vsplit)
    def vsplit(ary, indices_or_sections):
        return ArrayCollection({name: np.vsplit(ai, indices_or_sections)
                                for name, ai in ary._arrays.items()})

    @implements(np.dsplit)
    def dsplit(ary, indices_or_sections):
        return ArrayCollection({name: np.dsplit(ai, indices_or_sections)
                                for name, ai in ary._arrays.items()})

    @implements(np.tile)
    def tile(ary, reps):
        return ArrayCollection({name: np.tile(ai, reps)
                                for name, ai in ary._arrays.items()})

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

    @implements(np.delete)
    def delete(arr, obj, axis=None):
        return ArrayCollection({name: np.delete(ai, obj, axis)
                                for name, ai in arr._arrays.items()})

    @implements(np.insert)
    def insert(arr, obj, values, axis=None):
        vals = ArrayCollection(values, dtype=arr.dtype)
        return ArrayCollection({name: np.insert(ai, obj, vals[name], axis)
                                for name, ai in arr._arrays.items()})

    @implements(np.append)
    def append(arr, values, axis=None):
        vals = ArrayCollection(values, dtype=arr.dtype)
        return ArrayCollection({name: np.append(ai, obj, vals[name], axis)
                                for name, ai in arr._arrays.items()})

    @implements(np.extract, checked_args=(1,))
    def extract(condition, arr):
        return ArrayCollection({name: np.extract(condition, ai)
                                for name, ai in arr._arrays.items()})

    @implements(np.place, checked_args=(0,))
    def place(arr, mask, vals):
        vals = ArrayCollection(vals, dtype=arr.dtype)
        for name, ai in arr._arrays.items():
            np.place(ai, mask, vals[name])

    @implements(np.pad)
    def pad(array, pad_width, mode, **kwargs):
        return ArrayCollection({name: np.pad(ai, pad_width, mode, **kwargs)
                                for name, ai in array._arrays.items()})

    @implements(np.broadcast_to)
    def broadcast_to(array, shape, subok=False):
        return ArrayCollection({name: np.broadcast_to(ai, shape, subok)
                                for name, ai in a._arrays.items()})

    @implements(np.broadcast_arrays)
    def broadcast_arrays(*args, **kwargs):
        if kwargs:
            raise TypeError('broadcast_arrays() got an unexpected keyword '
                            'argument {!r}'.format(list(kwargs.keys())[0]))
        shape = _broadcast_shape(*args)

        if all(array.shape == shape for array in args._arrays.values()):
            return args

        return [np.broadcast_to(array, shape, subok=subok, readonly=False)
                for array in args._arrays.values()]

    @implements(np.empty_like)
    def empty_like(prototype, dtype=None, order='K', subok=True):
        # XXX order is ignored
        dtype = dtype or prototype.dtype
        return empty_collection(prototype.shape, dtype, 'C')

    @implements(np.ones_like)
    def ones_like(prototype, dtype=None, order='K', subok=True):
        arr = np.empty_like(prototype, dtype, order, subok)
        arr[...] = 1
        return arr

    @implements(np.zeros_like)
    def zeros_like(prototype, dtype=None, order='K', subok=True):
        arr = np.empty_like(prototype, dtype, order, subok)
        arr[...] = 0
        return arr

    @implements(np.full_like)
    def full_like(a, fill_value, dtype=None, order='K', subok=True):
        arr = np.empty_like(prototype, dtype, order, subok)
        arr[...] = fill_value
        return arr

    @implements(np.where)
    def where(condition, x=None, y=None):
        if x is None or y is None:
            raise ValueError('np.where can only be used in "nonzero" mode for '
                             'ArrayCollection. Supply x and y.')
        (x, y), dtype = check_common_fields((x, y))
        fields = zip(dtype.names, x._arrays.values(), y._arrays.values())
        return ArrayCollection({name: np.where(condition, xi, yi)
                                for name, xi, yi in fields})

    @implements(np.choose, checked_args=(1,)) #XXX checkedargs doesn't work here
    def choose(a, choices, out=None, mode='raise'):
        pass

    @implements(np.shape)
    def shape(a):
        return a.shape

    @implements(np.alen)
    def alen(a):
        return len(a)

    @implements(np.ndim)
    def ndim(a):
        return a.ndim

    @implements(np.size)
    def size(a):
        return a.size

    @implements(np.copyto, checked_args=(0))
    def copyto(dst, src, casting='same_kind', where=True):
        pass

    @implements(np.putmask)
    def putmask(a, mask, values):
        pass

setup_ducktype()
