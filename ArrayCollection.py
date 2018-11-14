#!/usr/bin/env python
import numpy as np
import warnings
from duckprint2 import repr_implementation, str_implementation


HANDLED_FUNCTIONS = {}

class ArrayCollection:
    """
    NDarray-like type which stores a set of named arrays with a common shape
    but which may be different dtypes. Allows numpy-style indexing of them as a
    group. An ArrayCollection looks and behaves very similarly to a structured
    ndarray, but will have better performance because it has better memory
    layout.

    Can be thought of as a "lite" version of pandas dataframe, or Xarray
    dataset.  Look to those projects if more features are desired, such
    as named axes.

    Tip: One can use `np.broadcast_to` to broadcast the input arrays to the
    same shape in a memory-efficient way. Some caution is needed when writing
    to an ArrayCollection created this way though. See the `np.broadcast_to`
    docstring for details.

    Parameters
    ----------
    data : tuple of (str, arraylike) pairs, structured ndarray,
           or Arraycollection
        Set of data to make a collection from.
    """
    def __init__(self, data):
        # if data is a list of (name, arr) tuples:
        if isinstance(data, list):
            self.names, arrays = zip(*data)

        # if data is a structured array
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            # unpack a structured array
            self.names = data.dtype.names
            arrays = []
            for n in self.names:
                if data[n].dtype.names is not None:
                    # unpack nested types recursively
                    arrays.append(ArrayCollection(data[n]))
                else:
                    arrays.append(data[n].copy())
                    # note that this folds in dims of subarrays

        # if data is another ArrayCollection
        elif isinstance(data, ArrayCollection):
            self.names = data.names.copy()
            arrays = data.arrays
        else:
            raise Exception("Expected either a list of (name, arr) pairs"
                            "or a structured array")

        # check all arrays have the same shape
        shapes = [a.shape for a in arrays]
        if shapes.count(shapes[0]) != len(shapes):
            raise Exception('All arrays in a collection must have the '
                            'same shape')

        self.arrays = dict(zip(self.names, arrays))

        # for now, hijack structured dtypes to represent our dtype
        self.dtype = np.dtype([(n, self.arrays[n].dtype) for n in self.names])

        arr0 = self.arrays[self.names[0]]
        self.shape = arr0.shape
        self.size = arr0.size
        self.ndim = arr0.ndim
        self.strides = None

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, ArrayCollection) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    #def __array_ufunc__():
    #    # TODO: ArrayCollection supports only one ufunc: np.equal

    def __getitem__(self, ind):
        # for a single field name, return the bare ndarray
        if isinstance(ind, str):
            return self.arrays[ind]

        # for a list of field names return an arraycollection
        if is_list_of_strings(ind):
            return ArrayCollection([(n, self.arrays[n]) for n in ind])

        # single integers get converted to tuple
        if isinstance(ind, (int, np.integer)):
            ind = (ind,)

        # fall through: Use ndarray indexing
        out = [(n, self.arrays[n][ind]) for n in self.names]

        # scalars get returned as a CollectionScalar
        if out[0][1].shape == ():
            return CollectionScalar(tuple(a for n,a in out), self.dtype)

        return ArrayCollection(out)

    def __setitem__(self, ind, val):
        # for a single field name, assign to that array
        if isinstance(ind, str):
            self.arrays[ind][:] = val

        # for a list of field names, assign to those arrays
        elif is_list_of_strings(ind):
            for name in ind:
                self.arrays[name][:] = val
                return

        # for a tuple, assign values to each array in order
        elif isinstance(val, tuple):
            if len(val) != len(self.names):
                raise ValueError("wrong number of values")
            for n,v in zip(self.names, val):
                self.arrays[n][ind] = v

        # for a structured ndarray, fail
        elif isinstance(val, np.ndarray) and val.dtype.names is not None:
            raise Exception("Convert structured arrays to "
                            "ArrayCollection first")

        # for another arraycollection, assign successive arrays
        elif isinstance(val, ArrayCollection):
            if len(self.names) != len(val.names):
                raise ValueError("wrong number of values")
            for n, nother in zip(self.names, val.names):
                self.arrays[n][ind] = val[nother]

        # otherwise, try to assign val to each array (includes scalars and
        # unstructured ndarrays)
        else:
            for n in self.names:
                self.arrays[n][ind] = val

    def __str__(self):
        return str_implementation(self)

    def __repr__(self):
        return repr_implementation(self)

    def __len__(self):
        return len(self.arrays[self.names[0]])

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        kwds = {'order': order, 'casting': casting, 'subok': subok,
                'copy': copy}

        dtype = np.dtype(dtype)
        types = [(n, dtype.fields[n][0]) for n in dtype.names]
        if len(types) != len(self.names):
            raise Exception("Cannot change number of fields")

        # force a copy if the names changed
        if self.names != [n for n,dt in types]:
            copy = True

        mapping = [(n, self.arrays[no], self.arrays[no].astype(dt, *kwds))
                   for no,(n,dt) in zip(self.names, types)]

        # only return self if no array needed a copy
        if copy is False:
            iscopy = [ai is ao for (n, ai, ao) in mapping]
            if not any(iscopy):
                return self
            # otherwise fall back to a copy for all arrays
            mapping = [((n, ai, ao.copy() if ai is ao else ao)
                       for (n, ai, ao) in mapping)]

        return ArrayCollection([(n, ao) for (n, ai, ao) in mapping])

    def view(dtype=None):
        # note: "type" kwd of ndarrays is not yet supported: How to mix
        # ducktyping and subclassing?

        dtype = np.dtype(dtype)
        newtypes = [(n, dtype.fields[n][0]) for n in dtype.names]
        if len(newtypes) != len(self.names):
            raise Exception("Cannot change number of fields")

        out = [(n, self.arrays[no].view(dt))
               for no,(n,dt) in zip(self.names, types)]

        return ArrayCollection(out)

    def copy(self, order='C'):
        return ArrayCollection([(n, self.arrays[n].copy(order=order))
                                for n in self.names])

    def reshape(self, shape, order='C'):
        for n in self.names:
            self.arrays[n] = self.arrays[n].reshape(shape, order=order)
        self.shape = shape
        return self

    #def repeat
    #def resize
    #def squeeze
    #def swapaxes
    #def T
    #def take
    #def transpose

    def argsort(self, axis=-1, order=None):
        if order is None:
            order = self.names

        return np.lexsort([self.arrays[n] for n in order], axis=axis)

    def sort(self, axis=-1, order=None):
        inds = self.argsort(axis=axis, order=order)
        for n in self.names:
            self.arrays[n] = np.take(self.arrays[n], inds, axis=axis)


    #def fill
    #def ravel

# Interesting Fact: The numpy arrayprint machinery (for one) depends on having
# a separate scalar type associated with any new ducktype (or subclass). This
# is partly why both MaskedArray and recarray have to define associated scalar
# types. I don't currently see a way to avoid this: All ducktypes will need
# to create a scalar type, and return it (and not a 0d array) when indexed with
# an integer.
class CollectionScalar:
    def __init__(self, vals, dtype=None):
        self.data = vals
        self.dtype = np.dtype(dtype)
        self.shape = ()
        self.size = 1
        self.ndim = 0

    def __getitem__(self, ind):
        # for a single field name, return the bare ndarray
        if isinstance(ind, str):
            return self.data[self.dtype.names.index(ind)]

        # for a list of field names return an arraycollection
        if is_list_of_strings(ind):
            new_dtype = np.dtype([(n, self.dtype.fields[n][0]) for n in ind])
            new_data = tuple(self.data[self.dtype.names.index(n)] for n in inds)
            return CollectionScalar(new_data, new_dtype)

        # integer index
        return self.data[ind]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return "CollectionScalar({}, dtype={})".format(str(self.data), str(self.dtype))

def is_list_of_strings(val):
    if not isinstance(val, list):
        return False
    for v in val:
        if not isinstance(v, str):
            return False
    return True

def empty_collection(shape, dtype, order='C'):
    dtype = np.dtype(dtype)

    arrays = []
    for n in dtype.names:
        dt = dtype.fields[n][0]
        nshape = shape + dt.shape
        if dt.names:
            arrays.append((n, empty_collection(nshape, dt, order=order)))
        else:
            arrays.append((n, np.empty(nshape, dt, order=order)))

    return ArrayCollection(arrays)

def implements(numpy_function):
    """Register an __array_function__ implementation for MaskedArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

@implements(np.reshape)
def reshape(a, newshape, order='C'):
    return a.reshape(newshape, order)

@implements(np.transpose)
def transpose(a, axes=None):
    return a.transpose(axes)

@implements(np.sort)
def sort(a, axis=-1, kind='quicksort', order=None):
    return a.sort(axis, kind, order)

@implements(np.broadcast_to)
def broadcast_to(array, shape, subok=False):
    pass

@implements(np.asarray)
def asarray(val):
    return val

@implements(np.array_repr)
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    return array_repr_impl(arr, max_line_width, precision, suppress_small)

@implements(np.array2string)
def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=np._NoValue, formatter=None, threshold=None,
                 edgeitems=None, sign=None, floatmode=None, suffix="",
                 **kwarg):
    return array2string_impl(a, max_line_width, precision, suppress_small, separator,
                      prefix, style, formatter, threshold, edgeitems, sign,
                      floatmode, suffix, **kwarg)

@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    # validate the dtypes are similar enough (same field names)
    dtype = None
    for a in arrays:
        if a.dtype.names is not None:
            a = ArrayCollection(a)

        if not isinstance(a, ArrayCollection):
            raise Exception("mismatched number of fields")

        if dtype is None:
            dtype = a.dtype

        if a.dtype.names != dtype.names:
            raise Exception("mismatched field names")

    if out is not None:
        if not isinstance(out, ArrayCollection):
            raise Exception("out must be an arraycollection")

        if out.dtype.names != dtype.names:
            raise Exception("out has mismatched field names")

        arrs = [np.concatenate(x, axis)
                for x in zip(*[a.arrays for a in arrays])]
        for a,b in zip(arrs, out.arrays):
            b[:] = a

        return out
    else:
        arrs = [np.concatenate(x, axis)
                for x in zip(*[a.arrays for a in arrays])]
        return ArrayCollection(list(zip(dtype.names, arrs)))


if __name__ == '__main__':
    # note: printing requires a modification of numpy to work:
    # In numpy/core/arrayprint.py, in _array2string(), replace the first line
    # "data = asarray(a)" by
    #if issubclass(a, np.ndarray):
    #    data = asarray(a)
    #else:
    #    data = a

    a = np.arange(10, dtype='u2')
    b = np.arange(10, 20, dtype='f8')
    A = ArrayCollection([('a', a), ('b', b)])

    print(A[0])
    print(repr(A))

    B = ArrayCollection(np.ones((2,3), dtype='u1,S3'))
    print(repr(B.reshape((3,2))))
    print(repr(empty_collection((3,4), 'f4,u1,u1')))

    # does not work yet
    #print("Concatenate:")
    #print(np.concatenate([A, A]))
    print(repr_implementation(np.arange(12).reshape((4,3))))
