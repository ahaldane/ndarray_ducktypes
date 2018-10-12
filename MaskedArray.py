#!/usr/bin/env python
import numpy as np
from numpy.core.arrayprint import array_repr_impl, array_str_impl, array2string_impl

HANDLED_FUNCTIONS = {}

class MaskedArray:
    def __init__(self, data=None, mask=None, dtype=None, copy=False,
                order=None, subok=True, ndmin=0, fill_value=None, **options):
        self.data = np.array(data, dtype, copy=copy, order=order, subok=subok, 
                             ndmin=ndmin)
        
        # Note: We do not try to mask individual fields structured types.
        # Instead you get one mask alue per struct element. Use an
        # ArrayCollection of MaskedArrays if you want to mask individual
        # fields.

        if mask is None:
            self.mask = np.zeros(data.shape, dtype='bool', order=data.order)
        else:
            mask = mask.astype('bool', copy=False)
            self.mask = np.broadcast_to(mask, self.data.shape)

        self.fill_value = fill_value

        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, MaskedArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, ind):
        data = self.data[ind]
        mask = self.data[ind]
        if ret.shape == ():
            return MaskedScalar(data, mask)
        return MaskedArray(data, mask)

    def __setitem__(self, ind, val):
        if isinstance(val, [MaskedArray, MaskedScalar]):
            self.data[ind] = val.data
            self.mask[ind] = val.mask
        else:
            self.data[ind] = val
            self.mask[ind] = False

    def __str__(self):
        return np.array2string(self, formatter=_format_provider)

    def __repr__(self):
        return np.array2string(self, formatter=_format_provider)

    def reshape(self, shape, order='C'):
        return MaskedArray(self.data.reshape(shape, order),
                           self.mask.reshape(shape, order))

# this format-provider code must be used with Eric-Wieser's
# "single-formatter" branch:
# https://github.com/eric-wieser/numpy/tree/single-formatter
# a custom format provider for masked arrays, that inserts --
def _format_provider(data, **options):
    """
    Custom format provider for masked arrays.

    This is passed as the `formatter` argument to array2string.

    This is used to insert the masked_print_option into the repr,
    without using the old trick of casting everything to object first.
    """
    # Here we pass only the raw data, to avoid warnings from
    # `float(np.ma.masked)`
    default = np.core.arrayprint.default_format_provider(data.data, **options)
    any_masked = data.mask.any()
    masked_str = str(masked_print_option)

    def formatter(x):
        res = default(x.data)
        # no need to adjust padding if this data isn't masked
        if not any_masked:
            return res

        # pad the columns to align when including the masked string
        col_width = builtins.max(len(res), len(masked_str))
        if x.mask:
            res = masked_str
        return res.rjust(col_width)

    return formatter

def implements(numpy_function):
    """Register an __array_function__ implementation for MaskedArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

#@implements(np.concatenate)
#def concatenate(arrays, axis=0, out=None):
#    ...  # implementation of concatenate for MyArray objects

#@implements(np.broadcast_to)
#def broadcast_to(array, shape):
#    ...  # implementation of broadcast_to for MyArray objects

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

def array(data, dtype=None, copy=False, order=None,
          mask=None, fill_value=None, keep_mask=True,
          hard_mask=False, shrink=True, subok=True, ndmin=0):
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy, order=order,
                       subok=subok, ndmin=ndmin, fill_value=fill_value)

if __name__ == '__main__':
    A = MaskedArray(np.arange(10), np.arange(10)%2)
    print(hasattr(A, '__array_function__'))
    print(A)
    print(A[2:4].data)
