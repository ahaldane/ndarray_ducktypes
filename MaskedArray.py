#!/usr/bin/env python
import numpy as np
from duckprint import (repr_implementation, str_implementation,
    default_duckprint_options, default_duckprint_formatters, FormatDispatcher)
import builtins

def as_masked_fmt(formattercls):
    class MaskedFormatter(formattercls):
        def get_format_func(self, elem, **options):
            default_fmt = super().get_format_func(elem, **options)

            if not elem.mask.any():
                return lambda x: default_fmt(x.data)

            masked_str = options['masked_str']

            # pad the columns to align when including the masked string
            def fmt(x):
                res = default_fmt(x.data)
                reslen = builtins.max(len(res), len(masked_str))
                if x.mask:
                    res = masked_str
                return res.rjust(reslen)

            return fmt

    return MaskedFormatter

MASK_STR = '_'  # make this more configurable later

masked_formatters = [as_masked_fmt(f) for f in default_duckprint_formatters]
default_options = default_duckprint_options.copy()
default_options['masked_str'] = MASK_STR
masked_dispatcher = FormatDispatcher(masked_formatters, default_options)

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
        self.ndim = self.data.ndim
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
        mask = self.mask[ind]
        if data.shape == ():
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
        return str_implementation(self, dispatcher=masked_dispatcher)

    def __repr__(self):
        return repr_implementation(self, dispatcher=masked_dispatcher)

    def reshape(self, shape, order='C'):
        return MaskedArray(self.data.reshape(shape, order=order),
                           self.mask.reshape(shape, order=order))

class MaskedScalar(object):
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
            return MASK_STR
        return repr(self.data)

    def __format__(self, format_spec):
        if self.mask:
            return MASK_STR
        return format(self.data, format_spec)

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

@implements(np.max)
def max(array):
    return array.data[~array.mask].max()

@implements(np.min)
def min(array):
    return array.data[~array.mask].min()

if __name__ == '__main__':
    A = MaskedArray(np.arange(12), np.arange(12)%2).reshape((4,3))
    print(hasattr(A, '__array_function__'))
    print(A)
    print(repr(A))
    print(A[2:4])
