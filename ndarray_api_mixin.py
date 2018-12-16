import operator
from functools import reduce
import numpy as np

#XXX how to take care of docstrings?

class NDArrayAPIMixin:
    def all(self, axis=None, out=None, keepdims=False):
        return np.all(self, axis, out, keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return np.any(self, axis, out, keepdims)

    def argmax(self, axis=None, out=None):
        return np.argmax(self, axis, out)

    def argmin(self, axis=None, out=None):
        return np.argmin(self, axis, out)

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        return np.argpartition(self, kth, axis, kind, order)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        return np.argsort(self, axis, kind, order)

    def choose(self, choices, out=None, mode='raise'):
        return np.choose(self, choices, out, mode)

    def clip(self, a_min, a_max, out=None):
        return np.clip(self, a_min, a_max, out)

    def compress(self, condition, axis=None, out=None):
        return np.compress(condition, self, axis, out)

    def copy(self, order='K'):
        return np.copy(self, order='K')

    def cumprod(self, axis=None, dtype=None, out=None):
        return np.cumprod(self, axis, dtype, out)

    def cumsum(self, axis=None, dtype=None, out=None):
        return np.cumsum(self, axis, dtype, out)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return np.diagonal(self, offset, axis1, axis2)

    def dot(self, b, out=None):
        return np.dot(self, b, out)

    def max(self, axis=None, out=None, keepdims=False):
        return np.max(self, axis, out, keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        return np.mean(self, axis, dtype, out, keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return np.min(self, axis, out, keepdims)

    def nonzero(self):
        return np.nonzero(self)

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        return np.partition(self, kth, axis, kind, order)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        return np.prod(self, axis, dtype, out, keepdims)

    def ptp(self, axis=None, out=None, keepdims=False):
        return np.ptp(self, axis, out, keepdims)

    def put(self, indices, values, mode='raise'):
        return np.put(self, indices, values, mode)

    def ravel(self, order='C'):
        return np.ravel(self, order)

    def repeat(self, repeats, axis=None):
        return np.repeat(self, repeats, axis)
    
    # unlike np.reshape, allows shape to be passed as separate args
    def reshape(self, *shape, **kwargs):
        if len(shape) > 1:
            shape = (shape,)
        return np.reshape(self, *shape, **kwargs)

    def resize(self, new_shape, refcheck=True): #XXX what is this refcheck?
        return np.resize(self, new_shape)

    def round(self, decimals=0, out=None):
        return np.round(self, decimals, out)

    def searchsorted(self, v, side='left', sorter=None):
        return np.searchsorted(self, v, side, sorter)

    def sort(self, axis=-1, kind='quicksort', order=None):
        return np.sort(self, axis, kind, order)

    def squeeze(self, axis=None):
        return np.squeeze(self, axis)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        return np.std(self, axis, dtype, out, ddof, keepdims)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        return np.sum(self, axis, dtype, out, keepdims)

    def swapaxes(self, axis1, axis2):
        return np.swapaxes(self, axis1, axis2)

    def take(self, indices, axis=None, out=None, mode='raise'):
        return np.take(self, indices, axis, out, mode)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return np.trace(self, offset, axis1, axis2, dtype, out)

    def transpose(self, *axes):
        return np.transpose(self, *axes)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        return np.var(self, axis, dtype, out, ddof, keepdims)

    # not part of the numpy api, but implementation is obvious:
    def conj(self):
        return np.conjugate(self)

    def conjugate(self):
        return np.conjugate(self)

    def fill(self, value):
        self[:] = value

    def flatten(self, order='C'):
        return np.copy(self, order).reshape(self.size, order)

    @property
    def T(self):
        return self.transpose()

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.dtype.itemsize*self.size

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self):
        return len(self.shape)

# Attributes not implemented:
#  Basic Attributes:
#    dtype, shape, strides, flags, base
#  Basic methods:
#    astype, view, item
#  Misc:
#   byteswap, ctypes, data, dump, dumps, flat, getfield,  itemset, itemsize,
#   newbyteorder, setfield, setflags, tobytes, tofile, tolist, tostring
