

# Idea: A lazy array-like which would help optimize out intermediate
# storage. You would do:
#     >>> a = lazyarray(rand(10)
#     >>> b = lazyarray(rand(10))
#     >>> c = lazyarray(rand(10))
#     >>> val = a * b + c
#     >>> val
#     unevaluated_lazyarray: x[0] * x[1] + x[2]
#     >>> val.eval()
#     array([0.68474366, 0.93214605, 1.1390915 , 0.86786507, 0.820504  ,
#            0.70097947, 0.4085696 , 0.72455566, 0.25511482, 0.93304744])
# 
# And the eval() function could optimize out all intermediate out arrays.


class LazyArray:
    def __init__(self, data=None, mask=None, dtype=None, copy=False,
                order=None, subok=True, ndmin=0, fill_value=None, **options):
        self.data = np.array(data, dtype, copy, order, subok, ndmin)

        if mask is None:
            self.mask = np.zeros(data.shape, dtype='bool', order=data.order)
        else:
            self.mask = np.broadcast_to(self.data, mask)

        self.fill_value = fill_value

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, MaskedArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, ind):
        # note: we always return 0d arrays instead of scalars
        return MaskedArray(self.data[ind], self.mask[ind])

    def __setitem__(self, ind, val):
        if isinstance(val, MaskedArray):
            self.data[ind] = val.data
            self.mask[ind] = val.mask
        else:
            self.data[ind] = val

    def reshape(self, shape, order='C'):
        return MaskedArray(self.data.reshape(shape, order),
                           self.mask.reshape(shape, order))
