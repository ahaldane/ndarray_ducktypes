

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

class AST(object):
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UniOp(AST):
    def __init__(self, left, op, right):
        if isinstance(left, LazyArray):
            left = left.val
        if isinstance(left, LazyArray):
            right = right.val

        self.left = left
        self.op = op
        self.right = right

class LazyArray:
    def __init__(self, val):
        self.val = val

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, MaskedArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Cannot handle items that have __array_ufunc__ (other than our own).
        outputs = kwargs.get('out', ())
        for item in inputs + outputs:
            if (hasattr(item, '__array_ufunc__') and
                    type(item).__array_ufunc__ is not ndarray.__array_ufunc__):
                return NotImplemented

        if ufunc is np.add:
            return LazyArray(BinOp(inputs[0], np.add, inputs[1]))
        if ufunc is np.mul:
            return LazyArray(BinOp(inputs[0], np.mul, inputs[1]))

        return NotImplemented
    
    def eval(self):
        

