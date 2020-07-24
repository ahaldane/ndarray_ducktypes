import builtins
from inspect import signature
from collections.abc import Iterable
import numpy as np

def is_ndducktype(val):
    return hasattr(val, '__array_function__')

# Interesting Fact: The numpy arrayprint machinery (for one) depends on having
# a separate scalar type associated with any new ducktype (or subclass). This
# is partly why both MaskedArray and recarray have to define associated scalar
# types. I don't currently see a way to avoid this: All ducktypes will need
# to create a scalar type, and return it (and not a 0d array) when indexed with
# an integer.

#XXX consider defining an abstract "duck_scalar" class which all duck-scalar
# implementors would be required to inherit. Then the function below could
#be coded as "isinstance(val, [np.generic, duck_scalar])"

# Or, can we just subclass np.generic?

def is_duckscalar(val):
    # These files assume that a scalar is a type separate from the main ducktype
    # which also has an __array_function__ attribute.
    # A simple test of whether a numpy-like type is a scalar and not a 0d array
    # is that indexing with an empty tuple gives back a scalar. Hopefully that
    # is not too fragile.
    return (isinstance(val, np.generic) or
            (is_ndducktype(val) and val.shape == () and
             type(val[()]) is type(val)))

class _implements:
    """
    Register an __array_function__ method for a ducktype implementation.

    checked_args : iterable of strings, function. optional
        If not provided, all entries in the "types" list from numpy
        dispatch are checked to be of known class.

        If an iterable of argument names, those arguments are checked to be of
        known class if supplied.

        If a function, should take four arguments, (args, kwds, types,
        known_types) where args, kwds, types are the same as supplied to
        __array_function__, and "known_types" is a list of compatible types. An
        iterable of types may be returned to be checked to be of known class,
        or None to signify all args were checked, or a NotImplementedError may
        be raised to signal no match.

    Notes
    -----
    Goal is to allow both control and convenience. The callable form of
    checked_args is most powerful since it can be used to implement any
    desired behavior based purely on the dispatch "args" by raising
    NotImplementedError. This includes the behaviors obtained by giving a
    callable return an iterable of , of using a tuple, or not providing
    checked_args, so these latter forms are for convenience only.
    """
    def __init__(self, numpy_function, checked_args=None):
        self.npfunc = numpy_function
        self.checked_args = checked_args

    @classmethod
    def check_types(cls, types, known_types):
        # returns true if all types are known types or ndarray/scalar.
        return builtins.all((issubclass(t, known_types) or
                             t is np.ndarray or np.isscalar(t)) for t in types)

    def __call__(self, func):
        checked_args = self.checked_args

        if isinstance(checked_args, Iterable):
            sig = signature(func)
            def checked_args_func(args, kwargs, types, known_types):
                bound = sig.bind(*args, *kwargs).arguments
                types = (type(bound[a]) for a in checked_args if a in bound)
                return self.check_types(types, known_types)

        elif callable(checked_args):
            def checked_args_func(args, kwargs, types, known_types):
                try:
                    types = checked_args(args, kwargs, types, known_types)
                except NotImplementedError:
                    return False
                if types is None:
                    return True
                return self.check_types(types, known_types)

        elif checked_args is None:
            checked_args_func = lambda a, k, t, n: self.check_types(t, n)

        else:
            raise ValueError("invalid checked_args")

        self.handled_functions[self.npfunc] = (func, checked_args_func)

        return func

def new_ducktype_implementation():
    class new_impl(_implements):
        handled_functions = {}
        def __init__(self, numpy_function, checked_args=None):
            super().__init__(numpy_function, checked_args)

    return new_impl

def ducktype_linkscalar(arraytype, scalartype):
    arraytype.ArrayType = scalartype.ArrayType = arraytype
    arraytype.ScalarType = scalartype.ScalarType = scalartype

def get_duck_cls(*args):
    """
    Helper to make ducktypes Subclass-friendly.

    Finds the most derived class of MaskedArray/MaskedScalar.
    If given both an Array and a Scalar, convert the Scalar to an array first.
    In the case of two non-inheriting subclasses, raise TypeError.

    Parameters
    ==========
    *args : nested list/tuple or ndarray ducktype
        The bottom elements can be either ndarrays, scalars, ducktypes of
        either, or type objects of any of these.

    Returns
    =======
    arraytype : type
        The derived class of all of the inputs
    """
    cls = None
    for arg in args:
        acl = arg if isinstance(arg, type) else type(arg)

        if issubclass(acl, (np.ndarray, np.generic)):
            continue
        elif is_ndducktype(acl):
            atype, stype = acl.ArrayType, acl.ScalarType

            if cls is None or issubclass(acl, cls):
                cls = acl
                continue
            elif issubclass(cls, cls.ScalarType) and issubclass(acl, atype):
                cls = cls.ArrayType
            elif issubclass(cls, cls.ArrayType) and issubclass(acl, stype):
                acl = acl.ArrayType

            if issubclass(acl, cls):
                cls = acl
            elif not issubclass(cls, acl):
                raise TypeError(("Ambiguous mix of ducktypes {} and {}"
                                ).format(cls, acl))
        elif issubclass(acl, (list, tuple)):
            tmpcls = get_duck_cls(*arg)
            if tmpcls is not None and (cls is None or issubclass(cls, tmpcls)):
                cls = tmpcls

    if cls is None:
        return None
    return cls.ArrayType

