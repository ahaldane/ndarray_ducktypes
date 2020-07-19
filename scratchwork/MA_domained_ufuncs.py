
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
