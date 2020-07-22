

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


# Flags object for MaskedArray only supports "WRITEABLE" flag
class MaskedFlagsObj:
    def __init__(writeable=True):
        self._writeable = writeable

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, val):
        self[attr] = val

    def __getitem__(self, ind):
        if ind == 'WRITEABLE':
            return self._writeable
        raise KeyError('Unknown flag')

    def __setitem__(self, ind, val):
        if ind == 'WRITEABLE':
            self._writeable = bool(val)
        raise KeyError('Unknown flag')

    def __repr__(self):
        return repr({'WRITEABLE': self.writeable})
#
# Far-out ideas for making these choice configurable: Since the mask is stored
# as a byte anyway, maybe we could have two kinds of masked values: Sticky and
# nonsticky masks? So the mask would be stored as 'u1', 0=unmasked, 1=unsticky,
# 2=sticky. For the invalid domain conversions, someone might also want for
# this not to happen. Maybe instead we should implement these choices as
# subclasses, so we would have a subclass without invalid domin conversion.

def test_coverage():
    # temporary code to figure out our api coverage
    api = ['np.empty_like', 'np.concatenate', 'np.inner', 'np.where',
    'np.lexsort', 'np.can_cast', 'np.min_scalar_type', 'np.result_type',
    'np.dot', 'np.vdot', 'np.bincount', 'np.ravel_multi_index',
    'np.unravel_index', 'np.copyto', 'np.putmask', 'np.packbits',
    'np.unpackbits', 'np.shares_memory', 'np.may_share_memory', 'np.is_busday',
    'np.busday_offset', 'np.busday_count', 'np.datetime_as_string',
    'np.zeros_like', 'np.ones_like', 'np.full_like', 'np.count_nonzero',
    'np.argwhere', 'np.flatnonzero', 'np.correlate', 'np.convolve', 'np.outer',
    'np.tensordot', 'np.roll', 'np.rollaxis', 'np.moveaxis', 'np.cross',
    'np.allclose', 'np.isclose', 'np.array_equal', 'np.array_equiv', 'np.take',
    'np.reshape', 'np.choose', 'np.repeat', 'np.put', 'np.swapaxes',
    'np.transpose', 'np.partition', 'np.argpartition', 'np.sort', 'np.argsort',
    'np.argmax', 'np.argmin', 'np.searchsorted', 'np.resize', 'np.squeeze',
    'np.diagonal', 'np.trace', 'np.ravel', 'np.nonzero', 'np.shape',
    'np.compress', 'np.clip', 'np.sum', 'np.any', 'np.all', 'np.cumsum',
    'np.ptp', 'np.amax', 'np.amin', 'np.alen', 'np.prod', 'np.cumprod',
    'np.ndim', 'np.size', 'np.around', 'np.mean', 'np.std', 'np.var',
    'np.round_', 'np.product', 'np.cumproduct', 'np.sometrue', 'np.alltrue',
    'np.rank', 'np.array2string', 'np.array_repr', 'np.array_str',
    'np.char.equal', 'np.char.not_equal', 'np.char.greater_equal',
    'np.char.less_equal', 'np.char.greater', 'np.char.less', 'np.char.str_len',
    'np.char.add', 'np.char.multiply', 'np.char.mod', 'np.char.capitalize',
    'np.char.center', 'np.char.count', 'np.char.decode', 'np.char.encode',
    'np.char.endswith', 'np.char.expandtabs', 'np.char.find', 'np.char.index',
    'np.char.isalnum', 'np.char.isalpha', 'np.char.isdigit', 'np.char.islower',
    'np.char.isspace', 'np.char.istitle', 'np.char.isupper', 'np.char.join',
    'np.char.ljust', 'np.char.lower', 'np.char.lstrip', 'np.char.partition',
    'np.char.replace', 'np.char.rfind', 'np.char.rindex', 'np.char.rjust',
    'np.char.rpartition', 'np.char.rsplit', 'np.char.rstrip', 'np.char.split',
    'np.char.splitlines', 'np.char.startswith', 'np.char.strip',
    'np.char.swapcase', 'np.char.title', 'np.char.translate', 'np.char.upper',
    'np.char.zfill', 'np.char.isnumeric', 'np.char.isdecimal', 'np.atleast_1d',
    'np.atleast_2d', 'np.atleast_3d', 'np.vstack', 'np.hstack', 'np.stack',
    'np.block', 'np.einsum_path', 'np.einsum', 'np.fix', 'np.isposinf',
    'np.isneginf', 'np.asfarray', 'np.real', 'np.imag', 'np.iscomplex',
    'np.isreal', 'np.iscomplexobj', 'np.isrealobj', 'np.nan_to_num',
    'np.real_if_close', 'np.asscalar', 'np.common_type', 'np.fliplr',
    'np.flipud', 'np.diag', 'np.diagflat', 'np.tril', 'np.triu', 'np.vander',
    'np.histogram2d', 'np.tril_indices_from', 'np.triu_indices_from',
    'np.linalg.tensorsolve', 'np.linalg.solve', 'np.linalg.tensorinv',
    'np.linalg.inv', 'np.linalg.matrix_power', 'np.linalg.cholesky',
    'np.linalg.qr', 'np.linalg.eigvals', 'np.linalg.eigvalsh', 'np.linalg.eig',
    'np.linalg.eigh', 'np.linalg.svd', 'np.linalg.cond',
    'np.linalg.matrix_rank', 'np.linalg.pinv', 'np.linalg.slogdet',
    'np.linalg.det', 'np.linalg.lstsq', 'np.linalg.norm',
    'np.linalg.multi_dot', 'np.histogram_bin_edges', 'np.histogram',
    'np.histogramdd', 'np.rot90', 'np.flip', 'np.average', 'np.piecewise',
    'np.select', 'np.copy', 'np.gradient', 'np.diff', 'np.interp', 'np.angle',
    'np.unwrap', 'np.sort_complex', 'np.trim_zeros', 'np.extract', 'np.place',
    'np.cov', 'np.corrcoef', 'np.i0', 'np.sinc', 'np.msort', 'np.median',
    'np.percentile', 'np.quantile', 'np.trapz', 'np.meshgrid', 'np.delete',
    'np.insert', 'np.append', 'np.digitize', 'np.broadcast_to',
    'np.broadcast_arrays', 'np.ix_', 'np.fill_diagonal',
    'np.diag_indices_from', 'np.nanmin', 'np.nanmax', 'np.nanargmin',
    'np.nanargmax', 'np.nansum', 'np.nanprod', 'np.nancumsum', 'np.nancumprod',
    'np.nanmean', 'np.nanmedian', 'np.nanpercentile', 'np.nanquantile',
    'np.nanvar', 'np.nanstd', 'np.take_along_axis', 'np.put_along_axis',
    'np.apply_along_axis', 'np.apply_over_axes', 'np.expand_dims',
    'np.column_stack', 'np.dstack', 'np.array_split', 'np.split', 'np.hsplit',
    'np.vsplit', 'np.dsplit', 'np.kron', 'np.tile', 'np.lib.scimath.sqrt',
    'np.lib.scimath.log', 'np.lib.scimath.log10', 'np.lib.scimath.logn',
    'np.lib.scimath.log2', 'np.lib.scimath.power', 'np.lib.scimath.arccos',
    'np.lib.scimath.arcsin', 'np.lib.scimath.arctanh', 'np.poly', 'np.roots',
    'np.polyint', 'np.polyder', 'np.polyfit', 'np.polyval', 'np.polyadd',
    'np.polysub', 'np.polymul', 'np.polydiv', 'np.ediff1d', 'np.unique',
    'np.intersect1d', 'np.setxor1d', 'np.in1d', 'np.isin', 'np.union1d',
    'np.setdiff1d', 'np.save', 'np.savez', 'np.savez_compressed', 'np.savetxt',
    'np.fv', 'np.pmt', 'np.nper', 'np.ipmt', 'np.ppmt', 'np.pv', 'np.rate',
    'np.irr', 'np.npv', 'np.mirr', 'np.pad', 'np.fft.fftshift',
    'np.fft.ifftshift', 'np.fft.fft', 'np.fft.ifft', 'np.fft.rfft',
    'np.fft.irfft', 'np.fft.hfft', 'np.fft.ihfft', 'np.fft.fftn',
    'np.fft.ifftn', 'np.fft.fft2', 'np.fft.ifft2', 'np.fft.rfftn',
    'np.fft.rfft2', 'np.fft.irfftn', 'np.fft.irfft2']

    n_implemented, n_skipped, n_missing = 0, 0, 0
    for a in api:
        if a.startswith('np.char.'):
            n_skipped += 1
            continue
        if a.startswith('np.fft.'):
            n_skipped += 1
            continue

        parts = a.split('.')[1:]
        f = np
        while parts and f:
            f = getattr(f, parts.pop(0), None)
        if f is None:
            print("Missing", a)
            continue
        if f not in HANDLED_FUNCTIONS:
            n_missing += 1
            print("Missing:", a)
            pass
        else:
            n_implemented += 1
        #    print("Have", a)
    print("Total api:   ", len(api))
    print("Skipped:     ", n_skipped)
    print("Implemented: ", n_implemented)
    print("Missing:     ", n_missing)
