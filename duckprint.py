from __future__ import division, absolute_import, print_function

import sys
import functools
if sys.version_info[0] >= 3:
    try:
        from _thread import get_ident
    except ImportError:
        from _dummy_thread import get_ident
else:
    try:
        from thread import get_ident
    except ImportError:
        from dummy_thread import get_ident

import numpy as np
from numpy import (concatenate, errstate, array, format_float_positional,
                   format_float_scientific, datetime_as_string, datetime_data,
                   ndarray, ravel, any, longlong, intc, int_, float_, complex_,
                   bool_, flexible)
from numpy.core import umath
import warnings
import contextlib
import os

from numpy.lib import NumpyVersion
if (NumpyVersion(np.__version__) < '1.15.10' or
    os.environ.get('NUMPY_EXPERIMENTAL_ARRAY_FUNCTION', 0) == 0):
    raise Exception("numpy __array_function__ must be enabled")

# WIP: Notes.
#
# The get/set_printoptions functionality is temporarily removed and being
# reworked. That is the purpose of all the `check_options` methods defined
# below, which are currently largely unused.
#

class FormatDispatcher(object):
    """
    Class used to determine which formatter to use to print an array's
    elements. The main method is `get_format_func`, called on an array,
    which will return the best formatter to use to print that array's elements.
    """
    def __init__(self, formatter_cls, options=None):
        self.formatters = [f() for f in formatter_cls]
        self.options = options or {}
        self.check_options()

    def get_format_func(self, elem, **options):
        opt = self.options.copy()
        if options:
            opt.update(options)

        opt['dispatch'] = self

        for f in self.formatters:
            if f.will_dispatch(elem):
                return f.get_format_func(elem, **opt)

        raise Exception("No dispatcher found for this array")

    def check_options(self, **kwds):
        if not kwds:
            kwds = self.options

        unset_opts = set(kwds.keys())

        for f in self.formatters:
            seen = f.check_options(**kwds)
            unset_opts = unset_opts.difference(set(seen))

        # return unknown options
        return list(unset_opts)

    def set_options(self, **kwds):
        self.check_options(**kwds)
        self.options.update(kwds)

class ElementFormatter(object):
    """
    Base class for the array element formatters.

    These have three methods:
      * `will_dispatch` is used by the dispatcher (above) to determine
        whether to use this formatter for an array. It should be passed the set
        of elements that will ultimately be printed.
      * `get_format_func` returns a function taking a single argument, an array
        element, which will format that element as a string.
      * `check_options` for validating options. This will validate any relevant
        options, and return a list of options it didn't handle.
    """
    def will_dispatch(self, elem):
        return False

    def get_format_func(self, elem, options=None):
        return lambda x: ''

    def check_options(self, **options):
        return []


class BoolFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.bool_)

    def get_format_func(self, elem, **options):
        truestr = ' True' if elem.shape != () else 'True'
        return lambda x: truestr if x else "False"


class IntegerFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.integer)

    def get_format_func(self, elem, **options):
        max_str_len = 0
        if elem.size > 0:
            max_str_len = max(len(str(np.max(elem))),
                              len(str(np.min(elem))))
        fmt = '{{:{}d}}'.format(max_str_len)
        return lambda x: fmt.format(x)


class FloatingFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.floating)

    def get_format_func(self, elem, **options):
        missing_opt = self.check_options(**options)
        if missing_opt:
            raise Exception("Missing options: {}".format(missing_opt))

        floatmode = options['floatmode']
        precision = None if floatmode == 'unique' else options['precision']
        suppress_small = options['suppress_small']
        sign = options['sign']
        infstr = options['infstr']
        nanstr = options['nanstr']
        exp_format = False
        pad_left, pad_right = 0, 0

        # only the finite values are used to compute the number of digits
        finite = umath.isfinite(elem)
        finite_vals = elem[finite]
        nonfinite_vals = elem[~finite]

        # choose exponential mode based on the non-zero finite values:
        abs_non_zero = umath.absolute(finite_vals[finite_vals != 0])
        if len(abs_non_zero) != 0:
            max_val = np.max(abs_non_zero)
            min_val = np.min(abs_non_zero)
            with np.errstate(over='ignore'):  # division can overflow
                if max_val >= 1.e8 or (not suppress_small and
                        (min_val < 0.0001 or max_val/min_val > 1000.)):
                    exp_format = True

        # do a first pass of printing all the numbers, to determine sizes
        if len(finite_vals) == 0:
            trim, exp_size, unique = '.', -1, True
        elif exp_format:
            trim, unique = '.', True
            if floatmode == 'fixed':
                trim, unique = 'k', False
            strs = (format_float_scientific(x, precision=precision,
                               unique=unique, trim=trim, sign=sign == '+')
                    for x in finite_vals)
            frac_strs, _, exp_strs = zip(*(s.partition('e') for s in strs))
            int_part, frac_part = zip(*(s.split('.') for s in frac_strs))
            exp_size = max(len(s) for s in exp_strs) - 1

            trim = 'k'
            precision = max(len(s) for s in frac_part)

            # this should be only 1 or 2. Can be calculated from sign.
            pad_left = max(len(s) for s in int_part)
            # pad_right is only needed for nan length calculation
            pad_right = exp_size + 2 + precision

            unique = False
        else:
            trim, unique = '.', True
            if floatmode == 'fixed':
                trim, unique = 'k', False
            strs = (format_float_positional(x, precision=precision,
                                       fractional=True,
                                       unique=unique, trim=trim,
                                       sign=sign == '+')
                    for x in finite_vals)
            int_part, frac_part = zip(*(s.split('.') for s in strs))
            pad_left = max(len(s) for s in int_part)
            pad_right = max(len(s) for s in frac_part)
            exp_size = -1

            if floatmode in ['fixed', 'maxprec_equal']:
                precision = pad_right
                unique = False
                trim = 'k'
            else:
                unique = True
                trim = '.'

        # account for sign = ' ' by adding one to pad_left
        if sign == ' ' and not any(np.signbit(finite_vals)):
            pad_left += 1

        # account for nan and inf in pad_left
        if len(nonfinite_vals) != 0:
            nanlen, inflen = 0, 0
            if any(umath.isinf(nonfinite_vals)):
                neginf = sign != '-' or any(isneginf(nonfinite_vals))
                inflen = len(infstr) + neginf
            if any(umath.isnan(elem)):
                nanlen = len(nanstr)
            offset = pad_right + 1  # +1 for decimal pt
            pad_left = max(nanlen - offset, inflen - offset, pad_left)

        def print_nonfinite(x):
            with errstate(invalid='ignore'):
                if umath.isnan(x):
                    ret = ('+' if sign == '+' else '') + nanstr
                else:  # isinf
                    infsgn = '-' if x < 0 else '+' if sign == '+' else ''
                    ret = infsgn + infstr
                return ' '*(pad_left + pad_right + 1 - len(ret)) + ret

        if exp_format:
            def print_finite(x):
                return format_float_scientific(x, precision=precision,
                           unique=unique, trim=trim, sign=sign == '+',
                           pad_left=pad_left, exp_digits=exp_size)
        else:
            def print_finite(x):
                return format_float_positional(x, precision=precision,
                           unique=unique, fractional=True, trim=trim,
                           sign=sign == '+',
                           pad_left=pad_left, pad_right=pad_right)

        def fmt(x):
            if umath.isfinite(x):
                return print_finite(x)
            else:
                return print_nonfinite(x)

        return fmt

    def check_options(self, **options):
        opt = set(['floatmode', 'precision', 'sign', 'infstr', 'nanstr',
               'suppress_small', 'floatnotation'])

        if 'floatmode' in options:
            if options['floatmode'] not in ['unique', 'unique_equal',
                                           'maxprec', 'maxprec_equal', 'fixed']:
                # note: Unique is the same as maxprec with infinite precision.
                # Unique_equal is the same as maxprec_equal w/ infinite prec.
                raise Exception()
        if 'floatnotation' in options:
            if options['floatnotation'] not in ['exponential',
                                                'positional', 'auto']:
                raise Exception()
        if 'precision' in options:
            if options['precision'] != None and options['precision'] < 0:
                raise Exception('precision must be positive or None')
        if 'sign' in options:
            if options['sign'] not in " +-":
                raise Exception("sign must be one of ' +-'")

        return list(opt.difference(options))


class ComplexFloatingFormatter(FloatingFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.complexfloating)

    def get_format_func(self, elem, **options):
        imag_opts = options.copy()
        imag_opts['sign'] = '+'

        formatter = FloatingFormatter()

        def fmt(x):
            r = formatter(x.real, **options)
            i = formatter(x.imag, **imag_opts)

            # add the 'j' before the terminal whitespace in i
            sp = len(i.rstrip())
            i = i[:sp] + 'j' + i[sp:]
            return r + i

        return fmt

class _TimelikeFormatter(ElementFormatter):
    def _datetime_fmt(self, elem, fmt_non_nat):
        non_nat = elem[~umath.isnat(elem)]
        if len(non_nat) > 0:
            # Max str length of non-NaT elements
            max_str_len = max(len(fmt_non_nat(np.max(non_nat))),
                              len(fmt_non_nat(np.min(non_nat))))
        else:
            max_str_len = 0
        if len(non_nat) < elem.size:
            # data contains a NaT
            max_str_len = max(max_str_len, 5)
        fmt = '%{}s'.format(max_str_len)
        nats = "'NaT'".rjust(max_str_len)

        return lambda x: nats if isnat(x) else fmt % fmt_non_nat(x)

class DatetimeFormatter(_TimelikeFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.datetime64)

    def get_format_func(self, elem, **options):
        # Get the unit from the dtype
        unit = options['unit']
        if options['unit'] is None:
            if elem.dtype.kind == 'M':
                unit = datetime_data(elem.dtype)[0]
            else:
                unit = 's'

        timezone = options['timezone']
        if timezone is None:
            timezone = 'naive'

        casting = options.get('casting', 'same_kind')
        if casting is None:
            casting = 'same_kind'

        def fmt_non_nat(x):
            return "'%s'" % datetime_as_string(x, unit=unit, timezone=timezone,
                                               casting=casting)

        return self._datetime_fmt(elem, fmt_non_nat)

    def check_options(self, **options):
        opt = set(['unit', 'timezone', 'casting'])
         #XXX do some more serious checking here?
        return list(opt.difference(options))


class TimedeltaFormatter(_TimelikeFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.timedelta64)

    def get_format_func(self, elem, **options):
        return self._datetime_fmt(elem, lambda x: str(x.astype('i8')))


class SubArrayFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return elem.dtype.shape is not ()

    def get_format_func(self, elem, **options):
        #XXX incorporate threshold?
        inner_fmt = options['fmt']

        def fmt(arr):
            if arr.ndim <= 1:
                return "[" + ", ".join(inner_fmt(a) for a in arr) + "]"
            return "[" + ", ".join(fmt(a) for a in arr) + "]"

        return fmt

class StructuredFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return elem.dtype.names is not None

    def get_format_func(self, elem, **options):
        format_funcs = []
        for field_name in elem.dtype.names:
            farr = elem[field_name]

            dispatcher = get_duckprint_dispatcher(farr)
            format_func = dispatcher.get_format_func(farr, **options)
            if elem.dtype[field_name].shape != ():
                fmttr = SubArrayFormatter()
                func = fmttr.get_format_func(elem, fmt=format_func, **options)
                format_funcs.append(func)
            else:
                format_funcs.append(format_func)

        if len(elem.dtype.names) == 1:
            return lambda x: "({},)".format(format_funcs[0](x[0]))

        def fmt(x):
            fieldstr = (fmt(f) for fmt,f in zip(format_funcs, x))
            return "({})".format(", ".join(fieldstr))
        return fmt

class ObjectFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtyp.typee, np.object_)

    def get_format_func(self, elem, **options):
        def fmt(x):
            fmtstr = 'list({!r})' if type(x) is list else '{!r}'
            return fmtstr.format(x)
        return fmt

class StringFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, (np.unicode_, np.string_))

    def get_format_func(self, elem, **options):
        return repr

class VoidFormatter(ElementFormatter):
    def will_dispatch(self, elem):
        return issubclass(elem.dtype.type, np.void)

    def get_format_func(self, elem, **options):
        return repr

# don't modify this
default_duckprint_options = {
    'floatmode': 'maxprec',
    'floatnotation': 'auto',
    'precision': 8,  # precision of floating point representations
    'suppress_small': False,  # don't print small floating values in exp format
    'nanstr': 'nan',
    'infstr': 'inf',
    'sign': '-',
    }

# Note: Order matters. Timedelta must go before Integer because they have the
# same underlying type.
default_duckprint_formatters = [BoolFormatter, TimedeltaFormatter,
    IntegerFormatter, FloatingFormatter, ComplexFloatingFormatter,
    StringFormatter, DatetimeFormatter, StructuredFormatter, VoidFormatter,
    ObjectFormatter]

default_duckprint_dispatcher = FormatDispatcher(default_duckprint_formatters,
                                                default_duckprint_options)

def get_duckprint_dispatcher(arr):
    if hasattr(arr, '__nd_duckprint_dispatcher__'):
        return arr.__nd_duckprint_dispatcher__()
    else:
        return default_duckprint_dispatcher

def _leading_trailing(a, edgeitems, index=()):
    """
    Keep only the N-D corners (leading and trailing edges) of an array.
    """
    axis = len(index)
    if axis == a.ndim:
        return a[index]

    if a.shape[axis] > 2*edgeitems:
        return concatenate((
            _leading_trailing(a, edgeitems, index + np.index_exp[ :edgeitems]),
            _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:])
        ), axis=axis)
    else:
        return _leading_trailing(a, edgeitems, index + np.index_exp[:])

# only needed for recursive object arrays. See if we might only use it then?
def _recursive_guard(fillvalue='...'):
    """
    Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs

    Decorates a function such that if it calls itself with the same first
    argument, it returns `fillvalue` instead of recursing.

    Largely copied from reprlib.recursive_repr
    """

    def decorating_function(f):
        repr_running = set()

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                return f(self, *args, **kwargs)
            finally:
                repr_running.discard(key)

        return wrapper

    return decorating_function


# gracefully handle recursive calls, when object arrays contain themselves
@_recursive_guard()
def _array2string(a, dispatcher, options, separator, prefix, suffix, linewidth,
                  threshold, edgeitems):
    if a.size > threshold:
        summary_insert = "..."
        data = _leading_trailing(data, edgeitems)
    else:
        summary_insert = ""

    # find the right formatting function for the array
    format_function = dispatcher.get_format_func(a, **options)

    # skip over "["
    next_line_prefix = " "
    # skip over array(
    next_line_prefix += " "*len(prefix)

    lst = _formatArray(a, format_function, linewidth, next_line_prefix,
                       separator, edgeitems, summary_insert)
    return lst

def duck_array2string(a, separator=' ', prefix="", suffix="", linewidth=75,
                      threshold=1000, edgeitems=3, dispatcher=None, **options):
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    separator : str, optional
        Inserted between elements.
    prefix : str, optional
    suffix: str, optional
        The length of the prefix and suffix strings are used to respectively
        align and wrap the output. An array is typically printed as::

          prefix + array2string(a) + suffix

        The output is left-padded by the length of the prefix string, and
        wrapping is forced at the column ``max_line_width - len(suffix)``.
        It should be noted that the content of prefix and suffix strings are
        not included in the output.
    linewidth : int, optional
        The maximum number of columns the string should span. Newline
        characters splits the string appropriately after array elements.
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension.
    dispatcher : FormatDispatcher
        FormatDispatcher object used to print array elements. Uses
        the default dispatcher if None.

    All other keyword options are passed on to the FormatDispatcher.

    Returns
    -------
    array_str : str
        String representation of the array.

    Raises
    ------
    TypeError
        if a callable in `formatter` does not return a string.

    Notes
    -----
    If a formatter is specified for a certain type, the `precision` keyword is
    ignored for that type.

    This is a very flexible function; `array_repr` and `array_str` are using
    `array2string` internally so keywords with the same name should work
    identically in all three functions.

    Examples
    --------
    >>> x = np.array([1e-16,1,2,3])
    >>> print(np.array2string(x, precision=2, separator=',',
    ...                       suppress_small=True))
    [ 0., 1., 2., 3.]

    >>> x  = np.arange(3.)
    >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
    '[0.00 1.00 2.00]'

    >>> x  = np.arange(3)
    >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
    '[0x0L 0x1L 0x2L]'

    """
    if dispatcher is None:
        dispatcher = get_duckprint_dispatcher(a)

    # treat as a null array if any of shape elements == 0
    if a.size == 0:
        return "[]"

    return _array2string(a, dispatcher, options, separator, prefix, suffix,
                         linewidth, threshold, edgeitems)


def _extendLine(s, line, word, line_width, next_line_prefix):
    needs_wrap = len(line) + len(word) > line_width

    if needs_wrap:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    line += word
    return s, line


def _formatArray(a, format_function, line_width, next_line_prefix,
                 separator, edge_items, summary_insert):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output

    """
    def recurser(index, hanging_indent, curr_width):
        """
        By using this local function, we don't need to recurse with all the
        arguments. Since this function is not created recursively, the cost is
        not significant
        """
        axis = len(index)
        axes_left = a.ndim - axis

        if axes_left == 0:
            return format_function(a[index])

        # when recursing, add a space to align with the [ added, and reduce the
        # length of the line by 1
        next_hanging_indent = hanging_indent + ' '
        next_width = curr_width - len(']')

        a_len = a.shape[axis]
        show_summary = summary_insert and 2*edge_items < a_len
        if show_summary:
            leading_items = edge_items
            trailing_items = edge_items
        else:
            leading_items = 0
            trailing_items = a_len

        # stringify the array with the hanging indent on the first line too
        s = ''

        # last axis (rows) - wrap elements if they would not fit on one line
        if axes_left == 1:
            # the length up until the beginning of the separator / bracket
            elem_width = curr_width - max(len(separator.rstrip()), len(']'))

            line = hanging_indent
            for i in range(leading_items):
                word = recurser(index + (i,), next_hanging_indent, next_width)
                s, line = _extendLine(
                    s, line, word, elem_width, hanging_indent)
                line += separator

            if show_summary:
                s, line = _extendLine(
                    s, line, summary_insert, elem_width, hanging_indent)
                line += separator

            for i in range(trailing_items, 1, -1):
                word = recurser(index + (-i,), next_hanging_indent, next_width)
                s, line = _extendLine(
                    s, line, word, elem_width, hanging_indent)
                line += separator

            word = recurser(index + (-1,), next_hanging_indent, next_width)
            s, line = _extendLine(
                s, line, word, elem_width, hanging_indent)

            s += line

        # other axes - insert newlines between rows
        else:
            s = ''
            line_sep = separator.rstrip() + '\n'*(axes_left - 1)

            for i in range(leading_items):
                nested = recurser(index + (i,), next_hanging_indent, next_width)
                s += hanging_indent + nested + line_sep

            if show_summary:
                s += hanging_indent + summary_insert + line_sep

            for i in range(trailing_items, 1, -1):
                nested = recurser(index + (-i,), next_hanging_indent,
                                  next_width)
                s += hanging_indent + nested + line_sep

            nested = recurser(index + (-1,), next_hanging_indent, next_width)
            s += hanging_indent + nested

        # remove the hanging indent, and wrap in []
        s = '[' + s[len(hanging_indent):] + ']'
        return s

    try:
        # invoke the recursive part with an initial index and prefix
        return recurser(index=(),
                        hanging_indent=next_line_prefix,
                        curr_width=line_width)
    finally:
        # recursive closures have a cyclic reference to themselves, which
        # requires gc to collect (gh-10620). To avoid this problem, for
        # performance and PyPy friendliness, we break the cycle:
        recurser = None


_typelessdata = [int_, float_, complex_, bool_]
if issubclass(intc, int):
    _typelessdata.append(intc)
if issubclass(longlong, int):
    _typelessdata.append(longlong)

def dtype_short_repr(dtype):
    """
    Convert a dtype to a short form which evaluates to the same dtype.

    The intent is roughly that the following holds

    >>> from numpy import *
    >>> assert eval(dtype_short_repr(dt)) == dt
    """
    if dtype.names is not None:
        # structured dtypes give a list or tuple repr
        return str(dtype)
    elif issubclass(dtype.type, flexible):
        # handle these separately so they don't give garbage like str256
        return "'%s'" % str(dtype)

    typename = dtype.name
    # quote typenames which can't be represented as python variable names
    if typename and not (typename[0].isalpha() and typename.isalnum()):
        typename = repr(typename)

    return typename

def duck_repr(arr, **options):
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.

    Returns
    -------
    string : str
      The string representation of an array.

    Examples
    --------
    >>> repr(np.array([1,2]))
    'array([1, 2])'
    >>> repr(np.ma.array([0.]))
    'MaskedArray([ 0.])'
    >>> repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> repr(x, precision=6, suppress_small=True)
    'array([ 0.000001,  0.      ,  2.      ,  3.      ])'

    """
    linewidth = 75

    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = "array"

    skipdtype = False
    if arr.dtype.type in _typelessdata and arr.dtype.names is None:
        skipdtype = True

    prefix = class_name + "("
    suffix = ")" if skipdtype else ","

    if arr.size > 0 or arr.shape == (0,):
        lst = duck_array2string(arr, separator=', ', prefix=prefix,
                                suffix=suffix, **options)
    else:  # show zero-length shape unless it is (0,)
        lst = "[], shape=%s" % (repr(arr.shape),)

    arr_str = prefix + lst + suffix

    if skipdtype:
        return arr_str

    dtype_str = "dtype={})".format(dtype_short_repr(arr.dtype))

    # compute whether we should put dtype on a new line: Do so if adding the
    # dtype would extend the last line past max_line_width.
    # Note: This line gives the correct result even when rfind returns -1.
    last_line_len = len(arr_str) - (arr_str.rfind('\n') + 1)
    spacer = " "
    if last_line_len + len(dtype_str) + 1 > linewidth:
        spacer = '\n' + ' '*len(class_name + "(")

    return arr_str + spacer + dtype_str


def duck_str(a, **options):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Parameters
    ----------
    a : ndarray
        Input array.

    Examples
    --------
    >>> str(np.arange(3))
    '[0 1 2]'

    """
    return duck_array2string(a, separator=' ', prefix="", **options)

