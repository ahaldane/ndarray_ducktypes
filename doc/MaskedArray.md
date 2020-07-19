MaskedArray
===========

This `MaskedArray` module provides "missing value" support on top of NumPy's `ndarray` type. You can construct a `MaskedArray` from any `ndarray` and then mask selected elements, which will be ignored in subsequent NumPy operations:

```python
>>> import numpy as np
>>> from MaskedArray import MaskedArray, MaskedScalar, X
>>> m = MaskedArray(np.arange(5))
>>> m[2:4] = X  # X denotes mask
>>> m
MaskedArray([0, 1, X, X, 4])
>>> np.sum(m)
MaskedScalar(5)
>>> m + MaskedArray([X, 5, 6, 1, 2])
MaskedArray([X, 6, X, X, 6])
```

`MaskedArray` is implemented as an `ndarray` ducktype, and so `MaskedArrays` can be substituted into almost any NumPy expression, for instance involving `np.sum`, `np.mean`, `np.concatenate`, and many other vectorized mathematical operations. There are a small number of cases in which MaskedArrays cannot be substituted, described further below.

Historical Relationship to NumPy's `MaskedArray`s
-------------------------------------------------

This module should not be confused with NumPy's builtin `np.ma.MaskedArray` class, which has the same name and similar functionality but is otherwise independent. This module was developed by forking NumPy's `np.ma.MaskedArray` codebase and updating it to take advantage of NumPy's new and powerful `__array_function__` interface for creating `ndarray` ducktypes. It is intended as a more modern replacement for NumPy's `np.ma.MaskedArray`.

Quickstart
==========

This module introduces three main new types on top of `NumPy`'s types:
 * `MaskedArray`, which behaves like NumPy's `ndarray` type but with masked elements
 * `MaskedScalar`, which behaves like NumPy's scalar types but which can be masked
 * `X`, a special variable representing a masked value for use in `MaskedArray` construction and assignment.

`MaskedArray` and `MaskedScalar` are implemented by storing a "data" NumPy `ndarray` of any NumPy datatype along with an auxiliary boolean "mask" `ndarray` of the same shape describing which elements of the data array are masked. Both the data and mask `ndarrays` are normally hidden from the user, but it is useful to know this to understand `MaskedArray` behavior.

MaskedArray construction
------------------------

`MaskedArray`s can be created in multiple ways. One way is using Python lists,
in which case `X` can be used to signify masked elements:

```python
>>> a = MaskedArray([[1, X, 3], [X, X, 2], [X, 4, 1]])
>>> a
MaskedArray([[1, X, 3],
             [X, X, 2],
             [X, 4, 1]])
```

Another way is by specifying separate `data` and `mask` objects as the first two arguments of the constructor. The mask can be any object which NumPy can cast to a boolean array and broadcast to the same shape as the data:

```python
>>> MaskedArray(np.ones(4), [0, 1, 0, 1])
MaskedArray([1., X , 1., X ])
```

`MaskedArray`s can also be constructed from other `MaskedArray`s, resulting in a view of the original:

```python
>>> MaskedArray(a)
MaskedArray([[1, X, 3],
             [X, X, 2],
             [X, 4, 1]])
```

`MaskedArray` Basic Attributes
-----------------------------

`MaskedArray` and `MaskedScalar` support two new attributes relative to `ndarray`:

The `.filled` method  returns a copy of the `MaskedArray`s data as a plain `ndarray`, with all masked elements replaced by a fill value, which is 0 by default. The `.filled()` method is the primary way of converting a `MaskedArray` back to an `ndarray`.

```python
>>> a.filled()
array([[1, 0, 3],
       [0, 0, 2],
       [0, 4, 1]])
```

The `.mask` attribute gives a readonly view of the boolean `ndarray` representing which elements are masked.

```python
>>> a.mask
array([[False,  True, False],
       [ True,  True, False],
       [ True, False, False]])
```
This is readonly to prevent exposure of uninitialized values to the user. To mask elements you should use assignment with `X` as described below. 

Otherwise `MaskedArray` supports most of the basic attributes of `ndarrays` like `.shape`, `.dtype`, `.strides` and others, with a few exceptions such as `.base`.

MaskedScalars and the masked input marker `X`
---------------------------------------------

When operating on a `MaskedArray`, NumPy will return a `MaskedScalar` object in  situations which ordinarily return a NumPy scalar object, for instance when indexing an array to get a single element. While `MaskedScalar`s support almost all the methods and attributes of NumPy scalars there are some important differences. Notably, `MaskedScalar`s are not part of the NumPy scalar type hierarchy, and so for instance code of the form `isinstance(var, np.inexact)` will fail, and `np.isscalar` will return `False`. In such situations you can use `.filled` first to obtain a NumPy scalar.  Like NumPy scalars, `MaskedScalar`s are immutable (except for structured scalars) and can be used as dict keys.

The special `X` variable is not a `MaskedScalar`, and does not share attributes with NumPy types and does not have a `dtype`. It is a singleton of a special `MaskedX` class which acts as a masked-value signifier. When it is used in `MaskedArray` operations and assignments it is effectively promoted to a masked `MaskedScalar` of the appropriate `dtype`. As a convenience the `X` variable can be explicitly promoted to a masked `MaskedScalar` of a specified `dtype` by calling it with the `dtype` as argument. The `repr` of `MaskedScalar`s whose mask is `True` is represented using this `X(dtype)` construction:

```python
>>> X
masked_input
>>> MaskedScalar(1)
MaskedScalar(1)
>>> MaskedScalar(1, mask=True)
X(int64)
>>> X(np.float64)
X(float64)
```

General Principles of Behavior
==============================

NumPy operations on `MaskedArray`s follow a few general principles of behavior.

Unary and Binary operations
---------------------------

For simple unary and binary operations, such as addition of two arrays, output elements will be masked at positions where either input element is masked:

```python
>>> a = MaskedArray([1, X, X])
>>> b = MaskedArray([1, 2, X])
>>> a + b
MaskedArray([2, X, X])
>>> a == b
MaskedArray([True, X, X])
```

Note that after invalid mathematical operations such as division-by-zero the resulting output element is *not* automatically masked and can lead to `inf` or `nan` in your data, with the standard numpy warning in such cases. To avoid invalid operations you must apply appropriate masks, depending on the situation:
```python
>>> arr = MaskedArray([2, 0, 4, X])
>>> 1.0 / arr
RuntimeWarning: divide by zero encountered in true_divide
MaskedArray([0.5 ,  inf, 0.25, X   ])
>>> arr[arr == 0] = X
>>> 1.0 / arr
MaskedArray([0.5 , X   , 0.25, X   ])
```

Reductions
----------

For "reduction"-like operations involving many elements, such as summing an array, the masked elements do not contribute to the reduction. The result will be masked only if there were no unmasked input elements:

```python
>>> np.sum(MaskedArray([1, X, 2])
MaskedScalar(3)
>>> np.sum(MaskedArray([X, X, X], dtype=int))
X(int64)
```

`MaskedArray` reductions are implemented by replacing all masked elements by the appropriate identity element and then performing the NumPy reduction. For instance, `np.sum(arr)` is implemented using the "ufunc" reduction  `np.add.reduce(arr)` after replacing all masked elements by `np.add.identity`, which is 0. In consequence `MaskedArray` does not support reductions for "ufuncs" that do not have an identity element.

Truthiness of Masked Values
----------------------------

 When a `MaskedScalar` is tested for truth, for instance in if-statements, it will return `False` if masked, or the same as the corresponding NumPy scalar if not masked. To get masked values to test as `True`, use `scalar.filled(True)` first.

`np.nonzero` and `np.where` will similarly treat masked values as `False` or zero-like, and so do not return the indices of masked values.

`np.any` treats masked values as `False` (since it is implemented as `np.logical_or.reduce`, whose identity value is `False`), and `np.all` treats masked values as `True`. Use `.filled` to get different behaviors.

Fancy Indexing
--------------

`MaskedArray` supports fancy-indexing with boolean or integer array indexes, with some differences relative to `ndarray`.

For integer array fancy-indexing, the index array must be an `ndarray` and not a `MaskedArray`. Masked integer fancy indexes are not currently supported. Otherwise `MaskedArray` indexing with an integer array behaves just like for `ndarray`.

For boolean array fancy-indexing the index array may be a `MaskedArray`, in which case masked values are treated as `False`. Thus `arr[bool_index]` behaves the same as `arr[np.nonzero(bool_index)]`.

`MaskedArray` in indexing-like operations
-----------------------------------------

Unlike most other NumPy API functions, NumPy functions which perform indexing-like operations or produce index arrays, such as `np.take`, `np.nonzero`, `np.argsort` will generally use `ndarray`s instead of `MaskedArrays` for index variables.  For operations with integer-index inputs, eg `np.take`, the input index array should should be an `ndarray`. For operations with integer-index outputs, eg `np.nonzero`, the returned array will be an `ndarray`.

`out` arguments
---------------

For NumPy functions which takes an `out` argument, such as `np.sum` or `np.add`, if a `MaskedArray` is one of the inputs then the out argument must also be a `MaskedArray` if given, with exceptions for index-like operations. This is to prevent situations in which the mask is lost and uninitialized data is exposed to the user. 

Sorting masked values
---------------------

In sorting operations such as `np.sort`, `np.lexsort`, `np.argsort`, `np.partition`, masked elements will be treated as greater than the greatest value of the `MaskedArray`'s dtype. In other words they will sort to the "end" of the sorted array.

For `np.max`, `np.min`, `np.argmin`, and `np.argmax`, non-masked values are returned before any masked value. In other words, these will only return a masked value, or index of a masked value, if all input values were masked. The `arg*` methods will return `ndarrays` or `int` scalars, following the discussion of indexing-like operations above.

Views and Copies
----------------

It is important to understand when a NumPy operation returns a view or a copy, and this is doubly true for `MaskedArray`s as either the mask or data may be views. `MaskedArray` frequently uses views for performance reasons, but this means that with improper use the user can sometimes get access to the uninitialized values at masked positions.

During `MaskedArray` construction, when supplying the data and mask as `ndarrays`,  in most cases these are viewed by the `MaskedArray`and will be modified by operations on the `MaskedArray`. This can lead to confusing or undesirable behavior if you separately make use of the `ndarray` supplied as data, particularly as `MaskedArray` may fill the data `ndarray` with nonsense values at masked positions. To avoid this you can use the `copy=True` argument to the `MaskedArray` constructor.

The `.mask` attribute of a `MaskedArray` is a readonly view of the internal mask data, and will be updated if the `MaskedArray` is assigned to. The `.filled()` method returns a copy of the data by default, but as an optimization for careful users the `view` argument of `.filled` can be set to `True` to return a readonly view. Use caution with this view, because any subsequent operation with the original `MaskedArray` may put nonsense values at masked positions of its internal data array.

Unlike `ndarray`s, `MaskedArray`s do not support a `.base` attribute which can be used to tell if an array is a view. However, it is possible to check the `.base` attribute of the `ndarray`s returned by `.mask`, or `.filled` with `view=True`.

Differences in behavior between MaskedArray and ndarray
-------------------------------------------------------

There are a few types of expressions which will fail for MaskedArrays even though they work for ndarrays, which are noted here since they prevent substitution of MaskedArrays for ndarrays in these cases.

.base and .flags attributes
---------------------------

MaskedArrays do not have a .base attribute as described above.

The `.flags` attribute of a MaskedArray  is equivalent to the `.flags` attribute of the internal data array.

The .real and .imag attributes of Complex arrays and Fields of Structured Arrays
--------------------------------------------------------------------------------

`MaskedArray`s of complex `dtype` have `.real` and `.imag` attributes, like `ndarrays`, which can also be obtained using `np.real` and `np.imag`. These return views into the original array. However, unlike ndarrays, these attributes return readonly arrays and attempting to assign to them will raise an exception because it is unclear how to update the mask of the original complex array when its individual real or imaginary components are modified or masked. As a workaround one can make a copy of the real or imag part to get a writeable array.

If having writeable masked `.real/.imag` attributes is important to you, you might consider a few alternatives to `MaskedArray` which have better support for this kind of behavior. One option is to use numpy's `np.nan*` functions instead of `MaskedArray`, treating `np.nan` as a masked value signifier. Another possibility is to use an "ArrayCollection" with dtype `'[('real', 'f8'), ('imag', 'f8')]`.

Individuals fields of structured array behave similarly, as `MaskedArray` does not support masking individual fields of a structured datatype. Each structured element as a whole can be masked. Accessing an individual field of a masked structured array will give a readonly view of the original masked array.

If you wish to mask individual fields, as an alternative consider using the `MaskedArrayCollection` class which behaves similarly to structured arrays but allows the user to separately mask each named array in the collection.


Advanced MaskedArray Usage and implementation details
=====================================================

Implementation Details
----------------------

Describe using ._data, ._mask

Using MaskedArray to mask other ndarray ducktypes
-------------------------------------------------

It is sometimes desirable to mask ducktypes of ndarray, rather than plain ndarrays. For instance, one might want to make a masked-unit type which supports both masked values and keeps track of scientific units associated to the array.

MaskedArray is designed to support such behavior though either composition or encapsulation . 

The simplest is encapsulation: The 'data' argument to the MaskedArray constructor can be any ndarray ducktype or subclass which follows numpy's indexing and broadcasting behavior, and this ducktype will be preserved in the course of all masked operations. In other words, `MaskedArray` can act as a container type which encapsulates the other ducktype. You can always get back access to the encapsulated ducktyped array using `.filled()`. For instance:

```python

class MySubtype(np.ndarray):
    def __init__(self, *args, **kwargs):
        self.new_attr = 10
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super().repr + str(self.new_attr)

my_arr = MySubtype([1,2,3])
my_marr = MaskedArray(my_arr, mask=[0,1,0])
my_arr
(2 + my_arr).filled()
```

Encapsulation is easy to implement, but it is missing  some desirable features: Attributes of the encapsulated type are not exposed on the masked instance, and the `repr` and `str` of the `MaskedArray` do not display anything about the contained type, and numpy functions which create new masked arrays may lose the contained ducktype class information. 

To control these behaviors we recommend you combine or subclass `MaskedArray` and the other ducktype to create a composite type. [TODO: example subclass]


Statistical Functions
=====================

It is often ambiguous how to implement missing values in many of the more complex statistical functions, such correlation coefficients. This MaskedArray implementation makes choices which can have benefits and drawbacks in different situations, described here.

First in `np.cov` MaskedArray computes each covariance value separately by simply leaving out the masked datapoints when taking each expectation value. This can lead to a covariance matrix which is not positive semi-definite. In many cases a more involved approach is needed to estimate covariance matrices with missing data, many ways are documented in the scientific literature.



Appendix: Behavior Changes relative to np.ma.MaskedArray
========================================================

 * No support for masking individual fields of structured dtypes. Use an `ArrayCollection` of `MaskedArray`s instead, or (better) use `MaskedArrayCollection`.
 * No more `nomask`: the mask is always stored as a full ndarray.
 * No more returned `masked` singleton: Instead scalar values are returned as MaskedScalar instances. See comments in MaskedArray.py for discussion. A special masked singleton called `X` with no dtype exists for array construction purposes only.
 * No more `fill_value` stored with instances: You must always supply the desired fill as an argument to `filled()`
 * No more attempt to preserve the hidden data behind masked values. MaskedArray is free to modify these elements arbitrarily.
 * Features preserved from numpy's maskedarray: 1. The mask is not "sticky", it behaves as the "ignore" or "skipna" style described in the MaskedArray NEP. "na" style will not be supported.
 * no checking of ufunc domains: invalid operations do not lead to mask.
  Instead user must explicitly add masks to avoid invalid operations
 * out arguments to ufuncs/methods must be MaskedArrays too
 * more careful preservation of dtype (old MA would often cast to float64)
 * np.sort now doesn't mix maxval and masked vals up.
