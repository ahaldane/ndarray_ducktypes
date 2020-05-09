ArrayCollection
===============

`ArrayCollection` is a NumPy `ndarray`-like type which stores an ordered set of
named NumPy arrays with a common shape but which may be different dtypes. It
supports NumPy-style indexing of the arrays as a group.

```python
>>> a = ArrayCollection({'age': [8, 10, 7, 8], 'weight': [10, 20, 13, 15]})
>>> a
ArrayCollection([( 8, 10), (10, 20), ( 7, 13), ( 8, 15)],
                dtype=[('age', '<i8'), ('weight', '<i8')])
>>> a[:2]
ArrayCollection([( 8, 10), (10, 20)],
                dtype=[('age', '<i8'), ('weight', '<i8')])
>>> a['age']
array([ 8, 10,  7,  8])
```

Comparison with similar Python types
------------------------------------

An ArrayCollection has a very similar interface as a NumPy structured
`ndarray` and they can often be used interchangeably. But while NumPy's
structured arrays are designed for lowlevel manipulation of binary blobs and
for interfacing with C programs, `ArrayCollection` is meant for simple
manipulation of multidimensional, multi-datatype datasets. At the technical
level, `ArrayCollection` is a container for a set of ndarrays, while structured
ndarrays are arrays of C-structs.  This means that users of `ArrayCollection`
do not need to worry about 'padding' bytes present in structured arrays, and
`ArrayCollection` will often be more cache-friendly because the individual arrays are not strided in memory as the fields are are in structured ndarrays.

`ArrayCollection` can be used as a "lite" version of Pandas `DataFrame` or
Xarray `Dataset`. You are encouraged to consider using these other projects
instead as they provide many additional high-level features useful for tabular
data analysis, such as named axes. `ArrayCollection` in contrast provides a simpler more direct interface to a set of ndarrays.

Quickstart
==========

`ArrayCollection` Construction
------------------------------

`ArrayCollection` uses the same datatype specification as `NumPy`'s structured arrays.

Indexing and Assignment
-----------------------

Tips
----

One can use `np.broadcast_to` to broadcast the input arrays to the
same shape in a memory-saving way, though this makes the arrays readonly.
See the `np.broadcast_to` docstring for details.
