#!/usr/bin/env python
from ArrayCollection import ArrayCollection
from MaskedArray import MaskedArray
import numpy as np

# Tests for Masked ArrayCollections.
#
# First try: Simply make an arraycollection of MaskedArrays. Downside: this
# strategy does not give a "filled" method. Probably to get a masked
# ArrayCollection we should really subclass ArrayCollection to have a
# fill_value and a filled() method

a = MaskedArray(np.arange(10), np.arange(10)%3)
b = MaskedArray(np.arange(10.) + 13, np.arange(10)%2)

c = ArrayCollection([('age', a), ('weight', b)])
print(repr(c))
c['age'] += 100
print(repr(c))

# second try: Subclass of ArrayCollection

class MaskedArrayCollection(ArrayCollection):
    def __init__(self, data, skip_validation=False):
        if isinstance(data, list):
            data = [(name, MaskedArray(arr)) for name, arr in data]
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            names = data.dtype.names
            arrays = []
            for n in self.names:
                if data[n].dtype.names is not None:
                    # unpack nested types recursively
                    arrays.append(MaskedArrayCollection(data[n]))
                else:
                    arrays.append(MaskedArray(data[n].copy()))
            data = list(zip(names, arrays))
        elif isinstance(data, ArrayCollection):
            data = [(name, MaskedArray(data.arrays[name]))
                    for name, arr in data.names]
        else:
            raise Exception("Expected either a list of (name, arr) pairs"
                            "or a structured array")

        super().__init__(data, skip_validation)

    def filled(self, fill_value=0):
        if not isinstance(fill_value, tuple):
            fill_value = (fill_value,)*len(self.names)

        data = [(name, self.arrays[name].filled(fill))
                for name, fill in zip(self.names, fill_value)]

        return ArrayCollection(data)

c = MaskedArrayCollection([('age', a), ('weight', b)])
print(repr(c))
c['age'] += 100
print(repr(c))
print(repr(c.filled()))
