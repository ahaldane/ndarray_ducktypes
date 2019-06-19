#!/usr/bin/env python
from MaskedArray import X, MaskedArray, MaskedScalar
from unyt import m, km, s, unyt_array, unyt_quantity
import numpy as np
import duckprint

class MaskedUnyt(MaskedArray):
    known_types = MaskedArray.known_types + (unyt_array, unyt_quantity)

    def __init__(self, data, *args, units=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._data = unyt_array(self._data, units=units)

    def __str__(self):
        return duckprint.duck_str(self) + " " + str(self.units)

    def __repr__(self):
        return duckprint.duck_repr(self, extra_args=['units={}'.format(self.units)])

    @property
    def units(self):
        return self._data.units

    @units.setter
    def units(self, val):
        self._data.units = val

class MaskedUnytScalar(MaskedScalar):
    pass

MaskedUnyt.ArrayType = MaskedUnyt
MaskedUnyt.ScalarType = MaskedUnytScalar
MaskedUnytScalar.ArrayType = MaskedUnyt
MaskedUnytScalar.ScalarType = MaskedUnytScalar

# TODO: allow binops with unyt.unit_object.Unit

if __name__ == '__main__':
    uarr = MaskedUnyt([1, X, 1], units=km)
    print(repr(uarr))
    print(repr(uarr + (1*m)))
    print(repr(uarr/(1*s)))
    print(repr((uarr*(1*m))[1:]))
    print(repr(np.add.outer(uarr, uarr)))
    print(str(uarr))
