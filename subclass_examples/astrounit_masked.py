#!/usr/bin/env python
from MaskedArray import X, MaskedArray, MaskedScalar
from astropy import units as u
from astropy.units import m, km, s
import numpy as np
import duckprint

class MaskedQ(MaskedArray):
    known_types = MaskedArray.known_types + (u.quantity.Quantity, u.core.PrefixUnit)

    def __init__(self, data, *args, units=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        if not isinstance(self._data, u.quantity.Quantity) and units is None:
            units = u.dimensionless_unscaled
        self._data = u.quantity.Quantity(self._data, unit=units)

    def __str__(self):
        s = MaskedArray(np.array(self._data), self._mask)
        return duckprint.duck_str(s) + " " + str(self.units)

    def __repr__(self):
        s = MaskedArray(np.array(self._data), self._mask)
        return duckprint.duck_repr(s, name='MaskedQ', 
                                   extra_args=['units={}'.format(self.units)])

    @property
    def units(self):
        return self._data.unit

    @units.setter
    def units(self, val):
        self._data.unit = val

# TODO: scalars
class MaskedQScalar(MaskedScalar):
    pass

MaskedQ.ArrayType = MaskedQ
MaskedQ.ScalarType = MaskedQScalar
MaskedQScalar.ArrayType = MaskedQ
MaskedQScalar.ScalarType = MaskedQScalar


# TODO: allow binops directly with astropy.units.core.PrefixUnit

if __name__ == '__main__':
    uarr = MaskedQ([1, X, 1], units=km)
    print(repr(uarr))
    print(repr(uarr + (1*m)))
    print(repr(uarr/(1*s)))
    print(repr((uarr*(1*m))[1:]))
    print(repr(np.add.outer(uarr, uarr)))
    print(str(uarr))
