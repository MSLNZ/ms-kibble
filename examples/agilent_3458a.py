"""Agilent3458A example."""

from equipment_register import records  # type: ignore[import-not-found]

from kibble import Agilent3458A

dmm = Agilent3458A(records["3458a"])

dt = dmm.configure(aperature=0.1)
dmm.initiate()

data = dmm.fetch()
print(f"approx. time between samples: {dt:.3f}")
print(f"samples: {data}")

dmm.disconnect()
