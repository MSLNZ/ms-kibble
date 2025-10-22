"""Agilent3458A example."""

from __future__ import annotations

from msl.equipment import Connection, Equipment

from kibble import Agilent3458A

equipment = Equipment(connection=Connection("GPIB::22"))

dmm = Agilent3458A(equipment)

dt = dmm.configure(aperture=0.1)
dmm.initiate()

data = dmm.fetch()
print(f"approx. time between samples: {dt:.3f}")
print(f"samples: {data}")

dmm.disconnect()
