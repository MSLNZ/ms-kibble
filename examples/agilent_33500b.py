"""Agilent33500B example."""

from equipment_register import records  # type: ignore[import-not-found]

from kibble import Agilent33500B

awg = Agilent33500B(records["awg"])

# turn Channel 1 output off
awg.output(channel=1, state=False)

# configure Channel 1 to output a 3 Vpp sine wave at a frequency of 12 kHz
awg.sine(1, frequency=12e3, amplitude=3)

awg.disconnect()
