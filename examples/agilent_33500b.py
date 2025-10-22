"""Agilent33500B example."""

from msl.equipment import Connection, Equipment

from kibble import Agilent33500B

equipment = Equipment(connection=Connection("GPIB::10"))

awg = Agilent33500B(equipment)

# turn Channel 1 output off
awg.output(channel=1, state=False)

# configure Channel 1 to output a 3 Vpp sine wave at a frequency of 12 kHz
awg.sine(1, frequency=12e3, amplitude=3)

awg.disconnect()
