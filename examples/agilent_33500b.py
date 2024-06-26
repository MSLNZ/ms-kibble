from equipment_register import records
from kibble import Agilent33500B

siggen = Agilent33500B(records['siggen'])

# turn Channel 1 output off
siggen.output(channel=1, state=False)

# configure Channel 1 to output a 3 Vpp sine wave at a frequency of 12 kHz
siggen.configure_sine(1, frequency=12e3, amplitude=3)

siggen.disconnect()
