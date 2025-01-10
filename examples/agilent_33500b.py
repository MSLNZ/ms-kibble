"""Agilent33500B example."""

from equipment_register import records  # type: ignore[import-not-found]
from kibble import Agilent33500B


awg = Agilent33500B(records["awg"])

x = [0.1, 0.2, 0.1, 0.2, 0.3, 0.5, 0.1, 0.2, 0.1, 0.5, -0.5, -1]

#x = str(x)
#x = x.replace('[', '').replace(']', '')


print(x)
print(type(x))
print(len(x))



#y = x

#if any(y < -1) is True or any(y > 1) is True:
#max_val = np.max(np.abs(y))
#y = y / max_val



#y = list(y)
#y1 = str(y)
#y2 = y1.replace('[', '').replace(']', '')
#print(y)
#print(type(y))

awg.arb_input(channel=2, sample_rate=100, pt_peak=2, amplitude=1, offset=0, data=x)
# turn Channel 1 output off
awg.output(channel=2, state=True)

# configure Channel 1 to output a 3 Vpp sine wave at a frequency of 12 kHz
#awg.sine(1, frequency=12e3, amplitude=3)



awg.disconnect()
