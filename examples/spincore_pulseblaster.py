"""PulseBlaster example."""

from msl.equipment import Connection, Equipment

from kibble import PulseBlaster

equipment = Equipment(manufacturer="SpinCore", model="PulseBlaster", connection=Connection("SDK::libspinapi"))

blaster = PulseBlaster(equipment)

blaster.one_pulse_two_channels(width=10e-6, delay=5e-6)

while True:
    blaster.trigger()
    x = input("Press ENTER to send trigger, -1 to exit: ")
    if x == "-1":
        break

blaster.stop()
blaster.disconnect()
