"""PulseBlaster example."""

from equipment_register import records  # type: ignore[import-not-found]

from kibble import PulseBlaster

blaster = PulseBlaster(records["blaster"])

blaster.configure_two_pulses(width=10e-6, delay=5e-6)

while True:
    blaster.trigger()
    x = input("Press ENTER to send trigger, -1 to exit: ")
    if x == "-1":
        break

blaster.stop()
blaster.disconnect()
