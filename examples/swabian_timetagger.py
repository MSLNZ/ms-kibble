"""TimeTagger example."""

from kibble import TriggeredTIA

# We can specify the trigger, start and stop channels as integers, or, to have
# more control over the configuraton of each channel, as a `Channel` object
start = TriggeredTIA.create_channel(2, deadtime=2000, delay=100, frequency=2.96e6, level=0.6)

with TriggeredTIA(trigger=1, start=start, stop=3, duration=1) as tia:
    # Start the time-interval analyser measurement, this is not a blocking call and this script will continue
    tia.start()

    # You could implement code to send a trigger pulse to Channel 1 here.
    # Ideally, there would also be start-stop signals connected to Channels 2 and 3.

    # Calling tia.intervals() will block until the measurement is done.
    # Specify a timeout, since there is no trigger signal.
    # Enabling debug will print the runtime and number of events of the measurement.
    #
    # If there was an actual trigger signal, a structured numpy array of
    # (time, amplitude) values would be returned
    print(tia.intervals(debug=True, timeout=2))
