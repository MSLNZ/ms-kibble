"""TimeTagger example."""

from kibble import TriggeredTIA

with TriggeredTIA(trigger=1, start=2, stop=3, duration=1) as tia:
    # Start the time-interval analyser measurement, this is not a blocking call and this script will continue
    tia.start()

    # Implement code to send a trigger pulse to Channel 1.
    # Ideally, there would also be start-stop signals connected to Channels 2 and 3.

    # Calling tia.intervals() will block until the measurement is done.
    # Specify a timeout, since there is no trigger signal.
    # Enabling debug will print the runtime and number of events of the measurement.
    #
    # If there was an actual trigger signal, a structured numpy array of
    # (time, amplitude) values would be returned
    print(tia.intervals(debug=True, timeout=2))
