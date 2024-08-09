"""TimeTagger example."""

from kibble import TriggeredTIA

with TriggeredTIA(trigger=1, start=2, stop=3, duration=1) as tia:
    # Start the time-interval analyser measurement, this is not a blocking call
    tia.start()

    # Calling wait() will block until the measurement is done.
    # Specify a timeout, since there is no trigger signal.
    # Enabling debug will print some info.
    status = tia.wait(debug=True, timeout=2)
    if status.code > 0:
        raise RuntimeError(status.message)

    # If there was an actual trigger signal, status.code == 0, and we
    # could get the (time, amplitude) values as a structured numpy array
    print(tia.intervals())
