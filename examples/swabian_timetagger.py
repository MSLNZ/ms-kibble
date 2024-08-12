"""TimeTagger example."""

from kibble import TimeIntervalAnalyser

# We can specify the start and stop channels as integers, or, to have
# more control over the configuration of each channel, as a `Channel` object
start = TimeIntervalAnalyser.create_channel(2, deadtime=2000, delay=0, frequency=3e6, level=0.5)

# If you want to synchronise the measurement with other equipment, the TimeIntervalAnalyser
# class also accepts a `gate` or a `trigger` channel as a keyword argument
tia = TimeIntervalAnalyser(start=start, stop=3, duration=1)

# Start the time-interval analyser measurement
# This is not a blocking call and this script will continue
tia.start()

# For example, you could communicate with other equipment here...

# When you have performed all other tasks, call tia.time_interval()
# This method will block until the measurement is done or until an error is raised
time, interval = tia.time_interval()
print(time)
print(interval)
