"""Swabian TimeTagger equipment."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from enum import IntEnum
from time import perf_counter, sleep
from typing import TYPE_CHECKING

import numpy as np
import TimeTagger  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import Any, Self

    from msl.equipment import EquipmentRecord  # type: ignore[import-untyped]
    from numpy.typing import NDArray


@dataclass
class Channel:
    """A time-tagger channel to measure events.

    Args:
        number: Channel number.

    Keyword Args:
        deadtime: Dead time (in picoseconds) of the channel. The minimum dead time is defined
            by the internal clock period (which is 2000 ps for Time Tagger Ultra).
        delay: Additional delay (in picoseconds) to add to the timestamp of every event on this channel.
        frequency: The expected maximum number of events (on this channel) per second during a measurement.
        level: Signal level (in Volts) that, when exceeded, defines an event.
    """

    number: int
    _: KW_ONLY
    deadtime: int = 2000
    delay: int = 0
    frequency: float = 3e6
    level: float = 0.5


class StatusCode(IntEnum):
    """Status codes for a measurement.

    Attributes:
        SUCCESS: `0`
        TIMEOUT: `1`
        OVERFLOW: `2`
    """

    SUCCESS = 0
    TIMEOUT = 1
    OVERFLOW = 2


@dataclass(frozen=True, kw_only=True)
class Status:
    """The measurement status.

    Args:
        code: A status code.
        message: A message.
        success: Whether the measurement finished without error.
    """

    code: StatusCode
    message: str
    success: bool


class TimeTagMeasurement(TimeTagger.CustomMeasurement):  # type: ignore[misc]
    """Custom TimeTagger measurement."""

    def __init__(
        self,
        channels: Sequence[Channel],
        *,
        duration: float = 10,
        record: EquipmentRecord | None = None,
        tagger: TimeTagger.TimeTagger | None = None,
    ) -> None:
        """Perform a time-tag measurement.

        Args:
            channels: The channels involved in the measurement.

        Keyword Args:
            duration: The expected duration (in seconds) that measurement events will occur.
                For gated and triggered measurements, the time before the rising edge of the
                gate/trigger pulse shall not be included in the duration.
            record: The equipment record. If specified and `tagger` is not specified, then the
                serial number in the record will be used to connect to a specific `TimeTagger`.
            tagger: A `TimeTagger` instance. If not specified, a new instance is created.
        """
        self._is_array_overflow = False  # overflow on the numpy array?
        self._is_tagger_overflow = False  # overflow on the time-tagger?
        self._is_done = False  # measurement done?
        self._events = 0  # number of events that were processed
        self._begin_time = 0  # the begin_time (in ps) of the first call to self.process()

        self._free_tagger = False
        if not tagger:
            serial = "" if record is None else record.serial
            tagger = TimeTagger.createTimeTagger(serial=serial)
            self._free_tagger = True
        self._tagger: TimeTagger.TimeTagger = tagger
        super().__init__(tagger)

        self._input_channels = channels
        for channel in channels:
            tagger.setTriggerLevel(channel.number, channel.level)
            tagger.setInputDelay(channel.number, channel.delay)
            tagger.setDeadtime(channel.number, channel.deadtime)
            self.register_channel(channel.number)

        self._channels: NDArray[np.int8]
        self._timestamps: NDArray[np.int64]
        self.duration = duration

        self.finalize_init()  # Must be called when done initialising a CustomMeasurement
        self.stop()  # However, finalize_init() automatically starts a measurement (which also calls self.on_start())

    def __del__(self) -> None:
        """Called when object reference count reaches zero."""
        self._cleanup()

    def __enter__(self) -> Self:
        """Enter a "with" statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit a "with" statement."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Stop the measurement and release the TimeTagger object from memory."""
        if not hasattr(self, "_free_tagger"):
            return

        self.stop()
        if self._free_tagger:
            TimeTagger.freeTimeTagger(self._tagger)
            self._free_tagger = False

    def _insert_tags(self, tags: NDArray[Any]) -> None:
        """Insert the data-stream tags into the numpy arrays."""
        i = self._events + len(tags)
        if i > self._channels.size:
            self._is_array_overflow = True
            return

        self._channels[self._events : i] = tags["channel"]
        self._timestamps[self._events : i] = tags["time"]
        self._events = i

    def _tags_okay(self, tags: NDArray[Any]) -> bool:
        """Check whether these data-stream tags should continue to be processed."""
        if self._is_done or self._is_tagger_overflow or self._is_array_overflow:
            # Cannot call self.stop() within self.process() due to SWIG threading
            # Simply ignore any new data chunks
            return False

        if np.any(tags["type"] != 0):
            self._is_overflow = True
            return False

        return True

    @property
    def channels(self) -> NDArray[np.int8]:
        """Return the channel number for every event.

        Returns:
            A 1D numpy array of channel numbers.
        """
        with self.mutex:
            return self._channels[: self._events]

    def configure_data_stream(self, *, events: int = 131072, latency: int = 20) -> None:
        """Configure the size of the data stream.

        Depending on which of the two parameters is exceeded first, the number of
        events to `process()` in the data stream is adjusted accordingly.

        Keyword Args:
            events: Maximum number of events before `process()` is called (256 - 32M).
            latency: Maximum latency (in milliseconds) before `process` is called (1 to 10000).
        """
        self._tagger.setStreamBlockSize(events, latency)

    def count_rate(self, channel: int) -> float | None:
        """Return the count rate for a particular channel.

        Args:
            channel: A measurement channel.

        Returns:
            The count rate or `None` if it cannot be determined (too few data points).
        """
        t = self.timestamps[self.channels == channel]
        if t.size < 1:
            return None

        dt = t[-1] - t[0]
        if dt == 0:
            return None

        return float(t.size / (1e-12 * dt))

    @staticmethod
    def create_channel(
        number: int, *, deadtime: int = 2000, delay: int = 0, frequency: float = 3e6, level: float = 0.5
    ) -> Channel:
        """Create a new channel for a time-tag measurement.

        Args:
            number: Channel number.

        Keyword Args:
            deadtime: Dead time (in picoseconds) of the channel. The minimum dead time is defined
                by the internal clock period (which is 2000 ps for Time Tagger Ultra).
            delay: Additional delay (in picoseconds) to add to the timestamp of every event on this channel.
            frequency: The expected maximum number of events (on this channel) per second during a measurement.
            level: Signal level (in Volts) that, when exceeded, defines an event.

        Returns:
            The time-tag channel.
        """
        return Channel(number, deadtime=deadtime, delay=delay, frequency=frequency, level=level)

    def done(self) -> bool:
        """Check if the measurement is done.

        Returns:
            Whether the measurement is done.
        """
        return self._is_done

    @property
    def duration(self) -> float:
        """The expected number of seconds that a measurement will take.

        Returns:
            The measurement duration.
        """
        return self._duration

    @duration.setter
    def duration(self, seconds: float) -> None:
        self._duration = float(seconds)

        size = 0
        for channel in self._input_channels:
            size += round(seconds * channel.frequency)
        size = round(size * 1.2)  # for safety, increase array size by 20%

        self._channels = np.empty(size, dtype=np.int8)

        # Use int64 instead of uint64 because it allows for negative values if
        # differences between timestamps are calculated. Also, int64 is the data
        # type for the "time" tags that are received in self.process().
        self._timestamps = np.empty(size, dtype=np.int64)

    def data(self) -> NDArray[Any]:
        """Return the raw data for all events during the measurement.

        Returns:
            A structured numpy array with the following field names:

            * `channel` (int): the channel number of each event.
            * `timestamp` (int): the timestamp (in picoseconds) of each event.
        """
        with self.mutex:
            dtype = [
                ("channel", self._channels.dtype),
                ("timestamp", self._timestamps.dtype),
            ]
            array = np.empty(self._events, dtype=dtype)
            array["channel"] = self._channels[: self._events]
            array["timestamp"] = self._timestamps[: self._events]
            return array

    def process(self, tags: NDArray[Any], begin_time: int, end_time: int) -> None:
        """Callback function to process a data stream from the time tagger.

        No filtering is performed. All events are considered valid until the measurement is complete.

        Args:
            tags: The time tags as a structred numpy array. Field names: type, missed_events, channel, time.
            begin_time: The begin time of the data chunk.
            end_time: The end time of the data chunk.
        """
        if self._begin_time < 0:
            self._begin_time = begin_time

        # Since this method is essentially free-running (no trigger/gate signal)
        # the accuracy of the measurement duration is not critical. If there are
        # ~10ms of extra events in the measurement, that is okay.
        self._is_done = 1e-12 * (end_time - self._begin_time) > self._duration
        if self._tags_okay(tags):
            self._insert_tags(tags)

    def start(self) -> None:
        """Start a measurement.

        This method does not block the calling routine. It will return as
        soon as the measurement is running.

        See Also:
            [wait][kibble.equipment.swabian_timetagger.TimeTagMeasurement.wait]
        """
        self._is_array_overflow = False
        self._is_tagger_overflow = False
        self._is_done = False
        self._begin_time = -1
        self._events = 0
        self._tagger.clearOverflows()
        self.clear()
        super().start()

    def tagger(self) -> TimeTagger.TimeTagger:
        """Return the time-tagger instance."""
        return self._tagger

    @property
    def timestamps(self) -> NDArray[np.int64]:
        """Return the timestamp for every event.

        Returns:
            A 1D numpy array of timestamps.
        """
        with self.mutex:
            return self._timestamps[: self._events]

    def wait(self, *, debug: bool = False, timeout: float | None = None) -> Status:
        """Wait until the measurement is done.

        This is a blocking call and will not return until the measurement finishes or there is an error.

        Args:
            debug: Whether to print the runtime and the number of events to the terminal.
            timeout: The maxmimum number of seconds to wait. If `None`, wait forever.

        Returns:
            The status when the measurement finished.
        """
        t0 = perf_counter()
        while True:
            if self.done():
                self.stop()
                return Status(code=StatusCode.SUCCESS, message="Success", success=True)
            if (timeout is not None) and (perf_counter() - t0 > timeout):
                self.stop()
                return Status(
                    code=StatusCode.TIMEOUT,
                    message=f"Time-tag measurement timeout after {timeout} seconds",
                    success=False,
                )
            if self._is_tagger_overflow:
                self.stop()
                return Status(code=StatusCode.OVERFLOW, message="TimeTagger has a buffer overflow", success=False)
            if self._is_array_overflow:
                self.stop()
                return Status(
                    code=StatusCode.OVERFLOW,
                    message="Buffered numpy array too small. Increase the expected frequency of a "
                    "channel or the expected measurement duration",
                    success=False,
                )

            sleep(self._duration * 0.05)
            if debug:
                print(  # noqa: T201
                    f"Time-tag measurement runtime: {perf_counter() - t0:.3f} seconds, events: {self._events:.3e}"
                )


class TimeTagGated(TimeTagMeasurement):
    """A gated time-tag measurement."""

    def __init__(
        self,
        *,
        events: Sequence[Channel],
        gate: Channel,
        duration: float = 10,
        record: EquipmentRecord | None = None,
        tagger: TimeTagger.TimeTagger | None = None,
    ) -> None:
        """Perform a gated measurement.

        Events are considered valid only during the `gate` pulse.

        Args:
            events: The channel(s) that contain the events to measure.
            gate: The gate channel. The `frequency` attribute is ignored.

        Keyword Args:
            duration: The expected duration (in seconds) that measurement events will occur. The
                time before the rising edge of the gate pulse shall not be included in the duration.
            record: The equipment record. If specified and `tagger` is not specified, then the
                serial number in the record will be used to connect to a specific `TimeTagger`.
            tagger: A `TimeTagger` instance. If not specified, a new instance is created.
        """
        self._is_gated = False
        self._gate_channel = gate.number

        channels = list(events)
        channels.append(Channel(gate.number, deadtime=gate.deadtime, level=gate.level, frequency=1))
        channels.append(Channel(-gate.number, deadtime=gate.deadtime, level=gate.level, frequency=1))
        super().__init__(channels, duration=duration, record=record, tagger=tagger)

    def on_start(self) -> None:
        """Do not call. Called automatically when `self.start()` is called."""
        self._is_gated = False

    def process(self, tags: NDArray[Any], begin_time: int, end_time: int) -> None:  # noqa: ARG002
        """Callback function to process a data stream from the time tagger.

        Args:
            tags: The time tags as a structred numpy array. Field names: type, missed_events, channel, time.
            begin_time: The begin time of the data chunk.
            end_time: The end time of the data chunk.
        """
        # No need to use Numba since this method typically takes < 1ms to finish
        if not self._tags_okay(tags):
            return

        if self._is_gated:
            # Check for falling edge of GATE signal
            gate_indices = tags["channel"] == -self._gate_channel
        else:
            # Check for rising edge of GATE signal
            gate_indices = tags["channel"] == self._gate_channel

        gated_tags = None
        if np.any(gate_indices):
            if self._is_gated:
                self._is_gated = False
                self._is_done = True
                stop_time = tags[gate_indices]["time"][0]
                gated_tags = tags[tags["time"] <= stop_time]
            else:
                self._is_gated = True
                start_time = tags[gate_indices]["time"][0]
                gated_tags = tags[tags["time"] >= start_time]
        elif self._is_gated:
            gated_tags = tags

        if gated_tags is not None:
            self._insert_tags(gated_tags)


class TimeTagTriggered(TimeTagMeasurement):
    """A triggered time-tag measurement."""

    def __init__(
        self,
        *,
        events: Sequence[Channel],
        trigger: Channel,
        duration: float = 10,
        record: EquipmentRecord | None = None,
        tagger: TimeTagger.TimeTagger | None = None,
    ) -> None:
        """Perform a triggered measurement.

        Events are considered valid after the trigger signal and until the specified measurement duration.

        Args:
            events: The channel(s) that contain the events to measure.
            trigger: The trigger channel. The `frequency` attribute is ignored.

        Keyword Args:
            duration: The expected duration (in seconds) that measurement events will occur. The
                time before the rising edge of the trigger pulse shall not be included in the duration.
            record: The equipment record. If specified and `tagger` is not specified, then the
                serial number in the record will be used to connect to a specific `TimeTagger`.
            tagger: A `TimeTagger` instance. If not specified, a new instance is created.
        """
        self._finished_time = 0  # timestamp, in picoseconds, when the measurement is done
        self._trigger_channel: int = trigger.number

        channels = list(events)
        channels.append(
            Channel(
                trigger.number,
                deadtime=trigger.deadtime,
                level=trigger.level,
                frequency=1,
            )
        )
        super().__init__(channels, duration=duration, record=record, tagger=tagger)

    def on_start(self) -> None:
        """Do not call. Called automatically when `self.start()` is called."""
        self._finished_time = -1

    def process(self, tags: NDArray[Any], begin_time: int, end_time: int) -> None:  # noqa: ARG002
        """Callback function to process a data stream from the time tagger.

        Args:
            tags: The time tags as a structred numpy array. Field names: type, missed_events, channel, time.
            begin_time: The begin time of the data chunk.
            end_time: The end time of the data chunk.
        """
        # No need to use Numba since this method typically takes < 1ms to finish
        if not self._tags_okay(tags):
            return

        triggered_tags = None
        if self._finished_time > 0:
            if end_time < self._finished_time:
                triggered_tags = tags
            else:
                self._is_done = True
                triggered_tags = tags[tags["time"] <= self._finished_time]
        else:
            trigger_indices = tags["channel"] == self._trigger_channel
            if np.any(trigger_indices):
                start_time = tags[trigger_indices]["time"][0]
                triggered_tags = tags[tags["time"] >= start_time]
                self._finished_time = start_time + round(self._duration * 1e12)

        if triggered_tags is not None:
            self._insert_tags(triggered_tags)


class GatedTIA(TimeTagGated):
    """A gated time-interval analysis measurement."""

    def __init__(
        self,
        *,
        start: Channel | int,
        stop: Channel | int,
        gate: Channel | int,
        duration: float = 10,
        record: EquipmentRecord | None = None,
        tagger: TimeTagger.TimeTagger | None = None,
    ) -> None:
        """Perform a gated time-interval analysis measurement.

        The start-stop time differences are used to calculate amplitudes and the
        timestamp of each amplitude is relative to the rising edge of the gate signal.

        Args:
            start: The start channel.
            stop: The stop channel.
            gate: The gate channel.

        Keyword Args:
            duration: The expected duration (in seconds) that measurement events will occur. The
                time before the rising edge of the gate pulse shall not be included in the duration.
            record: The equipment record. If specified and `tagger` is not specified, then the
                serial number in the record will be used to connect to a specific `TimeTagger`.
            tagger: A `TimeTagger` instance. If not specified, a new instance is created.
        """
        if isinstance(start, int):
            start = Channel(start)
        if isinstance(stop, int):
            stop = Channel(stop)
        if isinstance(gate, int):
            gate = Channel(gate, frequency=1)
        self._start_channel = start.number
        self._stop_channel = stop.number
        super().__init__(events=[start, stop], gate=gate, duration=duration, record=record, tagger=tagger)

    def intervals(self, *, debug: bool = False, timeout: float | None = None) -> NDArray[Any]:
        """Get the time-interval data.

        This is a blocking call and will not return until the measurement finishes or there is an error.

        Args:
            debug: Whether to print the runtime and the number of events to the terminal.
            timeout: The maxmimum number of seconds to wait. If `None`, wait forever.

        Returns:
            A structured numpy array with the following field names:

            * `time` (float): Times (in seconds) of `start` events relative to the rising edge of the gate signal.
            * `ampltiude` (float): Difference between the `start` and `stop` timestamps.
        """
        status = self.wait(debug=debug, timeout=timeout)
        if not status.success:
            raise RuntimeError(status.message)
        channels = self.channels
        assert channels[0] == self._gate_channel  # noqa: S101
        assert channels[-1] == -self._gate_channel  # noqa: S101
        return _intervals(
            start=self._start_channel,
            stop=self._stop_channel,
            channels=channels,
            timestamps=self.timestamps,
        )


class TriggeredTIA(TimeTagTriggered):
    """A triggered time-interval analysis measurement."""

    def __init__(
        self,
        *,
        start: Channel | int,
        stop: Channel | int,
        trigger: Channel | int,
        duration: float = 10,
        record: EquipmentRecord | None = None,
        tagger: TimeTagger.TimeTagger | None = None,
    ) -> None:
        """Perform a triggered time-interval analysis measurement.

        The start-stop time differences are used to calculate amplitudes and the
        timestamp of each amplitude is relative to the rising edge of the trigger signal.

        Args:
            start: The start channel.
            stop: The stop channel.
            trigger: The trigger channel.

        Keyword Args:
            duration: The expected duration (in seconds) that measurement events will occur. The
                time before the rising edge of the trigger pulse shall not be included in the duration.
            record: The equipment record. If specified and `tagger` is not specified, then the
                serial number in the record will be used to connect to a specific `TimeTagger`.
            tagger: A `TimeTagger` instance. If not specified, a new instance is created.
        """
        if isinstance(start, int):
            start = Channel(start)
        if isinstance(stop, int):
            stop = Channel(stop)
        if isinstance(trigger, int):
            trigger = Channel(trigger, frequency=1)
        self._start_channel = start.number
        self._stop_channel = stop.number
        super().__init__(events=[start, stop], trigger=trigger, duration=duration, record=record, tagger=tagger)

    def intervals(self, *, debug: bool = False, timeout: float | None = None) -> NDArray[Any]:
        """Get the time-interval data.

        This is a blocking call and will not return until the measurement finishes or there is an error.

        Args:
            debug: Whether to print the runtime and the number of events to the terminal.
            timeout: The maxmimum number of seconds to wait. If `None`, wait forever.

        Returns:
            A structured numpy array with the following field names:

            * `time` (float): Times (in seconds) of `start` events relative to the rising edge of the trigger signal.
            * `ampltiude` (float): Difference between the `start` and `stop` timestamps.
        """
        status = self.wait(debug=debug, timeout=timeout)
        if not status.success:
            raise RuntimeError(status.message)
        channels = self.channels
        assert channels[0] == self._trigger_channel  # noqa: S101
        return _intervals(
            start=self._start_channel,
            stop=self._stop_channel,
            channels=channels,
            timestamps=self.timestamps,
        )


def _intervals(*, start: int, stop: int, channels: NDArray[np.int8], timestamps: NDArray[np.int64]) -> NDArray[Any]:
    """Get the time-interval data.

    Only considers a start event followed by a stop event as a valid time interval.
    """
    # arbitrarily chose to append 100 since it cannot be a valid TimeTagger channel but it is a valid int8
    diff = np.diff(channels, append=100)
    # must also check "channels==start" in addition to the "stop-start" difference since "start-trigger",
    # "stop-trigger", "abs(start-gate)", "abs(stop-gate)" may also equal the "stop-start" difference
    start_indices = np.logical_and(channels == start, diff == stop - start)
    stop_indices = np.roll(start_indices, 1)
    t1 = timestamps[start_indices]
    t2 = timestamps[stop_indices]
    data = np.empty(t1.size, dtype=[("time", np.float64), ("amplitude", np.float64)])
    data["amplitude"] = 1e-12 * (t1 - t2)
    data["time"] = 1e-12 * (t1 - timestamps[0])
    return data
