"""Swabian TimeTagger equipment."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from enum import IntEnum
from time import perf_counter, sleep
from typing import TYPE_CHECKING

import numpy as np

try:
    import TimeTagger
except ModuleNotFoundError:
    import os
    import sys

    # Assume running on Windows
    arch = "x64" if sys.maxsize > 2**32 else "x86"
    sys.path.append(os.path.join(os.environ.get("TIMETAGGER_INSTALL_PATH", ""), "driver", arch))  # noqa: PTH118
    try:
        import TimeTagger  # type: ignore  # noqa: PGH003
    except ModuleNotFoundError:

        class TimeTagger:  # type: ignore[no-redef]
            """Mocked TimeTagger module."""

            class Resolution:
                """Mocked Resolution enum."""

                Standard = 0

            class CustomMeasurement:
                """Mocked CustomMeasurement class."""

            @staticmethod
            def createTimeTagger(serial: str = "", resolution: int = 0) -> None:  # noqa: ARG004, N802
                """Mocked createTimeTagger function."""
                msg = (
                    "The Swabian Instruments TimeTagger module cannot be found.\n"
                    "Install it from https://www.swabianinstruments.com/time-tagger/downloads/\n"
                    "and ensure that the directory containing the TimeTagger module is available on sys.path"
                )
                raise ModuleNotFoundError(msg)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import Any, Self

    from msl.equipment.record_types import EquipmentRecord  # type: ignore[import-untyped]
    from numpy.typing import NDArray


@dataclass
class Channel:
    """A time-tagger channel to measure events.

    Args:
        number: Channel number. A positive value corresponds to a timestamp event on a rising edge,
            a negative value corresponds to a timestamp event on a falling edge. See the manual from
            Swabian Instruments for more details.
        deadtime: Dead time (in picoseconds) of the channel. The minimum dead time is defined
            by the internal clock period (which is 2000 ps for Time Tagger Ultra).
        delay: Additional delay (in picoseconds) to add to the timestamp of every event on this channel.
        frequency: The expected maximum number of events (on this channel) per second during a measurement.
            This value helps to determine the size of the numpy arrays to initialise.
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

    SUCCESS: int = 0
    TIMEOUT: int = 1
    OVERFLOW: int = 2


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


@dataclass(frozen=True, kw_only=True)
class Displacement:
    """A displacement measurement.

    Attributes:
        beat_freq: Beat frequency, in MHz.
        folding: Folding factor.
        wavelength: Wavelength, in nm.
        x: Corresponding time (in seconds) of each accumulated change in displacement.
        y: Accumulated change in displacement.
    """

    beat_freq: float
    folding: float
    wavelength: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]


class TimeTag(TimeTagger.CustomMeasurement):  # type: ignore[misc]
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
            duration: The expected duration (in seconds) that measurement events will occur.
                For gated and triggered measurements, the time before the rising edge of the
                gate/trigger pulse shall not be included in the duration.
            record: The equipment record. If specified and `tagger` is not specified, then the
                serial number in the `record` will be used to connect to a specific `TimeTagger`.
                A _resolution_ key-value pair in
                [record.connection.properties][msl.equipment.record_types.ConnectionRecord.properties]
                may be used to set the resolution of the `TimeTagger` (i.e., for HighResA set `resolution=1`).
            tagger: A Swabian `TimeTagger` instance. If not specified, a new instance is created.
        """
        self._is_array_overflow = False  # overflow on the numpy array?
        self._is_tagger_overflow = False  # overflow on the time-tagger?
        self._is_done = False  # measurement done?
        self._events = 0  # number of events that were processed
        self._begin_time = 0  # the begin_time (in ps) of the first call to self.process()

        self._free_tagger = False
        if not tagger:
            if record is None:
                serial, resolution = "", 0
            else:
                serial = record.serial
                if record.connection is None:
                    resolution = TimeTagger.Resolution.Standard
                else:
                    resolution = record.connection.properties.get("resolution", TimeTagger.Resolution.Standard)
            tagger = TimeTagger.createTimeTagger(serial=serial, resolution=resolution)
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

        self.finalize_init()  # Must be called when done configuring a CustomMeasurement
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
        if not hasattr(self, "_free_tagger") or not hasattr(self, "this"):
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

        Args:
            events: Maximum number of events before `process()` is called (256 - 32M).
            latency: Maximum latency (in milliseconds) before `process` is called (1 to 10000).
        """
        self._tagger.setStreamBlockSize(events, latency)

    @staticmethod
    def create_channel(
        number: int, *, deadtime: int = 2000, delay: int = 0, frequency: float = 3e6, level: float = 0.5
    ) -> Channel:
        """Create a new channel for a time-tag measurement.

        Args:
            number: Channel number. A positive value corresponds to a timestamp event on a rising edge,
                a negative value corresponds to a timestamp event on a falling edge. See the manual from
                Swabian Instruments for more details.
            deadtime: Dead time (in picoseconds) of the channel. The minimum dead time is defined
                by the internal clock period (which is 2000 ps for Time Tagger Ultra).
            delay: Additional delay (in picoseconds) to add to the timestamp of every event on this channel.
            frequency: The expected maximum number of events (on this channel) per second during a measurement.
            level: Signal level (in Volts) that, when exceeded, defines an event.

        Returns:
            A time-tag channel.
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

        The duration shall not include the time before a gate nor a trigger event
        (if used). Only the expected duration of the measurement of interest need
        be specified.

        For a triggered measurement, the `duration` corresponds to the measurement
        time after the trigger edge. For a gated measurement, the `duration`
        corresponds to the width of the gate pulse. The `duration` value also helps
        to determine the size of the numpy arrays to initialise.

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

    def process(self, tags: NDArray[Any], begin_time: int, end_time: int) -> None:
        """Callback function to process a data stream from the time tagger.

        No filtering is performed. All events are considered valid until the measurement is complete.

        Args:
            tags: The time tags as a structured numpy array. Field names: type, missed_events, channel, time.
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
            [wait][kibble.equipment.swabian_timetagger.TimeTag.wait]
        """
        self._is_array_overflow = False
        self._is_tagger_overflow = False
        self._is_done = False
        self._begin_time = -1
        self._events = 0
        self._tagger.clearOverflows()
        self.clear()
        super().start()

    def stop(self) -> None:
        """Stop a measurement."""
        super().stop()

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

    def wait(self, *, timeout: float | None = None) -> Status:
        """Wait until the measurement is done.

        This is a blocking call and will not return until the measurement finishes or there is an error.

        Args:
            timeout: The maximum number of seconds to wait for the measurement to be done.
                If `None`, wait forever.

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

            sleep(0.01)


class TimeTagGated(TimeTag):
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
            gate: The gate channel. The [frequency][kibble.equipment.swabian_timetagger.Channel] attribute is ignored.
            duration: The expected duration (in seconds) that measurement events will occur. See
                [duration][kibble.equipment.swabian_timetagger.TimeTag.duration] for more details.
            record: The equipment record. See the constructor of
                [TimeTag][kibble.equipment.swabian_timetagger.TimeTag] for more details.
            tagger: A Swabian `TimeTagger` instance. If not specified, a new instance is created.
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
            tags: The time tags as a structured numpy array. Field names: type, missed_events, channel, time.
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


class TimeTagTriggered(TimeTag):
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
            trigger: The trigger channel. The [frequency][kibble.equipment.swabian_timetagger.Channel]
                attribute is ignored.
            duration: The expected duration (in seconds) that measurement events will occur. See
                [duration][kibble.equipment.swabian_timetagger.TimeTag.duration] for more details.
            record: The equipment record. See the constructor of
                [TimeTag][kibble.equipment.swabian_timetagger.TimeTag] for more details.
            tagger: A Swabian `TimeTagger` instance. If not specified, a new instance is created.
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
            tags: The time tags as a structured numpy array. Field names: type, missed_events, channel, time.
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


class TimeIntervalAnalyser:
    """A time-interval analysis measurement based on start-stop events."""

    def __init__(
        self,
        *,
        start: Channel | int,
        stop: Channel | int,
        duration: float = 10,
        gate: Channel | int | None = None,
        record: EquipmentRecord | None = None,
        tagger: TimeTagger.TimeTagger | None = None,
        trigger: Channel | int | None = None,
    ) -> None:
        """Perform a time-interval analysis measurement based on start-stop events.

        The intervals are calculated as stop-start time differences and the time of each interval is relative to:

        * the timestamp of the first start or stop event if neither gate nor trigger are specified
        * the first edge of the gate pulse, or
        * the trigger signal

        Args:
            start: The start channel.
            stop: The stop channel.
            duration: The expected duration (in seconds) that measurement events will occur. See
                [duration][kibble.equipment.swabian_timetagger.TimeTag.duration] for more details.
            gate: The gate channel.
            record: The equipment record. See the constructor of
                [TimeTag][kibble.equipment.swabian_timetagger.TimeTag] for more details.
            tagger: A Swabian `TimeTagger` instance. If not specified, a new instance is created.
            trigger: The trigger channel.
        """
        if gate is not None and trigger is not None:
            msg = "Cannot specify both a gate and a trigger channel"
            raise ValueError(msg)

        self._start: Channel = Channel(start) if isinstance(start, int) else start
        self._stop: Channel = Channel(stop) if isinstance(stop, int) else stop
        self._gate: Channel | None = None
        self._trigger: Channel | None = None

        self._measurement: TimeTag
        if gate is not None:
            self._gate = Channel(gate, frequency=1) if isinstance(gate, int) else gate
            self._measurement = TimeTagGated(
                events=[self._start, self._stop], gate=self._gate, duration=duration, record=record, tagger=tagger
            )
        elif trigger is not None:
            self._trigger = Channel(trigger, frequency=1) if isinstance(trigger, int) else trigger
            self._measurement = TimeTagTriggered(
                events=[self._start, self._stop], trigger=self._trigger, duration=duration, record=record, tagger=tagger
            )
        else:
            self._measurement = TimeTag([self._start, self._stop], duration=duration, record=record, tagger=tagger)

    @staticmethod
    def create_channel(
        number: int, *, deadtime: int = 2000, delay: int = 0, frequency: float = 3e6, level: float = 0.5
    ) -> Channel:
        """Create a new channel for a time-tag measurement.

        Args:
            number: Channel number. A positive value corresponds to a timestamp event on a rising edge,
                a negative value corresponds to a timestamp event on a falling edge. See the manual from
                Swabian Instruments for more details.
            deadtime: Dead time (in picoseconds) of the channel. The minimum dead time is defined
                by the internal clock period (which is 2000 ps for Time Tagger Ultra).
            delay: Additional delay (in picoseconds) to add to the timestamp of every event on this channel.
            frequency: The expected maximum number of events (on this channel) per second during a measurement.
            level: Signal level (in Volts) that, when exceeded, defines an event.

        Returns:
            A time-tag channel.
        """
        return Channel(number, deadtime=deadtime, delay=delay, frequency=frequency, level=level)

    def displacement(
        self,
        *,
        beat_freq: float | None = None,
        folding: float = 1.0,
        subset: slice | None = None,
        timeout: float | None = None,
        wavelength: float = 633.24567,
    ) -> Displacement:
        """Get the displacement data.

        Follows the algorithm in [XXX](). Only considers a start event followed by a stop event as valid data.

        This is a blocking call and will not return until the measurement finishes or there is an error.

        Args:
            beat_freq: Beat frequency, in MHz, of heterodyne beams. If `None`, calculate the beat
                frequency as the average difference between neighbouring _start_ timestamps.
            folding: Beam path folding number.
            subset: Slice the data to only use a subset. For example, `subset=slice(5, -10)` will
                ignore the first 5 values and the last 10 values from the displacement calculation.
            timeout: The maximum number of seconds to wait for the measurement to be done.
                If `None`, wait forever.
            wavelength: Wavelength, in nanometres, of laser.

        Returns:
            The displacement data.
        """
        x, y = self.time_interval(timeout=timeout)
        if subset is not None:
            x, y = x[subset], y[subset]

        if beat_freq is None:
            channels = self._measurement.channels
            timestamps = self._measurement.timestamps
            beat_freq = float(np.mean(1e6 / np.diff(timestamps[channels == self._start.number])))

        phase = np.diff(y * 2.0 * np.pi * beat_freq * 1e6)

        # If change in phase exceeds pi, assume 0/2*pi was crossed
        phase = np.where(np.abs(phase) > np.pi, phase - np.sign(phase) * 2.0 * np.pi, phase)

        # Accumulate change in displacement with time
        y_displacement = np.cumsum(phase * (wavelength * 1e-9) / (4.0 * np.pi * folding))
        return Displacement(
            beat_freq=beat_freq,
            folding=folding,
            wavelength=wavelength,
            x=x[1 : len(y_displacement) + 1],
            y=y_displacement,
        )

    def time_interval(self, *, timeout: float | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the time-interval data.

        Only considers a start event followed by a stop event as valid data.

        This is a blocking call and will not return until the measurement finishes or there is an error.

        Args:
            timeout: The maximum number of seconds to wait for the measurement to be done.
                If `None`, wait forever.

        Returns:
            A tuple of two numpy array's (time, interval).

                * `time`: Time (in seconds) of `start` events relative to the first timestamp event.
                * `interval`: Difference (in seconds) between the `stop` and `start`  timestamps.
        """
        status = self._measurement.wait(timeout=timeout)
        if not status.success:
            raise RuntimeError(status.message)

        start = self._start.number
        stop = self._stop.number
        channels = self._measurement.channels
        timestamps = self._measurement.timestamps

        if self._gate is not None:
            if channels[0] != self._gate.number:
                msg = f"The first channel, {channels[0]}, is not equal to the gate channel, {self._gate.number}"
                raise RuntimeError(msg)
            if channels[-1] != -self._gate.number:
                msg = f"The last channel, {channels[0]}, is not equal to the gate channel, {-self._gate.number}"
                raise RuntimeError(msg)

        if self._trigger is not None and channels[0] != self._trigger.number:
            msg = f"The first channel, {channels[0]}, is not equal to the trigger channel, {self._trigger.number}"
            raise RuntimeError(msg)

        # arbitrarily chose to append 100 since it cannot be a valid TimeTagger channel but it is a valid int8
        diff = np.diff(channels, append=100)

        # must also check "channels==start" in addition to the "stop-start" difference since "start-trigger",
        # "stop-trigger", "abs(start-gate)", "abs(stop-gate)" may also equal the "stop-start" difference
        start_indices = np.logical_and(channels == start, diff == stop - start)
        stop_indices = np.roll(start_indices, 1)

        t1 = timestamps[start_indices]
        t2 = timestamps[stop_indices]
        if t1.size == 0 or t2.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        intervals = 1e-12 * (t1 - t2).astype(np.float64)
        times = 1e-12 * (t1 - timestamps[0]).astype(np.float64)
        return times, intervals

    def start(self) -> None:
        """Start a measurement.

        This method does not block the calling routine. It will return as soon as the measurement is running.
        """
        return self._measurement.start()
