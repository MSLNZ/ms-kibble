"""Agilent (or Hewlett Packard or Keysight) 3458A digital multimeter."""

# cSpell: words INBUF TARM NRDGS AZERO LFREQ DISP MFORMAT OFORMAT DREAL ERRSTR RMEM ACAL fixedz
from __future__ import annotations

import warnings
from time import sleep
from typing import TYPE_CHECKING

import numpy as np
from msl.equipment import MSLConnectionError

if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    from msl.equipment import GPIB, Equipment


class Agilent3458A:
    """Agilent (or Hewlett Packard or Keysight) 3458A digital multimeter."""

    def __init__(self, equipment: Equipment, *, reset: bool = True, clear: bool = True) -> None:
        """Communicate with an Agilent (or Hewlett Packard or Keysight) 3458A digital multimeter.

        Args:
            equipment: An equipment instance.
            reset: Whether to automatically send the RESET command.
            clear: Whether to automatically send the GPIB CLEAR command.
        """
        self._initiate_cmd: str = "<gets updated in configure>"
        self._check_revision: bool = True
        self._num_readings: int = -1
        self._scale: float = 1.0

        self._cxn: GPIB = equipment.connect()
        self._cxn.read_termination = "\r"
        self._cxn.write_termination = "\r"

        if reset:
            self.reset()
        if clear:
            self.clear()

    def _wait_until_done(self, seconds: float) -> None:
        """Wait until the multimeter is done processing the command.

        Args:
            seconds: The number of seconds to sleep before rechecking if done.
        """
        while True:
            try:
                # From the "Using the Input Buffer" section of the manual (page 75):
                #   When using the input buffer, it may be necessary to know when all
                #   buffered commands have been executed. The multimeter provides this
                #   information by setting bit 4 (0b00010000 = 16) in the status register
                val = self._cxn.serial_poll()
                if val & 16:
                    return
            except TypeError:  # serial_poll() received an empty reply
                pass
            else:
                sleep(seconds)

    def abort(self) -> None:
        """Abort a measurement in progress."""
        self.clear()

    def calibrate(self, *, wait: bool = True) -> None:
        """Instructs the digital multimeter to perform a self calibration for DC voltage gain and offset.

        Always disconnect the input signal before performing a self calibration.

        The multimeter should be in a thermally stable environment with its power turned on for at
        least 2 hours before performing a self calibration. For maximum accuracy, you should perform
        the self calibration once every 24 hours or when the multimeter's temperature changes by ±1°C
        from when it was last externally calibrated or from the last self calibration.

        After performing the self calibration, let the instrument sit for 15 minutes before acquiring
        readings.

        Args:
            wait: Whether to wait for the self calibration to finish before returning to the calling program.
        """
        _ = self._cxn.write("ACAL DCV")
        if wait:
            self._wait_until_done(1.0)

    def clear(self) -> None:
        """Clears the event registers in all register groups and the error queue."""
        _ = self._cxn.clear()

    def configure(
        self,
        *,
        range: float = 10,  # noqa: A002
        nsamples: int = 10,
        aperture: float = 0.01,
        auto_zero: Literal["ONCE", "ON", "OFF"] = "ONCE",
        trigger: Literal["IMMEDIATE", "BUS", "EXTERNAL"] = "IMMEDIATE",
        ntriggers: int = 1,
        delay: float | None = None,
    ) -> float:
        """Configure the digital multimeter.

        Args:
            range: The range to use for the measurement.
            nsamples: The number of samples to acquire after a trigger event.
            aperture: The A/D converter integration time in seconds.
            auto_zero: The auto-zero mode. Either ONCE, ON or OFF.
            trigger: The trigger mode. Either IMMEDIATE, BUS or EXTERNAL.
            ntriggers: The number of triggers that are accepted before
                returning to the wait-for-trigger state.
            delay: The number of seconds to wait after a trigger event before
                acquiring samples. A value of `None` is equivalent to zero.

        Returns:
            The actual A/D converter integration time in seconds.
        """
        if auto_zero not in ["ONCE", "ON", "OFF"]:
            msg = f"Auto zero must be ONCE, ON or OFF. Got {auto_zero!r}"
            raise ValueError(msg)

        if trigger not in ["IMMEDIATE", "BUS", "EXTERNAL"]:
            msg = f"Trigger mode must be IMMEDIATE, BUS or EXTERNAL. Got {trigger!r}"
            raise ValueError(msg)

        # TARM  -> AUTO, EXT, HOLD,              SGL, SYN
        # TRIG  -> AUTO, EXT, HOLD, LEVEL, LINE, SGL, SYN
        # NRDGS -> AUTO, EXT,     , LEVEL, LINE       SYN, TIMER
        trig_event = "AUTO"
        if trigger == "IMMEDIATE":
            self._initiate_cmd = f"MEM FIFO;TARM SGL,{ntriggers};MEM OFF"
        elif trigger == "BUS":
            self._initiate_cmd = "TARM HOLD"
        else:
            self._initiate_cmd = f"MEM FIFO;TARM SGL,{ntriggers};MEM OFF"
            trig_event = "EXT"
            if self._check_revision:
                self._check_revision = False
                rev = tuple(map(int, self._cxn.query("REV?").split(",")))
                if rev < (9, 2):
                    msg = (
                        f"Trigger {trigger} works with firmware revision "
                        f"(9, 2), but revision (6, 2) does not work. "
                        f"The revision is {rev}."
                    )
                    warnings.warn(msg, stacklevel=2)

        # Turning the INBUF ON/OFF is required because the GPIB write()
        # method waits for the count() return value. Therefore, when
        # self.initiate() or self.trigger() is called, it blocks until a
        # timeout error is raised or until count() receives a return value.
        #
        # Used the NI GPIB-USB-HS+ adapter to communicate with the DMM
        # to determine this caveat.
        buff = "INBUF ON;INBUF OFF;"
        self._initiate_cmd = buff + self._initiate_cmd

        self._num_readings = nsamples * ntriggers
        if self._num_readings > 16_777_215:  # noqa: PLR2004
            msg = f"Too many samples requested, {self._num_readings}. Must be <= 16,777,215"
            raise ValueError(msg)

        message = (
            f"TARM HOLD;"
            f"TRIG {trig_event};"
            f"MEM FIFO;"
            f"FUNC DCV,{range};"
            f"APER {aperture};"
            f"AZERO {auto_zero};"
            f"NRDGS {nsamples},AUTO;"
            f"DELAY {delay or 0};"
            f"LFREQ LINE;"
            f"FIXEDZ ON;"
            f"MATH OFF;"
            f"DISP OFF;"
            f"MFORMAT DINT;"
            f"OFORMAT DINT;"
            f"END ALWAYS;"
        )
        _ = self._cxn.write(message)

        message = self._cxn.query("ERRSTR?")
        if not message.startswith("0,"):
            raise MSLConnectionError(self._cxn, message)

        dt = float(self._cxn.query("APER?"))
        self._scale = float(self._cxn.query("ISCALE?"))

        self._cxn.read_termination = None  # DMM returns binary data in fetch() using EOI termination
        return dt

    def disconnect(self) -> None:
        """Turn the display back on and disconnect from the digital multimeter."""
        _ = self._cxn.write("DISP ON")
        self._cxn.disconnect()

    def fetch(self, *, initiate: bool = False) -> npt.NDArray[np.float64]:
        """Fetch the samples.

        This is a blocking call and will not return to the calling program until
        all samples have been acquired.

        Args:
            initiate: Whether to call `initiate()` before fetching the samples.

        Returns:
            The samples.
        """
        if initiate:
            self.initiate()

        self._wait_until_done(0.1)

        # From the RMEM documentation on page 336 of manual (Edition 10, March 2023):
        #   The multimeter assigns a number to each reading in reading memory. The most
        #   recent reading is assigned the lowest number (1) and the oldest reading has the
        #   highest number. Numbers are always assigned in this manner regardless of
        #   whether you're using the FIFO or LIFO mode.
        # This means that samples is an array of [latest reading, ..., first reading]
        # Want FIFO, so reverse array to be [first reading, ..., latest reading]
        buffer = bytearray(self._cxn.query(f"END ON;RMEM 1,{self._num_readings},1", decode=False))

        # Occasionally the returned buffer had the wrong length, could not reproduce it reliably
        # Not sure if this 'while' loop fixes the issue
        size = self._num_readings * 4
        while len(buffer) < size:
            buffer.extend(self._cxn.read(decode=False))

        return self._scale * np.frombuffer(buffer, dtype=">i4")[::-1]

    def initiate(self) -> None:
        """Put the digital multimeter in the wait-for-trigger state (arm the trigger).

        If the digital multimeter has been configured for trigger mode `IMMEDIATE`,
        then the digital multimeter will start acquiring data once this method is called.
        """
        _ = self._cxn.write(self._initiate_cmd)

    def reset(self) -> None:
        """Resets the digital multimeter to the factory default state."""
        _ = self._cxn.write("RESET;TARM HOLD")

    def serial_poll(self) -> int:
        """Read the status byte by serial polling the digital multimeter.

        Returns:
            The status byte.
        """
        return self._cxn.serial_poll()

    def trigger(self) -> None:
        """Send a software trigger.

        If the digital multimeter has been configured for trigger mode `BUS`, then
        the digital multimeter will start acquiring data once this method is called.
        """
        _ = self._cxn.write("MEM FIFO;TARM SGL")
