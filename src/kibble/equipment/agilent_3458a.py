"""Agilent (or Hewlett Packard or Keysight) 3458A digital multimeter."""

from __future__ import annotations

import warnings
from time import sleep
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    from msl.equipment.record_types import EquipmentRecord  # type: ignore[import-untyped]


class Agilent3458A:
    """Agilent (or Hewlett Packard or Keysight) 3458A digital multimeter."""

    def __init__(self, record: EquipmentRecord, *, reset: bool = True, clear: bool = True) -> None:
        """Communicate with an Agilent (or Hewlett Packard or Keysight) 3458A digital multimeter.

        Args:
            record: The equipment record.
            reset: Whether to automatically send the RESET command.
            clear: Whether to automatically send the GPIB CLEAR command.
        """
        self._initiate_cmd: str = "<gets updated in configure>"
        self._check_revision: bool = True
        self._nreadings: int = -1

        record.connection.properties.setdefault("termination", "\r")
        self._cxn = record.connect()

        if reset:
            self.reset()
        if clear:
            self.clear()

    def abort(self) -> None:
        """Abort a measurement in progress."""
        self.clear()

    def clear(self) -> None:
        """Clears the event registers in all register groups and the error queue."""
        self._cxn.clear()

    def configure(  # noqa: C901
        self,
        *,
        function: Literal["DCV"] = "DCV",
        range: float = 10,  # noqa: A002
        nsamples: int = 10,
        aperature: float = 0.01,
        auto_zero: Literal["ONCE", "ON", "OFF"] = "ONCE",
        trigger: Literal["IMMEDIATE", "BUS", "EXTERNAL"] = "IMMEDIATE",
        edge: Literal["FALLING"] = "FALLING",
        ntriggers: int = 1,
        delay: float | None = None,
    ) -> float:
        """Configure the digital multimeter.

        Args:
            function: The measurement function. Currently, only DCV is allowed.
            range: The range to use for the measurement.
            nsamples: The number of samples to acquire after a trigger event.
            aperature: The A/D converter integration time in seconds.
            auto_zero: The auto-zero mode. Either ONCE, ON or OFF.
            trigger: The trigger mode. Either IMMEDIATE, BUS or EXTERNAL.
            edge: The trigger edge (only used if `trigger` is EXTERNAL).
                Must always be FALLING.
            ntriggers: The number of triggers that are accepted before
                returning to the wait-for-trigger state.
            delay: The number of seconds to wait after a trigger event before
                acquiring samples. A value of `None` is equivalent to zero.

        Returns:
            The actual A/D converter integration time in seconds.
        """
        if function != "DCV":
            msg = f"Only DCV is implemented, not {function!r}"
            raise ValueError(msg)

        if edge != "FALLING":
            msg = f"Can only trigger on FALLING edge, got {edge!r}"
            raise ValueError(msg)

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
                    warnings.warn(
                        f"Trigger {trigger} works with firmware revision "
                        f"(9, 2), but revision (6, 2) does not work. "
                        f"The revision is {rev}.",
                        stacklevel=2,
                    )

        # Turning the INBUF ON/OFF is required because the GPIB write()
        # method waits for the count() return value. Therefore, when
        # self.initiate() or self.trigger() is called, it blocks until a
        # timeout error is raised or until count() receives a return value.
        #
        # Used the NI GPIB-USB-HS+ adapter to communicate with the DMM
        # to determine this caveat.
        buff = "INBUF ON;INBUF OFF;"
        self._initiate_cmd = buff + self._initiate_cmd

        fixedz = "ON" if function in ["DCV", "OHM", "OHMF"] else "OFF"

        self._nreadings = nsamples * ntriggers
        if self._nreadings > 16_777_215:  # noqa: PLR2004
            msg = f"Too many samples requested, {self._nreadings}. Must be <= 16,777,215"
            raise ValueError(msg)

        self._cxn.write(
            f"TARM HOLD;"
            f"TRIG {trig_event};"
            f"MEM FIFO;"
            f"FUNC {function},{range};"
            f"APER {aperature};"
            f"AZERO {auto_zero};"
            f"NRDGS {nsamples},AUTO;"
            f"DELAY {delay or 0};"
            f"LFREQ LINE;"
            f"FIXEDZ {fixedz};"
            f"MATH OFF;"
            f"DISP OFF;"
            # f'MFORMAT DREAL;'  TODO not working yet
            # f'OFORMAT DREAL;'
        )

        message = self._cxn.query("ERRSTR?")
        if not message.startswith("0,"):
            self._cxn.raise_exception(message)

        return float(self._cxn.query("APER?"))

    def disconnect(self) -> None:
        """Turn the display back on and disconnect from the digital multimeter."""
        self._cxn.write("DISP ON")
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

        while True:
            try:
                # From the "Using the Input Buffer" section of the manual (page 75):
                #   When using the input buffer, it may be necessary to know when all
                #   buffered commands have been executed. The multimeter provides this
                #   information by setting bit 4 (0b00010000 = 16) in the status register
                val = self._cxn.serial_poll()
                if val & 16:
                    break
            except TypeError:  # serial_poll() received an empty reply
                pass
            else:
                sleep(0.1)

        # From the RMEM documentation on page 336 of manual (Edition 10, March 2023):
        #   The multimeter assigns a number to each reading in reading memory. The most
        #   recent reading is assigned the lowest number (1) and the oldest reading has the
        #   highest number. Numbers are always assigned in this manner regardless of
        #   whether you're using the FIFO or LIFO mode.
        # This means that samples is an array of [latest reading, ..., first reading]
        samples = self._cxn.query(f"RMEM 1,{self._nreadings},1")
        # Want FIFO, so reverse to be [first reading, ..., latest reading]
        return np.array(samples.split(",")[::-1], dtype=np.float64)

    def initiate(self) -> None:
        """Put the digital multimeter in the wait-for-trigger state (arm the trigger).

        If the digital multimeter has been configured for trigger mode `IMMEDIATE`,
        then the digital multimeter will start acquiring data once this method is called.
        """
        self._cxn.write(self._initiate_cmd)

    def reset(self) -> None:
        """Resets the digital multimeter to the factory default state."""
        self._cxn.write("RESET;TARM HOLD;")

    def trigger(self) -> None:
        """Send a software trigger.

        If the digital multimeter has been configured for trigger mode `BUS`, then
        the digital multimeter will start acquiring data once this method is called.
        """
        self._cxn.write("MEM FIFO;TARM SGL")
