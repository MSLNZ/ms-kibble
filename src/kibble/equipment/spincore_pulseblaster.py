"""SpinCore PulseBlaster."""

from __future__ import annotations

import re
from ctypes import c_char_p, c_double, c_int
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from msl.equipment.connection_sdk import ConnectionSDK  # type: ignore[import-untyped]
from msl.equipment.resources import register  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from msl.equipment.record_types import EquipmentRecord  # type: ignore[import-untyped]


@dataclass
class Version:
    """Board software and firmware version numbers."""

    software: str
    firmware: str


class Status(IntEnum):
    """Board status."""

    STOPPED = 1 << 0
    RESET = 1 << 1
    RUNNING = 1 << 2
    WAITING = 1 << 3
    SCANNING = 1 << 4


class Code(IntEnum):
    """Program instruction codes."""

    CONTINUE = 0
    STOP = 1
    LOOP = 2
    END_LOOP = 3
    JSR = 4
    RTS = 5
    BRANCH = 6
    LONG_DELAY = 7
    WAIT = 8
    RTI = 9


@register(manufacturer=r"Spin\s*Core", model=r"Pulse\s*Blaster", flags=re.IGNORECASE)
class PulseBlaster(ConnectionSDK):  # type: ignore[misc]
    """SpinCore PulseBlaster."""

    CODE = Code
    STATUS = Status

    MIN_DURATION = 50e-9  # 50ns is the minimum duration for a pulse-program instruction

    def __init__(self, record: EquipmentRecord) -> None:  # noqa: PLR0915
        """Communicate with a SpinCore PulseBlaster.

        See [SpinAPI](http://www.spincore.com/support/spinapi/reference/production/2013-09-25/spinapi_8c.html)
        for the API reference.

        Args:
            record: The equipment record.
        """
        self._closed = False
        super().__init__(record=record, libtype="cdll")

        # initialize function declarations in spinapi64.dll
        self.sdk.pb_get_error.argtype = []
        self.sdk.pb_get_error.restype = c_char_p

        self.sdk.pb_read_status.argtype = []
        self.sdk.pb_read_status.restype = c_int

        self.sdk.pb_count_boards.argtype = []
        self.sdk.pb_count_boards.restype = c_int

        self.sdk.pb_select_board.argtype = [c_int]
        self.sdk.pb_select_board.restype = c_int
        self.sdk.pb_select_board.errcheck = self._errcheck

        self.sdk.pb_init.argtype = []
        self.sdk.pb_init.restype = c_int
        self.sdk.pb_init.errcheck = self._errcheck

        self.sdk.pb_close.argtype = []
        self.sdk.pb_close.restype = c_int
        self.sdk.pb_close.errcheck = self._errcheck

        self.sdk.pb_get_version.argtype = []
        self.sdk.pb_get_version.restype = c_char_p

        self.sdk.pb_get_firmware_id.argtype = []
        self.sdk.pb_get_firmware_id.restype = c_int

        self.sdk.pb_core_clock.argtype = [c_double]
        self.sdk.pb_core_clock.restype = None

        self.sdk.pb_start_programming.argtype = [c_int]
        self.sdk.pb_start_programming.restype = c_int
        self.sdk.pb_start_programming.errcheck = self._errcheck

        self.sdk.pb_stop_programming.argtype = []
        self.sdk.pb_stop_programming.restype = c_int
        self.sdk.pb_stop_programming.errcheck = self._errcheck

        self.sdk.pb_inst_pbonly.argtype = [c_int, c_int, c_int, c_double]
        self.sdk.pb_inst_pbonly.restype = c_int
        self.sdk.pb_inst_pbonly.errcheck = self._errcheck

        self.sdk.pb_reset.argtype = []
        self.sdk.pb_reset.restype = c_int
        self.sdk.pb_reset.errcheck = self._errcheck

        self.sdk.pb_start.argtype = []
        self.sdk.pb_start.restype = c_int
        self.sdk.pb_start.errcheck = self._errcheck

        self.sdk.pb_stop.argtype = []
        self.sdk.pb_stop.restype = c_int
        self.sdk.pb_stop.errcheck = self._errcheck

        # configure the default PulseBlaster settings
        n_boards: int = self.sdk.pb_count_boards()
        if n_boards <= 0:
            msg = "No PulseBlaster boards are available"
            raise RuntimeError(msg)
        if n_boards > 1:
            msg = f"{n_boards} PulseBlaster boards are available."
            raise RuntimeError(msg)

        self.sdk.pb_select_board(0)
        self.sdk.pb_init()
        self.sdk.pb_core_clock(c_double(100))  # 100 MHz clock frequency

    def _errcheck(self, result: int, func: Any, args: tuple[Any, ...]) -> int:  # noqa: ANN401
        if result >= 0:
            return result

        msg = self.sdk.pb_get_error().decode() or "Unknown reason"
        msg = f"PulseBlaster.{func.__name__}{args} {msg} [code: {result}]"
        raise RuntimeError(msg)

    def add_instruction(
        self,
        *,
        bits: Sequence[int] | None = None,
        code: Code | int = Code.CONTINUE,
        duration: float = 1e-3,
        data: int = 0,
    ) -> int:
        """Add an instruction to the ``PULSE_PROGRAM``.

        Args:
            bits: A sequence of bits to set to be TTL high. Each value must
                be between 0 and 23, inclusive.
            code: Operation code for the instruction.
            duration: Number of seconds to use for this instruction.
            data: The corresponding data for the `code` parameter.

        Returns:
            The address of the created instruction. This address can be
            used as the branch address for any branch instructions.
        """
        flags = 0
        if bits is not None:
            for bit in bits:
                if bit < 0 or bit > 23:  # noqa: PLR2004
                    msg = f"A bit must be in the range 0..23, got {bit}"
                    raise ValueError(msg)
                flags |= 1 << bit

        return int(self.sdk.pb_inst_pbonly(flags, code, data, c_double(duration * 1e9)))

    def configure_two_pulses(
        self,
        *,
        pulse1: int = 0,
        pulse2: int = 1,
        width: float = 1e-3,
        delay: float = 0,
        single: bool = True,
        period: float | None = None,
    ) -> None:
        """Configure a new ``PULSE_PROGRAM`` that creates two pulses.

        Args:
            pulse1: The `bit#` to use for the first pulse.
            pulse2: The `bit#` to use for the second pulse.
            width: The width, in seconds, of each pulse.
            delay: The delay, in seconds, of the second pulse.
            single: Whether the pulses are output in single-shot mode. If enabled,
                the `trigger()` method must be called before the pulses are output.
            period: The time, in seconds, between the rising edge of the
                first pulses. Only used if `single` is `False`.
        """
        if delay < 0:
            msg = f"Only positive delays are allowed, got {delay}"
            raise ValueError(msg)

        self.start_programming()

        if delay == 0:
            total = width
            start = self.add_instruction(bits=[pulse1, pulse2], duration=width)
        elif delay < width:
            d = max(width - delay, self.MIN_DURATION)
            total = delay + d + delay
            start = self.add_instruction(bits=[pulse1], duration=delay)
            self.add_instruction(bits=[pulse1, pulse2], duration=d)
            self.add_instruction(bits=[pulse2], duration=delay)
        else:
            d = max(delay - width, self.MIN_DURATION)
            total = width + d + width
            start = self.add_instruction(bits=[pulse1], duration=width)
            self.add_instruction(duration=d)
            self.add_instruction(bits=[pulse2], duration=width)

        self.add_instruction(duration=self.MIN_DURATION)
        if single:
            self.add_instruction(code=Code.STOP, duration=self.MIN_DURATION)
        else:
            if period:
                if period < total:
                    msg = f"period ({period}) < total time for pulses ({total})"
                    raise ValueError(msg)
                period = max(period - total, self.MIN_DURATION)
            else:
                period = self.MIN_DURATION
            self.add_instruction(code=Code.BRANCH, data=start, duration=period)

        self.stop_programming()

    def disconnect(self) -> None:
        """Disconnect from the PulseBlaster board."""
        if self._closed:
            return
        self.sdk.pb_close()
        self._closed = True

    def reset(self) -> None:
        """Stops the output of board and resets the PulseBlaster."""
        self.sdk.pb_reset()

    def start(self) -> None:
        """Send a software trigger to the board to start the pulse program."""
        self.sdk.pb_start()

    def start_programming(self) -> None:
        """Start a ``PULSE_PROGRAM``."""
        self.sdk.pb_start_programming(0)  # PULSE_PROGRAM = 0

    def status(self) -> Status:
        """Read status from the board."""
        return Status(self.sdk.pb_read_status())

    def stop(self) -> None:
        """Stops output of board.

        Analog output will return to ground, and TTL outputs will either remain in the same
        state they were in when the reset command was received or return to ground.
        """
        self.sdk.pb_stop()

    def stop_programming(self) -> None:
        """Stop programming the ``PULSE_PROGRAM``, which was started by `start_programming()`."""
        self.sdk.pb_stop_programming()

    def trigger(self) -> None:
        """Restart the ``PULSE_PROGRAM``."""
        self.reset()
        self.start()

    def version(self) -> Version:
        """Get the version information.

        Returns:
            Version information.
        """
        fw = self.sdk.pb_get_firmware_id()

        # See: C:\SpinCore\SpinAPI\examples\General\pb_read_firmware.c
        device = (fw & 0xFF00) >> 8
        revision = fw & 0x00FF

        return Version(software=self.sdk.pb_get_version().decode(), firmware=f"{device}-{revision}")
