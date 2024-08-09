"""Agilent 33500B Waveform Generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    from msl.equipment import EquipmentRecord  # type: ignore[import-untyped]


class Agilent33500B:
    """Agilent 33500B Waveform Generator."""

    def __init__(self, record: EquipmentRecord, *, reset: bool = True, clear: bool = True) -> None:
        """Communicate with an Agilent 33500B Waveform Generator.

        :param record: The equipment record.
        :param reset: Whether to automatically send the *RST command.
        :param clear: Whether to automatically send the *CLS command.
        """
        self._cxn = record.connect()

        if reset:
            self.reset()
        if clear:
            self.clear()

    def _check_error(self) -> None:
        """Check the waveform generator for an internal error."""
        reply: str = self._cxn.query("SYSTEM:ERROR?")
        if not reply.startswith("+0,"):
            self._cxn.raise_exception(reply)

    def sine(
        self,
        channel: int,
        *,
        amplitude: float = 3,
        frequency: float = 1,
        load: float | None = 50,
        offset: float = 0,
        phase: float = 0,
        unit: Literal["VPP", "VRMS", "DBM"] = "VPP",
    ) -> None:
        """Configure a SINUSOID waveform for a particular channel.

        :param channel: The channel number to configure.
        :param amplitude: The amplitude of the waveform.
        :param frequency: The frequency, in Hz, of the waveform.
        :param load: The load termination, in Ohms. In the range 1 to 10 kOhm,
            or :data:`None` for infinite (High Z).
        :param offset: The offset of the waveform.
        :param phase: The phase, in degrees, of the waveform. In the range
            0 to 360 degrees.
        :param unit: The `amplitude` and `offset` unit.
        """
        if channel not in [1, 2]:
            msg = f"Channel must be either 1 or 2, got {channel}"
            raise ValueError(msg)

        if unit not in ["VPP", "VRMS", "DBM"]:
            msg = f"Only VPP, VRMS or DBM units are currently allowed, got {unit}"
            raise ValueError(msg)

        if phase < 0 or phase > 360:  # noqa: PLR2004
            msg = f"The phase must be between 0 and 360 degrees, got {phase}"
            raise ValueError(msg)

        if load is not None and (load < 1 or load > 10e3):  # noqa: PLR2004
            msg = f"The terminal load must be between 1 and 10 kOhm or None, got {load}"
            raise ValueError(msg)

        self._cxn.write(
            f":OUTPUT{channel}:LOAD {load or 'INFINITY'};"  # OUTPUT must come before SOURCE
            f":SOURCE{channel}:FUNCTION SINUSOID;"
            f":SOURCE{channel}:FREQUENCY {frequency};"
            f":SOURCE{channel}:VOLT:UNIT {unit};"
            f":SOURCE{channel}:VOLT {amplitude};"
            f":SOURCE{channel}:VOLT:OFFSET {offset};"
            f":SOURCE{channel}:PHASE {phase};"
        )
        self._check_error()

    def clear(self) -> None:
        """Clears the event registers in all register groups and the error queue."""
        self._cxn.write("*CLS")

    def disconnect(self) -> None:
        """Disconnect from the waveform generator."""
        self._cxn.disconnect()

    def output(self, *, channel: int, state: bool) -> None:
        """Turn the output of a channel on or off.

        :param channel: The channel number, 1 or 2.
        :param state: Either on (True) or off (False)
        """
        if channel not in [1, 2]:
            msg = f"Channel must be either 1 or 2, got {channel}"
            raise ValueError(msg)
        s = "ON" if state else "OFF"
        self._cxn.write(f"OUTPUT{channel} {s}")

    def reset(self) -> None:
        """Resets the waveform generator to the factory default state."""
        self._cxn.write("*RST")
