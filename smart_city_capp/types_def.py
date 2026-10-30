"""Shared data types for the Smart City Computing Framework simulator.

These lightweight models are intentionally independent of simulator modules so
components can exchange packets, instructions, coherence states, and metrics
without introducing import cycles.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union


class MESIState(Enum):
    """Possible states of a cache line in the MESI coherence protocol."""

    MODIFIED = "M"
    EXCLUSIVE = "E"
    SHARED = "S"
    INVALID = "I"


class MOESIState(Enum):
    """Possible states of a cache line in the MOESI protocol."""

    MODIFIED = "M"
    OWNED = "O"
    EXCLUSIVE = "E"
    SHARED = "S"
    INVALID = "I"


@dataclass
class Packet:
    """A bus transaction submitted by a sensor or other device.

    Args:
        dev_id: Stable identifier for the requesting device.
        name: Human-readable device or transaction name.
        is_emergency: Whether the request uses the emergency-priority channel.
        data: Payload transported by the bus; its shape is component-specific.
    """

    dev_id: int
    name: str
    is_emergency: bool
    data: Any


@dataclass
class Instruction:
    """One instruction flowing through the simulated processor pipeline."""

    pc: int
    opcode: str
    dest: Optional[Union[int, str]]
    src1: Optional[Union[int, str]]
    src2: Optional[Union[int, str]] = None
    is_branch: bool = False
    branch_target: Optional[int] = None

    def __repr__(self) -> str:
        """Return the compact register-oriented representation used by the UI."""
        sources = f", R{self.src2}" if self.src2 is not None else ""
        return f"{self.opcode} R{self.dest}, R{self.src1}{sources}"


@dataclass
class SystemMetrics:
    """Cumulative architectural counters used by the performance scorecard.

    Counters are updated by the owning simulator components. Derived metrics
    return ``0.0`` when their denominator is zero.
    """

    cycles: int = 0
    instructions_retired: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    energy_units: float = 0.0
    reference_energy_units: float = 0.0

    @property
    def cpi(self) -> float:
        """Return cycles per retired instruction."""
        return self.cycles / self.instructions_retired if self.instructions_retired else 0.0

    @property
    def ipc(self) -> float:
        """Return retired instructions per cycle."""
        return self.instructions_retired / self.cycles if self.cycles else 0.0

    @property
    def cache_hit_percent(self) -> float:
        """Return the cache hit rate as a percentage."""
        accesses = self.cache_hits + self.cache_misses
        return 100.0 * self.cache_hits / accesses if accesses else 0.0

    @property
    def energy_factor(self) -> float:
        """Return energy relative to a reference design, when provided."""
        return self.energy_units / self.reference_energy_units if self.reference_energy_units else 0.0