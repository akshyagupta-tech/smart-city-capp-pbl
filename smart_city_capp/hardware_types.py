"""Core typed interfaces shared by the Smart City hardware simulator.

This module contains data contracts only. Keeping protocol state and packet
shape definitions here avoids import cycles between the bus, core, ALU, and
memory/interconnect models.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional, Union


class MESIState(Enum):
    """MOESI-compatible cache-line coherence states."""

    MODIFIED = "M"
    EXCLUSIVE = "E"
    SHARED = "S"
    INVALID = "I"
    OWNER = "O"


class PacketPriority(IntEnum):
    """Lower numeric values represent more urgent bus traffic."""

    EMERGENCY = 0
    HIGH_TRAFFIC = 1
    ROUTINE_TELEMETRY = 2


class Opcode(Enum):
    """Instruction operations supported by the architectural simulator."""

    LOAD = "LOAD"
    STORE = "STORE"
    ADD = "ADD"
    SUB = "SUB"
    MUL_BOOTH = "MUL_BOOTH"
    DIV_NONRESTORE = "DIV_NONRESTORE"
    FMA = "FMA"
    BRANCH = "BRANCH"
    FENCE = "FENCE"
    ATOMIC_LL = "ATOMIC_LL"
    ATOMIC_SC = "ATOMIC_SC"


@dataclass(frozen=True)
class TelemetryPacket:
    """Sensor payload and metadata submitted to the shared bus.

    ``timestamp`` is a simulator timestamp in seconds or cycles as selected by
    the driver. ``token_cost`` allows larger packets to consume more rate-limit
    budget than small status updates.
    """

    id: int
    device_id: int
    source_zone: str
    priority: PacketPriority
    payload: Any
    timestamp: float
    token_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("packet id cannot be negative")
        if self.device_id < 0:
            raise ValueError("device_id cannot be negative")
        if self.token_cost <= 0:
            raise ValueError("token_cost must be positive")


@dataclass(frozen=True)
class MicroInstruction:
    """Decoded micro-operation passed between processor pipeline stages."""

    pc: int
    opcode: Opcode
    dest: Optional[Union[int, str]] = None
    src1: Optional[Union[int, str]] = None
    src2: Optional[Union[int, str]] = None
    imm: Optional[int] = None
    is_branch: bool = False
    target_pc: Optional[int] = None


@dataclass
class CacheLine:
    """Cache-line state and integrity metadata for L1/L2 and directory models."""

    tag: int
    state: MESIState = MESIState.INVALID
    data: Any = None
    dirty: bool = False
    last_access: int = 0
    error_syndrome: int = 0
