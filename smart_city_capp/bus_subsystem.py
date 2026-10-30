"""Preemptive bus, split transactions, SECDED, and scatter-gather DMA.

All components use only Python's standard library. The models are cycle
stepped: callers decide when to advance arbitration or DMA service cycles.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, MutableMapping, Optional, Sequence, Tuple

from hardware_types import PacketPriority, TelemetryPacket


@dataclass
class TokenBucket:
    """Per-device token bucket with cycle-based refill."""

    capacity: float
    refill_per_cycle: float
    tokens: Optional[float] = None
    last_cycle: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.refill_per_cycle < 0:
            raise ValueError("bucket capacity must be positive and refill non-negative")
        if self.tokens is None:
            self.tokens = self.capacity
        elif not 0 <= self.tokens <= self.capacity:
            raise ValueError("initial tokens must be within bucket capacity")

    def consume(self, cost: float, cycle: int) -> bool:
        """Refill through ``cycle`` and atomically consume ``cost`` tokens."""
        if cost <= 0:
            raise ValueError("token cost must be positive")
        elapsed = max(0, cycle - self.last_cycle)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_cycle)
        self.last_cycle = max(self.last_cycle, cycle)
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True


@dataclass(frozen=True)
class BusAdmission:
    """Result of submitting a telemetry packet to the bus subsystem."""

    accepted: bool
    reason: str
    packet_id: int


class DualTierPreemptiveBusArbiter:
    """Emergency-preemptive bus with rate limiting and weighted fair polling.

    Emergency packets use a dedicated FIFO that is not charged against the
    routine buffer capacity. Consequently, capacity pressure never evicts an
    admitted emergency packet. High and routine packets share the bounded
    buffer. Above the high-water mark, routine traffic is dropped first.

    When the emergency FIFO is continuously occupied, ``max_emergency_burst``
    bounds consecutive grants before a waiting normal packet gets service.
    Remaining normal traffic follows a 2:1 weighted high:routine schedule.
    Separate emergency/routine token buckets are maintained for each device so
    routine telemetry cannot consume a device's emergency allowance.
    """

    _WEIGHTED_NORMAL_SCHEDULE = (
        PacketPriority.HIGH_TRAFFIC,
        PacketPriority.HIGH_TRAFFIC,
        PacketPriority.ROUTINE_TELEMETRY,
    )

    def __init__(
        self,
        buffer_capacity: int = 128,
        high_watermark: float = 0.85,
        token_capacity: float = 8.0,
        token_refill_per_cycle: float = 1.0,
        emergency_token_capacity: float = 4.0,
        emergency_refill_per_cycle: float = 1.0,
        max_emergency_burst: int = 8,
    ) -> None:
        if buffer_capacity < 1:
            raise ValueError("buffer_capacity must be positive")
        if not 0.0 < high_watermark <= 1.0:
            raise ValueError("high_watermark must be in (0, 1]")
        if max_emergency_burst < 1:
            raise ValueError("max_emergency_burst must be positive")
        self.buffer_capacity = buffer_capacity
        self.high_watermark = high_watermark
        self.token_capacity = token_capacity
        self.token_refill_per_cycle = token_refill_per_cycle
        self.emergency_token_capacity = emergency_token_capacity
        self.emergency_refill_per_cycle = emergency_refill_per_cycle
        self.max_emergency_burst = max_emergency_burst

        self.emergency_queue: Deque[TelemetryPacket] = deque()
        self.high_queue: Deque[TelemetryPacket] = deque()
        self.routine_queue: Deque[TelemetryPacket] = deque()
        self._buckets: Dict[Tuple[int, bool], TokenBucket] = {}
        self._normal_schedule_index = 0
        self._consecutive_emergency_grants = 0
        self.cycle = 0
        self.bus_master: Optional[TelemetryPacket] = None
        self.emergency_interrupt_pending = False
        self.emergency_preemptions = 0
        self.accepted_packets = 0
        self.throttled_packets = 0
        self.dropped_packets = 0
        self.grants_by_priority: Dict[PacketPriority, int] = defaultdict(int)

    @property
    def normal_occupancy(self) -> int:
        """Number of packets occupying the bounded high/routine queues."""
        return len(self.high_queue) + len(self.routine_queue)

    @property
    def buffer_occupancy(self) -> int:
        """Total queued packets, including the independent emergency FIFO."""
        return self.normal_occupancy + len(self.emergency_queue)

    @property
    def normal_utilization(self) -> float:
        """Bounded-buffer utilization; emergency storage is out-of-band."""
        return self.normal_occupancy / self.buffer_capacity

    def _bucket_for(self, packet: TelemetryPacket) -> TokenBucket:
        is_emergency = packet.priority == PacketPriority.EMERGENCY
        key = (packet.device_id, is_emergency)
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=(self.emergency_token_capacity if is_emergency else self.token_capacity),
                refill_per_cycle=(
                    self.emergency_refill_per_cycle
                    if is_emergency
                    else self.token_refill_per_cycle
                ),
                last_cycle=self.cycle,
            )
        return self._buckets[key]

    def _drop_oldest_routine(self) -> Optional[TelemetryPacket]:
        """Evict the oldest routine packet, never an emergency packet."""
        if not self.routine_queue:
            return None
        dropped = self.routine_queue.popleft()
        self.dropped_packets += 1
        print(f"[Bus Drop] Evicted routine packet {dropped.id} to protect higher priority traffic")
        return dropped

    def submit(self, packet: TelemetryPacket) -> BusAdmission:
        """Rate-limit and enqueue one packet, returning its admission decision."""
        bucket = self._bucket_for(packet)
        if not bucket.consume(packet.token_cost, self.cycle):
            self.throttled_packets += 1
            print(
                f"[Token Bucket] Throttled device {packet.device_id}, "
                f"packet {packet.id} ({packet.priority.name})"
            )
            return BusAdmission(False, "rate_limited", packet.id)

        if packet.priority == PacketPriority.EMERGENCY:
            if self.bus_master is not None and not self.emergency_interrupt_pending:
                self.emergency_preemptions += 1
            self.emergency_queue.append(packet)
            self.emergency_interrupt_pending = True
            self.accepted_packets += 1
            print(
                f"[IRQ] Emergency line asserted by device {packet.device_id}; "
                f"packet {packet.id} queued out-of-band"
            )
            return BusAdmission(True, "emergency_interrupt", packet.id)

        projected = (self.normal_occupancy + 1) / self.buffer_capacity
        if packet.priority == PacketPriority.ROUTINE_TELEMETRY and projected > self.high_watermark:
            self.dropped_packets += 1
            print(
                f"[Bus Drop] Routine packet {packet.id} rejected at "
                f"{projected:.0%} projected occupancy"
            )
            return BusAdmission(False, "high_watermark_drop", packet.id)

        if self.normal_occupancy >= self.buffer_capacity:
            if packet.priority == PacketPriority.HIGH_TRAFFIC and self._drop_oldest_routine() is not None:
                pass
            else:
                self.dropped_packets += 1
                print(f"[Bus Drop] Packet {packet.id} rejected; normal buffer full")
                return BusAdmission(False, "buffer_full", packet.id)

        if packet.priority == PacketPriority.HIGH_TRAFFIC and projected > self.high_watermark:
            self._drop_oldest_routine()
        target = self.high_queue if packet.priority == PacketPriority.HIGH_TRAFFIC else self.routine_queue
        target.append(packet)
        self.accepted_packets += 1
        return BusAdmission(True, "queued", packet.id)

    def _grant_weighted_normal(self) -> Optional[TelemetryPacket]:
        """Select a normal packet using the repeating 2:1 weighted schedule."""
        for offset in range(len(self._WEIGHTED_NORMAL_SCHEDULE)):
            slot = (self._normal_schedule_index + offset) % len(self._WEIGHTED_NORMAL_SCHEDULE)
            priority = self._WEIGHTED_NORMAL_SCHEDULE[slot]
            queue = self.high_queue if priority == PacketPriority.HIGH_TRAFFIC else self.routine_queue
            if queue:
                self._normal_schedule_index = (slot + 1) % len(self._WEIGHTED_NORMAL_SCHEDULE)
                self._consecutive_emergency_grants = 0
                return queue.popleft()
        return None

    def arbitrate_cycle(self) -> Optional[TelemetryPacket]:
        """Advance one cycle and grant the next preemptive/weighted request."""
        self.cycle += 1
        normal_waiting = bool(self.high_queue or self.routine_queue)
        if self.emergency_queue and not (
            normal_waiting and self._consecutive_emergency_grants >= self.max_emergency_burst
        ):
            granted = self.emergency_queue.popleft()
            self._consecutive_emergency_grants += 1
            self.bus_master = granted
            self.grants_by_priority[granted.priority] += 1
            self.emergency_interrupt_pending = bool(self.emergency_queue)
            print(f"[Bus Grant C{self.cycle}] EMERGENCY packet {granted.id} (dedicated IRQ path)")
            return granted

        granted = self._grant_weighted_normal()
        if granted is not None:
            self.bus_master = granted
            self.grants_by_priority[granted.priority] += 1
            print(f"[Bus Grant C{self.cycle}] {granted.priority.name} packet {granted.id}")
            return granted

        if self.emergency_queue:
            granted = self.emergency_queue.popleft()
            self._consecutive_emergency_grants += 1
            self.bus_master = granted
            self.grants_by_priority[granted.priority] += 1
            self.emergency_interrupt_pending = bool(self.emergency_queue)
            print(f"[Bus Grant C{self.cycle}] EMERGENCY packet {granted.id} (dedicated IRQ path)")
            return granted

        self.bus_master = None
        self._consecutive_emergency_grants = 0
        print(f"[Bus Grant C{self.cycle}] idle")
        return None

    def request_bus(self, packet: TelemetryPacket) -> bool:
        """Compatibility wrapper returning only the admission boolean."""
        return self.submit(packet).accepted


@dataclass(frozen=True)
class SplitTransaction:
    """Address phase metadata reserved until the matching data return."""

    tag_id: int
    packet_id: int
    address: int
    operation: str
    request_cycle: int


@dataclass(frozen=True)
class TransactionCompletion:
    """Data phase response for a completed split transaction."""

    transaction: SplitTransaction
    data: Any
    completion_cycle: int


class SplitTransactionBus:
    """Address/data-decoupled bus with 16 independently tracked transaction tags."""

    TAG_COUNT = 16

    def __init__(self) -> None:
        self._free_tags: Deque[int] = deque(range(self.TAG_COUNT))
        self.outstanding: Dict[int, SplitTransaction] = {}
        self.cycle = 0
        self.address_phases = 0
        self.data_returns = 0

    def begin_transaction(
        self,
        packet_id: int,
        address: int,
        operation: str = "READ",
    ) -> Optional[SplitTransaction]:
        """Issue an address phase or return ``None`` if all tags are occupied."""
        if not self._free_tags:
            return None
        tag_id = self._free_tags.popleft()
        transaction = SplitTransaction(tag_id, packet_id, address, operation, self.cycle)
        self.outstanding[tag_id] = transaction
        self.address_phases += 1
        return transaction

    def complete_transaction(self, tag_id: int, data: Any) -> TransactionCompletion:
        """Complete a data return and recycle its transaction tag."""
        if tag_id not in self.outstanding:
            raise KeyError(f"unknown or already completed transaction tag {tag_id}")
        transaction = self.outstanding.pop(tag_id)
        self._free_tags.append(tag_id)
        self.data_returns += 1
        self.cycle += 1
        return TransactionCompletion(transaction, data, self.cycle)


@dataclass(frozen=True)
class SECDEDDecodeResult:
    """SECDED decode output for one 64-bit bus transfer."""

    data: int
    syndrome: int
    corrected: bool
    double_error: bool
    corrected_codeword: int


class SECDEDHamming72_64:
    """Extended Hamming [72,64] code for single-error correction/double detection."""

    DATA_BITS = 64
    HAMMING_BITS = 71
    CODE_BITS = 72
    _PARITY_POSITIONS = frozenset(1 << bit for bit in range(7))

    @classmethod
    def encode(cls, data: int) -> int:
        """Encode an unsigned 64-bit payload into a 72-bit SECDED codeword."""
        if not 0 <= data < (1 << cls.DATA_BITS):
            raise ValueError("SECDED payload must be an unsigned 64-bit integer")
        code = [0] * (cls.HAMMING_BITS + 1)
        data_index = 0
        for position in range(1, cls.HAMMING_BITS + 1):
            if position not in cls._PARITY_POSITIONS:
                code[position] = (data >> data_index) & 1
                data_index += 1

        for parity_position in cls._PARITY_POSITIONS:
            parity = 0
            for position in range(1, cls.HAMMING_BITS + 1):
                if position & parity_position and position != parity_position:
                    parity ^= code[position]
            code[parity_position] = parity

        overall_parity = 0
        for bit in code[1:]:
            overall_parity ^= bit
        codeword = 0
        for position in range(1, cls.HAMMING_BITS + 1):
            codeword |= code[position] << (position - 1)
        codeword |= overall_parity << (cls.CODE_BITS - 1)
        return codeword

    @classmethod
    def decode(cls, codeword: int) -> SECDEDDecodeResult:
        """Correct a single flipped code bit or flag an uncorrectable double."""
        if not 0 <= codeword < (1 << cls.CODE_BITS):
            raise ValueError("codeword must be a 72-bit unsigned integer")
        code = [0] * (cls.HAMMING_BITS + 1)
        for position in range(1, cls.HAMMING_BITS + 1):
            code[position] = (codeword >> (position - 1)) & 1

        syndrome = 0
        for parity_position in cls._PARITY_POSITIONS:
            parity = 0
            for position in range(1, cls.HAMMING_BITS + 1):
                if position & parity_position:
                    parity ^= code[position]
            if parity:
                syndrome |= parity_position

        overall_parity = (codeword >> (cls.CODE_BITS - 1)) & 1
        for bit in code[1:]:
            overall_parity ^= bit

        corrected = False
        double_error = False
        corrected_word = codeword
        if syndrome and overall_parity:
            if syndrome <= cls.HAMMING_BITS:
                corrected_word ^= 1 << (syndrome - 1)
                corrected = True
            else:
                double_error = True
        elif syndrome and not overall_parity:
            double_error = True
        elif not syndrome and overall_parity:
            corrected_word ^= 1 << (cls.CODE_BITS - 1)
            corrected = True

        corrected_data = 0
        data_index = 0
        for position in range(1, cls.HAMMING_BITS + 1):
            if position not in cls._PARITY_POSITIONS:
                bit = (corrected_word >> (position - 1)) & 1
                corrected_data |= bit << data_index
                data_index += 1
        return SECDEDDecodeResult(
            corrected_data,
            syndrome,
            corrected,
            double_error,
            corrected_word,
        )

    @classmethod
    def inject_bit_error(cls, codeword: int, bit_index: int) -> int:
        """Return a test codeword with one selected bit flipped (0 through 71)."""
        if not 0 <= bit_index < cls.CODE_BITS:
            raise ValueError("SECDED bit index must be in [0, 71]")
        return codeword ^ (1 << bit_index)


@dataclass
class DMADescriptor:
    """Scatter-gather transfer descriptor stored in the circular DMA ring."""

    source: Sequence[Any]
    destination_address: int
    length: Optional[int] = None
    offset: int = 0

    def __post_init__(self) -> None:
        if self.destination_address < 0:
            raise ValueError("destination_address cannot be negative")
        if self.length is None:
            self.length = len(self.source)
        if self.length < 0 or self.length > len(self.source):
            raise ValueError("descriptor length must fit within source data")


class ScatterGatherDMAController:
    """Circular descriptor DMA engine independent of CPU pipeline cycles."""

    def __init__(
        self,
        memory: Optional[MutableMapping[int, Any]] = None,
        ring_capacity: int = 16,
        words_per_cycle: int = 4,
        ecc_engine: Optional[SECDEDHamming72_64] = None,
    ) -> None:
        if ring_capacity < 1 or words_per_cycle < 1:
            raise ValueError("DMA ring capacity and bandwidth must be positive")
        self.memory: MutableMapping[int, Any] = memory if memory is not None else {}
        self.ring_capacity = ring_capacity
        self.words_per_cycle = words_per_cycle
        self.ecc_engine = ecc_engine or SECDEDHamming72_64()
        self._ring: List[Optional[DMADescriptor]] = [None] * ring_capacity
        self._read_index = 0
        self._write_index = 0
        self._queued = 0
        self.ecc_sidecar: Dict[int, int] = {}
        self.dma_cycles = 0
        self.transferred_words = 0
        self.overrun_count = 0
        self.cpu_cycles_stolen = 0

    @property
    def pending_descriptors(self) -> int:
        """Number of descriptors queued or partially active."""
        return self._queued

    def submit(self, descriptor: DMADescriptor) -> bool:
        """Append a scatter-gather descriptor; return false when ring is full."""
        if self._queued == self.ring_capacity:
            self.overrun_count += 1
            return False
        self._ring[self._write_index] = descriptor
        self._write_index = (self._write_index + 1) % self.ring_capacity
        self._queued += 1
        return True

    def transfer_cycle(self) -> int:
        """Transfer at most ``words_per_cycle`` words and return word count.

        Descriptor processing proceeds in ring order, while the modeled CPU
        cycle-steal counter remains zero because DMA owns its independent path.
        """
        self.dma_cycles += 1
        remaining_bandwidth = self.words_per_cycle
        transferred_this_cycle = 0
        while remaining_bandwidth and self._queued:
            descriptor = self._ring[self._read_index]
            if descriptor is None:
                raise RuntimeError("DMA ring metadata is inconsistent")
            if descriptor.offset >= int(descriptor.length):
                self._ring[self._read_index] = None
                self._read_index = (self._read_index + 1) % self.ring_capacity
                self._queued -= 1
                continue

            source_index = descriptor.offset
            destination = descriptor.destination_address + source_index
            value = descriptor.source[source_index]
            self.memory[destination] = value
            if isinstance(value, int) and 0 <= value < (1 << SECDEDHamming72_64.DATA_BITS):
                self.ecc_sidecar[destination] = self.ecc_engine.encode(value)
            descriptor.offset += 1
            remaining_bandwidth -= 1
            transferred_this_cycle += 1
            self.transferred_words += 1

            if descriptor.offset >= int(descriptor.length):
                self._ring[self._read_index] = None
                self._read_index = (self._read_index + 1) % self.ring_capacity
                self._queued -= 1

        return transferred_this_cycle

    def run_until_idle(self) -> int:
        """Service descriptors until the ring is empty and return DMA cycles."""
        initial_cycles = self.dma_cycles
        while self._queued:
            self.transfer_cycle()
        return self.dma_cycles - initial_cycles


__all__ = [
    "BusAdmission",
    "DMADescriptor",
    "DualTierPreemptiveBusArbiter",
    "SECDEDDecodeResult",
    "SECDEDHamming72_64",
    "ScatterGatherDMAController",
    "SplitTransaction",
    "SplitTransactionBus",
    "TokenBucket",
    "TransactionCompletion",
]