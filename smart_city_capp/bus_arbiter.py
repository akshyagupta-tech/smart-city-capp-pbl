"""Dual-tier preemptive arbitration for emergency and routine bus traffic."""

from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from types_def import Packet


class DualTierBusArbiter:
    """Arbitrate a shared bus with emergency priority and fair sensor polling.

    Emergency packets are serviced FIFO and take priority over routine traffic.
    To prevent either class from starving, at most ``max_emergency_burst``
    consecutive emergencies are granted while routine requests are pending.
    Routine packets are queued per device and selected round-robin by device ID.

    Args:
        max_emergency_burst: Maximum consecutive emergency grants while routine
            traffic is waiting. Must be at least one.
    """

    def __init__(
        self,
        max_emergency_burst: int = 4,
        buffer_capacity: int = 128,
        drop_threshold: float = 0.85,
        token_bucket_capacity: float = 8.0,
        token_refill_rate: float = 1.0,
    ) -> None:
        if max_emergency_burst < 1:
            raise ValueError("max_emergency_burst must be at least 1")
        if buffer_capacity < 1:
            raise ValueError("buffer_capacity must be at least 1")
        if not 0.0 < drop_threshold <= 1.0:
            raise ValueError("drop_threshold must be in (0, 1]")
        if token_bucket_capacity < 1.0 or token_refill_rate < 0.0:
            raise ValueError("token bucket capacity must be positive and refill non-negative")
        self.emergency_queue: Deque[Packet] = deque()
        self.routine_queues: Dict[int, Deque[Packet]] = defaultdict(deque)
        self._routine_rotation: Deque[int] = deque()
        self.max_emergency_burst = max_emergency_burst
        self.buffer_capacity = buffer_capacity
        self.drop_threshold = drop_threshold
        self.token_bucket_capacity = token_bucket_capacity
        self.token_refill_rate = token_refill_rate
        self._device_tokens: Dict[int, float] = {}
        self._token_update_cycle: Dict[int, int] = {}
        self._consecutive_emergencies = 0
        self.bus_master: Optional[Packet] = None
        self.cycle_count = 0
        self.throttled_count = 0
        self.queue_drop_count = 0
        self.dropped_by_device: Dict[int, int] = defaultdict(int)

    @property
    def buffer_occupancy(self) -> int:
        """Return the number of packets currently waiting for the bus."""
        return len(self.emergency_queue) + sum(map(len, self.routine_queues.values()))

    @property
    def buffer_utilization(self) -> float:
        """Return current shared-buffer utilization in the inclusive range [0, 1]."""
        return self.buffer_occupancy / self.buffer_capacity

    def _consume_device_token(self, dev_id: int) -> bool:
        tokens = self._device_tokens.get(dev_id, self.token_bucket_capacity)
        last_cycle = self._token_update_cycle.get(dev_id, self.cycle_count)
        elapsed = max(0, self.cycle_count - last_cycle)
        tokens = min(self.token_bucket_capacity, tokens + elapsed * self.token_refill_rate)
        self._token_update_cycle[dev_id] = self.cycle_count
        if tokens < 1.0:
            self._device_tokens[dev_id] = tokens
            return False
        self._device_tokens[dev_id] = tokens - 1.0
        return True

    def _drop_oldest_routine(self) -> Optional[Packet]:
        """Drop the oldest packet of the lowest-priority tier, if one exists."""
        while self._routine_rotation:
            dev_id = self._routine_rotation[0]
            queue = self.routine_queues[dev_id]
            if not queue:
                self._routine_rotation.popleft()
                self.routine_queues.pop(dev_id, None)
                continue
            packet = queue.popleft()
            self.queue_drop_count += 1
            self.dropped_by_device[packet.dev_id] += 1
            if not queue:
                self._routine_rotation.popleft()
                del self.routine_queues[dev_id]
            return packet
        return None

    @property
    def routine_queue(self) -> Deque[Packet]:
        """Return pending routine packets in current device-rotation order.

        This compatibility view keeps existing callers that inspect
        ``routine_queue`` working while internal storage remains per-device.
        """
        return deque(
            packet
            for dev_id in self._routine_rotation
            for packet in self.routine_queues[dev_id]
        )

    def request_bus(self, packet: Packet) -> bool:
        """Admit a packet unless rate or buffer protection rejects it.

        Every device has an independent token bucket. Above the configured
        high-water mark, new routine traffic is dropped first; emergency
        arrivals may evict a routine packet to preserve critical capacity.
        Returns ``True`` only when the packet is queued.
        """
        if not self._consume_device_token(packet.dev_id):
            self.throttled_count += 1
            self.dropped_by_device[packet.dev_id] += 1
            print(f"[Bus Protection] Throttled device {packet.dev_id} ({packet.name}): token bucket empty")
            return False
        projected_utilization = (self.buffer_occupancy + 1) / self.buffer_capacity
        above_high_watermark = projected_utilization > self.drop_threshold
        if packet.is_emergency:
            if above_high_watermark:
                self._drop_oldest_routine()
            if self.buffer_occupancy >= self.buffer_capacity:
                self.queue_drop_count += 1
                self.dropped_by_device[packet.dev_id] += 1
                print(f"[Bus Protection] Emergency buffer full; rejected {packet.name}")
                return False
            self.emergency_queue.append(packet)
            print(
                f"[Cycle {self.cycle_count}] !! EMERGENCY INTERRUPT ASSERTED "
                f"by {packet.name} (ID: {packet.dev_id})"
            )
            return True

        if above_high_watermark or self.buffer_occupancy >= self.buffer_capacity:
            self.queue_drop_count += 1
            self.dropped_by_device[packet.dev_id] += 1
            print(
                f"[Bus Protection] Queue pressure {projected_utilization:.0%}; "
                f"dropped routine packet from {packet.name}"
            )
            return False

        device_queue = self.routine_queues[packet.dev_id]
        if not device_queue:
            self._routine_rotation.append(packet.dev_id)
        device_queue.append(packet)
        return True

    def _grant_routine(self) -> Packet:
        """Grant one packet from the next active routine device."""
        dev_id = self._routine_rotation.popleft()
        device_queue = self.routine_queues[dev_id]
        packet = device_queue.popleft()
        if device_queue:
            self._routine_rotation.append(dev_id)
        else:
            del self.routine_queues[dev_id]
        self._consecutive_emergencies = 0
        return packet

    def arbitrate_cycle(self) -> Optional[Packet]:
        """Advance one bus cycle and return the granted packet, if any."""
        self.cycle_count += 1
        routine_waiting = bool(self._routine_rotation)

        if self.emergency_queue and not (
            routine_waiting
            and self._consecutive_emergencies >= self.max_emergency_burst
        ):
            self.bus_master = self.emergency_queue.popleft()
            self._consecutive_emergencies += 1
            print(
                f"[Cycle {self.cycle_count}] Bus Granted -> EMERGENCY CHANNEL: "
                f"{self.bus_master.name} | Data: {self.bus_master.data}"
            )
        elif routine_waiting:
            self.bus_master = self._grant_routine()
            print(
                f"[Cycle {self.cycle_count}] Bus Granted -> Routine Round-Robin: "
                f"{self.bus_master.name} | Data: {self.bus_master.data}"
            )
        elif self.emergency_queue:
            self.bus_master = self.emergency_queue.popleft()
            self._consecutive_emergencies += 1
            print(
                f"[Cycle {self.cycle_count}] Bus Granted -> EMERGENCY CHANNEL: "
                f"{self.bus_master.name} | Data: {self.bus_master.data}"
            )
        else:
            self.bus_master = None
            self._consecutive_emergencies = 0
            print(f"[Cycle {self.cycle_count}] Bus Idle (No active requests)")

        return self.bus_master
