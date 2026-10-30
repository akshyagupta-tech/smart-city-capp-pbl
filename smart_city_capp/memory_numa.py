"""Directory-coherent NUMA memory, synchronization, and mesh routing models."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Deque, Dict, List, Optional, Tuple

from types_def import MOESIState


class DirectoryState(str, Enum):
    """Coherence state tracked by a home-node directory entry."""

    UNCACHED = "U"
    SHARED = "S"
    EXCLUSIVE = "E"
    MODIFIED = "M"
    OWNED = "O"


@dataclass
class DirectoryEntry:
    """Directory metadata for one memory line.

    ``sharers`` is a node-indexed bit vector for compatibility with the first
    simulator prototype. ``owner`` is set for exclusive or modified lines.
    """

    num_nodes: int
    state: DirectoryState = DirectoryState.UNCACHED
    sharers: List[bool] = field(init=False)
    owner: Optional[int] = None

    def __post_init__(self) -> None:
        self.sharers = [False] * self.num_nodes

    @property
    def active_sharers(self) -> List[int]:
        """Return the node IDs currently recorded as sharers or owners."""
        return [node_id for node_id, active in enumerate(self.sharers) if active]


@dataclass(frozen=True)
class SECDEDWord:
    """Data and parity bits for one extended-Hamming SECDED word."""

    parity_bits: int
    overall_parity: int
    data_bits: int
    signed: bool


@dataclass(frozen=True)
class ECCCheckResult:
    """Result of checking a received word against its stored ECC bits."""

    data: int
    syndrome: int
    corrected: bool
    double_error: bool


class SECDEDECC:
    """Extended-Hamming SECDED encoder/checker for integer data words."""

    @staticmethod
    def _parity_count(data_bits: int) -> int:
        parity_count = 0
        while (1 << parity_count) < data_bits + parity_count + 1:
            parity_count += 1
        return parity_count

    @staticmethod
    def _is_parity_position(position: int) -> bool:
        return position > 0 and position & (position - 1) == 0

    @classmethod
    def encode(cls, data: int, data_bits: int = 64) -> SECDEDWord:
        """Calculate Hamming parity plus an overall even-parity bit."""
        if data_bits < 1:
            raise ValueError("data_bits must be at least 1")
        minimum = -(1 << (data_bits - 1))
        maximum = (1 << data_bits) - 1
        if not minimum <= data <= maximum:
            raise ValueError("data does not fit configured ECC word width")

        signed = data < 0
        data_mask = (1 << data_bits) - 1
        data_pattern = data & data_mask
        parity_count = cls._parity_count(data_bits)
        code_length = data_bits + parity_count
        code = [0] * (code_length + 1)
        data_index = 0
        for position in range(1, code_length + 1):
            if not cls._is_parity_position(position):
                code[position] = (data_pattern >> data_index) & 1
                data_index += 1

        packed_parity = 0
        for parity_index in range(parity_count):
            parity_position = 1 << parity_index
            parity = 0
            for position in range(1, code_length + 1):
                if position & parity_position and position != parity_position:
                    parity ^= code[position]
            code[parity_position] = parity
            packed_parity |= parity << parity_index
        overall = 0
        for bit in code[1:]:
            overall ^= bit
        return SECDEDWord(packed_parity, overall, data_bits, signed)

    @classmethod
    def check(cls, word: SECDEDWord, received_data: int) -> ECCCheckResult:
        """Correct a single flipped bit or report an uncorrectable double error."""
        parity_count = cls._parity_count(word.data_bits)
        code_length = word.data_bits + parity_count
        data_mask = (1 << word.data_bits) - 1
        pattern = received_data & data_mask
        code = [0] * (code_length + 1)
        data_index = 0
        for position in range(1, code_length + 1):
            if cls._is_parity_position(position):
                parity_index = position.bit_length() - 1
                code[position] = (word.parity_bits >> parity_index) & 1
            else:
                code[position] = (pattern >> data_index) & 1
                data_index += 1

        syndrome = 0
        for parity_index in range(parity_count):
            parity_position = 1 << parity_index
            parity = 0
            for position in range(1, code_length + 1):
                if position & parity_position:
                    parity ^= code[position]
            syndrome |= parity << parity_index

        overall_mismatch = word.overall_parity
        for bit in code[1:]:
            overall_mismatch ^= bit

        corrected = False
        double_error = False
        if syndrome and overall_mismatch:
            if syndrome <= code_length:
                if not cls._is_parity_position(syndrome):
                    data_index = 0
                    for position in range(1, syndrome + 1):
                        if not cls._is_parity_position(position):
                            data_index += 1
                    pattern ^= 1 << (data_index - 1)
                corrected = True
            else:
                double_error = True
        elif syndrome and not overall_mismatch:
            double_error = True
        elif not syndrome and overall_mismatch:
            corrected = True

        if word.signed and pattern & (1 << (word.data_bits - 1)):
            corrected_data = pattern - (1 << word.data_bits)
        else:
            corrected_data = pattern
        return ECCCheckResult(corrected_data, syndrome, corrected, double_error)


class AdvancedMESICache:
    """Small MOESI cache with LRU line replacement."""

    def __init__(self, node_id: int, num_lines: int = 4) -> None:
        if num_lines < 1:
            raise ValueError("num_lines must be at least 1")
        self.node_id = node_id
        self.num_lines = num_lines
        self.lines: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.ecc_corrections = 0
        self.ecc_double_errors = 0

    def access(self, tag: int, is_write: bool = False) -> str:
        """Record a local access and return its hit or upgrade status."""
        line = self.lines.get(tag)
        if line is None or line["state"] == MOESIState.INVALID.value:
            self.misses += 1
            return "MISS"

        ecc_word = line.get("ecc")
        if ecc_word is not None:
            check = SECDEDECC.check(ecc_word, line["data"])
            if check.double_error:
                line["state"] = MOESIState.INVALID.value
                self.ecc_double_errors += 1
                self.misses += 1
                return "ECC_DOUBLE_ERROR"
            if check.corrected:
                line["data"] = check.data
                line["ecc"] = SECDEDECC.encode(check.data, ecc_word.data_bits)
                self.ecc_corrections += 1

        self.hits += 1
        self.lines.move_to_end(tag)
        if is_write and line["state"] in (MOESIState.SHARED.value, MOESIState.OWNED.value):
            return "WRITE_UPGRADE_REQ"
        return f"HIT_{line['state']}"

    def install(self, tag: int, data: Any, state: MOESIState) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Install or update a line; return an evicted tag and line if needed."""
        evicted = None
        if tag not in self.lines and len(self.lines) >= self.num_lines:
            evicted = self.lines.popitem(last=False)
        ecc = SECDEDECC.encode(data) if isinstance(data, int) else None
        self.lines[tag] = {"state": state.value, "data": data, "ecc": ecc}
        self.lines.move_to_end(tag)
        return evicted

    def inject_data_bit_error(self, tag: int, bit_index: int) -> None:
        """Flip one data bit in a valid integer cache line for fault testing."""
        line = self.lines[tag]
        ecc_word = line.get("ecc")
        if ecc_word is None or not 0 <= bit_index < ecc_word.data_bits:
            raise ValueError("line has no integer ECC word or bit index is out of range")
        pattern = line["data"] & ((1 << ecc_word.data_bits) - 1)
        pattern ^= 1 << bit_index
        if ecc_word.signed and pattern & (1 << (ecc_word.data_bits - 1)):
            line["data"] = pattern - (1 << ecc_word.data_bits)
        else:
            line["data"] = pattern


class NUMADirectorySystem:
    """Directory-based MOESI coherence over a configurable NUMA node array.

    At least four nodes are required to model the intended multi-node system.
    The home node for an address is selected with ``tag % num_nodes``.
    """

    def __init__(self, num_nodes: int = 4, cache_lines: int = 4) -> None:
        if num_nodes < 4:
            raise ValueError("NUMA system requires at least four nodes")
        self.num_nodes = num_nodes
        self.directory: Dict[int, DirectoryEntry] = {}
        self.caches = [AdvancedMESICache(node, cache_lines) for node in range(num_nodes)]
        self.memory: Dict[int, Any] = {}
        self.memory_ecc: Dict[int, SECDEDWord] = {}
        self.ecc_corrections = 0
        self.ecc_double_errors = 0
        self.interconnect_traffic = 0
        self.invalidation_count = 0
        self.memory_writebacks = 0
        self.transaction_count = 0
        self.line_versions: Dict[int, int] = {}

    def _validate_node(self, node_id: int) -> None:
        if not 0 <= node_id < self.num_nodes:
            raise ValueError(f"node_id must be in [0, {self.num_nodes - 1}]")

    def get_home_node(self, tag: int) -> int:
        """Return the deterministic home node for a memory-line tag."""
        return tag % self.num_nodes

    def _entry(self, tag: int) -> DirectoryEntry:
        if tag not in self.directory:
            self.directory[tag] = DirectoryEntry(self.num_nodes)
        return self.directory[tag]

    def _install_line(self, node_id: int, tag: int, data: Any, state: MOESIState) -> None:
        evicted = self.caches[node_id].install(tag, data, state)
        if evicted is not None:
            evicted_tag, evicted_line = evicted
            if evicted_line["state"] in (MOESIState.MODIFIED.value, MOESIState.OWNED.value):
                self.memory[evicted_tag] = evicted_line["data"]
                if isinstance(evicted_line["data"], int):
                    self.memory_ecc[evicted_tag] = SECDEDECC.encode(evicted_line["data"])
                self.memory_writebacks += 1
            self._remove_sharer(evicted_tag, node_id)

    def _remove_sharer(self, tag: int, node_id: int) -> None:
        entry = self.directory.get(tag)
        if entry is None:
            return
        entry.sharers[node_id] = False
        if entry.owner == node_id:
            entry.owner = None
        remaining = entry.active_sharers
        if not remaining:
            entry.state = DirectoryState.UNCACHED
            entry.owner = None
        elif len(remaining) == 1:
            survivor = remaining[0]
            entry.state = DirectoryState.EXCLUSIVE
            entry.owner = survivor
            line = self.caches[survivor].lines.get(tag)
            if line is not None:
                line["state"] = MOESIState.EXCLUSIVE.value
        else:
            entry.state = DirectoryState.SHARED
            entry.owner = None

    def _read_backing_memory(self, tag: int) -> Any:
        """Transfer and ECC-check one backing-memory word."""
        data = self.memory.get(tag, 0)
        ecc_word = self.memory_ecc.get(tag)
        if ecc_word is None or not isinstance(data, int):
            return data
        result = SECDEDECC.check(ecc_word, data)
        if result.double_error:
            self.ecc_double_errors += 1
            raise MemoryError(f"uncorrectable SECDED error on memory line 0x{tag:X}")
        if result.corrected:
            self.memory[tag] = result.data
            self.memory_ecc[tag] = SECDEDECC.encode(result.data, ecc_word.data_bits)
            self.ecc_corrections += 1
            print(f"[ECC] Corrected a single-bit memory error at line 0x{tag:X}")
        return self.memory[tag]

    def read_transaction(self, requester_id: int, tag: int) -> Any:
        """Read a line, fetching it or downgrading an existing owner as needed."""
        self._validate_node(requester_id)
        self.transaction_count += 1
        cache = self.caches[requester_id]
        status = cache.access(tag)
        if status == "ECC_DOUBLE_ERROR":
            self.ecc_double_errors += 1
            entry = self._entry(tag)
            if entry.owner == requester_id and entry.state in (
                DirectoryState.MODIFIED,
                DirectoryState.OWNED,
            ):
                raise MemoryError(f"uncorrectable ECC error in dirty owner line 0x{tag:X}")
            self._remove_sharer(tag, requester_id)
        if status.startswith("HIT"):
            value = cache.lines[tag]["data"]
            if cache.ecc_corrections:
                self.ecc_corrections = sum(item.ecc_corrections for item in self.caches)
            print(
                f"[Node {requester_id}] Cache HIT ({cache.lines[tag]['state']}) "
                f"for 0x{tag:X}: {value}"
            )
            return value

        entry = self._entry(tag)
        print(
            f"[Node {requester_id}] Cache MISS for 0x{tag:X} "
            f"| Home Node {self.get_home_node(tag)}"
        )
        data = self._read_backing_memory(tag)
        active = entry.active_sharers
        if entry.state in (
            DirectoryState.EXCLUSIVE,
            DirectoryState.MODIFIED,
            DirectoryState.OWNED,
        ) and entry.owner is not None and entry.owner != requester_id:
            owner = entry.owner
            owner_line = self.caches[owner].lines.get(tag)
            if owner_line is not None:
                data = owner_line["data"]
                self.interconnect_traffic += 1
                if entry.state == DirectoryState.MODIFIED:
                    owner_line["state"] = MOESIState.OWNED.value
                    entry.state = DirectoryState.OWNED
                    print(f" -> Forward dirty data from Node {owner}; owner retains Owned (O)")
                elif entry.state == DirectoryState.OWNED:
                    print(f" -> Forward from Owned Node {owner}; memory writeback avoided")
                else:
                    owner_line["state"] = MOESIState.SHARED.value
                    entry.state = DirectoryState.SHARED
                    entry.owner = None
                    self.memory[tag] = data
                    print(f" -> Forward from Node {owner}; clean owner downgraded to Shared (S)")
            else:
                entry.state = DirectoryState.SHARED if entry.active_sharers else DirectoryState.UNCACHED
                entry.owner = None

        entry.sharers[requester_id] = True
        if entry.state == DirectoryState.UNCACHED and not active:
            entry.state = DirectoryState.EXCLUSIVE
            entry.owner = requester_id
            line_state = MOESIState.EXCLUSIVE
        elif entry.state == DirectoryState.OWNED and entry.owner is not None:
            line_state = MOESIState.SHARED
        else:
            entry.state = DirectoryState.SHARED
            entry.owner = None
            line_state = MOESIState.SHARED
        self._install_line(requester_id, tag, data, line_state)
        print(
            f" -> Directory {entry.state.value}; Node {requester_id} now "
            f"{line_state.name.title()} ({line_state.value})"
        )
        return data

    def write_transaction(self, requester_id: int, tag: int, val: Any) -> Any:
        """Acquire write ownership, invalidate all other sharers, and store data."""
        self._validate_node(requester_id)
        self.transaction_count += 1
        cache = self.caches[requester_id]
        status = cache.access(tag, is_write=True)
        entry = self._entry(tag)
        print(
            f"\n[Node {requester_id}] WRITE 0x{tag:X} = {val} "
            f"| Home Node {self.get_home_node(tag)}"
        )

        for node_id, is_sharer in enumerate(entry.sharers):
            if not is_sharer or node_id == requester_id:
                continue
            line = self.caches[node_id].lines.get(tag)
            if line is not None:
                line["state"] = MOESIState.INVALID.value
                print(f" -> DIRECTORY INVALIDATE: Node {node_id} -> Invalid (I)")
            entry.sharers[node_id] = False
            self.interconnect_traffic += 1
            self.invalidation_count += 1

        entry.sharers = [False] * self.num_nodes
        entry.sharers[requester_id] = True
        entry.state = DirectoryState.MODIFIED
        entry.owner = requester_id
        self.line_versions[tag] = self.line_versions.get(tag, 0) + 1
        self._install_line(requester_id, tag, val, MOESIState.MODIFIED)
        if status == "MISS":
            print(" -> Write miss; ownership acquired in Modified (M)")
        else:
            print(" -> Write ownership acquired in Modified (M)")
        return val

    @property
    def total_cache_hits(self) -> int:
        """Return aggregate hits across all node caches."""
        return sum(cache.hits for cache in self.caches)

    @property
    def total_cache_misses(self) -> int:
        """Return aggregate misses across all node caches."""
        return sum(cache.misses for cache in self.caches)

    def seed_memory(self, tag: int, value: Any) -> None:
        """Initialize a backing-memory line before the workload begins."""
        self.memory[tag] = value
        if isinstance(value, int):
            self.memory_ecc[tag] = SECDEDECC.encode(value)

    def inject_memory_bit_error(self, tag: int, bit_index: int) -> None:
        """Flip one data bit in a backing-memory word for resilience tests."""
        if tag not in self.memory or tag not in self.memory_ecc:
            raise KeyError(f"memory line 0x{tag:X} has no integer ECC word")
        ecc_word = self.memory_ecc[tag]
        if not 0 <= bit_index < ecc_word.data_bits:
            raise ValueError("bit index is outside the encoded data word")
        mask = (1 << ecc_word.data_bits) - 1
        pattern = (self.memory[tag] & mask) ^ (1 << bit_index)
        if ecc_word.signed and pattern & (1 << (ecc_word.data_bits - 1)):
            self.memory[tag] = pattern - (1 << ecc_word.data_bits)
        else:
            self.memory[tag] = pattern

    def inject_cache_bit_error(self, node_id: int, tag: int, bit_index: int) -> None:
        """Flip one bit in a cached word for SECDED correction tests."""
        self._validate_node(node_id)
        self.caches[node_id].inject_data_bit_error(tag, bit_index)


class CircularDMA:
    """Circular DMA ring transferring queued telemetry into a memory sink."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("DMA capacity must be at least one entry")
        self.capacity = capacity
        self._ring: List[Optional[Any]] = [None] * capacity
        self.read_index = 0
        self.write_index = 0
        self.count = 0
        self.transferred_items = 0
        self.overrun_count = 0

    @property
    def available(self) -> int:
        """Number of occupied entries currently waiting for DMA service."""
        return self.count

    @property
    def free_slots(self) -> int:
        """Number of unused entries in the circular descriptor ring."""
        return self.capacity - self.count

    def submit(self, item: Any) -> bool:
        """Queue one item unless the ring is full or the item is ``None``."""
        if item is None:
            raise ValueError("None is reserved as the empty-slot marker")
        if self.count == self.capacity:
            self.overrun_count += 1
            return False
        self._ring[self.write_index] = item
        self.write_index = (self.write_index + 1) % self.capacity
        self.count += 1
        return True

    def transfer_to(self, destination: List[Any], max_items: Optional[int] = None) -> int:
        """Transfer up to ``max_items`` queued items into a memory sink.

        Read and write indices wrap independently, allowing producer and DMA
        consumer progress to overlap without shifting ring contents.
        """
        if max_items is not None and max_items < 0:
            raise ValueError("max_items cannot be negative")
        transfer_count = self.count if max_items is None else min(self.count, max_items)
        for _ in range(transfer_count):
            item = self._ring[self.read_index]
            if item is None:
                raise RuntimeError("DMA ring metadata is inconsistent")
            destination.append(item)
            self._ring[self.read_index] = None
            self.read_index = (self.read_index + 1) % self.capacity
            self.count -= 1
            self.transferred_items += 1
        return transfer_count


class CacheAccessStatus(str, Enum):
    """Non-blocking cache request outcomes."""

    HIT = "hit"
    MISS_ALLOCATED = "miss_allocated"
    MISS_COALESCED = "miss_coalesced"
    NO_MSHR = "no_mshr"


@dataclass(frozen=True)
class CacheAccessResult:
    """Outcome and timing metadata for a non-blocking cache request."""

    status: CacheAccessStatus
    tag: int
    ready_cycle: Optional[int] = None
    waiter_count: int = 0


@dataclass
class _OutstandingMiss:
    ready_cycle: int
    waiters: int = 1


class NonBlockingCache:
    """Non-blocking cache with finite MSHRs and same-line miss coalescing."""

    def __init__(
        self,
        memory: Optional[Dict[int, Any]] = None,
        cache_lines: int = 16,
        mshr_count: int = 4,
        miss_latency: int = 8,
    ) -> None:
        if mshr_count < 1 or miss_latency < 1:
            raise ValueError("MSHR count and miss latency must be positive")
        self.cache = AdvancedMESICache(node_id=0, num_lines=cache_lines)
        self.memory = memory if memory is not None else {}
        self.mshr_count = mshr_count
        self.miss_latency = miss_latency
        self.cycle = 0
        self.mshrs: Dict[int, _OutstandingMiss] = {}
        self.completed_misses = 0
        self.blocked_requests = 0
        self.coalesced_requests = 0

    def access(self, tag: int) -> CacheAccessResult:
        """Perform a non-blocking lookup or allocate/coalesce an MSHR."""
        line = self.cache.lines.get(tag)
        if line is not None and line["state"] != MOESIState.INVALID.value:
            self.cache.access(tag)
            return CacheAccessResult(CacheAccessStatus.HIT, tag, self.cycle)

        outstanding = self.mshrs.get(tag)
        if outstanding is not None:
            outstanding.waiters += 1
            self.coalesced_requests += 1
            return CacheAccessResult(
                CacheAccessStatus.MISS_COALESCED,
                tag,
                outstanding.ready_cycle,
                outstanding.waiters,
            )
        if len(self.mshrs) >= self.mshr_count:
            self.blocked_requests += 1
            return CacheAccessResult(CacheAccessStatus.NO_MSHR, tag)

        self.cache.access(tag)
        outstanding = _OutstandingMiss(self.cycle + self.miss_latency)
        self.mshrs[tag] = outstanding
        return CacheAccessResult(
            CacheAccessStatus.MISS_ALLOCATED,
            tag,
            outstanding.ready_cycle,
            outstanding.waiters,
        )

    def tick(self, cycles: int = 1) -> List[int]:
        """Advance time and complete any ready MSHRs, returning filled tags."""
        if cycles < 0:
            raise ValueError("cycles cannot be negative")
        completed: List[int] = []
        for _ in range(cycles):
            self.cycle += 1
            ready_tags = [
                tag for tag, miss in self.mshrs.items() if miss.ready_cycle <= self.cycle
            ]
            for tag in ready_tags:
                value = self.memory.get(tag, 0)
                self.cache.install(tag, value, MOESIState.EXCLUSIVE)
                del self.mshrs[tag]
                self.completed_misses += 1
                completed.append(tag)
        return completed


class HardwareBarrierUnit:
    """Reusable N-party barrier with generation-based release."""

    def __init__(self, participants: int) -> None:
        if participants < 1:
            raise ValueError("barrier requires at least one participant")
        self.participants = participants
        self.generation = 0
        self._arrived: set[int] = set()
        self.completed_generations = 0

    @property
    def waiting_count(self) -> int:
        """Number of unique participants waiting in the current generation."""
        return len(self._arrived)

    def arrive(self, participant_id: int) -> bool:
        """Register an arrival; return true only to the generation's last arrival.

        A duplicate arrival in the same generation raises ``ValueError`` so
        one core cannot accidentally satisfy more than one barrier slot.
        """
        if not 0 <= participant_id < self.participants:
            raise ValueError("participant_id is outside barrier membership")
        if participant_id in self._arrived:
            raise ValueError("participant has already arrived in this generation")
        self._arrived.add(participant_id)
        if len(self._arrived) < self.participants:
            return False
        self._arrived.clear()
        self.generation += 1
        self.completed_generations += 1
        return True


@dataclass(frozen=True)
class NVMJournalEntry:
    """One dirty cache line persisted during early brownout recovery."""

    tag: int
    data: int
    source_node: int
    ecc: SECDEDWord


class BrownoutRecoveryJournal:
    """Persist dirty MOESI owners into a simple NVM journal on brownout."""

    def __init__(self) -> None:
        self.nvm: Dict[int, int] = {}
        self.entries: List[NVMJournalEntry] = []
        self.trigger_count = 0
        self.recovery_count = 0

    def trigger_brownout(
        self,
        system: NUMADirectorySystem,
        supply_voltage: float,
        threshold_voltage: float = 0.8,
    ) -> int:
        """Persist each unique dirty M/O cache line when voltage is low."""
        if supply_voltage > threshold_voltage:
            return 0
        self.trigger_count += 1
        persisted = 0
        seen: set[int] = set()
        for cache in system.caches:
            for tag, line in cache.lines.items():
                if tag in seen or line["state"] not in (
                    MOESIState.MODIFIED.value,
                    MOESIState.OWNED.value,
                ):
                    continue
                if not isinstance(line["data"], int):
                    raise TypeError("NVM journal supports integer cache-line payloads")
                check = SECDEDECC.check(line["ecc"], line["data"])
                if check.double_error:
                    raise MemoryError(f"cannot journal corrupt cache line 0x{tag:X}")
                data = check.data
                ecc = SECDEDECC.encode(data, line["ecc"].data_bits)
                self.nvm[tag] = data
                self.entries.append(NVMJournalEntry(tag, data, cache.node_id, ecc))
                seen.add(tag)
                persisted += 1
                print(
                    f"[Brownout Journal] Persisted dirty line 0x{tag:X} "
                    f"from Node {cache.node_id} to NVM"
                )
        print(f"[Brownout Journal] Voltage {supply_voltage:.2f} V; {persisted} lines protected")
        return persisted

    def recover(self, system: NUMADirectorySystem) -> int:
        """Restore journaled words and invalidate volatile copies after restart."""
        restored = 0
        for tag, data in self.nvm.items():
            system.memory[tag] = data
            system.memory_ecc[tag] = SECDEDECC.encode(data)
            entry = system.directory.get(tag)
            if entry is not None:
                entry.state = DirectoryState.UNCACHED
                entry.owner = None
                entry.sharers = [False] * system.num_nodes
            for cache in system.caches:
                line = cache.lines.get(tag)
                if line is not None:
                    line["state"] = MOESIState.INVALID.value
            restored += 1
        if restored:
            self.recovery_count += 1
        print(f"[Brownout Journal] Recovered {restored} lines from NVM")
        return restored


class LoadLinkedStoreConditional:
    """Atomic LL/SC operations guarded by per-line write generations."""

    def __init__(self, system: NUMADirectorySystem) -> None:
        self.system = system
        self.reservations: Dict[int, Tuple[int, int]] = {}

    def load_linked(self, node_id: int, tag: int) -> Any:
        """Read a line and reserve it for a conditional store by this node."""
        value = self.system.read_transaction(node_id, tag)
        self.reservations[node_id] = (tag, self.system.line_versions.get(tag, 0))
        print(f"[LL/SC] Node {node_id} reserved line 0x{tag:X}")
        return value

    def store_conditional(self, node_id: int, tag: int, value: Any) -> bool:
        """Store only if this node's reservation is still current."""
        reservation = self.reservations.pop(node_id, None)
        success = reservation == (tag, self.system.line_versions.get(tag, 0))
        if not success:
            print(f"[LL/SC] Node {node_id} SC FAILED for 0x{tag:X}")
            return False
        self.system.write_transaction(node_id, tag, value)
        print(f"[LL/SC] Node {node_id} SC SUCCEEDED for 0x{tag:X}")
        return True


AtomicLLSC = LoadLinkedStoreConditional
AdvancedMOESICache = AdvancedMESICache


class HardwareLockSync:
    """Owner-checked hardware Test-and-Set lock model."""

    def __init__(self) -> None:
        self.owner: Optional[int] = None

    @property
    def locked(self) -> bool:
        """Whether a node currently owns the lock."""
        return self.owner is not None

    def acquire(self, node_id: int) -> bool:
        """Atomically acquire an unlocked lock; otherwise report contention."""
        if self.owner is None:
            self.owner = node_id
            print(f"[Sync Unit] Lock ACQUIRED by Node {node_id} via Test-and-Set")
            return True
        print(f"[Sync Unit] Lock BUSY (owner Node {self.owner}); Node {node_id} stalls")
        return False

    def release(self, node_id: int) -> bool:
        """Release only when called by the owning node."""
        if self.owner != node_id:
            print(f"[Sync Unit] Release rejected for non-owner Node {node_id}")
            return False
        self.owner = None
        print(f"[Sync Unit] Lock RELEASED by Node {node_id}")
        return True


@dataclass(frozen=True)
class MeshNode:
    """A named router location in a two-dimensional mesh."""

    zone_name: str
    x: int
    y: int


class VirtualChannel(IntEnum):
    """Independent mesh queues for routine and emergency messages."""

    ROUTINE = 0
    EMERGENCY = 1


@dataclass(frozen=True)
class RoutedMessage:
    """Payload and XY route assigned to one virtual channel."""

    payload: Any
    source: Tuple[int, int]
    destination: Tuple[int, int]
    path: Tuple[Tuple[int, int], ...]
    virtual_channel: VirtualChannel


class MeshInterconnect:
    """XY mesh with separate per-VC queues to avoid head-of-line blocking."""

    def __init__(
        self,
        failed_nodes: Optional[List[Tuple[int, int]]] = None,
        failed_links: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
        detour_margin: int = 4,
        mesh_width: int = 8,
        mesh_height: int = 8,
    ) -> None:
        if detour_margin < 1:
            raise ValueError("detour_margin must be at least 1")
        if mesh_width < 1 or mesh_height < 1:
            raise ValueError("mesh dimensions must be positive")
        self._channels: Dict[VirtualChannel, Deque[RoutedMessage]] = {
            VirtualChannel.ROUTINE: deque(),
            VirtualChannel.EMERGENCY: deque(),
        }
        self.failed_nodes = set(failed_nodes or ())
        self.failed_links = {
            self._link_key(first, second) for first, second in (failed_links or ())
        }
        self.detour_margin = detour_margin
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height

    @staticmethod
    def _link_key(
        first: Tuple[int, int], second: Tuple[int, int]
    ) -> frozenset[Tuple[int, int]]:
        """Normalize an undirected physical link for failure lookup."""
        return frozenset((first, second))

    def fail_link(self, first: Tuple[int, int], second: Tuple[int, int]) -> None:
        """Mark an undirected mesh link as severed."""
        self.failed_links.add(self._link_key(first, second))

    def fail_node(self, coordinate: Tuple[int, int]) -> None:
        """Mark a router coordinate unavailable."""
        self.failed_nodes.add(coordinate)

    def _in_bounds(self, coordinate: Tuple[int, int]) -> bool:
        """Return whether a coordinate belongs to this finite mesh."""
        x, y = coordinate
        return 0 <= x < self.mesh_width and 0 <= y < self.mesh_height

    def route_hot_potato(self, src_node: Any, dst_node: Any) -> List[Tuple[int, int]]:
        """Deflect around failed routers/links with bounded XY-prioritized BFS.

        Candidate hops favor the X-then-Y minimal route. If that link or node
        is unavailable, the packet is deflected to the next healthy neighbor.
        The finite search region is expanded by ``detour_margin`` around the
        source/destination bounding box.
        """
        source = (src_node.x, src_node.y)
        destination = (dst_node.x, dst_node.y)
        if not self._in_bounds(source) or not self._in_bounds(destination):
            raise RuntimeError("mesh route endpoint is outside configured dimensions")
        if source in self.failed_nodes or destination in self.failed_nodes:
            raise RuntimeError("mesh route endpoint is unavailable")
        if source == destination:
            return [source]

        min_x = max(0, min(source[0], destination[0]) - self.detour_margin)
        max_x = min(self.mesh_width - 1, max(source[0], destination[0]) + self.detour_margin)
        min_y = max(0, min(source[1], destination[1]) - self.detour_margin)
        max_y = min(self.mesh_height - 1, max(source[1], destination[1]) + self.detour_margin)

        x_step = 1 if destination[0] > source[0] else -1 if destination[0] < source[0] else 0
        y_step = 1 if destination[1] > source[1] else -1 if destination[1] < source[1] else 0
        preferred = [(x_step, 0), (0, y_step)]
        fallback = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        directions = list(dict.fromkeys(delta for delta in preferred + fallback if delta != (0, 0)))

        frontier: Deque[Tuple[int, int]] = deque([source])
        previous: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {source: None}
        while frontier:
            current = frontier.popleft()
            if current == destination:
                break
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not min_x <= neighbor[0] <= max_x or not min_y <= neighbor[1] <= max_y:
                    continue
                if neighbor in self.failed_nodes or neighbor in previous:
                    continue
                if self._link_key(current, neighbor) in self.failed_links:
                    continue
                previous[neighbor] = current
                frontier.append(neighbor)

        if destination not in previous:
            raise RuntimeError("no healthy mesh route found within detour bounds")
        path: List[Tuple[int, int]] = []
        cursor: Optional[Tuple[int, int]] = destination
        while cursor is not None:
            path.append(cursor)
            cursor = previous[cursor]
        path.reverse()
        baseline_hops = abs(destination[0] - source[0]) + abs(destination[1] - source[1])
        deflections = len(path) - 1 - baseline_hops
        src_name = getattr(src_node, "zone_name", str(source))
        dst_name = getattr(dst_node, "zone_name", str(destination))
        print(
            f"[Hot-Potato] {src_name} -> {dst_name}: {path}; "
            f"{deflections} deflection hops around failed topology"
        )
        return path

    @staticmethod
    def route_xy(
        src_node: Any,
        dst_node: Any,
        is_emergency: bool = False,
    ) -> List[Tuple[int, int]]:
        """Return a minimal X-then-Y route and identify its VC in the trace."""
        current_x, current_y = src_node.x, src_node.y
        path = [(current_x, current_y)]
        while current_x != dst_node.x:
            current_x += 1 if dst_node.x > current_x else -1
            path.append((current_x, current_y))
        while current_y != dst_node.y:
            current_y += 1 if dst_node.y > current_y else -1
            path.append((current_x, current_y))
        src_name = getattr(src_node, "zone_name", str(path[0]))
        dst_name = getattr(dst_node, "zone_name", str(path[-1]))
        channel = VirtualChannel.EMERGENCY if is_emergency else VirtualChannel.ROUTINE
        print(
            f"Routing 2D-Mesh {src_name} -> {dst_name} via VC{channel.value}: {path} "
            f"| {len(path) - 1} router hops"
        )
        return path

    def enqueue(
        self,
        payload: Any,
        src_node: Any,
        dst_node: Any,
        is_emergency: bool = False,
    ) -> RoutedMessage:
        """Route and enqueue a packet without sharing another class's VC buffer."""
        channel = VirtualChannel.EMERGENCY if is_emergency else VirtualChannel.ROUTINE
        if self.failed_links or self.failed_nodes:
            path = self.route_hot_potato(src_node, dst_node)
            print(f"Routed via VC{channel.value}")
        else:
            path = self.route_xy(src_node, dst_node, is_emergency=is_emergency)
        message = RoutedMessage(
            payload=payload,
            source=(src_node.x, src_node.y),
            destination=(dst_node.x, dst_node.y),
            path=tuple(path),
            virtual_channel=channel,
        )
        self._channels[channel].append(message)
        return message

    def dequeue(self, channel: VirtualChannel = VirtualChannel.ROUTINE) -> Optional[RoutedMessage]:
        """Remove a message from the selected virtual channel only."""
        selected = VirtualChannel(channel)
        return self._channels[selected].popleft() if self._channels[selected] else None

    def pending(self, channel: Optional[VirtualChannel] = None) -> int:
        """Return queued messages for one VC or the total across all VCs."""
        if channel is not None:
            return len(self._channels[VirtualChannel(channel)])
        return sum(len(queue) for queue in self._channels.values())


__all__ = [
    "AdvancedMESICache",
    "AdvancedMOESICache",
    "AtomicLLSC",
    "CacheAccessResult",
    "CacheAccessStatus",
    "CircularDMA",
    "BrownoutRecoveryJournal",
    "DirectoryEntry",
    "DirectoryState",
    "ECCCheckResult",
    "HardwareLockSync",
    "HardwareBarrierUnit",
    "LoadLinkedStoreConditional",
    "MeshInterconnect",
    "MeshNode",
    "NVMJournalEntry",
    "NUMADirectorySystem",
    "NonBlockingCache",
    "RoutedMessage",
    "SECDEDECC",
    "SECDEDWord",
    "VirtualChannel",
]