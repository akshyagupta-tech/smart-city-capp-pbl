"""Deterministic integrated workload for the Smart City architecture demo."""

from bus_arbiter import DualTierBusArbiter
from bus_arbiter import Packet
from alu_unit import DeltaXORCompressionUnit
from alu_unit import FusedMultiplyAddPipeline
from alu_unit import IEEE754FloatConverter
from alu_unit import NonRestoringDivider
from alu_unit import Q16_16FixedPointDivider
from alu_unit import Radix4BoothMultiplier
from alu_unit import SaturatingArithmeticUnit
from cache_numa import BrownoutRecoveryJournal
from cache_numa import CacheAccessStatus
from cache_numa import CircularDMA
from cache_numa import HardwareLockSync
from cache_numa import HardwareBarrierUnit
from cache_numa import LoadLinkedStoreConditional
from cache_numa import MeshInterconnect
from cache_numa import MeshNode
from cache_numa import NUMADirectorySystem
from cache_numa import NonBlockingCache
from cache_numa import VirtualChannel
from pipeline_core import RISCPipeline5Stage
from types_def import Instruction
from types_def import SystemMetrics
from typing import Tuple
from contextlib import redirect_stdout
from io import StringIO


def run_telemetry_workload(arbiter: DualTierBusArbiter) -> Tuple[int, int]:
    """Queue and service one repeatable rush-hour telemetry batch."""
    samples = [
        (1, "WATER_GRID_PUMP_2", "pressure_kpa=412"),
        (2, "TRAFFIC_CAM_JUNCTION_4", "vehicle_count=83"),
        (3, "AQI_SECTOR_9", "pm25_ug_m3=31"),
        (4, "POWER_SUBSTATION_3", "load_kw=728"),
        (1, "WATER_GRID_PUMP_2", "flow_l_min=940"),
        (2, "TRAFFIC_CAM_JUNCTION_4", "queue_seconds=54"),
        (3, "AQI_SECTOR_9", "co2_ppm=612"),
        (4, "POWER_SUBSTATION_3", "voltage_v=231"),
    ]
    for dev_id, name, data in samples:
        arbiter.request_bus(Packet(dev_id, name, False, data))
    arbiter.request_bus(
        Packet(90, "AMBULANCE_GREEN_CORRIDOR", True, "junction=4;priority=critical")
    )
    arbiter.request_bus(
        Packet(91, "FIRE_GRID_ISOLATION", True, "substation=3;action=isolate")
    )

    print("\n[MODULE 1] Dual-tier telemetry bus")
    serviced = 0
    emergency_grants = 0
    while arbiter.emergency_queue or arbiter.routine_queue:
        packet = arbiter.arbitrate_cycle()
        serviced += 1
        if packet is not None and packet.is_emergency:
            emergency_grants += 1
    print(f"Serviced {serviced} packets; routine sources use round-robin polling.")
    return serviced, emergency_grants


def run_alu_demo() -> None:
    """Exercise the arithmetic blocks on representative sensor values."""
    print("\n[MODULE 2] Arithmetic datapath")
    multiplier = Radix4BoothMultiplier(bit_width=8)
    product = multiplier.multiply(14, -5)
    quotient, remainder = NonRestoringDivider.divide(-100, 8, bits=8)
    encoded_float = IEEE754FloatConverter.to_hex(23.75)
    print(f"ALU check: 14 * -5 = {product}; divide -100 / 8 = {quotient} R {remainder}")
    print(f"IEEE 754 binary32 encoding for 23.75: 0x{encoded_float}")
    density = Q16_16FixedPointDivider.divide(83, 1000)
    compressor = DeltaXORCompressionUnit()
    first = compressor.compress(550, stream_id=2)
    delta = compressor.compress(552, stream_id=2)
    print(f"Q16.16 traffic-density normalization: 83/1000 = {density:.6f}")
    print(
        f"Delta/XOR telemetry sizing: {first.compressed_length_bytes} B initial, "
        f"{delta.compressed_length_bytes} B delta"
    )


def run_phase2_demo() -> dict[str, int]:
    """Exercise Phase 2 DMA, FMA, saturation, MSHR, and barrier features."""
    print("\nPHASE 2: EDGE-PROCESSING AND PARALLEL MEMORY FEATURES")

    dma = CircularDMA(capacity=4)
    for sample in range(5):
        dma.submit(Packet(sample, f"DMA_SENSOR_{sample}", False, f"sample={sample}"))
    dma_memory: list[Packet] = []
    dma.transfer_to(dma_memory, max_items=2)
    dma.transfer_to(dma_memory)
    if len(dma_memory) != 4 or dma.overrun_count != 1:
        raise RuntimeError("circular DMA ring failed wrap or overrun handling")
    print(
        f"Circular DMA: {dma.transferred_items} packets transferred; "
        f"{dma.overrun_count} full-ring overrun rejected."
    )

    fma = FusedMultiplyAddPipeline()
    fma_inputs = [
        (1e16, 1.0000000000000002, -1e16),
        (72.0, 1.25, 4.0),
        (83.0, 0.5, 1.0),
    ]
    fma_results = []
    for operands in fma_inputs:
        completed = fma.step(*operands)
        if completed is not None:
            fma_results.append(completed)
    fma_results.extend(fma.drain())
    if len(fma_results) != 3 or fma_results[0].value != 2.220446049250313:
        raise RuntimeError("pipelined fused multiply-add produced an invalid result")
    print(
        f"FMA pipeline: {fma.retired} operations retired; fused cancellation result "
        f"{fma_results[0].value:.15g}."
    )

    saturating = SaturatingArithmeticUnit(bit_width=16)
    sensor_accumulator = saturating.add(32760, 100)
    if sensor_accumulator.value != 32767 or not sensor_accumulator.saturated:
        raise RuntimeError("saturating accumulator failed to clamp positive overflow")
    print(
        f"Saturating ALU: 32760 + 100 -> {sensor_accumulator.value} "
        f"(overflow={sensor_accumulator.saturated})."
    )

    memory = {0x100: 412, 0x104: 83, 0x108: 31}
    cache = NonBlockingCache(memory, mshr_count=2, miss_latency=3)
    allocated_a = cache.access(0x100)
    coalesced_a = cache.access(0x100)
    allocated_b = cache.access(0x104)
    blocked_c = cache.access(0x108)
    if (
        allocated_a.status != CacheAccessStatus.MISS_ALLOCATED
        or coalesced_a.status != CacheAccessStatus.MISS_COALESCED
        or allocated_b.status != CacheAccessStatus.MISS_ALLOCATED
        or blocked_c.status != CacheAccessStatus.NO_MSHR
    ):
        raise RuntimeError("non-blocking cache MSHR behavior was inconsistent")
    completed_misses = cache.tick(3)
    retried_c = cache.access(0x108)
    cache.tick(3)
    if len(completed_misses) != 2 or retried_c.status != CacheAccessStatus.MISS_ALLOCATED:
        raise RuntimeError("non-blocking cache failed to release an MSHR after completion")
    print(
        f"Non-blocking cache: {cache.completed_misses} fills, "
        f"{cache.coalesced_requests} coalesced request, "
        f"{cache.blocked_requests} MSHR backpressure event."
    )

    barrier = HardwareBarrierUnit(participants=4)
    releases = [barrier.arrive(core_id) for core_id in range(4)]
    if releases != [False, False, False, True] or barrier.generation != 1:
        raise RuntimeError("four-core hardware barrier failed to release")
    print(f"Barrier unit: four cores released at generation {barrier.generation}.")

    return {
        "dma_transferred": dma.transferred_items,
        "fma_retired": fma.retired,
        "saturations": saturating.saturation_count,
        "coalesced_misses": cache.coalesced_requests,
        "barrier_generations": barrier.completed_generations,
    }


def run_pipeline_demo() -> RISCPipeline5Stage:
    """Run a load-use telemetry calculation and drain the five-stage pipeline."""
    print("\n[MODULE 3] Five-stage processor pipeline")
    pipeline = RISCPipeline5Stage()
    pipeline.registers["R0"] = 0x20
    pipeline.registers["R3"] = 8
    pipeline.registers["R5"] = 1
    pipeline.memory[0x20] = 72
    loop_branch = Instruction(
        0x10C,
        "BNE",
        dest=None,
        src1=5,
        src2=0,
        is_branch=True,
        branch_target=0x108,
    )
    program = [
        Instruction(0x100, "LOAD", dest=1, src1=0),
        Instruction(0x104, "ADD", dest=2, src1=1, src2=3),
        Instruction(0x108, "MUL", dest=4, src1=2, src2=3),
    ] + [loop_branch] * 5
    for instruction in program + [None] * 4:
        pipeline.clock_step(instruction)
    print(
        f"Computed telemetry result: R2={pipeline.registers['R2']} "
        f"and scaled value R4={pipeline.registers['R4']}"
    )
    print(
        "Loop branch penalty cycles avoided: "
        f"{pipeline.btb.branch_penalty_cycles_avoided}"
    )
    hot_state = pipeline.monitor_temperature(88.0)
    print(
        f"Thermal control at {hot_state.temperature_c:.1f} C: "
        f"{hot_state.frequency_ghz:.2f} GHz, power scale {hot_state.power_scale:.2f}"
    )

    watchdog_probe = RISCPipeline5Stage()
    watchdog_probe.IF_ID = Instruction(0x20, "ADD", 6, 1, 2)
    for _ in range(33):
        watchdog_probe.clock_step(stalled=True)
    if watchdog_probe.micro_rollback_count != 1:
        raise RuntimeError("pipeline watchdog failed to trigger micro-rollback")
    print(
        f"Pipeline micro-rollback completed for PC 0x{watchdog_probe.last_rollback_pc:X} "
        f"after {watchdog_probe.stall_cycles} held cycles."
    )
    return pipeline


def run_memory_demo() -> Tuple[NUMADirectorySystem, int]:
    """Exercise MOESI, atomics, synchronization, and virtual-channel routing."""
    print("\n[MODULES 4-5] MOESI NUMA, atomic synchronization, and mesh network")
    numa = NUMADirectorySystem(num_nodes=4)
    address = 0x3F2A
    numa.seed_memory(address, 100)
    numa.read_transaction(0, address)
    numa.write_transaction(0, address, 150)
    numa.read_transaction(1, address)
    print(
        "Backing memory remains at "
        f"{numa.memory[address]} while Node 0 owns dirty data in state O."
    )
    numa.read_transaction(2, address)
    numa.write_transaction(3, address, 999)
    numa.read_transaction(3, address)

    atomic = LoadLinkedStoreConditional(numa)
    counter_address = 0x4100
    numa.seed_memory(counter_address, 0)
    counter = atomic.load_linked(0, counter_address)
    atomic.store_conditional(0, counter_address, counter + 1)

    contested_address = 0x4104
    numa.seed_memory(contested_address, 1)
    atomic.load_linked(1, contested_address)
    numa.write_transaction(2, contested_address, 9)
    atomic.store_conditional(1, contested_address, 2)

    lock = HardwareLockSync()
    lock.acquire(2)
    lock.acquire(3)
    lock.release(2)
    lock.acquire(3)
    lock.release(3)

    source = MeshNode("North Traffic Zone", 0, 0)
    destination = MeshNode("East Power Substation", 2, 3)
    mesh = MeshInterconnect()
    mesh.enqueue("routine-meter-sample", source, destination)
    emergency = mesh.enqueue(
        "fire-isolation-alert", source, destination, is_emergency=True
    )
    delivered_emergency = mesh.dequeue(VirtualChannel.EMERGENCY)
    mesh.dequeue(VirtualChannel.ROUTINE)
    if delivered_emergency is None or delivered_emergency.payload != "fire-isolation-alert":
        raise RuntimeError("emergency virtual channel did not preserve packet ordering")
    print("VC1 emergency packet delivered independently of the queued VC0 sample.")
    return numa, len(emergency.path) - 1


def print_architectural_scorecard(
    pipeline: RISCPipeline5Stage,
    numa: NUMADirectorySystem,
    mesh_hops: int,
) -> None:
    """Print CPI, IPC, AMAT, dynamic power savings, and Amdahl speedups."""
    cache_accesses = numa.total_cache_hits + numa.total_cache_misses
    miss_rate = numa.total_cache_misses / cache_accesses if cache_accesses else 0.0
    hit_time_cycles = 1.0
    miss_penalty_cycles = 35.0
    amat = hit_time_cycles + miss_rate * miss_penalty_cycles
    metrics = SystemMetrics(
        cycles=pipeline.cycle_count,
        instructions_retired=pipeline.instructions_retired,
        cache_hits=numa.total_cache_hits,
        cache_misses=numa.total_cache_misses,
    )

    capacitance_f = 12e-9
    voltage_v = 0.9
    frequency_hz = 1e9
    baseline_alpha = 0.35
    optimized_alpha = 0.22
    baseline_power_w = baseline_alpha * capacitance_f * voltage_v**2 * frequency_hz
    optimized_power_w = optimized_alpha * capacitance_f * voltage_v**2 * frequency_hz
    energy_savings_percent = (
        (baseline_power_w - optimized_power_w) / baseline_power_w * 100.0
        if baseline_power_w
        else 0.0
    )

    rows = [
        ("Pipeline cycles", str(metrics.cycles)),
        ("Instructions retired", str(metrics.instructions_retired)),
        ("CPI", f"{metrics.cpi:.2f}"),
        ("IPC", f"{metrics.ipc:.2f}"),
        ("AMAT", f"{amat:.2f} cycles"),
        ("Cache hit rate", f"{metrics.cache_hit_percent:.1f}%"),
        ("Cache invalidations", str(numa.invalidation_count)),
        ("MOESI memory writebacks", str(numa.memory_writebacks)),
        ("Branch penalty cycles avoided", str(pipeline.btb.branch_penalty_cycles_avoided)),
        ("Mesh route", f"{mesh_hops} hops"),
        ("Dynamic energy savings", f"{energy_savings_percent:.1f}%"),
        ("Baseline dynamic power", f"{baseline_power_w * 1000:.3f} mW"),
        ("Optimized dynamic power", f"{optimized_power_w * 1000:.3f} mW"),
    ]
    width = max(len(label) for label, _ in rows)
    print("\n" + "=" * (width + 29))
    print("AUTOMATED ARCHITECTURAL SCORECARD")
    print("-" * (width + 29))
    for label, value in rows:
        print(f"{label:<{width}} | {value:>16}")
    print("=" * (width + 29))
    print(
        "Dynamic power model: P = alpha * C * V^2 * f "
        f"(C={capacitance_f * 1e9:.1f} nF, V={voltage_v:.2f} V, "
        f"f={frequency_hz / 1e9:.1f} GHz, alpha {baseline_alpha:.2f} -> {optimized_alpha:.2f})"
    )

    parallel_fraction = 0.90
    print("\nAmdahl's Law Speedup (parallel fraction = 90%)")
    print(f"{'Cores':>8} | {'Speedup':>10} | {'Parallel efficiency':>20}")
    print("-" * 46)
    for core_count in (4, 8, 16):
        speedup = 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / core_count)
        efficiency = speedup / core_count * 100.0
        print(f"{core_count:>8} | {speedup:>9.2f}x | {efficiency:>19.1f}%")


def run_ground_level_resilience_demo() -> dict[str, int]:
    """Inject four representative field failures and verify recovery behavior."""
    print("\nGROUND-LEVEL RESILIENCE DEMONSTRATION")

    arbiter = DualTierBusArbiter(
        buffer_capacity=16,
        token_bucket_capacity=2,
        token_refill_rate=0,
    )
    assert arbiter.request_bus(Packet(77, "FAULTY_WATER_SENSOR", False, "sample-1"))
    assert arbiter.request_bus(Packet(77, "FAULTY_WATER_SENSOR", False, "sample-2"))
    assert not arbiter.request_bus(Packet(77, "FAULTY_WATER_SENSOR", False, "sample-3"))
    print(f"(a) Babbling sensor throttled: {arbiter.throttled_count} excess packet")

    memory = NUMADirectorySystem(num_nodes=4)
    ecc_tag = 0x5200
    memory.seed_memory(ecc_tag, 0xA55A)
    assert memory.read_transaction(0, ecc_tag) == 0xA55A
    memory.inject_cache_bit_error(0, ecc_tag, 5)
    assert memory.read_transaction(0, ecc_tag) == 0xA55A
    print(f"(b) SECDED corrected a cache bit flip: {memory.ecc_corrections} correction")

    brownout_tag = 0x5204
    memory.write_transaction(1, brownout_tag, 0xBEEF)
    journal = BrownoutRecoveryJournal()
    journaled_lines = journal.trigger_brownout(
        memory, supply_voltage=0.68, threshold_voltage=0.80
    )
    assert journaled_lines == 1 and journal.nvm[brownout_tag] == 0xBEEF
    assert journal.recover(memory) == 1
    assert memory.read_transaction(1, brownout_tag) == 0xBEEF
    print(f"(c) Brownout journal restored {journaled_lines} dirty line from NVM")

    mesh = MeshInterconnect()
    source = MeshNode("City Edge Router", 0, 0)
    destination = MeshNode("Emergency Dispatch", 2, 3)
    mesh.fail_link((0, 0), (1, 0))
    routed = mesh.enqueue("severed-link-alert", source, destination, is_emergency=True)
    assert routed.path[1] == (0, 1)
    assert ((0, 0), (1, 0)) not in list(zip(routed.path, routed.path[1:]))
    print(f"(d) Deflection avoided severed link: {len(routed.path) - 1} hops via VC1")

    return {
        "throttled_packets": arbiter.throttled_count,
        "ecc_corrections": memory.ecc_corrections,
        "journaled_lines": journaled_lines,
        "deflection_hops": len(routed.path) - 1,
    }


def main() -> None:
    """Run all architecture modules on a deterministic urban sensor workload."""
    print("INTEGRATED SMART CITY COMPUTING FRAMEWORK | SDG 11")
    print("Computer Architecture & Parallel Processing (CCSE 0304)")

    arbiter = DualTierBusArbiter(max_emergency_burst=3)
    serviced_packets, emergency_grants = run_telemetry_workload(arbiter)
    run_alu_demo()
    run_phase2_demo()
    pipeline = run_pipeline_demo()
    numa, mesh_hops = run_memory_demo()
    resilience_results = run_ground_level_resilience_demo()

    modeled_energy = (
        pipeline.cycle_count
        + arbiter.cycle_count * 0.25
        + numa.transaction_count * 2.0
        + numa.invalidation_count * 0.5
    )
    reference_energy = (
        pipeline.cycle_count * 1.5
        + arbiter.cycle_count * 0.5
        + numa.transaction_count * 2.5
        + numa.invalidation_count
    )
    metrics = SystemMetrics(
        cycles=pipeline.cycle_count,
        instructions_retired=pipeline.instructions_retired,
        cache_hits=numa.total_cache_hits,
        cache_misses=numa.total_cache_misses,
        energy_units=modeled_energy,
        reference_energy_units=reference_energy,
    )
    print_architectural_scorecard(pipeline, numa, mesh_hops)
    print("\nResilience verification summary")
    print(
        f"throttled={resilience_results['throttled_packets']}, "
        f"ecc_corrected={resilience_results['ecc_corrections']}, "
        f"nvm_journaled={resilience_results['journaled_lines']}, "
        f"deflection_hops={resilience_results['deflection_hops']}"
    )


if __name__ == "__main__":
    main()
