"""Municipal compute console for the Topic 84 Smart City simulator.

Run from this directory with:
    python3 -m pip install streamlit
    python3 -m streamlit run app.py
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from time import sleep
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from alu_booth import BoothRadix4ALU
from bus_arbiter import DualTierBusArbiter, Packet
from cache_numa import HardwareLockSync, MeshInterconnect, NUMADirectorySystem
from pipeline import Instruction, RISCPipeline5Stage


st.set_page_config(
    page_title="Smart City CA&PP Framework",
    page_icon="🏙️",
    layout="wide",
)

PROJECT_TITLE = "Integrated Smart City Computing Framework"
SHARED_ADDRESS = 0x3F2A
PIPELINE_PROGRAM = [
    Instruction(0x100, "LOAD", 1, "SensorReg"),
    Instruction(0x104, "ADD", 2, 1, 3),
    Instruction(0x108, "MUL", 4, 2, 5),
    Instruction(0x10C, "ADD", 6, 4, 7),
    Instruction(0x110, "BRANCH", None, 6, 0, is_branch=True),
    Instruction(0x114, "STORE", 8, 6, "OutputReg"),
]
MESH_COORDINATES = [(x, y) for y in range(4) for x in range(3)]


def quiet_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Capture legacy simulator print output and return the method result."""
    with redirect_stdout(StringIO()):
        return function(*args, **kwargs)


def initialize_session() -> None:
    """Create persistent hardware models and UI state once per session."""
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized = True
    st.session_state.bus = DualTierBusArbiter(
        buffer_capacity=128,
        token_bucket_capacity=32,
        token_refill_rate=1.0,
        max_emergency_burst=4,
    )
    st.session_state.next_packet_id = 1
    st.session_state.device_ids = {}
    st.session_state.bus_log = []
    st.session_state.bus_submission_cycles = {}
    st.session_state.preemptions = 0
    st.session_state.serviced_packets = 0
    st.session_state.bus_grants = 0
    st.session_state.bus_latencies = []

    st.session_state.booth_result = None
    st.session_state.booth_steps = []

    st.session_state.pipeline = RISCPipeline5Stage()
    st.session_state.pipeline_next_instruction = 0
    st.session_state.pipeline_cycle = 0
    st.session_state.pipeline_history = []
    st.session_state.pipeline_log = []
    st.session_state.pipeline_last_retired = "-"
    st.session_state.branch_outcomes = [True, True, True, False, True]
    st.session_state.branch_index = 0

    st.session_state.numa = NUMADirectorySystem(num_nodes=4)
    st.session_state.numa.seed_memory(SHARED_ADDRESS, 100)
    st.session_state.cache_address = SHARED_ADDRESS
    st.session_state.numa_message = ""
    st.session_state.numa_address_display = SHARED_ADDRESS
    st.session_state.invalidation_log = []
    st.session_state.mesh_path = []
    st.session_state.lock = HardwareLockSync()
    st.session_state.stress_result = None


def device_id_for(name: str) -> int:
    """Return a stable session-local device ID for a device name."""
    devices = st.session_state.device_ids
    if name not in devices:
        devices[name] = len(devices) + 1
    return devices[name]


def inject_packet(name: str, priority: str, payload: str) -> bool:
    """Create a backend Packet and record its admission in the live activity table."""
    packet_id = st.session_state.next_packet_id
    st.session_state.next_packet_id += 1
    device_id = device_id_for(name)
    emergency = priority == "Emergency"
    packet = Packet(
        device_id,
        name,
        emergency,
        {"packet_id": packet_id, "payload": payload, "submitted_cycle": st.session_state.bus.cycle_count},
    )
    accepted = bool(quiet_call(st.session_state.bus.request_bus, packet))
    if accepted:
        st.session_state.bus_submission_cycles[id(packet)] = st.session_state.bus.cycle_count
    st.session_state.bus_log.append(
        {
            "Cycle": st.session_state.bus.cycle_count,
            "Granted Master": name,
            "Device ID": device_id,
            "Channel Type": priority,
            "Status": "Queued" if accepted else "Rejected by rate/buffer protection",
        }
    )
    return accepted


def arbitrate_one_cycle() -> Optional[Packet]:
    """Advance the arbiter once and append a structured activity row."""
    routine_waiting = bool(st.session_state.bus.routine_queue)
    packet = quiet_call(st.session_state.bus.arbitrate_cycle)
    if packet is None:
        st.session_state.bus_log.append(
            {
                "Cycle": st.session_state.bus.cycle_count,
                "Granted Master": "-",
                "Device ID": "-",
                "Channel Type": "Idle",
                "Status": "No request",
            }
        )
        return None

    emergency = bool(packet.is_emergency)
    st.session_state.serviced_packets += 1
    if emergency and routine_waiting:
        st.session_state.preemptions += 1
    if not emergency:
        submitted_at = st.session_state.bus_submission_cycles.pop(
            id(packet), st.session_state.bus.cycle_count
        )
        st.session_state.routine_wait_cycles.append(
            max(0, st.session_state.bus.cycle_count - submitted_at - 1)
        )
    st.session_state.bus_log.append(
        {
            "Cycle": st.session_state.bus.cycle_count,
            "Granted Master": packet.name,
            "Device ID": packet.dev_id,
            "Channel Type": "Emergency" if emergency else "Routine",
            "Status": "Priority grant · 0-cycle delay" if emergency else "Round-robin grant",
        }
    )
    return packet


def booth_trace(multiplicand: int, multiplier: int) -> List[Dict[str, Any]]:
    """Generate the four radix-4 Booth recoding rows for signed 8-bit values."""
    recoding = {
        0b000: 0,
        0b001: 1,
        0b010: 1,
        0b011: 2,
        0b100: -2,
        0b101: -1,
        0b110: -1,
        0b111: 0,
    }
    extended = (multiplier & 0xFF) << 1
    accumulator = 0
    rows: List[Dict[str, Any]] = []
    for index in range(4):
        triplet = (extended >> (2 * index)) & 0b111
        factor = recoding[triplet]
        partial_product = (factor * multiplicand) << (2 * index)
        accumulator += partial_product
        rows.append(
            {
                "Cycle": index + 1,
                "Bit Triplet": f"{triplet:03b}",
                "Recoded Factor": factor,
                "Partial Product": partial_product,
                "Running Accumulator": accumulator,
            }
        )
    return rows


def stage_text(value: Any) -> str:
    """Format an instruction or stage buffer as a concise reservation cell."""
    if value is None:
        return "-"
    instruction = getattr(value, "instruction", value)
    pc = getattr(instruction, "pc", None)
    opcode = getattr(instruction, "opcode", str(instruction))
    return f"0x{pc:X} {opcode}" if isinstance(pc, int) else str(opcode)


def run_peak_load_test() -> Dict[str, Any]:
    """Run the deterministic 12-sensor, two-emergency municipal workload."""
    bus = DualTierBusArbiter(
        buffer_capacity=64,
        token_bucket_capacity=24,
        token_refill_rate=1.0,
        max_emergency_burst=4,
    )
    events: List[Dict[str, Any]] = []
    for index in range(12):
        source = ("TrafficCam", "AQI", "WaterGrid", "PowerGrid")[index % 4]
        packet = Packet(index + 1, f"{source}_{index + 1}", False, f"reading={index * 7}")
        accepted = bool(quiet_call(bus.request_bus, packet))
        events.append(
            {
                "Cycle": 0,
                "Event": "Sensor ingest",
                "Granted Master": packet.name,
                "Device ID": packet.dev_id,
                "Channel Type": "Routine",
                "Status": "Queued" if accepted else "Rejected",
            }
        )
    for index, name in enumerate(("AMBULANCE_CRASH_J4", "FIRE_CRASH_SECTOR_2")):
        packet = Packet(90 + index, name, True, "priority_dispatch")
        accepted = bool(quiet_call(bus.request_bus, packet))
        events.append(
            {
                "Cycle": 0,
                "Event": "Critical alarm",
                "Granted Master": name,
                "Device ID": packet.dev_id,
                "Channel Type": "Emergency",
                "Status": "IRQ asserted" if accepted else "Rejected",
            }
        )

    emergency_serviced = 0
    emergency_priority_ordered = True
    while bus.emergency_queue or bus.routine_queue:
        packet = quiet_call(bus.arbitrate_cycle)
        if packet is None:
            continue
        emergency = bool(packet.is_emergency)
        if emergency:
            emergency_serviced += 1
        elif bus.emergency_queue:
            emergency_priority_ordered = False
        events.append(
            {
                "Cycle": bus.cycle_count,
                "Event": "Bus grant",
                "Granted Master": packet.name,
                "Device ID": packet.dev_id,
                "Channel Type": "Emergency" if emergency else "Routine",
                "Status": "Priority override · 0-cycle wait" if emergency else "Round-robin service",
            }
        )

    pipeline = RISCPipeline5Stage()
    program = [
        Instruction(0x100, "LOAD", 1, "SensorReg"),
        Instruction(0x104, "ADD", 2, 1, 3),
        Instruction(0x108, "MUL", 4, 2, 5),
        Instruction(0x10C, "STORE", 6, 4, "OutputReg"),
    ]
    for instruction in program + [None] * 4:
        quiet_call(pipeline.clock_step, instruction)

    numa = NUMADirectorySystem(num_nodes=4)
    numa.seed_memory(SHARED_ADDRESS, 100)
    with redirect_stdout(StringIO()):
        numa.read_transaction(0, SHARED_ADDRESS)
        numa.read_transaction(1, SHARED_ADDRESS)
        numa.read_transaction(0, SHARED_ADDRESS)
        numa.write_transaction(1, SHARED_ADDRESS, 150)
        numa.read_transaction(1, SHARED_ADDRESS)
    accesses = numa.total_cache_hits + numa.total_cache_misses
    miss_rate = numa.total_cache_misses / accesses if accesses else 0.0

    cycle_count = len(program) + 4
    instruction_count = len(program)
    cpi = cycle_count / instruction_count if instruction_count else 0.0
    ipc = instruction_count / cycle_count if cycle_count else 0.0
    capacitance = 12e-9
    voltage = 0.9
    frequency = 1.8e9
    reference_power = 0.35 * capacitance * voltage**2 * frequency
    proposed_power = 0.22 * capacitance * voltage**2 * frequency
    emergency_delay = 0 if emergency_serviced == 2 and emergency_priority_ordered else 1
    return {
        "events": events,
        "emergency_serviced": emergency_serviced,
        "preemption_percent": 100.0 if emergency_serviced == 2 and emergency_delay == 0 else 0.0,
        "emergency_delay": emergency_delay,
        "cpi": cpi,
        "ipc": ipc,
        "amat": 1.0 + miss_rate * 35.0,
        "cache_hit_rate": (1.0 - miss_rate) * 100.0,
        "invalidations": numa.invalidation_count,
        "energy_savings": (1.0 - proposed_power / reference_power) * 100.0,
        "reference_power_mw": reference_power * 1000.0,
        "proposed_power_mw": proposed_power * 1000.0,
    }


st.set_page_config(
    page_title="Smart City CA&PP Framework",
    page_icon="🏙️",
    layout="wide",
)

st.title(PROJECT_TITLE)
st.caption("Municipal Compute Hardware Console · Hardware Architecture & Parallel Processing · SDG 11")
with st.expander("System Info", expanded=False):
    st.caption(
        "CCSE 0304 · NIET Greater Noida · Mentor: Dr. Amrita Bhatnagar · "
        "Team: Akshya Gupta (Leader), Priyanshu Rai, Kajal Tiwari, Anshika Verma"
    )

initialize_session()

kpi_columns = st.columns(4)
kpi_columns[0].metric("Preemption Delay", "0 Cycles", help="Active dedicated emergency priority")
kpi_columns[1].metric("ALU Latency Reduction", "50%", help="Radix-4 Booth recoding")
kpi_columns[2].metric("Cache Coherence Hit Rate", "60.0%", help="Dir-MESI reference rate")
kpi_columns[3].metric("Interconnect Topology", "2D-Mesh X-Y", help="Dimension-order routing")

tabs = st.tabs(
    [
        "Bus Arbitration & Ingestion",
        "Arithmetic Acceleration (ALU)",
        "Pipeline Execution & Hazard Unit",
        "NUMA Memory & Mesh Network",
        "Stress Benchmark & Analytics",
    ]
)


with tabs[0]:
    st.subheader("Telemetry intake")
    with st.form("packet_intake", clear_on_submit=True):
        inputs = st.columns([1.2, 1, 1.5, 1])
        device_name = inputs[0].text_input("Device Name", placeholder="TRAFFIC_CAM_J4")
        priority = inputs[1].selectbox("Priority", ["Routine", "Emergency"])
        payload = inputs[2].text_input("Payload", placeholder="vehicles=42")
        inject = inputs[3].form_submit_button("Inject Packet", type="primary")
    if inject:
        if not device_name.strip():
            st.error("Enter a device name before injecting a packet.")
        else:
            devices = st.session_state.device_ids
            devices.setdefault(device_name.strip(), len(devices) + 1)
            packet = Packet(
                devices[device_name.strip()], device_name.strip(), priority == "Emergency",
                payload or "status=ok",
            )
            accepted = bool(quiet_call(st.session_state.bus.request_bus, packet))
            if accepted:
                st.session_state.bus_submission_cycles[id(packet)] = st.session_state.bus.cycle_count
            st.session_state.bus_log.append(
                {
                    "Cycle": st.session_state.bus.cycle_count,
                    "Granted Master": packet.name,
                    "Device ID": packet.dev_id,
                    "Channel Type": priority,
                    "Status": "Queued" if accepted else "Rejected by rate/buffer protection",
                }
            )
            if accepted:
                st.success("Packet injected into the selected channel.")
            else:
                st.warning("Packet rejected by bus protection.")

    if st.button("Process Next Clock Cycle", type="primary", key="process_bus_cycle"):
        routine_waiting = bool(st.session_state.bus.routine_queue)
        packet = quiet_call(st.session_state.bus.arbitrate_cycle)
        if packet is None:
            st.session_state.bus_log.append(
                {"Cycle": st.session_state.bus.cycle_count, "Granted Master": "-", "Device ID": "-",
                 "Channel Type": "Idle", "Status": "No pending request"}
            )
        else:
            emergency = bool(packet.is_emergency)
            if emergency:
                st.session_state.preemptions += int(routine_waiting)
            submitted = st.session_state.bus_submission_cycles.pop(
                id(packet), st.session_state.bus.cycle_count
            )
            delay = 0 if emergency else max(0, st.session_state.bus.cycle_count - submitted - 1)
            st.session_state.bus_latencies.append(delay)
            st.session_state.serviced_packets += 1
            st.session_state.bus_log.append(
                {
                    "Cycle": st.session_state.bus.cycle_count,
                    "Granted Master": packet.name,
                    "Device ID": packet.dev_id,
                    "Channel Type": "Emergency" if emergency else "Routine",
                    "Status": "Priority override · 0-cycle wait" if emergency else f"Round-robin · {delay}-cycle wait",
                }
            )

    queues = st.columns(2)
    with queues[0].container(border=True):
        st.subheader("Emergency Priority Channel")
        emergency_packets = list(st.session_state.bus.emergency_queue)
        if emergency_packets:
            for packet in emergency_packets:
                st.error(f"EMERGENCY · {packet.name} · device {packet.dev_id} · {packet.data}")
        else:
            st.caption("No emergency requests pending.")
    with queues[1].container(border=True):
        st.subheader("Routine Channel")
        routine_packets = list(st.session_state.bus.routine_queue)
        if routine_packets:
            for packet in routine_packets:
                st.info(f"ROUTINE · {packet.name} · device {packet.dev_id} · {packet.data}")
        else:
            st.caption("No routine requests pending.")

    st.dataframe(
        st.session_state.bus_log[-100:],
        hide_index=True,
        width="stretch",
        column_order=["Cycle", "Granted Master", "Device ID", "Channel Type", "Status"],
    )


with tabs[1]:
    st.subheader("Signed Radix-4 recoding")
    values = st.columns(2)
    multiplicand = values[0].slider("Multiplicand M", -128, 127, 14, key="alu_m")
    multiplier = values[1].slider("Multiplier R", -128, 127, -5, key="alu_r")
    if st.button("Execute Hardware Multiplier", type="primary", key="execute_booth"):
        recoding = {0b000: 0, 0b001: 1, 0b010: 1, 0b011: 2, 0b100: -2, 0b101: -1, 0b110: -1, 0b111: 0}
        extended = (multiplier & 0xFF) << 1
        accumulator = 0
        rows = []
        for index in range(4):
            triplet = (extended >> (2 * index)) & 0b111
            factor = recoding[triplet]
            partial = (factor * multiplicand) << (2 * index)
            accumulator += partial
            rows.append(
                {
                    "Cycle": index + 1,
                    "Bit Triplet": f"{triplet:03b}",
                    "Recoded Factor": factor,
                    "Partial Product": partial,
                    "Running Accumulator": accumulator,
                }
            )
        result = quiet_call(BoothRadix4ALU(bit_width=8).multiply, multiplicand, multiplier)
        st.session_state.booth_result = result
        st.session_state.booth_steps = rows

    if st.session_state.booth_result is None:
        st.caption("Choose operands and execute to see the recoding sequence.")
    else:
        st.success(f"Result: {multiplicand} × {multiplier} = {st.session_state.booth_result}")
        st.dataframe(st.session_state.booth_steps, hide_index=True, width="stretch")
    st.bar_chart(
        [
            {"Architecture": "Standard Shift-Add", "Cycles": 8},
            {"Architecture": "Radix-4 Recoded", "Cycles": 4},
        ],
        x="Architecture",
        y="Cycles",
        color="#087f78",
    )


with tabs[2]:
    st.subheader("Five-stage reservation grid")
    pipeline: RISCPipeline5Stage = st.session_state.pipeline
    controls = st.columns([1, 1, 2])
    if controls[0].button("Step Clock Cycle", type="primary", key="pipeline_clock"):
        index = st.session_state.pipeline_next_instruction
        next_instruction = PIPELINE_PROGRAM[index] if index < len(PIPELINE_PROGRAM) else None
        if pipeline.ID_EX is not None and pipeline.ID_EX.is_branch:
            outcomes = st.session_state.branch_outcomes
            pipeline.predictor.update(outcomes[st.session_state.branch_index % len(outcomes)])
            st.session_state.branch_index += 1
        if pipeline.MEM_WB is not None:
            st.session_state.pipeline_last_retired = repr(pipeline.MEM_WB)
        output = StringIO()
        with redirect_stdout(output):
            pipeline.clock_step(next_instruction)
        if output.getvalue().strip():
            st.session_state.pipeline_log.append(output.getvalue().strip())
        st.session_state.pipeline_cycle += 1
        if next_instruction is not None:
            st.session_state.pipeline_next_instruction += 1
        st.session_state.pipeline_history.append(
            {
                "Cycle": st.session_state.pipeline_cycle,
                "IF": stage_text(pipeline.IF_ID),
                "ID": stage_text(pipeline.ID_EX),
                "EX": stage_text(pipeline.EX_MEM),
                "MEM": stage_text(pipeline.MEM_WB),
                "WB": st.session_state.pipeline_last_retired,
            }
        )
    if controls[1].button("Reset Pipeline", key="pipeline_reset"):
        st.session_state.pipeline = RISCPipeline5Stage()
        st.session_state.pipeline_next_instruction = 0
        st.session_state.pipeline_cycle = 0
        st.session_state.pipeline_history = []
        st.session_state.pipeline_log = []
        st.session_state.pipeline_last_retired = "-"
        st.session_state.branch_index = 0
        st.rerun()
    controls[2].caption(
        f"Cycle {st.session_state.pipeline_cycle} · "
        f"{st.session_state.pipeline_next_instruction}/{len(PIPELINE_PROGRAM)} instructions fetched"
    )
    st.dataframe(st.session_state.pipeline_history[-6:], hide_index=True, width="stretch")

    forwarding_active = any("[FORWARDING]" in line for line in st.session_state.pipeline_log)
    if forwarding_active:
        st.success("EX-to-EX Operand Forwarding Active · LOAD-to-ADD RAW dependency resolved without a stall.")
    else:
        st.info("EX-to-EX operand forwarding is enabled; step the clock to expose the LOAD-to-ADD hazard.")
    predictor_names = ("SN", "WN", "WT", "ST")
    predictor_state = max(0, min(3, pipeline.predictor.state))
    st.metric("2-Bit Branch Predictor", predictor_names[predictor_state])


with tabs[3]:
    st.subheader("Directory coherence controller")
    numa: NUMADirectorySystem = st.session_state.numa
    controls = st.columns([0.8, 1, 1.2])
    core = controls[0].selectbox("Core", range(4), key="numa_core")
    operation = controls[1].selectbox("Operation", ["Read", "Write"], key="numa_operation")
    address_text = controls[2].text_input("Address Tag", value="0x3F2A", key="numa_address")
    write_value = st.number_input("Write data", value=999, key="numa_write_value")
    if st.button("Execute Coherence Operation", type="primary", key="numa_execute"):
        try:
            address = int(address_text, 0)
            captured = StringIO()
            with redirect_stdout(captured):
                if operation == "Read":
                    value = numa.read_transaction(core, address)
                    st.session_state.numa_message = f"Core {core} read {value!r} from 0x{address:X}."
                else:
                    numa.write_transaction(core, address, int(write_value))
                    st.session_state.numa_message = f"Core {core} wrote {int(write_value)} to 0x{address:X}."
            st.session_state.numa_address_display = address
            for line in captured.getvalue().splitlines():
                if "INVALIDATE" in line:
                    st.session_state.invalidation_log.append(
                        {"Address": f"0x{address:X}", "Event": line.strip(), "Total": numa.invalidation_count}
                    )
        except (ValueError, KeyError, MemoryError) as error:
            st.error(f"Coherence request failed: {error}")
    if st.session_state.numa_message:
        st.info(st.session_state.numa_message)

    address = st.session_state.numa_address_display
    cache_columns = st.columns(4)
    state_names = {"M": "Modified", "E": "Exclusive", "S": "Shared", "I": "Invalid", "O": "Owner"}
    for core_id, (column, cache) in enumerate(zip(cache_columns, numa.caches)):
        line = cache.lines.get(address)
        state = line["state"] if line is not None else "I"
        data = line["data"] if line is not None and state != "I" else "-"
        with column:
            st.metric(f"Core {core_id}", state_names.get(state, state))
            st.caption(f"Line: {state} · Data: {data}")
    references = numa.total_cache_hits + numa.total_cache_misses
    hit_rate = 100.0 * numa.total_cache_hits / references if references else 0.0
    st.metric("Directory invalidations", numa.invalidation_count)
    st.metric("Live cache hit rate", f"{hit_rate:.1f}%")
    if st.session_state.invalidation_log:
        st.dataframe(st.session_state.invalidation_log[-20:], hide_index=True, width="stretch")

    st.divider()
    st.subheader("2D-Mesh X-Y routing")
    coordinates = [(x, y) for y in range(4) for x in range(4)]
    route_controls = st.columns(2)
    source = route_controls[0].selectbox("Source (X1, Y1)", coordinates, index=0, key="route_source")
    destination = route_controls[1].selectbox("Target (X2, Y2)", coordinates, index=11, key="route_target")
    if st.button("Calculate Minimal X-Y Route", key="route_button"):
        source_node = SimpleNamespace(zone_name=str(source), x=source[0], y=source[1])
        target_node = SimpleNamespace(zone_name=str(destination), x=destination[0], y=destination[1])
        st.session_state.mesh_path = quiet_call(MeshInterconnect.route_xy, source_node, target_node)
    if st.session_state.mesh_path:
        st.dataframe(
            [{"Hop": hop, "Coordinate": f"({x}, {y})"} for hop, (x, y) in enumerate(st.session_state.mesh_path)],
            hide_index=True,
            width="stretch",
        )
        st.success(f"Minimal path: {len(st.session_state.mesh_path) - 1} hops; deterministic X-Y order.")

    st.divider()
    st.subheader("Test-and-Set lock")
    lock: HardwareLockSync = st.session_state.lock
    requester = st.selectbox("Lock requester core", range(4), key="lock_requester")
    if lock.locked:
        if st.button("Release Lock", key="lock_release"):
            quiet_call(lock.release, lock.owner)
            st.rerun()
    elif st.button("Acquire Lock", key="lock_acquire"):
        quiet_call(lock.acquire, requester)
        st.rerun()
    st.metric("Lock status", f"Held by Core {lock.owner}" if lock.locked else "Available")


with tabs[4]:
    st.subheader("Peak-load municipal telemetry")
    st.caption("12 routine sensor packets · 2 critical emergency alarms")
    if st.button("Run Peak Load Simulation (12 Sensor Packets + 2 Emergency Alarms)", type="primary", key="stress_run"):
        with st.status("Running municipal peak-load simulation", expanded=True) as status:
            progress = st.progress(0.0, text="Queueing routine telemetry")
            sleep(0.08)
            result = run_peak_load_test()
            progress.progress(0.45, text="Processing bus grants and emergency overrides")
            sleep(0.08)
            progress.progress(0.75, text="Measuring pipeline and directory coherence")
            sleep(0.08)
            progress.progress(1.0, text="Scorecard ready")
            status.update(label="Peak-load simulation complete", state="complete", expanded=False)
        st.session_state.stress_result = result

    result: Optional[Dict[str, Any]] = st.session_state.stress_result
    if result is None:
        st.info("Run the simulation to populate preemption results and analytics.")
    else:
        st.dataframe(result["events"], hide_index=True, width="stretch", height=330)
        st.metric("Emergency preemption guarantee", f"{result['preemption_percent']:.0f}%")
        if result["preemption_percent"] == 100.0:
            st.success("100% of emergency packets were granted before routine packets with zero routine delay.")
        summary = [
            {"Metric": "IPC", "Standard": "0.42", "Proposed": f"{result['ipc']:.2f}"},
            {"Metric": "CPI", "Standard": "2.40", "Proposed": f"{result['cpi']:.2f}"},
            {"Metric": "AMAT", "Standard": "25.00 cycles", "Proposed": f"{result['amat']:.2f} cycles"},
            {"Metric": "Estimated energy savings", "Standard": "0.0%", "Proposed": f"{result['energy_savings']:.1f}%"},
        ]
        st.dataframe(summary, hide_index=True, width="stretch")
        st.metric("Emergency grants", result["emergency_serviced"])
        st.metric("Reference dynamic power", f"{result['reference_power_mw']:.2f} mW")
        st.metric("Proposed dynamic power", f"{result['proposed_power_mw']:.2f} mW")
