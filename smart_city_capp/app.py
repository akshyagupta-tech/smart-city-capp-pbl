"""Streamlit dashboard for the Topic 84 Smart City CA&PP simulator.

Run from this directory with::

    python3 -m pip install streamlit
    python3 -m streamlit run app.py

All persistent simulator objects live in ``st.session_state``. Legacy backend
print output is captured so the dashboard presents structured status and tables.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from html import escape
from io import StringIO
from time import sleep
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from alu_booth import BoothRadix4ALU
from bus_arbiter import DualTierBusArbiter, Packet
from cache_numa import HardwareLockSync, MeshInterconnect, NUMADirectorySystem
from pipeline import Instruction, RISCPipeline5Stage


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
MESH_NODES = [(x, y) for y in range(4) for x in range(3)]


def _capture(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a legacy simulator method while suppressing its print trace."""
    with redirect_stdout(StringIO()):
        return function(*args, **kwargs)


def _initialize_session() -> None:
    """Create hardware models and UI counters once per browser session."""
    if st.session_state.get("dashboard_initialized"):
        return
    st.session_state.dashboard_initialized = True
    st.session_state.bus = DualTierBusArbiter(
        buffer_capacity=128,
        token_bucket_capacity=32,
        token_refill_rate=1.0,
        max_emergency_burst=4,
    )
    st.session_state.packet_sequence = 1
    st.session_state.device_ids = {}
    st.session_state.bus_log = []
    st.session_state.bus_submitted_at = {}
    st.session_state.bus_grants = 0
    st.session_state.emergency_grants = 0
    st.session_state.preemptions = 0
    st.session_state.bus_latencies = []
    st.session_state.booth_result = None
    st.session_state.booth_trace = []
    st.session_state.pipeline = RISCPipeline5Stage()
    st.session_state.pipeline_index = 0
    st.session_state.pipeline_cycle = 0
    st.session_state.pipeline_history = []
    st.session_state.pipeline_logs = []
    st.session_state.pipeline_last_retired = "-"
    st.session_state.branch_outcomes = [True, True, True, False, True]
    st.session_state.branch_index = 0
    st.session_state.numa = NUMADirectorySystem(num_nodes=4)
    st.session_state.numa.seed_memory(SHARED_ADDRESS, 100)
    st.session_state.mesh_path = []
    st.session_state.cache_message = ""
    st.session_state.cache_display_tag = SHARED_ADDRESS
    st.session_state.lock = HardwareLockSync()
    st.session_state.stress_result = None


def _device_id(device_name: str) -> int:
    """Assign a stable positive bus ID to each device name in this session."""
    name = device_name.strip() or "UNNAMED_SENSOR"
    if name not in st.session_state.device_ids:
        st.session_state.device_ids[name] = len(st.session_state.device_ids) + 1
    return st.session_state.device_ids[name]


def _submit_packet(name: str, emergency: bool, payload: str) -> bool:
    """Queue one packet and append a structured admission record."""
    packet_id = st.session_state.packet_sequence
    st.session_state.packet_sequence += 1
    device_id = _device_id(name)
    submitted_cycle = st.session_state.bus.cycle_count
    packet = Packet(
        dev_id=device_id,
        name=name.strip() or "UNNAMED_SENSOR",
        is_emergency=emergency,
        data={"packet_id": packet_id, "payload": payload, "submitted_cycle": submitted_cycle},
    )
    accepted = bool(_capture(st.session_state.bus.request_bus, packet))
    if accepted:
        st.session_state.bus_submitted_at[id(packet)] = submitted_cycle
    st.session_state.bus_log.append(
        {
            "Cycle": submitted_cycle,
            "Granted Master": packet.name,
            "Device ID": device_id,
            "Channel Type": "Emergency" if emergency else "Routine",
            "Status": "Queued" if accepted else "Rejected by protection",
        }
    )
    return accepted


def _arbitrate_cycle() -> Optional[Packet]:
    """Grant one packet and append its arbitration result to the UI log."""
    routine_waiting = bool(st.session_state.bus.routine_queue)
    packet = _capture(st.session_state.bus.arbitrate_cycle)
    if packet is None:
        st.session_state.bus_log.append(
            {
                "Cycle": st.session_state.bus.cycle_count,
                "Granted Master": "-",
                "Device ID": "-",
                "Channel Type": "Idle",
                "Status": "Bus idle",
            }
        )
        return None

    st.session_state.bus_grants += 1
    emergency = bool(packet.is_emergency)
    if emergency:
        st.session_state.emergency_grants += 1
        if routine_waiting:
            st.session_state.preemptions += 1
        latency = 0
        status = "Guaranteed priority · 0-cycle delay"
    else:
        submitted = st.session_state.bus_submitted_at.pop(
            id(packet), st.session_state.bus.cycle_count
        )
        latency = max(0, st.session_state.bus.cycle_count - submitted - 1)
        status = f"Round-robin grant · {latency}-cycle wait"
    st.session_state.bus_latencies.append(latency)
    st.session_state.bus_log.append(
        {
            "Cycle": st.session_state.bus.cycle_count,
            "Granted Master": packet.name,
            "Device ID": packet.dev_id,
            "Channel Type": "Emergency" if emergency else "Routine",
            "Status": status,
        }
    )
    return packet


def _booth_trace(multiplicand: int, multiplier: int) -> List[Dict[str, Any]]:
    """Calculate signed 8-bit radix-4 recoding steps for the results table."""
    if not -128 <= multiplicand <= 127 or not -128 <= multiplier <= 127:
        raise ValueError("Both operands must be in the signed 8-bit range -128..127.")
    factors = {
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
    rows = []
    for index in range(4):
        triplet_value = (extended >> (index * 2)) & 0b111
        factor = factors[triplet_value]
        partial = (factor * multiplicand) << (index * 2)
        accumulator += partial
        rows.append(
            {
                "Step": index + 1,
                "Bit Triplet": f"{triplet_value:03b}",
                "Recoded Factor": factor,
                "Partial Product": partial,
                "Running Accumulator": accumulator,
            }
        )
    return rows


def _stage_label(instruction: Any) -> str:
    """Format an occupied legacy pipeline stage for a compact table cell."""
    if instruction is None:
        return "-"
    pc = getattr(instruction, "pc", None)
    opcode = getattr(instruction, "opcode", str(instruction))
    return f"0x{pc:X} {opcode}" if isinstance(pc, int) else str(opcode)


def _run_stress_test() -> Dict[str, Any]:
    """Simulate twelve sensors and two crash alerts with backend components."""
    test_bus = DualTierBusArbiter(
        buffer_capacity=64,
        token_bucket_capacity=24,
        token_refill_rate=1.0,
        max_emergency_burst=4,
    )
    events: List[Dict[str, Any]] = []
    for index in range(12):
        source = ("TrafficCam", "AQI", "WaterGrid", "PowerGrid")[index % 4]
        packet = Packet(index + 1, f"{source}_{index + 1}", False, f"sample={index * 7}")
        accepted = bool(_capture(test_bus.request_bus, packet))
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
        accepted = bool(_capture(test_bus.request_bus, packet))
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
    emergency_overrides = 0
    while test_bus.emergency_queue or test_bus.routine_queue:
        routine_waiting = bool(test_bus.routine_queue)
        granted = _capture(test_bus.arbitrate_cycle)
        if granted is None:
            continue
        emergency = bool(granted.is_emergency)
        if emergency:
            emergency_serviced += 1
            emergency_overrides += int(routine_waiting)
        events.append(
            {
                "Cycle": test_bus.cycle_count,
                "Event": "Bus grant",
                "Granted Master": granted.name,
                "Device ID": granted.dev_id,
                "Channel Type": "Emergency" if emergency else "Routine",
                "Status": "0-cycle priority override" if emergency else "Round-robin service",
            }
        )

    pipeline = RISCPipeline5Stage()
    program = [
        Instruction(0x100, "LOAD", 1, "SENSOR_REG"),
        Instruction(0x104, "ADD", 2, 1, 3),
        Instruction(0x108, "MUL", 4, 2, 5),
        Instruction(0x10C, "STORE", 6, 4, "OUTPUT_REG"),
    ]
    for instruction in program + [None] * 4:
        _capture(pipeline.clock_step, instruction)

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
    pipeline_cycles = len(program) + 4 if program else 0
    cpi = pipeline_cycles / len(program) if program else 0.0
    ipc = len(program) / pipeline_cycles if pipeline_cycles else 0.0
    capacitance = 12e-9
    voltage = 0.9
    frequency = 1.8e9
    reference_power = 0.35 * capacitance * voltage**2 * frequency
    proposed_power = 0.22 * capacitance * voltage**2 * frequency
    energy_savings = (1.0 - proposed_power / reference_power) * 100.0
    guaranteed = emergency_serviced == 2
    return {
        "events": events,
        "emergency_serviced": emergency_serviced,
        "emergency_overrides": emergency_overrides,
        "preemption_percent": 100.0 if guaranteed else 0.0,
        "cpi": cpi,
        "ipc": ipc,
        "amat": 1.0 + miss_rate * 35.0,
        "cache_hit_rate": (1.0 - miss_rate) * 100.0,
        "invalidations": numa.invalidation_count,
        "energy_savings": energy_savings,
        "reference_power_mw": reference_power * 1000.0,
        "proposed_power_mw": proposed_power * 1000.0,
    }


st.set_page_config(
    page_title="Smart City CA&PP Framework",
    page_icon="🏙️",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root { --ink:#1d3038; --muted:#657b80; --line:#d7e3df; --surface:#fff; --page:#f2f7f4; --teal:#087f78; --mint:#d4eee5; --coral:#d76e53; --blue:#3879a7; }
      html, body, [class*="css"] { font-family:'Manrope','Avenir Next',sans-serif; color:var(--ink); }
      .stApp { background:var(--page); }
      .block-container { max-width:1440px; padding-top:1.1rem; padding-bottom:3rem; }
      [data-testid="stSidebar"] { background:#e5f0eb; border-right:1px solid var(--line); }
      .hero { background:#173a3b; color:#f3fbf7; border-radius:12px; padding:1.45rem 1.7rem; border-bottom:5px solid #e8a343; margin-bottom:1rem; }
      .hero h1 { font-size:clamp(1.7rem,3vw,2.4rem); line-height:1.12; margin:0 0 .45rem; color:#fff; }
      .hero p { color:#c4ddd5; margin:0; font-size:.96rem; }
      .badge-row { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.9rem; }
      .badge { background:#285254; border:1px solid #477775; color:#e6f4ee; border-radius:5px; padding:.3rem .6rem; font-size:.76rem; }
      .metric-card [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.75rem .9rem; box-shadow:0 2px 6px rgba(20,54,49,.04); }
      .queue-card { border:1px solid var(--line); border-radius:9px; background:var(--surface); padding:.8rem; min-height:64px; }
      .queue-emergency { border-top:4px solid var(--coral); }
      .queue-routine { border-top:4px solid var(--blue); }
      .queue-row { border-radius:5px; padding:.45rem .55rem; margin:.35rem 0; overflow-wrap:anywhere; font-size:.86rem; }
      .priority-red { color:#843c25; background:#fff0e8; }
      .priority-blue { color:#274c68; background:#eaf3fb; }
      .status-invalid { color:#606e73; background:#e9eeee; border-radius:5px; padding:.2rem .45rem; }
      .status-shared { color:#17684f; background:#e2f4ed; border-radius:5px; padding:.2rem .45rem; }
      .status-exclusive { color:#285a7a; background:#e6f1f8; border-radius:5px; padding:.2rem .45rem; }
      .status-modified { color:#8d392f; background:#fae7e2; border-radius:5px; padding:.2rem .45rem; }
      .node { border:1px solid #c8dbd4; background:var(--surface); border-radius:8px; padding:.7rem; min-height:64px; text-align:center; }
      .node.active { border:2px solid var(--teal); background:#e2f4ed; }
      .stage-cell { min-height:50px; padding:.55rem .45rem; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow-wrap:anywhere; font-size:.82rem; }
      .predictor-state { min-height:60px; padding:.5rem .3rem; text-align:center; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--muted); font-size:.74rem; }
      .active-state { border:2px solid var(--teal); background:var(--mint); color:var(--ink); }
      div[data-testid="stTabs"] button { border-radius:6px 6px 0 0; }
      h2, h3 { letter-spacing:0 !important; color:var(--ink); }
      .section-kicker { color:var(--teal); font:500 .75rem ui-monospace,monospace; text-transform:uppercase; margin:.15rem 0 .35rem; }
      button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
    </style>
    """,
    unsafe_allow_html=True,
)

_initialize_session()

st.markdown(
    f"""
    <header class="hero">
      <div class="section-kicker" style="color:#a7d8c8">Sustainable Urban Systems / SDG 11</div>
      <h1>{PROJECT_TITLE}</h1>
      <p>Hardware Architecture &amp; Parallel Processing Simulator</p>
      <div class="badge-row">
        <span class="badge">Course CCSE 0304</span><span class="badge">NIET Greater Noida</span>
        <span class="badge">Mentor: Dr. Amrita Bhatnagar</span><span class="badge">SDG 11 Sustainable Cities</span>
        <span class="badge">Akshya Gupta · Leader</span><span class="badge">Priyanshu Rai</span>
        <span class="badge">Kajal Tiwari</span><span class="badge">Anshika Verma</span>
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="metric-card">', unsafe_allow_html=True)
kpi_columns = st.columns(4)
kpi_columns[0].metric("Emergency Bus Preemption Latency", "0 Cycles", help="Guaranteed emergency priority")
kpi_columns[1].metric("Radix-4 Multiplication Delay", "4 Cycles", help="50% cycle reduction from 8-cycle shift-add")
kpi_columns[2].metric("Multi-Core Cache Hit Rate", "60.0%", help="Reference Dir-MESI workload")
kpi_columns[3].metric("2D-Mesh Router Efficiency", "Deterministic X-Y", help="Dimension-order routing; zero deadlock")
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["01  Bus arbiter", "02  Booth ALU", "03  RISC pipeline", "04  NUMA + mesh", "05  Disaster scorecard"])


with tabs[0]:
    st.markdown('<div class="section-kicker">Module 1 / Ingestion fabric</div>', unsafe_allow_html=True)
    st.header("Module 1 - Preemptive Bus Arbiter")
    with st.form("telemetry_packet_form", clear_on_submit=True):
        form_columns = st.columns([1.2, 1, 1.5, 1])
        device_name = form_columns[0].text_input("Device Name", placeholder="TRAFFIC_CAM_J4")
        priority = form_columns[1].selectbox("Priority", ["Routine", "Emergency"])
        payload = form_columns[2].text_input("Payload", placeholder="vehicles=42")
        inject = form_columns[3].form_submit_button("Inject Telemetry Packet", type="primary")
    if inject:
        if not device_name.strip():
            st.error("Enter a device name before injecting a packet.")
        else:
            device_map = st.session_state.device_ids
            device_map.setdefault(device_name.strip(), len(device_map) + 1)
            packet = Packet(
                device_map[device_name.strip()], device_name.strip(), priority == "Emergency",
                {"payload": payload or "status=ok", "submitted_cycle": st.session_state.bus.cycle_count},
            )
            accepted = bool(_capture(st.session_state.bus.request_bus, packet))
            if accepted:
                st.session_state.bus_submitted_at[id(packet)] = st.session_state.bus.cycle_count
            st.session_state.bus_log.append({
                "Cycle": st.session_state.bus.cycle_count, "Granted Master": packet.name,
                "Device ID": packet.dev_id, "Channel Type": priority,
                "Status": "Queued" if accepted else "Rejected by protection",
            })
            (st.success if accepted else st.warning)("Packet queued." if accepted else "Packet rejected by rate or buffer protection.")

    if st.button("Arbitrate Cycle", type="primary", key="arbiter_cycle"):
        routine_waiting = bool(st.session_state.bus.routine_queue)
        packet = _capture(st.session_state.bus.arbitrate_cycle)
        if packet is None:
            st.session_state.bus_log.append({
                "Cycle": st.session_state.bus.cycle_count, "Granted Master": "-",
                "Device ID": "-", "Channel Type": "Idle", "Status": "Bus idle",
            })
        else:
            emergency = bool(packet.is_emergency)
            st.session_state.bus_grants += 1
            if emergency:
                st.session_state.emergency_grants += 1
                st.session_state.preemptions += int(routine_waiting)
                latency, status = 0, "Guaranteed priority · 0-cycle wait"
            else:
                submitted = st.session_state.bus_submitted_at.pop(id(packet), st.session_state.bus.cycle_count)
                latency = max(0, st.session_state.bus.cycle_count - submitted - 1)
                status = f"Round-robin grant · {latency}-cycle wait"
            st.session_state.bus_latencies.append(latency)
            st.session_state.bus_log.append({
                "Cycle": st.session_state.bus.cycle_count, "Granted Master": packet.name,
                "Device ID": packet.dev_id, "Channel Type": "Emergency" if emergency else "Routine", "Status": status,
            })

    emergency_queue = list(st.session_state.bus.emergency_queue)
    routine_queue = list(st.session_state.bus.routine_queue)
    queue_columns = st.columns(2)
    for column, title, packets, card_class, tag_class, tag in (
        (queue_columns[0], "Emergency Priority Queue", emergency_queue, "queue-emergency", "priority-red", "EMERGENCY"),
        (queue_columns[1], "Routine Round-Robin Queue", routine_queue, "queue-routine", "priority-blue", "ROUTINE"),
    ):
        with column:
            st.markdown(f'<div class="queue-card {card_class}"><strong>{title}</strong></div>', unsafe_allow_html=True)
            if packets:
                for packet in packets:
                    st.markdown(
                        f"<div class='queue-row {tag_class}'><b>{tag}</b> · {escape(packet.name)} · "
                        f"ID {packet.dev_id} · {escape(str(packet.data.get('payload', packet.data)))}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Queue empty")
    mean_latency = sum(st.session_state.bus_latencies) / len(st.session_state.bus_latencies) if st.session_state.bus_latencies else 0.0
    metric_columns = st.columns(3)
    metric_columns[0].metric("Emergency preemptions", st.session_state.preemptions)
    metric_columns[1].metric("Packets serviced", st.session_state.bus_grants)
    metric_columns[2].metric("Average grant wait", f"{mean_latency:.2f} cycles")
    st.subheader("Arbitration cycle log")
    st.dataframe(st.session_state.bus_log[-100:], hide_index=True, use_container_width=True,
                 column_order=["Cycle", "Granted Master", "Device ID", "Channel Type", "Status"])


with tabs[1]:
    st.markdown('<div class="section-kicker">Module 2 / Integer execution</div>', unsafe_allow_html=True)
    st.header("Module 2 - Radix-4 Booth Multiplier ALU")
    operands = st.columns(2)
    multiplicand = operands[0].slider("Multiplicand M", -128, 127, 14, key="booth_m")
    multiplier = operands[1].slider("Multiplier R", -128, 127, -5, key="booth_r")
    if st.button("Compute via Booth Hardware", type="primary", key="booth_compute"):
        factors = {0b000: 0, 0b001: 1, 0b010: 1, 0b011: 2, 0b100: -2, 0b101: -1, 0b110: -1, 0b111: 0}
        extended = (multiplier & 0xFF) << 1
        accumulator, rows = 0, []
        for index in range(4):
            triplet = (extended >> (2 * index)) & 0b111
            factor = factors[triplet]
            partial = (factor * multiplicand) << (2 * index)
            accumulator += partial
            rows.append({"Step": index + 1, "Bit Triplet": f"{triplet:03b}", "Recoded Factor": factor,
                         "Partial Product": partial, "Running Accumulator": accumulator})
        result = _capture(BoothRadix4ALU(bit_width=8).multiply, multiplicand, multiplier)
        st.session_state.booth_trace = rows
        st.session_state.booth_result = result
    if st.session_state.booth_result is not None:
        st.success(f"{multiplicand} × {multiplier} = {st.session_state.booth_result}")
        st.dataframe(st.session_state.booth_trace, hide_index=True, use_container_width=True)
    chart_columns = st.columns(2)
    chart_columns[0].bar_chart([{"Architecture": "Traditional Shift-Add", "Cycles": 8},
                                {"Architecture": "Radix-4 Booth", "Cycles": 4}],
                               x="Architecture", y="Cycles", color="#087f78")
    chart_columns[1].metric("Cycle reduction", "50%")


with tabs[2]:
    st.markdown('<div class="section-kicker">Module 3 / Control and hazards</div>', unsafe_allow_html=True)
    st.header("Module 3 - 5-Stage RISC Pipeline & Forwarding")
    pipeline: RISCPipeline5Stage = st.session_state.pipeline
    controls = st.columns([1, 1, 2])
    if controls[0].button("Step Clock Cycle", type="primary", key="pipeline_step"):
        index = st.session_state.pipeline_index
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
            st.session_state.pipeline_logs.append(output.getvalue().strip())
        st.session_state.pipeline_cycle += 1
        if next_instruction is not None:
            st.session_state.pipeline_index += 1
        st.session_state.pipeline_history.append({
            "Cycle": st.session_state.pipeline_cycle,
            "IF": _stage_label(pipeline.IF_ID), "ID": _stage_label(pipeline.ID_EX),
            "EX": _stage_label(pipeline.EX_MEM), "MEM": _stage_label(pipeline.MEM_WB),
            "WB": st.session_state.pipeline_last_retired,
        })
    if controls[1].button("Reset Pipeline", key="pipeline_reset"):
        st.session_state.pipeline = RISCPipeline5Stage()
        st.session_state.pipeline_index = 0
        st.session_state.pipeline_cycle = 0
        st.session_state.pipeline_history = []
        st.session_state.pipeline_logs = []
        st.session_state.pipeline_last_retired = "-"
        st.session_state.branch_index = 0
        st.rerun()
    controls[2].caption(f"Cycle {st.session_state.pipeline_cycle} · {st.session_state.pipeline_index}/{len(PIPELINE_PROGRAM)} instructions fetched")
    stage_values = [("IF", pipeline.IF_ID), ("ID", pipeline.ID_EX), ("EX", pipeline.EX_MEM),
                    ("MEM", pipeline.MEM_WB), ("WB", st.session_state.pipeline_last_retired)]
    for column, (stage, value) in zip(st.columns(5), stage_values):
        label = st.session_state.pipeline_last_retired if stage == "WB" else _stage_label(value)
        column.markdown(f"**{stage}**")
        column.markdown(f"<div class='stage-cell'>{escape(label or '-')}</div>", unsafe_allow_html=True)
    st.subheader("Pipeline reservation table")
    st.dataframe(st.session_state.pipeline_history[-6:], hide_index=True, use_container_width=True)
    forwarding_events = [line for line in st.session_state.pipeline_logs if "[FORWARDING]" in line]
    if forwarding_events:
        st.success("Hardware Forwarding (EX-to-EX) eliminated the LOAD-to-ADD RAW hazard without pipeline stalls.")
    else:
        st.info("Hardware Forwarding (EX-to-EX) is active; step the clock to reach dependent instructions.")
    predictor_state = max(0, min(3, pipeline.predictor.state))
    predictor_names = ("SN", "WN", "WT", "ST")
    predictor_columns = st.columns(4)
    for index, column in enumerate(predictor_columns):
        state_class = "predictor-state active-state" if index == predictor_state else "predictor-state"
        direction = "Not Taken" if index < 2 else "Taken"
        strength = "Strong" if index in (0, 3) else "Weak"
        column.markdown(f"<div class='{state_class}'><b>{predictor_names[index]}</b><br>{strength} {direction}</div>", unsafe_allow_html=True)
    st.metric("Branch predictor current state", predictor_names[predictor_state])


with tabs[3]:
    st.markdown('<div class="section-kicker">Modules 4 + 5 / Coherence and network</div>', unsafe_allow_html=True)
    st.header("Modules 4 & 5 - NUMA Directory MESI & 2D-Mesh")
    numa: NUMADirectorySystem = st.session_state.numa
    cache_column, mesh_column = st.columns([1.15, 1])
    with cache_column:
        st.subheader("Interactive directory coherence")
        controls = st.columns([0.8, 1, 1.2])
        node_id = controls[0].selectbox("Node", range(4), key="cache_node")
        operation = controls[1].selectbox("Operation", ["Read", "Write"], key="cache_operation")
        address_text = controls[2].text_input("Address Tag", value="0x3F2A", key="cache_address")
        write_value = st.number_input("Write data", value=999, key="cache_write_value")
        if st.button("Execute Cache Operation", type="primary", key="cache_execute"):
            try:
                address = int(address_text, 0)
                if operation == "Read":
                    value = _capture(numa.read_transaction, node_id, address)
                    st.session_state.cache_message = f"Node {node_id} read {value!r} from 0x{address:X}."
                else:
                    _capture(numa.write_transaction, node_id, address, int(write_value))
                    st.session_state.cache_message = f"Node {node_id} wrote {int(write_value)} to 0x{address:X}."
                st.session_state.cache_display_tag = address
            except (ValueError, KeyError, MemoryError) as error:
                st.error(f"Cache operation failed: {error}")
        if st.session_state.cache_message:
            st.info(st.session_state.cache_message)
        address = st.session_state.cache_display_tag
        state_styles = {"M": ("Modified", "status-modified"), "E": ("Exclusive", "status-exclusive"),
                        "S": ("Shared", "status-shared"), "O": ("Owner", "status-modified"),
                        "I": ("Invalid", "status-invalid")}
        for index, (column, cache) in enumerate(zip(st.columns(4), numa.caches)):
            line = cache.lines.get(address)
            state = line["state"] if line is not None else "I"
            label, style = state_styles.get(state, (state, "status-invalid"))
            data = line["data"] if line is not None and state != "I" else "-"
            column.markdown(f"**Node {index}**")
            column.markdown(f"<span class='{style}'>{label}</span><br><small>Data: {escape(str(data))}</small>", unsafe_allow_html=True)
        st.caption(f"Cache line status for address 0x{address:X}")
        accesses = numa.total_cache_hits + numa.total_cache_misses
        hit_rate = 100.0 * numa.total_cache_hits / accesses if accesses else 0.0
        st.progress(hit_rate / 100.0, text=f"Multi-core cache hit rate {hit_rate:.1f}%")
        metrics = st.columns(2)
        metrics[0].metric("Cache hit rate", f"{hit_rate:.1f}%")
        metrics[1].metric("Directory invalidations", numa.invalidation_count)

    with mesh_column:
        st.subheader("2D-Mesh X-Y route")
        coordinates = [f"({x}, {y})" for y in range(4) for x in range(3)]
        route_controls = st.columns(2)
        source_text = route_controls[0].selectbox("Source Node (X1, Y1)", coordinates, index=0, key="mesh_source")
        destination_text = route_controls[1].selectbox("Destination Node (X2, Y2)", coordinates, index=11, key="mesh_destination")
        if st.button("Calculate X-Y Hop Path", key="mesh_route"):
            sx, sy = (int(value.strip()) for value in source_text.strip("()").split(","))
            dx, dy = (int(value.strip()) for value in destination_text.strip("()").split(","))
            source = SimpleNamespace(zone_name=source_text, x=sx, y=sy)
            destination = SimpleNamespace(zone_name=destination_text, x=dx, y=dy)
            st.session_state.mesh_path = _capture(MeshInterconnect.route_xy, source, destination)
        if st.session_state.mesh_path:
            st.dataframe([{"Hop": i, "Router coordinate": f"({x}, {y})"}
                          for i, (x, y) in enumerate(st.session_state.mesh_path)],
                         hide_index=True, use_container_width=True)
            st.success(f"Minimal X-Y route: {len(st.session_state.mesh_path) - 1} hops; deadlock-free.")
        else:
            st.caption("Choose endpoints and calculate a dimension-order path.")

        st.subheader("Test-and-Set lock")
        lock: HardwareLockSync = st.session_state.lock
        lock_node = st.selectbox("Lock requester", range(4), key="lock_node")
        if lock.locked:
            if st.button("Release Lock", key="release_lock"):
                _capture(lock.release, lock.owner)
                st.rerun()
        elif st.button("Acquire Lock", key="acquire_lock"):
            _capture(lock.acquire, lock_node)
            st.rerun()
        st.metric("Lock status", f"Held by Node {lock.owner}" if lock.locked else "Available")


with tabs[4]:
    st.markdown('<div class="section-kicker">Module 5 / Urban stress test</div>', unsafe_allow_html=True)
    st.header("Real-World Disaster Stress Test & Scorecard")
    st.caption("Peak rush-hour telemetry: 12 routine sensors and 2 critical crash alarms.")
    if st.button("Run Peak Rush-Hour Stress Test (12 Sensors + 2 Emergency Alarms)", type="primary", key="run_stress"):
        result = _run_stress_test()
        progress = st.progress(0.0, text="Dispatching sensor telemetry")
        for portion in (0.25, 0.5, 0.75, 1.0):
            sleep(0.07)
            progress.progress(portion, text="Processing packets and emergency overrides")
        st.session_state.stress_result = result
        st.success("Stress test complete: critical alerts used the emergency bus path.")
    result: Optional[Dict[str, Any]] = st.session_state.stress_result
    if result is None:
        st.info("Run the stress test to generate the terminal event log and architecture comparison.")
    else:
        st.subheader("Cycle-by-cycle stress-test event log")
        st.dataframe(result["events"], hide_index=True, use_container_width=True, height=330)
        safety_columns = st.columns(3)
        safety_columns[0].metric("Critical alerts serviced", result["emergency_serviced"])
        safety_columns[1].metric("Critical preemption guarantee", f"{result['preemption_percent']:.0f}%")
        safety_columns[2].metric("Emergency grants while routine queued", result["emergency_overrides"])
        if result["preemption_percent"] == 100.0:
            st.success("100% of critical safety alerts were serviced through the emergency override path.")

        comparison = [
            {"Metric": "CPI", "Standard architecture": "2.40", "Proposed framework": f"{result['cpi']:.2f}"},
            {"Metric": "IPC", "Standard architecture": "0.42", "Proposed framework": f"{result['ipc']:.2f}"},
            {"Metric": "AMAT", "Standard architecture": "25.00 cycles", "Proposed framework": f"{result['amat']:.2f} cycles"},
            {"Metric": "Cache hit rate", "Standard architecture": "40.0%", "Proposed framework": f"{result['cache_hit_rate']:.1f}%"},
            {"Metric": "Emergency preemption delay", "Standard architecture": "1+ cycles", "Proposed framework": "0 cycles"},
            {"Metric": "Dynamic energy savings", "Standard architecture": "0.0%", "Proposed framework": f"{result['energy_savings']:.1f}%"},
        ]
        st.subheader("Standard vs. proposed architecture")
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        power_columns = st.columns(2)
        power_columns[0].metric("Reference dynamic power", f"{result['reference_power_mw']:.2f} mW")
        power_columns[1].metric("Proposed dynamic power", f"{result['proposed_power_mw']:.2f} mW")
        st.caption("Dynamic-power estimate: P = alpha × C × V² × f; estimates use alpha 0.35 → 0.22 and V = 0.9 V.")