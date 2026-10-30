"""Five-stage RISC pipeline with register forwarding and branch prediction."""

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from types_def import Instruction


def _register_key(register: Optional[object]) -> Optional[str]:
    """Normalize numeric and named register identifiers to file keys."""
    if register is None:
        return None
    name = str(register)
    if name.isdigit():
        return f"R{name}"
    return name.upper() if name.upper().startswith("R") else name


@dataclass
class StageBuffer:
    """Instruction and values carried between adjacent pipeline stages."""

    instruction: Instruction
    src1_value: int = 0
    src2_value: int = 0
    result: Optional[int] = None
    address: Optional[int] = None
    store_value: Optional[int] = None
    branch_prediction: Optional["BranchPrediction"] = None


@dataclass(frozen=True)
class BranchPrediction:
    """Prediction metadata captured when a control-flow instruction is fetched."""

    pc: int
    predicted_taken: bool
    predicted_target: Optional[int]
    btb_hit: bool


@dataclass
class _BTBEntry:
    target: Optional[int]
    counter: int = 0b01


class BranchTargetBuffer:
    """Per-PC target table with a two-bit saturating direction counter.

    Entries begin weakly not-taken. A correctly predicted taken branch avoids
    the configured redirect penalty; a wrong direction or target incurs it.
    """

    def __init__(self, branch_penalty_cycles: int = 2) -> None:
        if branch_penalty_cycles < 0:
            raise ValueError("branch_penalty_cycles cannot be negative")
        self.branch_penalty_cycles = branch_penalty_cycles
        self.entries: Dict[int, _BTBEntry] = {}
        self.predictions = 0
        self.correct_predictions = 0
        self.mispredictions = 0
        self.penalty_cycles_avoided = 0
        self.misprediction_penalty_cycles = 0

    @property
    def branch_penalty_cycles_avoided(self) -> int:
        """Return the cumulative cycles saved by correctly predicted taken branches."""
        return self.penalty_cycles_avoided

    def predict(self, pc: int) -> BranchPrediction:
        """Predict direction and target for the branch at ``pc``."""
        self.predictions += 1
        entry = self.entries.get(pc)
        if entry is None:
            return BranchPrediction(pc, False, None, False)
        taken = entry.counter >= 0b10
        return BranchPrediction(pc, taken, entry.target if taken else None, True)

    def update(
        self,
        pc: int,
        actual_taken: bool,
        actual_target: Optional[int] = None,
        prediction: Optional[BranchPrediction] = None,
    ) -> None:
        """Train the 2-bit counter and account for predicted branch penalties."""
        if prediction is None:
            prediction = self.predict(pc)
        entry = self.entries.setdefault(pc, _BTBEntry(target=actual_target))
        target_correct = not actual_taken or (
            actual_target is not None and prediction.predicted_target == actual_target
        )
        correct = prediction.predicted_taken == actual_taken and target_correct
        if correct:
            self.correct_predictions += 1
            if actual_taken:
                self.penalty_cycles_avoided += self.branch_penalty_cycles
        else:
            self.mispredictions += 1
            self.misprediction_penalty_cycles += self.branch_penalty_cycles

        if actual_taken:
            entry.counter = min(0b11, entry.counter + 1)
            if actual_target is not None:
                entry.target = actual_target
        else:
            entry.counter = max(0, entry.counter - 1)


class HardwareWatchdogTimer:
    """Detect an instruction held in a stall beyond its allowed cycle budget."""

    def __init__(self, timeout_cycles: int = 32) -> None:
        if timeout_cycles < 1:
            raise ValueError("timeout_cycles must be at least 1")
        self.timeout_cycles = timeout_cycles
        self.stalled_cycles = 0
        self.instruction_token: Optional[object] = None

    def observe(self, instruction_token: object, stalled: bool) -> bool:
        """Observe a cycle and return true once a stall exceeds the limit."""
        if not stalled:
            self.reset()
            return False
        if instruction_token != self.instruction_token:
            self.instruction_token = instruction_token
            self.stalled_cycles = 0
        self.stalled_cycles += 1
        return self.stalled_cycles > self.timeout_cycles

    def reset(self) -> None:
        """Clear the current stall interval."""
        self.stalled_cycles = 0
        self.instruction_token = None


@dataclass(frozen=True)
class DVFSState:
    """Frequency and relative dynamic-power operating point."""

    temperature_c: float
    frequency_ghz: float
    power_scale: float
    throttled: bool


class ThermalDVFSController:
    """Step core frequency/power down above a thermal limit with hysteresis."""

    def __init__(
        self,
        threshold_c: float = 85.0,
        hysteresis_c: float = 5.0,
        base_frequency_ghz: float = 2.0,
        minimum_frequency_ghz: float = 0.5,
        frequency_step_ghz: float = 0.25,
    ) -> None:
        if not 0 < minimum_frequency_ghz <= base_frequency_ghz:
            raise ValueError("frequency range must satisfy 0 < minimum <= base")
        if frequency_step_ghz <= 0 or hysteresis_c < 0:
            raise ValueError("frequency step must be positive and hysteresis non-negative")
        self.threshold_c = threshold_c
        self.hysteresis_c = hysteresis_c
        self.base_frequency_ghz = base_frequency_ghz
        self.minimum_frequency_ghz = minimum_frequency_ghz
        self.frequency_step_ghz = frequency_step_ghz
        self.frequency_ghz = base_frequency_ghz
        self.power_scale = 1.0
        self.transition_count = 0

    def update_temperature(self, temperature_c: float) -> DVFSState:
        """Apply one DVFS step if hot, or recover one step below hysteresis."""
        previous_frequency = self.frequency_ghz
        if temperature_c > self.threshold_c:
            self.frequency_ghz = max(
                self.minimum_frequency_ghz,
                self.frequency_ghz - self.frequency_step_ghz,
            )
        elif temperature_c < self.threshold_c - self.hysteresis_c:
            self.frequency_ghz = min(
                self.base_frequency_ghz,
                self.frequency_ghz + self.frequency_step_ghz,
            )
        if self.frequency_ghz != previous_frequency:
            self.transition_count += 1
        self.power_scale = self.frequency_ghz / self.base_frequency_ghz
        return DVFSState(
            temperature_c=temperature_c,
            frequency_ghz=self.frequency_ghz,
            power_scale=self.power_scale,
            throttled=self.frequency_ghz < self.base_frequency_ghz,
        )


class DynamicBranchPredictor:
    """Two-bit saturating counter predictor, initialized weakly taken."""

    def __init__(self, initial_state: int = 0b10) -> None:
        if not 0 <= initial_state <= 0b11:
            raise ValueError("initial_state must be between 0 and 3")
        self.state = initial_state

    def predict(self) -> bool:
        """Predict taken in either of the two taken states."""
        return self.state >= 0b10

    def update(self, actual_taken: bool) -> None:
        """Move the saturating counter one state toward the actual outcome."""
        if actual_taken:
            self.state = min(0b11, self.state + 1)
        else:
            self.state = max(0, self.state - 1)


class ForwardingUnit:
    """Resolve RAW dependencies using the nearest available pipeline result."""

    @staticmethod
    def _writes_register(buffer: Optional[StageBuffer]) -> bool:
        if buffer is None or buffer.instruction.dest is None or buffer.result is None:
            return False
        return buffer.instruction.opcode.upper() not in {"STORE", "BEQ", "BNE", "J", "JMP", "NOP"}

    @staticmethod
    def resolve(
        register: Optional[str],
        value: int,
        mem_stage: Optional[StageBuffer],
        wb_stage: Optional[StageBuffer],
    ) -> Tuple[int, Optional[str]]:
        """Return a forwarded value and source label, or the original value.

        MEM has priority over WB because it contains the younger producer.
        """
        if register is None:
            return value, None
        for source_name, source in (("MEM", mem_stage), ("WB", wb_stage)):
            if (
                ForwardingUnit._writes_register(source)
                and source.instruction.dest is not None
                and _register_key(source.instruction.dest) == _register_key(register)
            ):
                return int(source.result), source_name
        return value, None


class RISCPipeline5Stage:
    """Cycle-stepped IF/ID/EX/MEM/WB integer pipeline.

    ``clock_step`` accepts at most one fetched instruction per cycle. The
    pipeline advances every cycle and resolves RAW dependencies through MEM
    and WB forwarding paths, so ordinary ALU and load-use dependencies do not
    incur a stall in this functional timing model.
    """

    def __init__(self) -> None:
        self.predictor = DynamicBranchPredictor()
        self.btb = BranchTargetBuffer()
        self._branch_predictions: Dict[int, Deque[BranchPrediction]] = defaultdict(deque)
        self.registers: Dict[str, int] = {f"R{index}": 0 for index in range(32)}
        self.memory: Dict[int, int] = {}
        self.IF_ID: Optional[Instruction] = None
        self.ID_EX: Optional[StageBuffer] = None
        self.EX_MEM: Optional[StageBuffer] = None
        self.MEM_WB: Optional[StageBuffer] = None
        self.cycle_count = 0
        self.stall_cycles = 0
        self.instructions_retired = 0
        self.last_forwarding: Dict[str, str] = {}
        self.watchdog = HardwareWatchdogTimer(timeout_cycles=32)
        self.micro_rollback_count = 0
        self.last_rollback_pc: Optional[int] = None
        self._last_stall_cycle_count = 0
        self.thermal_controller = ThermalDVFSController()
        self.temperature_c = 45.0
        self.core_frequency_ghz = self.thermal_controller.base_frequency_ghz
        self.dynamic_power_scale = 1.0

    def monitor_temperature(self, temperature_c: float) -> DVFSState:
        """Apply thermal policy to a supplied simulated core temperature."""
        self.temperature_c = temperature_c
        state = self.thermal_controller.update_temperature(temperature_c)
        self.core_frequency_ghz = state.frequency_ghz
        self.dynamic_power_scale = state.power_scale
        return state

    def _micro_rollback(self, instruction_token: object) -> None:
        """Flush volatile pipeline state to the last architectural boundary."""
        self.last_rollback_pc = instruction_token if isinstance(instruction_token, int) else None
        self.micro_rollback_count += 1
        self.IF_ID = None
        self.ID_EX = None
        self.EX_MEM = None
        self.MEM_WB = None
        self._branch_predictions.clear()
        self.watchdog.reset()
        self._last_stall_cycle_count = 0
        print(
            f"[Watchdog] Stall exceeded {self.watchdog.timeout_cycles} cycles; "
            f"micro-rollback flushed transient pipeline state (token={instruction_token})"
        )

    @staticmethod
    def _register_name(register: Optional[object]) -> Optional[str]:
        return _register_key(register)

    def _read_register(self, register: Optional[object]) -> int:
        name = self._register_name(register)
        return self.registers.get(name, 0) if name is not None else 0

    def _writeback_stage(self, buffer: StageBuffer) -> None:
        instruction = buffer.instruction
        self.instructions_retired += 1
        if not ForwardingUnit._writes_register(buffer):
            return
        destination = self._register_name(instruction.dest)
        if destination is not None:
            self.registers[destination] = int(buffer.result)

    def _memory_stage(self, buffer: StageBuffer) -> StageBuffer:
        instruction = buffer.instruction
        opcode = instruction.opcode.upper()
        if opcode == "LOAD":
            address = buffer.address if buffer.address is not None else 0
            return StageBuffer(instruction, buffer.src1_value, buffer.src2_value,
                               result=self.memory.get(address, 0), address=address)
        if opcode == "STORE":
            address = buffer.address if buffer.address is not None else 0
            self.memory[address] = buffer.store_value if buffer.store_value is not None else 0
        return buffer

    def _execute_stage(
        self,
        buffer: StageBuffer,
        mem_stage: Optional[StageBuffer],
        wb_stage: Optional[StageBuffer],
    ) -> StageBuffer:
        instruction = buffer.instruction
        opcode = instruction.opcode.upper()
        src1_name = self._register_name(instruction.src1)
        src2_name = self._register_name(instruction.src2)
        src1, src1_source = ForwardingUnit.resolve(
            src1_name, buffer.src1_value, mem_stage, wb_stage
        )
        src2, src2_source = ForwardingUnit.resolve(
            src2_name, buffer.src2_value, mem_stage, wb_stage
        )
        if src1_source:
            self.last_forwarding["src1"] = src1_source
        if src2_source:
            self.last_forwarding["src2"] = src2_source

        if opcode == "ADD":
            result = src1 + src2
        elif opcode == "SUB":
            result = src1 - src2
        elif opcode == "MUL":
            result = src1 * src2
        elif opcode == "AND":
            result = src1 & src2
        elif opcode == "OR":
            result = src1 | src2
        elif opcode == "XOR":
            result = src1 ^ src2
        elif opcode == "LOAD":
            return StageBuffer(instruction, src1, src2, address=src1)
        elif opcode == "STORE":
            address = src2 if instruction.src2 is not None else int(instruction.dest or 0)
            return StageBuffer(instruction, src1, src2, address=address, store_value=src1)
        elif opcode in {"BEQ", "BNE", "J", "JMP"} or instruction.is_branch:
            if opcode in {"J", "JMP"}:
                taken = True
            else:
                taken = src1 == src2 if opcode != "BNE" else src1 != src2
            self.predictor.update(taken)
            self.btb.update(
                instruction.pc,
                taken,
                instruction.branch_target,
                buffer.branch_prediction,
            )
            return StageBuffer(instruction, src1, src2)
        else:
            result = None

        return StageBuffer(instruction, src1, src2, result=result)

    def clock_step(
        self,
        next_inst: Optional[Instruction] = None,
        stalled: bool = False,
    ) -> None:
        """Advance one cycle, or hold the pipeline while an external stall persists.

        A stall holds all pipeline buffers without allowing younger work to
        execute. On the 33rd consecutive stalled cycle for one instruction,
        the watchdog flushes transient buffers and returns to a clean boundary.
        """
        self.cycle_count += 1
        self.last_forwarding = {}

        active_instruction = self.ID_EX or self.IF_ID
        instruction_token: object = (
            active_instruction.instruction.pc
            if isinstance(active_instruction, StageBuffer)
            else active_instruction.pc
            if isinstance(active_instruction, Instruction)
            else next_inst.pc
            if next_inst is not None
            else "external-stall"
        )
        if stalled:
            self.stall_cycles += 1
            timed_out = self.watchdog.observe(instruction_token, True)
            if timed_out:
                self._micro_rollback(instruction_token)
            elif self.watchdog.stalled_cycles == 1:
                print(f"[Pipeline] Holding at token {instruction_token} (watchdog armed)")
            return
        self.watchdog.observe(instruction_token, False)

        active = any((next_inst, self.IF_ID, self.ID_EX, self.EX_MEM, self.MEM_WB))
        self.temperature_c += 0.20 if active else -0.05
        self.temperature_c = max(25.0, self.temperature_c)
        dvfs_state = self.monitor_temperature(self.temperature_c)
        if dvfs_state.throttled:
            print(
                f"[DVFS] {self.temperature_c:.1f} C: core at "
                f"{self.core_frequency_ghz:.2f} GHz, "
                f"power scale {self.dynamic_power_scale:.2f}"
            )

        old_if_id = self.IF_ID
        old_id_ex = self.ID_EX
        old_ex_mem = self.EX_MEM
        old_mem_wb = self.MEM_WB

        if old_mem_wb is not None:
            self._writeback_stage(old_mem_wb)
            print(f"   [WB Stage]  Retiring: {old_mem_wb.instruction}")

        next_mem_wb = self._memory_stage(old_ex_mem) if old_ex_mem is not None else None
        if old_ex_mem is not None:
            print(f"   [MEM Stage] Memory access: {old_ex_mem.instruction}")

        next_ex_mem = None
        if old_id_ex is not None:
            next_ex_mem = self._execute_stage(old_id_ex, next_mem_wb, old_mem_wb)
            print(f"   [EX Stage]  Executing: {old_id_ex.instruction}")
            if self.last_forwarding:
                sources = ", ".join(
                    f"{operand} from {stage}"
                    for operand, stage in self.last_forwarding.items()
                )
                print(f"   [FORWARDING] {sources} for {old_id_ex.instruction}")

        next_id_ex = None
        if old_if_id is not None:
            branch_prediction = None
            if self._is_control_flow(old_if_id):
                predictions = self._branch_predictions[old_if_id.pc]
                if predictions:
                    branch_prediction = predictions.popleft()
            next_id_ex = StageBuffer(
                old_if_id,
                src1_value=self._read_register(old_if_id.src1),
                src2_value=self._read_register(old_if_id.src2),
                branch_prediction=branch_prediction,
            )
            print(f"   [ID Stage]  Decoded: {old_if_id}")

        self.MEM_WB = next_mem_wb
        self.EX_MEM = next_ex_mem
        self.ID_EX = next_id_ex
        self.IF_ID = next_inst
        if next_inst is not None:
            if self._is_control_flow(next_inst):
                self._branch_predictions[next_inst.pc].append(self.btb.predict(next_inst.pc))
            print(f"   [IF Stage]  Fetched PC {next_inst.pc}: {next_inst}")

    @staticmethod
    def _is_control_flow(instruction: Instruction) -> bool:
        """Return whether an instruction participates in control flow."""
        return instruction.is_branch or instruction.opcode.upper() in {"BEQ", "BNE", "J", "JMP"}


__all__ = [
    "DynamicBranchPredictor",
    "BranchPrediction",
    "BranchTargetBuffer",
    "DVFSState",
    "ForwardingUnit",
    "HardwareWatchdogTimer",
    "RISCPipeline5Stage",
    "StageBuffer",
    "ThermalDVFSController",
]