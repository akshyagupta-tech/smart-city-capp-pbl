class Instruction:
    def __init__(self, pc, opcode, dest, src1, src2=None, is_branch=False):
        self.pc = pc
        self.opcode = opcode
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
        self.is_branch = is_branch

    def __repr__(self):
        return f"{self.opcode} R{self.dest}, R{self.src1}" + (f", R{self.src2}" if self.src2 is not None else "")

class DynamicBranchPredictor:
    def __init__(self):
        self.state = 0b10

    def predict(self):
        return self.state >= 0b10

    def update(self, actual_taken):
        if actual_taken and self.state < 3:
            self.state += 1
        elif not actual_taken and self.state > 0:
            self.state -= 1

class RISCPipeline5Stage:
    def __init__(self):
        self.predictor = DynamicBranchPredictor()
        self.registers = {f"R{i}": 0 for i in range(8)}
        self.IF_ID = None
        self.ID_EX = None
        self.EX_MEM = None
        self.MEM_WB = None

    def clock_step(self, next_inst=None):
        wb_inst = self.MEM_WB
        if wb_inst:
            print(f"   [WB Stage]  Retiring: {wb_inst}")

        self.MEM_WB = self.EX_MEM
        if self.EX_MEM:
            print(f"   [MEM Stage] Memory bus idle / buffer access: {self.EX_MEM}")

        if self.ID_EX and self.MEM_WB:
            if self.ID_EX.src1 == self.MEM_WB.dest or self.ID_EX.src2 == self.MEM_WB.dest:
                print(f"   [FORWARDING] Forwarded value from WB to EX for {self.ID_EX}")

        self.EX_MEM = self.ID_EX
        if self.ID_EX:
            print(f"   [EX Stage]  Executing ALU operation for: {self.ID_EX}")

        self.ID_EX = self.IF_ID
        if self.IF_ID:
            print(f"   [ID Stage]  Decoded instruction: {self.IF_ID}")

        self.IF_ID = next_inst
        if next_inst:
            print(f"   [IF Stage]  Fetched PC {next_inst.pc}: {next_inst}")
