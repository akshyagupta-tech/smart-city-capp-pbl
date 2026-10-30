import tkinter as tk
from tkinter import ttk, messagebox
from bus_arbiter import DualTierBusArbiter, Packet
from alu_booth import BoothRadix4ALU
from pipeline import RISCPipeline5Stage, Instruction
from cache_numa import NUMADirectorySystem, MeshInterconnect

class CompleteSmartCityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CA&PP PBL: Integrated Smart City Architecture Visualizer")
        self.root.geometry("980x760")
        self.root.configure(bg="#0A1128")

        self.arbiter = DualTierBusArbiter()
        self.alu = BoothRadix4ALU(bit_width=8)
        self.pipeline = RISCPipeline5Stage()
        self.numa = NUMADirectorySystem(num_nodes=4)
        self.pipe_cycle = 0

        # Header Title
        title = tk.Label(root, text="INTEGRATED SMART CITY COMPUTING FRAMEWORK", 
                         font=("Helvetica", 17, "bold"), fg="#00E5FF", bg="#0A1128")
        title.pack(pady=(10, 2))
        
        subtitle = tk.Label(root, text="Course: CCSE 0304 | SDG 11: Sustainable Cities | NIET Team 04", 
                            font=("Helvetica", 10), fg="#94A3B8", bg="#0A1128")
        subtitle.pack(pady=(0, 10))

        # Top Control Frame (Modules 1 & 2)
        top_frame = tk.Frame(root, bg="#0A1128")
        top_frame.pack(fill="x", padx=15)

        # Section 1: Module 1 Bus Arbiter
        f_bus = tk.LabelFrame(top_frame, text=" Module 1: Preemptive Bus Arbiter ", 
                              fg="#38BDF8", bg="#132247", font=("Helvetica", 10, "bold"))
        f_bus.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        btn_routine = tk.Button(f_bus, text="Send Routine Water Sensor", bg="#0284C7", fg="black", command=self.send_routine)
        btn_routine.grid(row=0, column=0, padx=6, pady=6)

        btn_emergency = tk.Button(f_bus, text="EMERGENCY FIRE ALARM", bg="#EF4444", fg="black", font=("Helvetica", 9, "bold"), command=self.send_emergency)
        btn_emergency.grid(row=0, column=1, padx=6, pady=6)

        btn_step_bus = tk.Button(f_bus, text="Step Bus Clock", bg="#10B981", fg="black", font=("Helvetica", 9, "bold"), command=self.step_bus)
        btn_step_bus.grid(row=0, column=2, padx=6, pady=6)

        # Section 2: Module 2 Radix-4 Booth ALU
        f_alu = tk.LabelFrame(top_frame, text=" Module 2: Radix-4 Booth ALU ", 
                              fg="#38BDF8", bg="#132247", font=("Helvetica", 10, "bold"))
        f_alu.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        tk.Label(f_alu, text="M:", fg="white", bg="#132247").grid(row=0, column=0, padx=2)
        self.entry_m = tk.Entry(f_alu, width=6)
        self.entry_m.insert(0, "14")
        self.entry_m.grid(row=0, column=1, padx=2)

        tk.Label(f_alu, text="R:", fg="white", bg="#132247").grid(row=0, column=2, padx=2)
        self.entry_r = tk.Entry(f_alu, width=6)
        self.entry_r.insert(0, "-5")
        self.entry_r.grid(row=0, column=3, padx=2)

        btn_calc = tk.Button(f_alu, text="Multiply (Booth)", bg="#00E5FF", fg="black", font=("Helvetica", 9, "bold"), command=self.compute_booth)
        btn_calc.grid(row=0, column=4, padx=6, pady=6)

        # Middle Control Frame (Modules 3, 4, 5)
        mid_frame = tk.Frame(root, bg="#0A1128")
        mid_frame.pack(fill="x", padx=15, pady=5)

        # Section 3: Module 3 Pipeline Stepper
        f_pipe = tk.LabelFrame(mid_frame, text=" Module 3: 5-Stage RISC Pipeline ", 
                               fg="#38BDF8", bg="#132247", font=("Helvetica", 10, "bold"))
        f_pipe.pack(side="left", fill="both", expand=True, padx=5)

        btn_pipe = tk.Button(f_pipe, text="Step Pipeline Clock (Hazard Forwarding)", bg="#F59E0B", fg="black", font=("Helvetica", 9, "bold"), command=self.step_pipeline)
        btn_pipe.pack(padx=8, pady=8)

        # Section 4: Modules 4 & 5 NUMA & Mesh
        f_numa = tk.LabelFrame(mid_frame, text=" Modules 4 & 5: MESI Coherence & 2D-Mesh ", 
                               fg="#38BDF8", bg="#132247", font=("Helvetica", 10, "bold"))
        f_numa.pack(side="right", fill="both", expand=True, padx=5)

        btn_read = tk.Button(f_numa, text="Node 0 Read (Shared)", bg="#A855F7", fg="black", command=self.sim_cache_read)
        btn_read.grid(row=0, column=0, padx=4, pady=8)

        btn_write = tk.Button(f_numa, text="Node 2 Write (Invalidate)", bg="#EC4899", fg="black", command=self.sim_cache_write)
        btn_write.grid(row=0, column=1, padx=4, pady=8)

        btn_mesh = tk.Button(f_numa, text="Route 2D-Mesh (X-Y)", bg="#06B6D4", fg="black", command=self.sim_mesh_route)
        btn_mesh.grid(row=0, column=2, padx=4, pady=8)

        # Bottom Frame: Live Cycle Console Log
        f_log = tk.LabelFrame(root, text=" Real-Time Architectural Activity & Cycle Logs ", 
                              fg="#00E5FF", bg="#132247", font=("Helvetica", 11, "bold"))
        f_log.pack(fill="both", expand=True, padx=15, pady=(5, 12))

        self.txt_log = tk.Text(f_log, bg="#050C1A", fg="#38BDF8", font=("Menlo", 10))
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=8)
        self.log("[System Clock 100MHz Initialized] All Functional Units Online. Ready for Interactive Evaluation.")

    def log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def send_routine(self):
        self.arbiter.request_bus(Packet(1, "Water_Sensor_Node", is_emergency=False, data="Level=72%"))
        self.log(">> [M1] Routine Sensor Queued in Round-Robin Buffer.")

    def send_emergency(self):
        self.arbiter.request_bus(Packet(99, "FIRE_ALARM_DISPATCH", is_emergency=True, data="EVACUATE_ZONE_1"))
        self.log("!! [M1] EMERGENCY INTERRUPT ASSERTED: High-Priority Line Triggered !!")

    def step_bus(self):
        pkt = self.arbiter.arbitrate_cycle()
        if pkt:
            prefix = "EMERGENCY PREEMPTION GRANTED ->" if pkt.is_emergency else "Routine Round-Robin Granted ->"
            self.log(f"[Cycle {self.arbiter.cycle_count}] {prefix} {pkt.name} | Data: {pkt.data}")
        else:
            self.log(f"[Cycle {self.arbiter.cycle_count}] Bus Idle (No pending requests).")

    def compute_booth(self):
        try:
            m = int(self.entry_m.get())
            r = int(self.entry_r.get())
            self.log(f"\n--- [M2] Starting Radix-4 Booth Multiplier: M={m}, R={r} ---")
            res = self.alu.multiply(m, r)
            self.log(f"-> Verified Result: {m} * {r} = {res} (4 Cycles via Radix-4 recoding)")
        except ValueError:
            messagebox.showerror("Error", "Enter valid integers (-128 to 127).")

    def step_pipeline(self):
        self.pipe_cycle += 1
        instructions = [
            Instruction(100, "LOAD", dest=1, src1="Sensor_Reg"),
            Instruction(104, "ADD",  dest=2, src1=1, src2=3), # RAW Hazard
            Instruction(108, "STORE", dest=4, src1=2),
            None, None
        ]
        inst = instructions[self.pipe_cycle - 1] if self.pipe_cycle <= len(instructions) else None
        self.log(f"\n--- [M3 Pipeline] Clock Cycle {self.pipe_cycle} ---")
        self.pipeline.clock_step(inst)

    def sim_cache_read(self):
        self.log("\n--- [M4/M5 NUMA] Memory Read Transaction ---")
        self.numa.read_transaction(requester_id=0, tag=0x3F2A)

    def sim_cache_write(self):
        self.log("\n--- [M4/M5 NUMA] Node 2 Write with Directory Broadcast ---")
        self.numa.write_transaction(requester_id=2, tag=0x3F2A, val=999)

    def sim_mesh_route(self):
        self.log("\n--- [M5 Interconnect] 2D-Mesh Dimension-Order (X-Y) Routing ---")
        class NodePos:
            def __init__(self, name, x, y):
                self.zone_name = name; self.x = x; self.y = y
        MeshInterconnect.route_xy(NodePos("North_Traffic(0,0)", 0, 0), NodePos("East_Substation(2,3)", 2, 3))

if __name__ == "__main__":
    root = tk.Tk()
    app = CompleteSmartCityGUI(root)
    root.mainloop()
