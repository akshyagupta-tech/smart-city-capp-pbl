import sys
from bus_arbiter import DualTierBusArbiter, Packet
from alu_booth import BoothRadix4ALU
from pipeline import RISCPipeline5Stage, Instruction
from cache_numa import NUMADirectorySystem, HardwareLockSync, MeshInterconnect

def menu():
    print("\n" + "=" * 70)
    print("   INTEGRATED SMART CITY COMPUTING ARCHITECTURE CONTROLLER")
    print("   Course: CCSE 0304 | Topic 84: Smart City Framework (SDG 11)")
    print("=" * 70)
    print("1. [Module 1] Inject Sensor Telemetry or Trigger Emergency Preemption")
    print("2. [Module 2] Interactive Radix-4 Booth's Multiplier Calculator")
    print("3. [Module 3] Trace 5-Stage RISC Pipeline with Hazard Forwarding")
    print("4. [Module 4 & 5] Simulate Multi-Core MESI Cache & NUMA Invalidation")
    print("5. [Module 5] Route Packet via 2D-Mesh (X-Y Dimension Order)")
    print("6. Run Complete Automated Benchmark & Stress Test")
    print("0. Exit")
    print("=" * 70)

def main():
    arbiter = DualTierBusArbiter()
    alu = BoothRadix4ALU(bit_width=8)
    numa = NUMADirectorySystem(num_nodes=4)

    while True:
        menu()
        choice = input("Enter hardware execution choice (0-6): ").strip()
        
        if choice == "1":
            print("\n--- Bus Arbiter Ingestion ---")
            p_type = input("Trigger (R)outine Sensor or (E)mergency Override? [R/E]: ").strip().upper()
            if p_type == "E":
                name = input("Enter Emergency Alert Name (e.g., AMBULANCE_ROUTE_9): ") or "AMBULANCE_DISPATCH"
                arbiter.request_bus(Packet(99, name, is_emergency=True, data="PRIORITY_GREEN_CORRIDOR"))
            else:
                name = input("Enter Routine Sensor Name (e.g., Water_Pump_3): ") or "Routine_Sensor_Node"
                val = input("Enter Telemetry Value (e.g., 85%): ") or "Data_Normal"
                arbiter.request_bus(Packet(1, name, is_emergency=False, data=val))
            
            print("\nClock Step Arbitration:")
            arbiter.arbitrate_cycle()

        elif choice == "2":
            print("\n--- Radix-4 Booth ALU Execution ---")
            try:
                m = int(input("Enter Signed Multiplicand M (-128 to 127): "))
                r = int(input("Enter Signed Multiplier R (-128 to 127): "))
                res = alu.multiply(m, r)
                print(f"\nFinal Calculated Value: {m} * {r} = {res}")
            except ValueError:
                print("Invalid integer entered!")

        elif choice == "3":
            print("\n--- 5-Stage Pipeline Cycle Stepper ---")
            pipe = RISCPipeline5Stage()
            code = [
                Instruction(100, "LOAD", dest=1, src1="Telemetry_Reg"),
                Instruction(104, "ADD",  dest=2, src1=1, src2=3), # RAW Hazard
                Instruction(108, "STORE", dest=4, src1=2),
                None, None
            ]
            for c, inst in enumerate(code, 1):
                input(f"Press Enter to advance to Clock Cycle {c}...")
                pipe.clock_step(inst)

        elif choice == "4":
            print("\n--- NUMA Directory MESI Simulation ---")
            node = int(input("Select Active Node ID (0-3): ") or "0")
            op = input("Select Operation (R)ead or (W)rite? [R/W]: ").strip().upper()
            tag = 0x5A10
            if op == "W":
                v = int(input("Enter updated integer value: ") or "500")
                numa.write_transaction(requester_id=node, tag=tag, val=v)
            else:
                numa.read_transaction(requester_id=node, tag=tag)

        elif choice == "5":
            print("\n--- 2D-Mesh Interconnect Router ---")
            class NodeCoord:
                def __init__(self, name, x, y):
                    self.zone_name = name; self.x = x; self.y = y
            try:
                x1 = int(input("Source X coord (0-3): "))
                y1 = int(input("Source Y coord (0-3): "))
                x2 = int(input("Destination X coord (0-3): "))
                y2 = int(input("Destination Y coord (0-3): "))
                MeshInterconnect.route_xy(NodeCoord(f"Zone({x1},{y1})", x1, y1), NodeCoord(f"Zone({x2},{y2})", x2, y2))
            except ValueError:
                print("Invalid coordinates!")

        elif choice == "6":
            from main import run_real_world_stress_test
            run_real_world_stress_test()

        elif choice == "0":
            print("Exiting simulator...")
            sys.exit(0)

if __name__ == "__main__":
    main()
