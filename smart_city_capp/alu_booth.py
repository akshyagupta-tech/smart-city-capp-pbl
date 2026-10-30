class CLAAdder:
    @staticmethod
    def add(a, b, bits=16):
        mask = (1 << bits) - 1
        return (a + b) & mask

class BoothRadix4ALU:
    def __init__(self, bit_width=8):
        self.n = bit_width

    def multiply(self, multiplicand_m, multiplier_r):
        m = multiplicand_m
        r = multiplier_r
        acc = 0
        r_extended = (r & ((1 << self.n) - 1)) << 1
        steps = self.n // 2
        
        print(f"\n--- Starting Radix-4 Booth Multiplication (M={m}, R={r}) ---")
        print(f"Total cycles required: {steps} (Traditional shift-add would take {self.n} cycles)")
        
        for i in range(steps):
            bits = (r_extended >> (2 * i)) & 0b111
            factor = 0
            if bits in [0b001, 0b010]:
                factor = 1
            elif bits == 0b011:
                factor = 2
            elif bits == 0b100:
                factor = -2
            elif bits in [0b101, 0b110]:
                factor = -1
            elif bits in [0b000, 0b111]:
                factor = 0

            partial_product = (factor * m) << (2 * i)
            acc += partial_product
            print(f" Cycle {i+1}: Bits={bin(bits)[2:].zfill(3)} -> Factor={factor:+d} | Partial Product={partial_product:+d} | Acc={acc}")

        return acc
