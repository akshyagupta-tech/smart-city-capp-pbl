"""Pure-Python arithmetic units for the Smart City architecture simulator."""

from dataclasses import dataclass
from fractions import Fraction
import math
import struct
from typing import List, Optional, Tuple, Union


class CLAAdder:
    """Fixed-width carry-lookahead adder for unsigned bit patterns."""

    @staticmethod
    def add_with_carry(a: int, b: int, bits: int = 16) -> Tuple[int, int]:
        """Return the ``bits``-wide sum and carry-out.

        Negative inputs are interpreted as their two's-complement bit patterns
        at the requested width. Carry-ins are expanded from the generate and
        propagate terms, rather than relying on Python's unbounded addition.
        """
        if bits < 1:
            raise ValueError("bits must be at least 1")

        mask = (1 << bits) - 1
        left = a & mask
        right = b & mask
        propagate = [((left >> index) ^ (right >> index)) & 1 for index in range(bits)]
        generate = [((left >> index) & (right >> index)) & 1 for index in range(bits)]

        carries = [0] * (bits + 1)
        for index in range(1, bits + 1):
            carry = 0
            for source in range(index):
                term = generate[source]
                for higher in range(source + 1, index):
                    term &= propagate[higher]
                carry |= term
            carries[index] = carry

        result = 0
        for index in range(bits):
            result |= (propagate[index] ^ carries[index]) << index
        return result, carries[bits]

    @staticmethod
    def add(a: int, b: int, bits: int = 16) -> int:
        """Return the fixed-width sum, discarding carry-out."""
        return CLAAdder.add_with_carry(a, b, bits)[0]

    @staticmethod
    def subtract(a: int, b: int, bits: int = 16) -> int:
        """Return the fixed-width two's-complement difference ``a - b``."""
        if bits < 1:
            raise ValueError("bits must be at least 1")
        mask = (1 << bits) - 1
        return CLAAdder.add(a, (~b) + 1, bits) & mask


class NonRestoringDivider:
    """Signed divider with a visible non-restoring, bit-serial recurrence."""

    @staticmethod
    def divide(
        dividend: int,
        divisor: int,
        bits: Optional[int] = None,
        show_steps: bool = True,
    ) -> Tuple[int, int]:
        """Return quotient and remainder, truncating quotient toward zero.

        The remainder has the sign of the dividend, matching integer hardware
        division conventions. ``bits`` controls the unsigned quotient scan
        width; when omitted it is sized to represent the dividend. With
        ``show_steps`` enabled, each shift, conditional add/subtract, and
        quotient-bit decision is printed.
        """
        if divisor == 0:
            raise ZeroDivisionError("hardware divider cannot divide by zero")
        scan_bits = bits if bits is not None else max(1, abs(dividend).bit_length())
        if scan_bits < 1:
            raise ValueError("bits must be at least 1")

        dividend_magnitude = abs(dividend)
        divisor_magnitude = abs(divisor)
        if dividend_magnitude >= (1 << scan_bits):
            raise ValueError("dividend does not fit the requested scan width")

        quotient = dividend_magnitude
        remainder = 0
        quotient_mask = (1 << scan_bits) - 1
        work_width = max(scan_bits + 1, divisor_magnitude.bit_length() + 2)
        work_mask = (1 << work_width) - 1

        def signed_binary(value: int) -> str:
            return f"{value & work_mask:0{work_width}b}"

        if show_steps:
            print(
                f"\n--- Non-Restoring Division ({dividend} / {divisor}, "
                f"scan width={scan_bits}) ---"
            )
            print(
                f"Initial: A={signed_binary(0)} Q={quotient:0{scan_bits}b} "
                f"M={divisor_magnitude:b}"
            )

        for cycle in range(1, scan_bits + 1):
            quotient_before = quotient
            next_dividend_bit = (quotient >> (scan_bits - 1)) & 1
            shifted_remainder = (remainder << 1) | next_dividend_bit
            shifted_quotient = (quotient << 1) & quotient_mask
            if shifted_remainder >= 0:
                operation = "A - M"
                remainder = shifted_remainder - divisor_magnitude
            else:
                operation = "A + M"
                remainder = shifted_remainder + divisor_magnitude
            quotient_bit = int(remainder >= 0)
            quotient = shifted_quotient | quotient_bit
            if show_steps:
                print(
                    f"Cycle {cycle}: shift left, input={next_dividend_bit}, "
                    f"A={signed_binary(shifted_remainder)}, "
                    f"Q={shifted_quotient:0{scan_bits}b}; "
                    f"{operation} -> A={signed_binary(remainder)}; "
                    f"Q0={quotient_bit} -> Q={quotient:0{scan_bits}b} "
                    f"(from {quotient_before:0{scan_bits}b})"
                )

        if remainder < 0:
            if show_steps:
                print(
                    f"Final correction: A = A + M -> "
                    f"{signed_binary(remainder + divisor_magnitude)}"
                )
            remainder += divisor_magnitude
        if (dividend < 0) != (divisor < 0):
            quotient = -quotient
        if dividend < 0:
            remainder = -remainder
        if show_steps:
            print(f"Signed result: quotient={quotient}, remainder={remainder}")
        return quotient, remainder


class Q16_16FixedPointDivider:
    """Signed Q16.16 divider using integer non-restoring long division."""

    FRACTION_BITS = 16
    SCALE = 1 << FRACTION_BITS
    WORD_BITS = 32

    @classmethod
    def to_fixed(cls, value: Union[int, float]) -> int:
        """Encode a real scalar into a signed Q16.16 integer."""
        scaled = round(float(value) * cls.SCALE)
        cls._validate_raw(scaled)
        return scaled

    @classmethod
    def from_fixed(cls, value: int) -> float:
        """Decode a signed Q16.16 integer to a Python float."""
        cls._validate_raw(value)
        return value / cls.SCALE

    @classmethod
    def divide_fixed(cls, dividend: int, divisor: int, show_steps: bool = False) -> int:
        """Divide two Q16.16 raw integers and return a Q16.16 raw quotient.

        The remainder recurrence is integer-only. Quotients truncate toward
        zero, and out-of-range signed Q16.16 results raise ``OverflowError``.
        """
        cls._validate_raw(dividend)
        cls._validate_raw(divisor)
        if divisor == 0:
            raise ZeroDivisionError("Q16.16 divider cannot divide by zero")

        numerator = abs(dividend) << cls.FRACTION_BITS
        denominator = abs(divisor)
        scan_bits = max(1, numerator.bit_length())
        quotient_mask = (1 << scan_bits) - 1
        quotient = numerator
        remainder = 0
        if show_steps:
            print(
                f"\n--- Q16.16 Non-Restoring Divide "
                f"({cls.from_fixed(dividend):.6f} / {cls.from_fixed(divisor):.6f}) ---"
            )

        for cycle in range(1, scan_bits + 1):
            input_bit = (quotient >> (scan_bits - 1)) & 1
            shifted_remainder = (remainder << 1) | input_bit
            quotient = (quotient << 1) & quotient_mask
            operation = "A - D" if shifted_remainder >= 0 else "A + D"
            remainder = (
                shifted_remainder - denominator
                if shifted_remainder >= 0
                else shifted_remainder + denominator
            )
            quotient_bit = int(remainder >= 0)
            quotient |= quotient_bit
            if show_steps:
                print(
                    f"Cycle {cycle}: shift input={input_bit}; {operation}; "
                    f"A={remainder}; Q0={quotient_bit}; Q=0x{quotient:X}"
                )

        if remainder < 0:
            remainder += denominator
        if (dividend < 0) != (divisor < 0):
            quotient = -quotient
        cls._validate_raw(quotient)
        if show_steps:
            print(f"Q16.16 quotient raw={quotient}, value={cls.from_fixed(quotient):.6f}")
        return quotient

    @classmethod
    def divide(
        cls,
        dividend: Union[int, float],
        divisor: Union[int, float],
        show_steps: bool = False,
    ) -> float:
        """Divide scalar values and return the decoded Q16.16 result."""
        dividend_raw = cls.to_fixed(dividend)
        divisor_raw = cls.to_fixed(divisor)
        return cls.from_fixed(cls.divide_fixed(dividend_raw, divisor_raw, show_steps))

    @classmethod
    def _validate_raw(cls, value: int) -> None:
        minimum = -(1 << (cls.WORD_BITS - 1))
        maximum = (1 << (cls.WORD_BITS - 1)) - 1
        if not minimum <= value <= maximum:
            raise OverflowError("value does not fit signed Q16.16 range")


FixedPointNonRestoringDivider = Q16_16FixedPointDivider


@dataclass(frozen=True)
class CompressedTelemetry:
    """Encoding and size summary for one telemetry integer."""

    stream_id: int
    encoded_value: int
    is_delta_xor: bool
    raw_length_bytes: int
    compressed_length_bytes: int

    @property
    def bytes_saved(self) -> int:
        """Return the number of bytes saved relative to the raw word."""
        return self.raw_length_bytes - self.compressed_length_bytes


class DeltaXORCompressionUnit:
    """Delta/XOR sensor compressor reporting compact encoded packet lengths."""

    def __init__(self, word_bits: int = 32, header_bytes: int = 1) -> None:
        if word_bits < 1 or header_bytes < 0:
            raise ValueError("word_bits must be positive and header_bytes non-negative")
        self.word_bits = word_bits
        self.header_bytes = header_bytes
        self._previous: Dict[int, int] = {}
        self.packets_encoded = 0
        self.total_raw_bytes = 0
        self.total_compressed_bytes = 0

    def compress(self, value: int, stream_id: int = 0) -> CompressedTelemetry:
        """XOR a sample with its stream predecessor and estimate byte length.

        The first sample is emitted as a full word. Later samples use their
        XOR delta when the header plus significant delta bytes are smaller
        than the raw word; otherwise the raw word is retained.
        """
        mask = (1 << self.word_bits) - 1
        if not -(1 << (self.word_bits - 1)) <= value < (1 << (self.word_bits - 1)):
            raise ValueError(f"value must fit signed {self.word_bits}-bit range")
        raw_length = (self.word_bits + 7) // 8
        previous = self._previous.get(stream_id)
        if previous is None:
            encoded = value & mask
            is_delta = False
            compressed_length = raw_length
        else:
            delta = (value & mask) ^ previous
            payload_length = max(1, (delta.bit_length() + 7) // 8)
            candidate_length = self.header_bytes + payload_length
            is_delta = candidate_length < raw_length
            encoded = delta if is_delta else value & mask
            compressed_length = candidate_length if is_delta else raw_length

        self._previous[stream_id] = value & mask
        self.packets_encoded += 1
        self.total_raw_bytes += raw_length
        self.total_compressed_bytes += compressed_length
        result = CompressedTelemetry(
            stream_id=stream_id,
            encoded_value=encoded,
            is_delta_xor=is_delta,
            raw_length_bytes=raw_length,
            compressed_length_bytes=compressed_length,
        )
        print(
            f"[Delta/XOR] stream={stream_id} encoded=0x{encoded:X} "
            f"mode={'delta' if is_delta else 'raw'} "
            f"packet={compressed_length}/{raw_length} bytes"
        )
        return result

    @property
    def compression_ratio(self) -> float:
        """Return compressed-to-raw byte ratio, or 1.0 before any packets."""
        if not self.total_raw_bytes:
            return 1.0
        return self.total_compressed_bytes / self.total_raw_bytes


@dataclass(frozen=True)
class SaturatingResult:
    """Signed saturating arithmetic result and overflow indicator."""

    value: int
    saturated: bool


class SaturatingArithmeticUnit:
    """Signed fixed-width add/subtract/multiply with two's-complement clamps."""

    def __init__(self, bit_width: int = 16) -> None:
        if bit_width < 1:
            raise ValueError("bit_width must be at least one")
        self.bit_width = bit_width
        self.minimum = -(1 << (bit_width - 1))
        self.maximum = (1 << (bit_width - 1)) - 1
        self.saturation_count = 0

    def _finish(self, exact_value: int) -> SaturatingResult:
        result = min(self.maximum, max(self.minimum, exact_value))
        saturated = result != exact_value
        if saturated:
            self.saturation_count += 1
        return SaturatingResult(result, saturated)

    def add(self, left: int, right: int) -> SaturatingResult:
        """Add signed values and clamp rather than wrap on overflow."""
        return self._finish(left + right)

    def subtract(self, left: int, right: int) -> SaturatingResult:
        """Subtract signed values and clamp rather than wrap on overflow."""
        return self._finish(left - right)

    def multiply(self, left: int, right: int) -> SaturatingResult:
        """Multiply signed values and clamp rather than wrap on overflow."""
        return self._finish(left * right)


@dataclass(frozen=True)
class FMARequest:
    """One fused multiply-add instruction accepted by the pipeline."""

    sequence: int
    multiplicand_a: float
    multiplicand_b: float
    addend_c: float


@dataclass(frozen=True)
class FMAResult:
    """Retired FMA result with its originating sequence number."""

    sequence: int
    value: float


class FusedMultiplyAddPipeline:
    """Three-stage pipelined FMA with exact intermediates and one final rounding.

    ``step`` accepts at most one operation each cycle and returns an operation
    that retires that cycle. A full input rate of one operation per cycle is
    supported after the initial three-cycle latency.
    """

    LATENCY = 3

    def __init__(self) -> None:
        self.cycle = 0
        self._input: Optional[FMARequest] = None
        self._product: Optional[Tuple[FMARequest, Fraction]] = None
        self._sum: Optional[Tuple[int, Fraction]] = None
        self.accepted = 0
        self.retired = 0

    def step(
        self,
        multiplicand_a: Optional[float] = None,
        multiplicand_b: Optional[float] = None,
        addend_c: Optional[float] = None,
    ) -> Optional[FMAResult]:
        """Advance one cycle and optionally inject one ``a*b+c`` operation."""
        supplied = (multiplicand_a, multiplicand_b, addend_c)
        if any(value is None for value in supplied) and not all(value is None for value in supplied):
            raise ValueError("FMA inputs must be supplied together")
        if multiplicand_a is not None and not all(math.isfinite(value) for value in supplied):
            raise ValueError("FMA operands must be finite values")

        self.cycle += 1
        completed: Optional[FMAResult] = None
        if self._sum is not None:
            sequence, exact_sum = self._sum
            try:
                rounded = float(exact_sum)
            except OverflowError:
                rounded = math.inf if exact_sum.numerator >= 0 else -math.inf
            completed = FMAResult(sequence, rounded)
            self.retired += 1

        next_sum: Optional[Tuple[int, Fraction]] = None
        if self._product is not None:
            request, exact_product = self._product
            exact_addend = Fraction.from_float(request.addend_c)
            next_sum = (request.sequence, exact_product + exact_addend)

        next_product: Optional[Tuple[FMARequest, Fraction]] = None
        if self._input is not None:
            request = self._input
            exact_product = (
                Fraction.from_float(request.multiplicand_a)
                * Fraction.from_float(request.multiplicand_b)
            )
            next_product = (request, exact_product)

        next_input = None
        if multiplicand_a is not None:
            self.accepted += 1
            next_input = FMARequest(
                self.accepted,
                float(multiplicand_a),
                float(multiplicand_b),
                float(addend_c),
            )

        self._sum = next_sum
        self._product = next_product
        self._input = next_input
        return completed

    def drain(self) -> List[FMAResult]:
        """Advance empty cycles until all in-flight operations retire."""
        results: List[FMAResult] = []
        while self._input is not None or self._product is not None or self._sum is not None:
            result = self.step()
            if result is not None:
                results.append(result)
        return results


class IEEE754FloatConverter:
    """Convert Python floats to and from IEEE 754 single-precision bit patterns."""

    @staticmethod
    def to_bits(value: float) -> str:
        """Return the IEEE 754 binary32 representation as 32 characters."""
        try:
            packed = struct.pack(">f", value)
        except OverflowError:
            packed = struct.pack(">f", math.copysign(math.inf, value))
        return f"{int.from_bytes(packed, byteorder='big'):032b}"

    @staticmethod
    def from_bits(bits: Union[int, str]) -> float:
        """Decode a 32-bit integer or 32-character binary string as a float."""
        if isinstance(bits, str):
            if len(bits) != 32 or any(character not in "01" for character in bits):
                raise ValueError("bits must be a 32-character binary string")
            bit_pattern = int(bits, 2)
        else:
            bit_pattern = bits
        if not 0 <= bit_pattern < (1 << 32):
            raise ValueError("bits must fit in an unsigned 32-bit word")
        packed = bit_pattern.to_bytes(4, byteorder="big")
        return struct.unpack(">f", packed)[0]

    @staticmethod
    def to_hex(value: float) -> str:
        """Return the IEEE 754 binary32 encoding as eight hexadecimal digits."""
        return f"{int(IEEE754FloatConverter.to_bits(value), 2):08X}"


@dataclass(frozen=True)
class BoothCycle:
    """Trace details for one radix-4 Booth multiplier cycle."""

    cycle: int
    triplet: str
    factor: int
    partial_product: int
    accumulator: int


class Radix4BoothMultiplier:
    """Signed radix-4 Booth multiplier with a cycle-by-cycle execution trace."""

    def __init__(self, bit_width: int = 8) -> None:
        if bit_width < 1:
            raise ValueError("bit_width must be at least 1")
        self.bit_width = bit_width
        self.last_trace: List[BoothCycle] = []

    def multiply(self, multiplicand: int, multiplier: int) -> int:
        """Multiply signed operands and print each recoding/accumulation cycle.

        Both operands must fit in the configured two's-complement width. An
        odd configured width uses a sign-extended final radix-4 group.
        """
        minimum = -(1 << (self.bit_width - 1))
        maximum = (1 << (self.bit_width - 1)) - 1
        if not minimum <= multiplicand <= maximum:
            raise ValueError(f"multiplicand must fit signed {self.bit_width}-bit range")
        if not minimum <= multiplier <= maximum:
            raise ValueError(f"multiplier must fit signed {self.bit_width}-bit range")

        cycles = (self.bit_width + 1) // 2
        recoded_width = cycles * 2
        multiplier_bits = multiplier & ((1 << recoded_width) - 1)
        extended_multiplier = multiplier_bits << 1
        accumulator = 0
        self.last_trace = []

        print(
            f"\n--- Starting Radix-4 Booth Multiplication "
            f"(M={multiplicand}, R={multiplier}, width={self.bit_width}) ---"
        )
        print(f"Total cycles required: {cycles}")

        for cycle_index in range(cycles):
            triplet_value = (extended_multiplier >> (cycle_index * 2)) & 0b111
            triplet = f"{triplet_value:03b}"
            factor = {
                0b000: 0,
                0b001: 1,
                0b010: 1,
                0b011: 2,
                0b100: -2,
                0b101: -1,
                0b110: -1,
                0b111: 0,
            }[triplet_value]
            partial_product = (factor * multiplicand) << (cycle_index * 2)
            accumulator += partial_product
            trace = BoothCycle(
                cycle=cycle_index + 1,
                triplet=triplet,
                factor=factor,
                partial_product=partial_product,
                accumulator=accumulator,
            )
            self.last_trace.append(trace)
            print(
                f" Cycle {trace.cycle}: Bits={triplet} -> Factor={factor:+d} | "
                f"Partial Product={partial_product:+d} | Acc={accumulator:+d}"
            )

        return accumulator


class BoothRadix4ALU(Radix4BoothMultiplier):
    """Compatibility name matching the original ``alu_booth`` module."""