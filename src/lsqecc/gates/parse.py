"""
Helper methods to parse qasm circuits.
"""

from fractions import Fraction
from typing import List, Sequence, Tuple

from lsqecc.gates import gates
from lsqecc.utils import QasmParseException


def get_index_arg(qreg_arg: str) -> int:
    return int(qreg_arg.split("[")[1].split("]")[0])


def split_instruciton_and_args(line: str) -> Tuple[str, List[str]]:
    if " " not in line:
        return line, []
    return line.split(" ")[0], line.split(" ")[1].split(",")


def parse_trivial_gate(instruction: str, args: List[str]) -> gates.Gate:
    if instruction == "h":
        return gates.H(get_index_arg(args[0]))
    elif instruction == "x":
        return gates.X(get_index_arg(args[0]))
    elif instruction == "s":
        return gates.S(get_index_arg(args[0]))
    elif instruction == "t":
        return gates.T(get_index_arg(args[0]))
    else:
        raise QasmParseException(f"Not a trivial gate: {instruction}")


def parse_gates_circuit(qasm: str) -> Sequence[gates.Gate]:

    instructions: List[Tuple[str, List[str]]] = list(
        map(split_instruciton_and_args, qasm.split(";\n"))
    )

    qregs = list(filter(lambda line: line[0] == "qreg", instructions))
    if len(qregs) != 1:
        raise QasmParseException(f"Need exactly one qreg, got {len(qregs)}")

    instructions = list(
        filter(lambda line: line[0] not in {"OPENQASM", "include", "barrier", "qreg"}, instructions)
    )

    ret_gates: List[gates.Gate] = []

    for instruction, args in instructions:
        if instruction and instruction in "hxzst":
            ret_gates.append(parse_trivial_gate(instruction, args))
        elif instruction[0:2] == "rz":
            if instruction[2:6] != "(pi/":
                raise QasmParseException(
                    f"Can only parse pi/n for n power of 2 angles as rz args, " f"got {instruction}"
                )
            phase_pi_frac_den = int(instruction[6:].split(")")[0])
            ret_gates.append(gates.RZ(get_index_arg(args[0]), Fraction(1, phase_pi_frac_den)))
        elif instruction[0:3] == "crz":
            if instruction[3:7] != "(pi/":
                raise QasmParseException(
                    f"Can only parse pi/n for n power of 2 angles as rz args, " f"got {instruction}"
                )
            phase_pi_frac_den = int(instruction[7:].split(")")[0])
            ret_gates.append(
                gates.CRZ(
                    control_qubit=get_index_arg(args[0]),
                    target_qubit=get_index_arg(args[1]),
                    phase=Fraction(1, phase_pi_frac_den),
                )
            )
        elif not instruction and not args:
            pass
        else:
            raise QasmParseException(f"Instruction {instruction} with args {args} not implemented")

    return ret_gates
