# Copyright (C) 2020-2021 - George Watkins and Alex Nguyen
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
import math
from typing import Dict, List, Optional, cast

import qiskit.exceptions as qkexcept
import qiskit.quantum_info as qkinfo
from qiskit import opflow as qkop


class TraceOverEntireStateException(Exception):
    def __init__(self):
        super().__init__(
            "Won't trace over the entire state, because it can't output a StateFn object"
        )


class StateSeparator:
    """Namespace for functions that deal with separating states."""

    @staticmethod
    def trace_dict_state(state: qkop.DictStateFn, trace_over: List[int]) -> qkop.DictStateFn:
        """Take a state comprised on n qubits and get the trace of the system over the subsystems
        specified by a list of indices.

        Assumes state is separable as a DictStateFn can only represent pure states.
        """
        if not trace_over:
            return state.copy()

        if set(trace_over) == set(range(state.num_qubits)):
            raise TraceOverEntireStateException()

        input_statevector = qkinfo.Statevector(state.to_matrix())
        traced_statevector = qkinfo.partial_trace(input_statevector, trace_over).to_statevector()
        return qkop.DictStateFn(traced_statevector.to_dict())

    @staticmethod
    def trace_to_density_op(state: qkop.DictStateFn, trace_over: List[int]) -> qkinfo.DensityMatrix:
        """Take a state comprised on n qubits and get the trace of the system over the subsystems
        specified by a list of indices.

        Makes no assumption about the separability of the traced subsystems and gives a density
        matrix as a result.
        """
        input_statevector = qkinfo.Statevector(state.to_matrix())
        return qkinfo.partial_trace(input_statevector, trace_over)

    @staticmethod
    def separate(qnum: int, dict_state: qkop.DictStateFn) -> Optional[qkop.DictStateFn]:
        """When a qubit is not entangled (up to a small tolerance) with the rest of the register,
        trace over the rest of the system, giving the qubits' pure state.

        If the selected qubit is entangled return None.
        """
        if dict_state.num_qubits == 1 and qnum == 0:
            return dict_state.copy()

        remaing_qubits = list(range(dict_state.num_qubits))
        remaing_qubits.remove(qnum)

        selected_qubit_maybe_mixed_state = StateSeparator.trace_to_density_op(
            dict_state, remaing_qubits
        )

        try:
            selected_qubit_pure_state = selected_qubit_maybe_mixed_state.to_statevector(
                rtol=10 ** (-10)
            )
            return qkop.DictStateFn(selected_qubit_pure_state.to_dict())

        except qkexcept.QiskitError as e:
            if e.message != "Density matrix is not a pure state":
                raise e
            return None

    @staticmethod
    def get_separable_qubits(dict_state: qkop.DictStateFn) -> Dict[int, qkop.DictStateFn]:
        """For each qubit, numerically detect if it's seprabale or not. If it is, add to
        the result dict, indexed by subsystem, the state traced over the remaining qubits.

        I.e. if a qubit is not entangled with the rest, its state shows up in the result.
        """
        out = {}
        for i in range(dict_state.num_qubits):
            maybe_state = StateSeparator.separate(i, dict_state)
            if maybe_state is not None:
                out[i] = maybe_state
        return out


def to_dict_fn(vector_state: qkop.OperatorBase) -> qkop.DictStateFn:
    # Switch two identical cases to keep mypy happy
    if isinstance(vector_state, qkop.VectorStateFn):
        return cast(qkop.DictStateFn, vector_state.to_dict_fn())
    elif isinstance(vector_state, qkop.SparseVectorStateFn):
        return cast(qkop.DictStateFn, vector_state.to_dict_fn())
    else:
        raise NotImplementedError(f"Conversion to_dict_fn of {repr(type(vector_state))}")


# For some reason, sometimes SparseVectorStateFn a nested vector
# ... maybe it's intended to be a column
def to_vector(state: qkop.OperatorBase):
    if len(state.to_matrix().shape) == 2:
        return state.to_matrix()[0]
    return state.to_matrix()


bell_pair = qkop.DictStateFn({"11": 1 / math.sqrt(2), "00": 1 / math.sqrt(2)})
