import random

from patches import *

import qiskit.aqua.operators as qk

from typing import *
import math
import cmath
import uuid


class ConvertersToQiskit:
    @staticmethod
    def pauli_op(op: PauliOperator) -> Optional[qiskit.aqua.operators.PrimitiveOp]:
        known_map: Dict[PauliOperator, qiskit.aqua.operators.PrimitiveOp] = {
            PauliOperator.I: qiskit.aqua.operators.I,
            PauliOperator.X: qiskit.aqua.operators.X,
            PauliOperator.Y: qiskit.aqua.operators.Y,
            PauliOperator.Z: qiskit.aqua.operators.Z
        }
        return known_map[op]


    @staticmethod
    def symbolic_state(s:SymbolicState)->qk.StateFn:
        if s == InitializeableState.Zero:
            return qk.Zero
        elif s == InitializeableState.Plus:
            return qk.One
        elif s == InitializeableState.YEigenState:
            return (qk.Zero - 1j*qk.One)/math.sqrt(2)
        elif s == InitializeableState.Magic:
            return (qk.Zero + cmath.exp(1j*math.pi/4)*qk.One)/math.sqrt(2)
        else:
            raise Exception("State cannot be converted to qiskit: "+repr(s))


def circuit_add_op_to_qubit(circ : qk.CircuitOp, op: qk.PrimitiveOp, idx: int) -> qk.CircuitOp:
    """"Take a local operator (applied to a single qubit) and apply it to the given circuit."""
    new_op = op
    if idx > 0:
        new_op = (qk.I ^ idx ) ^ new_op
    if circ.num_qubits-idx-1 > 0:
        new_op = new_op ^ (qk.I ^ (circ.num_qubits - idx - 1))
    return new_op @ circ

class ProjectiveMeasurement:

        class BinaryMeasurementOutcome:
            def __init__(self, resulting_state: qk.DictStateFn, corresponding_eigenvalue: int):
                assert corresponding_eigenvalue in {-1,1}
                self.resulting_state = resulting_state
                self.corresponding_eigenvalue = corresponding_eigenvalue

        @staticmethod
        def borns_rule(projector: qk.PrimitiveOp, state: qk.OperatorBase) -> float:
            # https://qiskit.org/documentation/tutorials/operators/01_operator_flow.html#listop
            compute_states = lambda s: s.to_matrix_op().eval()

            return qk.StateFn(projector).adjoint().eval(compute_states(state))

        @staticmethod
        def compute_outcome_state(projector: qk.PrimitiveOp, state: qk.OperatorBase) -> Tuple[qk.OperatorBase, float]:
            prob = ProjectiveMeasurement.borns_rule(projector, state)
            assert prob.imag < 10 ** (-8)
            prob = prob.real
            state = (projector @ state) / math.sqrt(prob)
            return state, prob

        @staticmethod
        def get_projectors_from_pauli_observable(pauli_observable: qk.OperatorBase) -> Tuple[
            qk.PrimitiveOp, qk.PrimitiveOp]:
            eye = qk.I ^ pauli_observable.num_qubits
            return (eye + pauli_observable) / 2, (eye - pauli_observable) / 2


        @staticmethod
        def pauli_product_measurement_distribution(pauli_observable: qk.OperatorBase, state: qk.OperatorBase) \
                -> Iterable[Tuple[BinaryMeasurementOutcome, float]] :
            p_plus, p_minus = ProjectiveMeasurement.get_projectors_from_pauli_observable(pauli_observable)

            out = []
            for proj, eigenv in [[p_plus, +1], [p_minus, -1]]:
                out_state, prob = ProjectiveMeasurement.compute_outcome_state(proj, state)
                numerical_out_state = out_state.eval()
                if not isinstance(numerical_out_state,qk.DictStateFn):
                    raise Exception("Composed ops do not eval to single state, but to "+str(numerical_out_state))
                out.append((ProjectiveMeasurement.BinaryMeasurementOutcome(numerical_out_state, eigenv), prob))
            return out


T = TypeVar('T')
def proportional_choice(assoc_data_prob : List[Tuple[T, float]]) -> T:
    return random.choices([val for val, prob in assoc_data_prob],
                          weights=[prob for val, prob in assoc_data_prob],
                          k=1)[0]

class PatchToQubitMapper:
    def __init__(self, slices: List[Lattice]):
        self.patch_location_to_logical_idx: Dict[uuid.UUID, int] = dict()
        for slice in slices:
            self._add_patches_from_slice(slice)

    def get_idx(self, patch: uuid.UUID) -> int:
        return self.patch_location_to_logical_idx[patch]

    def max_num_patches(self) ->int:
        return len(self.patch_location_to_logical_idx)

    def get_uuid(self, target_idx: int) -> uuid.UUID:
        for quiid, idx in self.patch_location_to_logical_idx.items():
            if idx==target_idx:
                return quiid

    def enumerate_patches_by_index(self) -> Iterable[Tuple[int, uuid.UUID]]:
        for idx in range(self.max_num_patches()):
            yield (idx, self.get_uuid(idx))

    def _add_patches_from_slice(self, lattice: Lattice):
        for p in PatchToQubitMapper._get_all_operating_patches(lattice.logical_ops):
            if self.patch_location_to_logical_idx.get(p) is None:
                self.patch_location_to_logical_idx[p] = self.max_num_patches()

    @staticmethod
    def _get_all_operating_patches(logical_ops: List[LogicalLatticeOperation]) -> List[uuid.UUID]:
        patch_set = set()
        for op in logical_ops:
            patch_set = patch_set.union(op.get_operating_patches())
        return list(patch_set)



def tensor_list(l):
    t = l[0]
    for s in l[1:]:
        t = t ^ s
    return t


class PatchSimulator:
    def __init__(self, slices: List[Lattice]):
        self.slices = slices
        self.mapper = PatchToQubitMapper(self.slices)
        self.intermediate_states: List[List[qk.StateFn]] = self._simulate_patch_operations(self.slices)


    def _make_initial_logical_state(self):
        """Every patch, when initialized, is considered a new logical qubit.
        So all patch initializations and magic state requests are handled ahead of time"""
        initial_ancilla_states:Dict[uuid.UUID,SymbolicState] = dict()

        def add_initial_ancilla_state(quuid, symbolic_state):
            if initial_ancilla_states.get(quuid) is not None:
                raise Exception("Initializing patch "+str(quuid)+" twice")
            initial_ancilla_states[quuid] = symbolic_state

        def get_init_state(quuid:uuid.UUID) -> qk.StateFn:
            return ConvertersToQiskit.symbolic_state(initial_ancilla_states.get(quuid, InitializeableState.Zero))

        for slice in self.slices:
            for op in slice.logical_ops:
                if isinstance(op, AncillaQubitPatchInitialization):
                    add_initial_ancilla_state(op.qubit_uuid,op.qubit_state)
                elif isinstance(op, MagicStateRequest):
                    add_initial_ancilla_state(op.qubit_uuid,InitializeableState.Magic)

        d = list(self.mapper.enumerate_patches_by_index())
        all_init_states = [get_init_state(quuid) for idx, quuid in self.mapper.enumerate_patches_by_index()]
        return tensor_list(all_init_states)

    def _simulate_patch_operations(self, slices: List[Lattice]) -> List[List[qk.DictStateFn]]:
        """Returns a list of computation states"""
        mapper = PatchToQubitMapper(slices)
        logical_state = self._make_initial_logical_state()
        per_slice_intermediate_logical_states: List[List[qk.DictStateFn]] = [[logical_state]]


        for slice_num, slice in enumerate(slices):
            per_slice_intermediate_logical_states.append([])

            for current_op in slice.logical_ops:

                if not current_op.does_evaluate():
                    continue # Skip conditinal operations that do not execute

                if isinstance(current_op, SinglePatchMeasurement):
                    measure_idx = mapper.get_idx(current_op.qubit_uuid)
                    local_observable = ConvertersToQiskit.pauli_op(current_op.op)
                    global_observable = (qk.I ^ measure_idx) ^ local_observable ^ (
                                qk.I ^ (mapper.max_num_patches() - measure_idx - 1))
                    distribution = ProjectiveMeasurement.pauli_product_measurement_distribution(global_observable,
                                                                                                    logical_state)
                    outcome = proportional_choice(distribution)
                    logical_state = outcome.resulting_state
                    current_op.set_outcome(outcome.corresponding_eigenvalue)

                elif isinstance(current_op, PauliOperator):
                    for patch, op in current_op.patch_pauli_operator_map.items():
                        logical_state = circuit_add_op_to_qubit(logical_state, ConvertersToQiskit.pauli_op(op),
                                                                mapper.get_idx(patch))
                        logical_state = logical_state.eval()  # Convert to DictStateFn

                elif isinstance(current_op, MultiBodyMeasurement):
                    pauli_op_list : List[qk.PrimitiveOp] = []

                    for j,quuid in self.mapper.enumerate_patches_by_index():
                        pauli_op_list.append(current_op.patch_pauli_operator_map.get(quuid, PauliOperator.I))
                    global_observable = tensor_list(list(map(ConvertersToQiskit.pauli_op,pauli_op_list)))
                    distribution = list(ProjectiveMeasurement.pauli_product_measurement_distribution(global_observable,
                                                                                                 logical_state))
                    outcome = proportional_choice(distribution)
                    logical_state = outcome.resulting_state
                    current_op.set_outcome(outcome.corresponding_eigenvalue)



                per_slice_intermediate_logical_states[-1].append(logical_state)

        return per_slice_intermediate_logical_states


