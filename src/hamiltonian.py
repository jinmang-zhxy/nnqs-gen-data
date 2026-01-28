import itertools
import torch
import numpy as np
from typing import Union
from torch import nn

import src.utils.complex_helper as cplx
from src.utils.utils import read_binary_qubit_op, calculate_total_memory

try:
    import interface.python.eloc as eloc
except ImportError:
    pass

class Hamiltonian(nn.Module):
    def __init__(self, ham_path: str):
        super().__init__()
        self.n_qubits = -1

    def get_n_qubits(self):
        return self.n_qubits

    def calculate_local_energy(self, wf, states: Union[torch.tensor, np.ndarray], ln_psis: torch.tensor=None, *args, **kwargs):
        raise NotImplementedError()

    def free_hamiltonian(self):
        pass

class MolecularHamiltonianExactOpt(Hamiltonian):
    def __init__(self, ham_path: str, device="cpu", qubit_dtype=torch.int8):
        self.qubit_dtype = qubit_dtype
        self.n_qubits, ham_ops = read_binary_qubit_op(ham_path)
        operators, coefficients = self._parse_hamiltonian_string(ham_ops.__str__(), self.n_qubits)
        self.num_terms, self.input_dim = operators.shape
        assert coefficients.shape[0] == self.num_terms
        # product of identity operators by default, encoded as 0
        operators = torch.tensor(operators)
        self.coefficients = torch.tensor(coefficients)
        self.coefficients = torch.stack((self.coefficients.real, self.coefficients.imag), dim=-1)
        # find index of pauli X,Y,Z operators
        pauli_x_idx = (operators==1).to(self.qubit_dtype) # [num_terms, input_dim]
        pauli_y_idx = (operators==2).to(self.qubit_dtype) # [num_terms, input_dim]
        pauli_z_idx = (operators==3).to(self.qubit_dtype) # [num_terms, input_dim]
        del operators
        # track the exponential of -i
        self.num_pauli_y = pauli_y_idx.sum(-1) # [num_terms]
        # the unique element has flipped value if the corresponding pauli is x or y.
        flip_idx = pauli_x_idx + pauli_y_idx # [num_terms, input_dim]
        # self.flip_idx = flip_idx
        del pauli_x_idx
        # only the entry value with y or z pauli is multiplied
        self.select_idx = pauli_y_idx + pauli_z_idx
        del pauli_y_idx
        del pauli_z_idx
        unique_flips, unique_indices = np.unique(np.array(flip_idx), axis=0, return_inverse=True)
        self.unique_flips = torch.tensor(unique_flips).to(self.qubit_dtype)
        self.unique_indices = torch.tensor(unique_indices)
        self.unique_num_terms = self.unique_flips.shape[0]
        self.device = device
        self.set_device(device=device)
        print(f"[Hamiltonian] Number of terms is {self.num_terms} uniq_terms: {self.unique_num_terms}")
        total_bytes = calculate_total_memory([self.unique_flips, self.unique_indices, self.coefficients, self.select_idx, self.num_pauli_y])
        print(f"[Hamiltonian] total_bytes: {total_bytes} Byte({total_bytes/1000/1000} MB)")

    @torch.no_grad()
    def calculate_local_energy(self, wf, states: Union[torch.tensor, np.ndarray], is_permutation: bool=False, eloc_split_bs=512, **kwargs):
        r"""
        Args:
            wf: wavefunction
            states: samples
            is_permutation: whether to reverse the state order 
            eloc_split_bs: splited batch size of states to calculate local energy
        Return:
            elocs: local energies for states, with type np.ndarray[np.complex]
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        states = states.to(self.device)

        split_bs = eloc_split_bs
        split_states = torch.split(states, split_bs)
        # print(f"states num blocks: {len(split_states)} split_bs: {split_bs}")
        local_energy = []
        for i, states_i in enumerate(split_states):
            local_energy_i = self.calculate_local_energy_single(wf, states_i, is_permutation=is_permutation)
            local_energy.append(local_energy_i)
        local_energy = np.concatenate(local_energy, axis=0)
        return local_energy

    @torch.no_grad()
    def calculate_local_energy_single(self, wf, states: Union[torch.tensor, np.ndarray], is_permutation: bool=False):
        # see appendix B of https://arxiv.org/pdf/1909.12852.pdf
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        states = states.to(self.device)

        # x [bs, input_dim]
        # determine the unique element
        x = states * (-1) # ATTENTION: here -1 is occupy electron
        x = x.type(self.qubit_dtype)
        bs = x.shape[0]
        # coupled states
        x_k = x.unsqueeze(1) * (self.unique_flips.unsqueeze(0)*(-2) + 1) # [bs, unique_num_terms, input_dim]

        # inference: ln_psi_or_psi = wf(x_k)
        kwargs = dict() if is_permutation is None else dict(permutation=is_permutation)
        X = torch.cat((x_k.reshape(-1, self.input_dim), x))
        X = (-1) * X # ATTENTION: here +1 is occ electron
        # output = wf.ln_psi(X, **kwargs) if wf.is_complex_wf() else wf.psi(X, **kwargs)
        X_uniq, X_uniq_revidx = torch.unique(X, dim=0, return_inverse=True) # only infer for the unique states
        output_uniq = wf.ln_psi(X_uniq, **kwargs) if wf.is_complex_wf() else wf.psi(X_uniq, **kwargs)
        output = output_uniq[X_uniq_revidx]

        log_psi_k, log_psi = output[:-bs], output[-bs:]

        if wf.is_complex_wf():    
            log_psi_k = log_psi_k.reshape(bs, self.unique_num_terms, 2) # [bs, unique_num_terms, 2]
            log_psi_k = log_psi_k[:, self.unique_indices] # [bs, num_terms, 2]
            ratio = cplx.exp(log_psi_k-log_psi.unsqueeze(1)) # [bs, num_terms, 2], unrestricted states's prob. = 0
        else:
            log_psi_k = log_psi_k.reshape(bs, self.unique_num_terms, 1) # [bs, unique_num_terms, 1]
            log_psi_k[torch.isinf(log_psi_k)] = 0 # -inf/+inf -> 0, unrestricted states's prob. = 0
            log_psi_k = log_psi_k[:, self.unique_indices] # [bs, num_terms, 1]
            ratio = log_psi_k / log_psi.view(bs, 1, 1) # [bs, num_terms, 1]
            # construct complex for general return interface
            zero_padding = torch.zeros_like(ratio)
            ratio = torch.cat([ratio, zero_padding], dim=-1)

        # compute matrix element: coupled coefficients
        # Eq. B3
        part2 = (x.unsqueeze(1).repeat(1, self.num_terms, 1) * self.select_idx.unsqueeze(0) + (1-self.select_idx).unsqueeze(0)).prod(-1) # [bs, num_terms, input_dim]
        part2 = torch.stack((part2, torch.zeros_like(part2)), dim=-1)
        part1 = (1j)**self.num_pauli_y.detach().cpu().numpy()
        part1 = torch.stack((torch.tensor(part1.real), torch.tensor(part1.imag)), dim=-1).float().to(x.device)
        mtx_k = cplx.scalar_mult(part1, part2) # [bs, num_terms, 2]

        # total local energy
        local_energy = cplx.scalar_mult(self.coefficients.unsqueeze(0), cplx.scalar_mult(mtx_k, ratio)).sum(1) # [bs, 2]
        local_energy = cplx.torch_to_numpy(local_energy.cpu())
        return local_energy

    def set_device(self, device):
        self.coefficients = self.coefficients.to(device)
        self.num_pauli_y = self.num_pauli_y.to(device)
        self.select_idx = self.select_idx.to(device)
        self.unique_flips = self.unique_flips.to(device)
        self.unique_indices = self.unique_indices.to(device)

    # def free_hamiltonian(self):
    #     del self.coefficients
    #     del self.num_pauli_y
    #     del self.select_idx
    #     del self.unique_flips
    #     del self.unique_indices

    def _parse_hamiltonian_string(self, hamiltonian_string, num_sites, **kwargs):
        splitted_string = hamiltonian_string.split('+\n')
        num_terms = len(splitted_string)
        params = np.zeros([num_terms]).astype(np.complex128)
        hmtn_ops = np.zeros([num_terms, num_sites])
        for i,term in enumerate(splitted_string):
            params[i] = complex(term.split(' ')[0])
            ops = term[term.index('[')+1:term.index(']')]
            ops_lst = ops.split(' ')
            for op in ops_lst:
                if op == '':
                    continue
                pauli_type = op[0]
                idx = int(op[1:])
                if pauli_type == 'X':
                    encoding = 1
                elif pauli_type == 'Y':
                    encoding = 2
                elif pauli_type == 'Z':
                    encoding = 3
                elif pauli_type == 'I':
                    encoding = 0
                else:
                    raise "Unknown pauli_type!"
                hmtn_ops[i, idx] = encoding
        return hmtn_ops, params

def laplacian_to_mtx(laplacian):
    size = laplacian.shape[0]
    coef = []
    mtx = []
    for i in range(size):
        for j in range(size):
            if i == j:
                coef.append(-laplacian[i,j]/4)
                ops = np.zeros(size)
                ops[i] = 0
                mtx.append(ops)
            elif laplacian[i,j] != 0:
                coef.append(-laplacian[i,j]/4)
                ops = np.zeros(size)
                ops[i] = 3
                ops[j] = 3
                mtx.append(ops)
    return np.array(mtx), np.array(coef)

def tim_to_mtx(h, g, gg):
    size = gg.shape[0]
    coef = []
    mtx = []
    for i in range(size):
        for j in range(size):
            coef.append(gg[i,j])
            ops = np.zeros(size)
            ops[i] = 3
            ops[j] = 3
            mtx.append(ops)
    for i in range(size):
        coef.append(g[i])
        ops = np.zeros(size)
        ops[i] = 3
        mtx.append(ops)
    for i in range(size):
        coef.append(h[i])
        ops = np.zeros(size)
        ops[i] = 1
        mtx.append(ops)
    return np.array(mtx), np.array(coef)

def dense_hamiltonian(num_sites, hmtn_ops, coefs, unique_flips, select_idx, num_pauli_y):
    inputs = torch.tensor(np.array(list(itertools.product([0, 1], repeat=num_sites)))) * 2.0 - 1.0
    num_terms = coefs.shape[0]
    assert num_terms == hmtn_ops.shape[0]
    assert num_sites == hmtn_ops.shape[1]
    size = inputs.shape[0]
    mtx = np.zeros((size, size), dtype=np.complex128)
    pauli_x_idx = (hmtn_ops==1).int()
    pauli_y_idx = (hmtn_ops==2).int()
    pauli_z_idx = (hmtn_ops==3).int()
    for i in range(size):
        print(i)
        x = inputs[i]
        for j in range(size):
            xx = inputs[j]
            num_pauli_y = pauli_y_idx.sum(-1)
            flip_idx = pauli_x_idx + pauli_y_idx
            select_idx = pauli_y_idx + pauli_z_idx
            cond = ((x.unsqueeze(0) * (flip_idx*(-2)+1)) == xx.unsqueeze(0)).prod(-1).detach().cpu().numpy()
            xx_prod = x.unsqueeze(0).repeat(num_terms, 1) * select_idx
            xx_prod += 0.9 # for efficiency, we only care about the sign here
            val = (1j)**num_pauli_y.detach().cpu().numpy() * xx_prod.prod(-1).sign().detach().cpu().numpy()
            mtx[i,j] = (cond * val * coefs.detach().cpu().numpy()).sum()
    return mtx

class MolecularHamiltonianExact(Hamiltonian):
    def __init__(self, ham_path: str, device='cpu'):
        self.n_qubits, self.hamiltonian_qubit_op = read_binary_qubit_op(ham_path)
        self.device = device
        self._apply_dict = {
            ("X", -1): (+1.0, 1),
            ("X", 1): (+1.0, -1),
            ("Y", -1): (+1.j, 1),
            ("Y", 1): (-1.j, -1),
            ("Z", -1): (+1.0, -1),
            ("Z", 1): (-1.0, 1)
        }

    @torch.no_grad()
    def _coupled_states(self, state: torch.tensor):
        L = len(state)
        c_states = torch.ones_like(state, device=self.device).reshape(1, -1)
        c_coefs = torch.zeros((1,), device=self.device)

        for pauli_str, coeff in self.hamiltonian_qubit_op.terms.items():
            coeff_i = torch.tensor([coeff], device=self.device)
            new_state = state.detach().clone().reshape(1, -1).to(self.device)
            for (target_qubit_, pauli_symbol) in pauli_str:
                target_qubit = target_qubit_  # assuming target_qubit_ is already adjusted if necessary
                new_coeff, new_basis = self._apply_dict[(pauli_symbol, new_state[0][target_qubit].item())]
                new_state[0][target_qubit] = new_basis
                coeff_i *= new_coeff

            c_states = torch.cat((c_states, new_state), dim=0)
            c_coefs = torch.cat((c_coefs, coeff_i))
        return c_states, c_coefs

    @torch.no_grad()
    def _local_energy_single(self, wf, state: torch.tensor, ln_psi_or_psi: torch.tensor, is_permutation: bool=None):
        c_states, c_coefs = self._coupled_states(state)
        if len(c_coefs) > 0:
            # (wf(c_states) / psi) @ c_coefs
            kwargs = dict() if is_permutation is None else dict(permutation=is_permutation)
            ln_psis_or_psi_k = wf.ln_psi(c_states, **kwargs) if wf.is_complex_wf() else wf.psi(c_states, **kwargs)

            # eloc = exp(ln_psis_k - ln_psis) @ c_coefs
            if wf.is_complex_wf():
                eloc = cplx.scalar_mult(cplx.exp(ln_psis_or_psi_k - ln_psi_or_psi), cplx.to_complex(c_coefs.real, c_coefs.imag)).sum(axis=0)
                ret = eloc[0].item() + eloc[1].item()*1j
            else:
                eloc = (ln_psis_or_psi_k / ln_psi_or_psi) @ c_coefs.real.to(ln_psi_or_psi.dtype)
                ret = eloc.item() + 0*1j
            return ret
        return 0.

    @torch.no_grad()
    def calculate_local_energy(self, wf, states: Union[torch.tensor, np.ndarray], ln_psis_or_psi: torch.tensor=None, is_permutation: bool=None, **kwargs):
        r"""
        Args:
            wf: wavefunction
            states: samples
            ln_psis_or_psi: wf(states)
            is_permutation: whether to reverse the state order 
        Return:
            elocs: local energies for states, with type np.ndarray[np.complex]
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)

        if ln_psis_or_psi is None:
            _kwargs = dict() if is_permutation is None else dict(permutation=is_permutation)
            ln_psis_or_psi = wf.ln_psi(states, **_kwargs) if wf.is_complex_wf() else wf.psi(states, **_kwargs)
        assert states.shape[0] == len(ln_psis_or_psi)
        elocs = [self._local_energy_single(wf, states[i], ln_psis_or_psi[i], is_permutation=is_permutation) for i in range(states.shape[0])]
        elocs = np.array(elocs) # np.ndarray[np.complex]
        return elocs

class MolecularHamiltonianCPP(Hamiltonian):
    def __init__(self, ham_path: str):
        super().__init__(ham_path)
        self.n_qubits = eloc.init_hamiltonian(ham_path)

    def calculate_local_energy(self, states: Union[torch.tensor, np.ndarray], psis: torch.tensor = None, *args, **kwargs):
        return eloc.calculate_local_energy(states, psis, *args, **kwargs)

    def free_hamiltonian(self):
        return eloc.free_hamiltonian()

if __name__ == '__main__':
    pass