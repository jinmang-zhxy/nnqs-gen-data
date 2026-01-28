import pyscf, pyscf.fci, pyscf.ci
from pyscf.cc import ccsd
import numpy as np
import sys
sys.path.append("../molecules/")
import pyscf_helper, utils
import openfermion

# import error
import re

def get_fci(geometry, basis="sto3g", eps=1e-12, is_sorted=False, ci_type='fci'):
    mol = pyscf.gto.M(atom=geometry, basis=basis)
    #mol = pyscf.gto.M(atom=geometry, basis=basis, symmetry=True)
    mf = pyscf.scf.RHF(mol).run()

    # save ham
    #molecule = mol
    #mo_coeff = mf.mo_coeff
    #n_orb = mo_coeff.shape[1]  # molecule.nao_nr()
    #n_orb_occ = sum(molecule.nelec) // 2
    #occ_indices_spin = [i for i in range(molecule.nelectron)]
    #hcore = mf.get_hcore()
    #one_body_mo, two_body_mo = pyscf_helper.get_mo_integrals_from_molecule_and_hf_orb(
    #    molecule, mo_coeff)
    #core_correction = 0.0

    #n_procs = 2
    #energy_nuc = molecule.energy_nuc()
    #hamiltonian_ferm_op_1, hamiltonian_ferm_op_2 = \
    #pyscf_helper.get_hamiltonian_ferm_op_from_mo_ints_mp(
    #    one_body_mo, two_body_mo, eps,
    #    n_procs=n_procs)

    #hamiltonian_ferm_op = hamiltonian_ferm_op_1 + hamiltonian_ferm_op_2
    #hamiltonian_ferm_op += energy_nuc + core_correction
    #qubit_op = openfermion.jordan_wigner(hamiltonian_ferm_op)
    #filename = "c2-try-qubit_op.data"
    #utils.save_binary_qubit_op(qubit_op, filename=filename)
    #print("Qubit Hamiltonian saved to %s." % (filename))

    if ci_type == 'fci':
        # FCI
        mf_fci = pyscf.fci.FCI(mf).run()
        e_fci = pyscf.fci.FCI(mf).kernel()[0]
        print(f"e_fci: {e_fci}")
        fci_coeff_matrix = mf_fci.ci
    elif ci_type == 'cisd':
        # CISD
        mf_ci = pyscf.ci.CISD(mf).run()
        # fci_coeff_matrix = mf_ci.to_fcivec(mf_ci.ci)

        eris = ccsd._make_eris_outcore(mf_ci, mf.mo_coeff)
        ecisd, civec = mf_ci.kernel(eris=eris)
        fci_coeff_matrix = mf_ci.to_fcivec(civec)
    else:
        error(f'Unsupport ci_type: {ci_type}')
    print(f'Using ci_type: {ci_type} basis: {basis} eps: {eps}')

    n_orbitals = mol.nao_nr()
    n_electron_a, n_electron_b = mol.nelec
    # print(fci_coeff_matrix)
    n_orb_a = fci_coeff_matrix.shape[0]
    n_orb_b = fci_coeff_matrix.shape[1]
    print(f"n_orbitals: {n_orbitals} n_orb_a: {n_orb_a} n_org_b: {n_orb_b}")

    print("|aa...bb...>: coefficient")
    states = []
    coeffs = []
    import time
    for idx_a in range(n_orb_a):
        st = time.time()
        p = [0, 0, 0.]
        for idx_b in range(n_orb_b):
            coeff = fci_coeff_matrix[idx_a, idx_b]
            #if coeff*coeff < eps:
            if abs(coeff) < eps:
                continue
            st1 = time.time()
            bitstring_a = pyscf.fci.cistring.addr2str(n_orbitals, n_electron_a, idx_a)
            bitstring_b = pyscf.fci.cistring.addr2str(n_orbitals, n_electron_b, idx_b)
            ed1 = time.time()
            p[0] += ed1 - st1
            # print(f"{bitstring_a} {bitstring_b}")
            # Prepend zeros
            st2 = time.time()
            bitstring_a = "0"*(n_orbitals - len(bin(bitstring_a)) + 2) + bin(bitstring_a)[2:]
            bitstring_b = "0"*(n_orbitals - len(bin(bitstring_b)) + 2) + bin(bitstring_b)[2:]
            ed2 = time.time()
            p[1] += ed2 - st2

            st3 = time.time()
            bitstring = bitstring_a + bitstring_b
            bitstring = ""
            for i in range(n_orbitals):
                bitstring += bitstring_a[i] + bitstring_b[i]
            rev_bitstring = "".join(reversed(bitstring))
            #rev_bitstring = "".join((bitstring))
            # print(f"n: {n_orbitals}", bitstring, rev_bitstring)
            coeff = fci_coeff_matrix[idx_a, idx_b]
            #if coeff*coeff > eps:
            if abs(coeff) > eps:
                # print("|%s>: %20.16f %20.16f" % (rev_bitstring, coeff, coeff*coeff))
                states.append(rev_bitstring)
                coeffs.append(coeff)
            ed3 = time.time()
            p[2] += ed3 - st3
        ed = time.time()
        # print(f"Time of {idx_a}, tot: {ed-st} p1: {ed1-st1} p2: {ed2-st2} p3: {ed3-st3}")
        # print(f"Time of {idx_a}, tot: {ed-st} p1,p2,p3: {p}")
    states, coeffs = np.array(states), np.array(coeffs)
    # probs = coeffs*coeffs
    probs = coeffs
    prob_sum = np.sum(coeffs*coeffs)
    print(f"prob_sum: {prob_sum} tot_cnt: {coeffs.shape}")

    if is_sorted:
        idxs = np.argsort(probs)[::-1]
        sorted_states, sorted_probs = states[idxs], probs[idxs]
        return sorted_states, sorted_probs

    return states, probs

def get_data(filename='samples.txt', eps=1e-10):
    # 定义正则表达式
    n_samples_pattern = re.compile(r'n_samples_detail: \[(.*)\]')
    # ks_pattern = re.compile(r'ks_detail: (.*)')
    ks_pattern = re.compile(r'ks_detail: UInt64\[(.*)\]')

    # 读取文本文件
    with open(filename, 'r') as file:
        data = file.read()

    # 解析 n_samples_detail 数组
    n_samples_match = n_samples_pattern.search(data)
    n_samples_str = n_samples_match.group(1)
    n_samples = [int(x) for x in n_samples_str.split(', ')]
    # print('n_samples:', n_samples)

    # 解析 ks_detail 数组
    ks_match = ks_pattern.search(data)
    ks_str = ks_match.group(1)
    ks_hex = ks_str.split(' ')
    # ks = [str(int(x, 16)) for x in ks_hex]
    ks = [int(x, 16) for x in ks_hex]
    ks, n_samples = np.array(ks), np.array(n_samples)
    n_samples = n_samples / np.sum(n_samples)

    print(f"without filter shape:", n_samples.shape)

    # sorted
    idxs = np.argsort(n_samples)[::-1]
    ks, n_samples = ks[idxs], n_samples[idxs]

    ks = ks[n_samples > eps]
    n_samples = n_samples[n_samples > eps]

    data_dict = dict()
    for _k, _prob in zip(ks, n_samples):
        data_dict[_k] = _prob

    print(f"sum: {np.sum(n_samples)} shape: {ks.shape}")
    return ks, n_samples, data_dict

def dump_fci(states, probs):
    _idx_spin_basis_vec = np.array([2 ** n for n in range(len(states[0]))])
    for state, prob in zip(states, probs):
        # sid = np.sum(_idx_spin_basis_vec * state)
        sid = 0
        for ik, s in enumerate(state):
            sid += _idx_spin_basis_vec[ik] * int(s)

        print("|%s> %20d: %20.16f" % (state, sid, prob))

def get_ci_dict(states, probs):
    ci_dict = dict()
    id2state = dict()
    _idx_spin_basis_vec = np.array([2 ** n for n in range(len(states[0]))])
    for state, prob in zip(states, probs):
        sid = 0
        for ik, s in enumerate(state):
            sid += _idx_spin_basis_vec[ik] * (int(s) == 1)
        ci_dict[sid] = prob
        #id2state[sid] = torch.tensor([1 if int(s) == 1 else 0 for s in state])
        id2state[sid] = np.array([1 if int(s) == 1 else 0 for s in state])
    return ci_dict, id2state

def get_ci_dict_keystate(states, probs):
    ci_probs, ci_states = [], []
    for state, prob in zip(states, probs):
        _state = np.array([1 if s == '1' else -1 for s in state])
        ci_probs.append(prob)
        ci_states.append(_state)
    # print(f'inner ci_probs: {ci_probs} ci_states: {ci_states}')
    return ci_probs, ci_states

# def get_ci_dict_keystate(states, probs):
#     ci_dict = dict()
#     _idx_spin_basis_vec = np.array([2 ** n for n in range(len(states[0]))])
#     for state, prob in zip(states, probs):
#         sid = 0
#         for ik, s in enumerate(state):
#             sid += _idx_spin_basis_vec[ik] * int(s)
#         _state = np.array([1 if s == '1' else -1 for s in state])
#         ci_dict[sid] = [prob, _state]
#     return ci_dict

def dump_fcis(geometry_list, basis_list, eps=1e-10, is_sorted=False):
    for geometry, basis in zip(geometry_list, basis_list):
        print(f"\ngeometry: {geometry}, basis: {basis}")
        states, probs = get_fci(geometry, basis=basis, eps=eps, is_sorted=is_sorted)
        dump_fci(states, probs)

def fci_vs_my(myfile, basis="sto-3g", eps=1e-10, is_sorted=False):
    states, probs = get_fci(geometry, basis=basis, eps=eps, is_sorted=is_sorted)
    ks, n_samples, data_dict = get_data(myfile, eps=1e-16)
    _idx_spin_basis_vec = np.array([2 ** n for n in range(len(states[0]))])

    # print(f"ks: {ks}")
    # print(f"n_samples: {n_samples}")

    hit_cnt = 0
    for state, prob in zip(states, probs):
        # sid = np.sum(_idx_spin_basis_vec * state)
        sid = 0
        for ik, s in enumerate(state):
            sid += _idx_spin_basis_vec[ik] * int(s)

        myprob = data_dict.get(sid)
        if myprob == None:
            myprob = -1
        else:
            hit_cnt += 1

        print("|%s> %20d: %20.16f vs %20.16f" % (state, sid, prob, myprob))

    print(f"hit_cnt: {hit_cnt} ({states.shape[0]} -> {n_samples.shape[0]})")


def get_ci_dict_wrapper(geometry, basis='sto-3g', is_sorted=False, ci_type='fci', key_type='sid'):
    states, probs = get_fci(geometry, basis=basis, is_sorted=is_sorted, ci_type=ci_type)
    if key_type == 'sid':
        return get_ci_dict(states, probs)
    else:
        return get_ci_dict_keystate(states, probs)

if __name__ == "__main__":
    import sys
    ci_type = 'fci'
    #ci_type = 'cisd'
    if len(sys.argv) > 1:
        ci_type = sys.argv[1]

    #geometry = ["H 0.0 0.0 0.0", "H 0.0 0.0 1.0"]
    #geometry = ["H 0.0 0.0 0.0", "H 0.4 0.0 0.0"]
    #geometry = [('H', (2, 0, 0)), ('H', (3, 0, 0))] # H2
    #geometry = [('Li', (3, 0, 0)), ('H', (2, 0, 0))] # LiH
    #geometry = [('O', (0, 0, 0)), ('H', (0.2774, 0.8929, 0.2544)), ('H', (0.6068, -0.2383, -0.7169))] # H2O
    #geometry = [('C', (-0.825, 0, 0)), ('C', (0.825, 0, 0))]
    # geometry = [('H', (2, 0, 0)), ('H', (3, 0, 0))]
    # geometry = [('H', (0, 0, 0.0)), ('H', (0, 0, 2.5)), ('H', (0, 0, 5.0)), ('H', (0, 0, 7.5)), ('H', (0, 0, 10.0)), ('H', (0, 0, 12.5))]
    geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])] # C2
    # geometry = [('O', (-0.0007, 0.8141, 0)), ('C', (0.7509, -0.4065, 0)), ('C', (-0.7502, -0.4076, 0)), ('H', (1.2625, -0.6786, 0.9136)), ('H', (1.2625, -0.6787, -0.9136)), ('H', (-1.2614, -0.6806, -0.9136)), ('H', (-1.2614, -0.6805, 0.9136))] # Ethylene Oxide, C2H4O
    # geometry = [('S', (0.0002, -0.0002, -0.1181)), ('O', (1.2744, 0.0355, 0.8975)), ('O', (-1.2765, -0.0326, 0.8949)), ('O', (-0.0336, 1.2577, -0.8391)), ('O', (0.0354, -1.2604, -0.8351)), ('H', (1.2111, 0.6912, 1.6312)), ('H', (-1.2148, -0.6859, 1.6307))] # Sulfuric Acid, H2O4S
    # geometry = [('O', (-0.3035, 1.289, -0.0002)), ('O', (-0.98, -0.8878, -0.0002)), ('C', (1.3743, -0.3516, -0.0002)), ('C', (-0.0907, -0.0496, 0.0006)), ('H', (1.8368, 0.057, -0.9021)), ('H', (1.84, 0.0676, 0.8952)), ('H', (1.5207, -1.4356, 0.0064)), ('H', (-1.2598, 1.5081, -0.0008))] # Acetic Acid, C2H4O2
    # geometry = [('C', (1.2818, -0.2031, 0)), ('C', (-0.0643, 0.4402, 0)), ('C', (-1.2175, -0.2371, 0)), ('H', (1.8429, 0.1063, -0.8871)), ('H', (1.2188, -1.2959, 0)), ('H', (1.8429, 0.1063, 0.8871)), ('H', (-0.095, 1.5262, 0)), ('H', (-2.1647, 0.2911, 0)), ('H', (-1.239, -1.3212, 0))] # Propene, C3H6
    # geometry = [["H", [0., 0., i * 3.0]] for i in range(10)] # H10
    #geometry = [('O', (2.866, -0.25, 0)), ('Li', (3.732, 0.25, 0)), ('Li', (2, 0.25, 0))] # Li2O
    states, probs = get_fci(geometry, basis='sto-3g', is_sorted=True, ci_type=ci_type)
    #states, probs = get_fci(geometry, basis='sto-6g', is_sorted=True, ci_type=ci_type)
    # states, probs = get_fci(geometry, basis="ccpvtz", eps=1e-10)
    # states, probs = get_fci(geometry, basis="augccpvtz", eps=1e-10, is_sorted=True)
    # sorted_states, sorted_probs = get_fci(geometry, basis="augccpvtz", eps=1e-10, is_sorted=True)
    _states = []
    for state in states:
        _states.append([1 if s == '1' else 0 for s in state])
    _states = np.array(_states)
    np.savez(f'c2-try.npz', ci_states=_states, ci_probs=probs)
    dump_fci(states, probs)
    ci_dict = get_ci_dict(states, probs)
    for k in ci_dict.keys():
        print(f"k: {k} v: {ci_dict[k]}")
    # myfile = 'samplesh6_2.5-5178th.txt'
    # fci_vs_my(myfile, basis="sto-3g", is_sorted=True)
    # fci_vs_my(myfile, basis="ccpvtz", is_sorted=True, eps=1e-12)
    # fci_vs_my(myfile, basis="augccpvtz", is_sorted=True)
    # bonds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for bond in bonds:
    # geometry_list = [["H 0.0 0.0 0.0", f"H {bond} 0.0 0.0"] for bond in bonds]
    # basis_list = ["sto-3g"]*len(bonds)
    # basis_list = ["631g"]*len(bonds)
    # basis_list = ["631++g"]*len(bonds)
    # dump_fcis(geometry_list, basis_list, is_sorted=True, eps=1e-12)
    # print(f"geometry_list {geometry_list}")
    # print(f"basis_list {basis_list}")
