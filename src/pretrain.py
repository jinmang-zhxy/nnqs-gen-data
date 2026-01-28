"""
Copyright wuyangjun21@163.com 2023-02-10
"""
from datetime import datetime
import os
import numpy as np
import torch

from src.utils.utils import read_yaml, count_parameters
from NeuralNetworkQuantumState import wrapper_train
from src.utils import cisd
from src.utils.config import Config

def train_cisd(cfgs, trainer, n_sample=None, molecule='H2O', loss_version='v4', mol_dict=None, psi_type=1):
    print(f"------------------------")
    print(f"--------Pretrain--------")
    print(f"------------------------")
    device = trainer.device
    
    #cfgs = read_yaml(cfg_file)

    load_ci, save_ci = True, False
    data = cfgs.pretrain_data
    pretrain_save_path = cfgs.pretrain_save_path  
    pretrain_step = cfgs.pretrain_step

    ci_data = np.load(data)
    ci_probs, ci_states = ci_data['ci_probs'], ci_data['ci_states']
    print(f'load from data path: {data}')

    if not np.iscomplex(ci_probs[0]):
        ci_probs = ci_probs.astype(np.complex128)
    use_aabb_pretrain = True
    use_aabb_pretrain = False
    if use_aabb_pretrain:
        L = ci_states.shape[1]
        perm_idx=np.array([t for t in zip(np.arange(L//2), np.arange(L//2)+L//2)]).reshape(-1)
        ci_states = ci_states[:, perm_idx] # aabb -> abab
        print(f"using aabb pretrain")

    ci_probs, ci_states = torch.from_numpy(ci_probs), torch.from_numpy(ci_states)
    assert ci_probs.shape[0] > 0, "ci_probs number must > 0"
    print(f'ci_probs: {ci_probs.shape} {ci_probs} \nci_states: {ci_states.shape} {ci_states}', flush=True)
    #save_ci and (np.savez(f'{molecule}-{loss_version}-state.npz', ci_probs=ci_probs.numpy(), ci_states=ci_states.numpy()))
    #ci_dict = cisd.get_ci_dict(ci_states, ci_probs)
    ci_dict, id2state = cisd.get_ci_dict(ci_states, ci_probs)
    # n_sample = 1e7
    if n_sample is not None:
        _n_sample = n_sample
    else:
        _n_sample = 1e7

    def cross_entropy(p, q):
        res = 0
        for i, v in enumerate(q):
            res += p[i] * torch.log(p[i]/q[i])
        return res

    # lossv4: let phase real part consider +1/-1, imag part ->0, amps consider probs
    def get_loss_fn(amps, phases, ci_amps, weights, renorm=False):
        if renorm:
            amps = amps / (amps.norm())
        # (amps^2 - weight) -> 0
        loss_sign_err = (amps * phases.cos() - ci_amps.real).pow(2)
        loss_sign_imag_err = (amps * phases.sin() - ci_amps.imag).pow(2) # imag -> 0
        loss = torch.dot(loss_sign_err + loss_sign_imag_err, weights)
        return loss

    # loss = -overlap(wf1, wf2)
    def construct_complex(amps, phases):
        return torch.complex(amps * phases.cos(), amps * phases.sin())

    # c1x1 + c2x2 + ... + cnxn: \Psi(xn) = cn
    def get_loss_overlap(psi1: torch.complex, psi2: torch.complex):
        #psi1 = psi1 / psi1.norm()
        #psi2 = psi2 / psi2.norm()
        print(f"psi1: {psi1}")
        print(f"psi2: {psi2}", flush=True)
        print(f"ps1i-psi2: {psi1-psi2}", flush=True)
        print(f"|psi1|-|psi2|: {psi1.abs()-psi2.abs()}", flush=True)
        print(f"sum|psi1|^2: {psi1.abs().pow(2).sum()}")
        print(f"sum|psi2|^2: {psi2.abs().pow(2).sum()}")
        print(f"|psi1|^2: {psi1.abs().pow(2)}")
        print(f"|psi2|^2: {psi2.abs().pow(2)}")
        S = (psi1.conj() * psi2).sum().abs().pow(2)
        loss = -S
        return loss

    def get_overlap_wf(ci_dict, sids, amps, phases):
        _ci_probs, sel_idxs = [], []
        #print("sids:")
        for k, sid in enumerate(sids):
            #print(f"{sid}")
            ci_prob = ci_dict.get(sid.item(), None)
            if ci_prob:
                _ci_probs.append(ci_prob)
                sel_idxs.append(k)
        print(f"sel_samples: {sids[sel_idxs]}")
        sel_idxs = torch.tensor(sel_idxs, dtype=torch.int32)
        amps, phases = torch.index_select(amps, 0, sel_idxs), torch.index_select(phases, 0, sel_idxs)
        ci_probs = torch.from_numpy(np.array(_ci_probs, dtype=np.complex128))
        return amps, phases, ci_probs

    def get_overlap_wf_pad(ci_dict, sids, amps, phases):
        _ci_probs, sel_idxs = [], []
        ci_keys = set(ci_dict.keys())
        #print("sids:")
        sel_keys = []
        for k, sid in enumerate(sids):
            #print(f"{sid}")
            ci_prob = ci_dict.get(sid.item(), None)
            if ci_prob:
                _ci_probs.append(ci_prob)
                #sel_idxs.append(k)
                sel_keys.append(sid)
            else:
                _ci_probs.append(0)
                #sel_idxs.append(k)

        #print(f"sel_samples: {sids[sel_idxs]}")
        #sel_idxs = torch.tensor(sel_idxs, dtype=torch.int32)
        #amps, phases = torch.index_select(amps, 0, sel_idxs), torch.index_select(phases, 0, sel_idxs)
        ci_probs = torch.from_numpy(np.array(_ci_probs, dtype=np.complex128))
        return amps, phases, ci_probs

    def get_overlap_wf_pad2(ci_dict, sids, amps, phases):
        _ci_probs, sel_idxs = [], []
        #ci_keys = set(ci_dict.keys())
        ci_keys = set([key.item() for key in ci_dict.keys()])
        #print("sids:")
        sel_keys = []
        sel_samples = []
        for k, sid in enumerate(sids):
            #print(f"{sid}")
            ci_prob = ci_dict.get(sid.item(), None)
            if ci_prob:
                _ci_probs.append(ci_prob)
                sel_keys.append(sid.item())
            else:
                _ci_probs.append(0)

        #amps, phases = torch.index_select(amps, 0, sel_idxs), torch.index_select(phases, 0, sel_idxs)
        ci_keys_rem = ci_keys - set(sel_keys)
        print(f"ci_keys: {len(ci_keys)} sel_keys: {len(sel_keys)} ci_keys_rem: {len(ci_keys_rem)}")
        _ci_states_rem = []
        for ci_key in ci_keys_rem:
            _ci_probs.append(ci_dict[ci_key])
            _ci_states_rem.append(id2state[ci_key])
        _ci_states_rem = torch.from_numpy(np.array(_ci_states_rem))
        ln_amps_rem, phases_rem = trainer.wavefunction._forward_pad(_ci_states_rem, masking=None, permutation=True) # amps is ln(amps)
        amps_rem = torch.exp(ln_amps_rem)
        amps = torch.cat((amps, amps_rem), dim=0)
        phases = torch.cat((phases, phases_rem), dim=0)

        ci_probs = torch.from_numpy(np.array(_ci_probs, dtype=np.complex128))
        return amps, phases, ci_probs

    for i in range(pretrain_step):
        trainer.zero_grad(set_to_none=True)
        trainer.optimizer.zero_grad()

        if i > -1:
            # method 1
            # amps, phases = trainer.wavefunction._forward_pad(ci_states, masking=None, permutation=True)
            # ln_amps, phases = trainer.wavefunction._forward_pad(ci_states, masking=None, permutation=True) # amps is ln(amps)
            # amps = torch.exp(ln_amps)
            ci_amps = ci_probs
            # weights = ci_probs * ci_probs
            weights = (ci_probs * ci_probs.conj()).real
            # weights = torch.zeros(phases.shape[0])
            # weights[:] = 1. / phases.shape[0]

            if trainer.use_grad_accumulation:
                loss = 0
                acc_bs = trainer.grad_accumulation_width
                samples = ci_states.reshape(-1, trainer.n_qubits)
                samples_list, weights_list, ci_amps_list = torch.split(samples, acc_bs), torch.split(weights, acc_bs), torch.split(ci_amps, acc_bs)
                for samples_i, weights_i, ci_amps_i in zip(samples_list, weights_list, ci_amps_list):
                    ln_amps_i, phases_i = trainer.wavefunction._forward_pad(samples_i, masking=None, permutation=True)
                    amps_i = ln_amps_i if psi_type == 1 else torch.exp(ln_amps_i) # real/complex

                    #amps_i, phases_i = amps_i.to(device), phases_i.to(device)
                    #loss_i = get_loss_fn(amps_i, phases_i, ci_amps_i, weights_i)
                    loss_i = get_loss_fn(amps_i, phases_i, ci_amps_i, weights_i, renorm=True)
                    #loss_i = get_loss_overlap(construct_complex(amps_i, phases_i), ci_probs)
                    loss += loss_i.item()
                    loss_i.backward()
            else:
                ln_amps, phases = trainer.wavefunction._forward_pad(ci_states, masking=None, permutation=True) # amps is ln(amps)
                amps = ln_amps if psi_type == 1 else torch.exp(ln_amps) # real/complex
        else:
            # method 2
            trainer.wavefunction.sample()
            n_samples, samples, amps, phases = trainer.gen_samples(n_sample, ret_type="torch")
            weights = (n_samples / n_samples.sum()).type(torch.float64)
            samples = samples.reshape(-1, trainer.n_qubits)
            sids = trainer.wavefunction._state2id_batch(samples).type(torch.int64)
            print(f'sids: {sids.shape} samples: {samples.shape}', flush=True)
            if trainer.use_grad_accumulation:
                loss = 0
                ci_probs_cnt = 0
                acc_bs = trainer.grad_accumulation_width
                samples_list, weights_list, sids_list = torch.split(samples, acc_bs), torch.split(weights, acc_bs), torch.split(sids, acc_bs)
                for samples_i, weights_i, sids_i in zip(samples_list, weights_list, sids_list):
                    ln_amps_i, phases_i = trainer.wavefunction._forward_pad(samples_i, masking=None, permutation=True)
                    amps_i = torch.exp(ln_amps_i)
                    # ci_amps_i = [ci_dict.get(sid.item(), 0) for sid in sids_i]
                    # ci_amps_i = torch.from_numpy(np.array(ci_amps_i, dtype=np.complex128))

                    # loss_i = get_loss_fn(amps_i, phases_i, ci_amps_i, weights_i)
                    # TODO: https://pytorch.org/docs/stable/checkpoint.html
                    amps_i, phases_i, ci_probs_i = get_overlap_wf(ci_dict, sids_i, amps_i, phases_i)
                    if ci_probs_i.shape[0] > 0:
                        loss_i = get_loss_overlap(construct_complex(amps_i, phases_i), ci_probs_i)
                        ci_probs_cnt += ci_probs_i.shape[0]
                        loss += loss_i.item()
                        loss_i.backward()
                print(f"ci_probs_cnt: {ci_probs_cnt}", flush=True)
            else:
                # ci_amps = [ci_dict.get(sid.item(), 0) for sid in sids]
                # ci_amps = torch.from_numpy(np.array(ci_amps, dtype=np.complex128))
                amps_sum = amps.pow(2).sum()
                #amps, phases, ci_probs = get_overlap_wf(ci_dict, sids, amps, phases)
                #amps, phases, ci_probs = get_overlap_wf_pad(ci_dict, sids, amps, phases)
                amps, phases, ci_probs = get_overlap_wf_pad2(ci_dict, sids, amps, phases)
                # print(f"ci_probs_cnt: {ci_probs.shape[0]} amps_sum: {amps_sum.item()}", flush=True)
                print(f"ci_probs_cnt: {ci_probs.shape[0]} amps_sum: {amps_sum.item()}", flush=True)

            # if i%40 == 0:
            #     print(f"sids.shape: {sids.shape} vs. new_ci_probs.shape: {ci_probs.shape} org_ci_states: {ci_states.shape}")

        if not trainer.use_grad_accumulation:
            #loss = get_loss_fn(amps, phases, ci_amps, weights)
            loss = get_loss_fn(amps, phases, ci_amps, weights, renorm=True)
            #loss = get_loss_overlap(construct_complex(amps, phases), ci_probs)
            loss.backward()

        trainer.step()

        if i%10 == 0:
            print(f"{i}-th loss: {loss}, time: {datetime.now()}", flush=True)

        if i%1000 == 0:
            ckpt = f"{pretrain_save_path}/{molecule.lower()}-{i}th.pt"
            trainer.save_model(ckpt)
            print(f"save model into {ckpt}")

def pretrain(cfg_file="config-h4.yaml"):
    r"""Simple debug example."""
    #cfgs = read_yaml(cfg_file)
    cfgs = Config(cfg_file)
    #mol = cfgs['system']
    mol = cfgs.system
    #seed = int(cfgs['seed'])
    seed = cfgs.seed
    qubit_dict = {'h10_ring_hf': 20, 'h2': 4, 'lih': 12, 'lih_new': 6, 'h2o': 14, 'h10': 20, 'c2': 20, 'h2s': 22, 'c2h4o': 38, 'li2o': 30, 'c2h4o': 38, 'cna2o3': 76, 'h12_uhf_cisd': 24, 'h2_f': 64, 'h_h_f': 64, 'h_hf': 64, 'fe2s2_40': 40, 'fenton9': 52, 'fenton10': 52, 'fenton8': 52}
    n_qubits = qubit_dict[mol.lower()]
    wtrain = wrapper_train(cfgs, n_qubits=n_qubits, n_rank=1, rank=0, seed=seed, pos_independ=False)

    n_samples = 1e7
    train_cisd(cfgs, wtrain, n_samples, molecule=mol, loss_version='v4', mol_dict=None, psi_type=wtrain.psi_type)

if __name__ == "__main__":
    pretrain("config-pretrain.yaml")

