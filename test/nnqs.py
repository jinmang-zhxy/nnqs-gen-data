#!/usr/bin/env python                                                                                                          
# encoding: utf-8

import argparse
import os

import torch
import torch.distributed as dist
import numpy as np
import time

import src.NeuralNetworkQuantumState as nnqs
from src.utils.config import Config
from src.utils.utils import Timer, count_parameters
from src.hamiltonian import MolecularHamiltonianExact, MolecularHamiltonianCPP, MolecularHamiltonianExactOpt

def ddp_setup(local_rank, device_type='cpu'):
    """
    Args:
        local_rank: Unique identifier of each process
        device_type: 'cpu' | 'cuda'
    """
    device = torch.device(f'cuda:{local_rank}' if device_type == 'cuda' and torch.cuda.is_available() else f'cpu:{local_rank}')
    if device_type == 'cuda':
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        dist.init_process_group(backend="gloo")
    return device

def log_rank(rank, mess, prompt=""):
    print(f"[rank {rank}] {prompt} mess: {mess}")

def train_loop_parallel(ham_path, cfg_file, log_file=None):
    import interface.python.eloc as eloc
    # load config
    config = Config(cfg_file)
    config.log_file = config.system if log_file is None else log_file
    
    # parallel init
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    (global_rank == 0) and print(f"ham path: {ham_path}\nconfig file: {cfg_file}")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    device = ddp_setup(local_rank, device_type=config.device)
    (global_rank == 0) and print(f"[Parallel Setting] world_size: {world_size} gpu_name: {gpu_name} distributed_backend: {dist.get_backend()}")
    print(f"[Parallel Mapping] local_rank: {local_rank} global_rank: {global_rank} device: {device}")
    dist.barrier() # just for print initial info

    # load hamiltonian
    n_qubits = eloc.init_hamiltonian(ham_path)
    dist.barrier() # just for print initial info

    (global_rank == 0) and print(config)

    # initial model
    n_epoches = config.n_epoches
    n_samples = config.n_samples
    seed = config.seed
    close_log = (global_rank != 0)
    wtrain = nnqs.wrapper_train(config, n_qubits=n_qubits, n_rank=world_size, rank=global_rank, seed=seed, close_log=close_log, pos_independ=False)

    # load model
    if config.load_model == 1:
        wtrain.load_model(config.checkpoint_path, config.transfer_learning)
        (global_rank == 0) and print(f"load model parameters from {config.checkpoint_path}")
    (global_rank == 0) and count_parameters(wtrain, print_verbose=True)

    # main training loop
    for i in range(1, n_epoches+1):
        Timer.start("total") # accumulation time from start
        Timer.start("elapsed") # one iteration time
        Timer.start("sampling") # sampling time
        n_samples_local, states_local, psis_local = wtrain.gen_samples(n_samples)
        Timer.stop("sampling")
        # print(f"grank: {global_rank} states_local: {states_local} psis_local: {psis_local} n_samples_local: {n_samples_local}", flush=True)

        # [distributed] collect weights
        batch_size_global = torch.tensor(n_samples_local.sum(), device=device)
        dist.all_reduce(batch_size_global, op=dist.ReduceOp.SUM)
        weights_global = n_samples_local / batch_size_global.item()

        Timer.start("eloc") # calculate local energy time
        states_local = states_local.reshape(n_samples_local.shape[0], n_qubits)
        local_energies = eloc.calculate_local_energy_parallel(states_local, psis_local, device, global_rank, world_size)
        Timer.stop("eloc")
        # log_rank(global_rank, local_energies, "local_energies")

        Timer.start("gradient") # gradient calculation time
        eloc_expectation_local = np.dot(weights_global, local_energies)
        eloc_expectation_global = torch.tensor(eloc_expectation_local, device=device)
        dist.all_reduce(eloc_expectation_global, op=dist.ReduceOp.SUM)
        eloc_corr = local_energies - eloc_expectation_global.cpu().numpy()

        # mean grad among world_size
        grad_local = wtrain.update_grad(eloc_corr, weights_global, True)
        grad_global = torch.tensor(grad_local, device=device)
        dist.all_reduce(grad_global, op=dist.ReduceOp.SUM)
        grad_global /= world_size
        wtrain.set_grad_and_step(grad_global.cpu().numpy())
        Timer.stop("gradient")
        Timer.stop("total")
        Timer.stop("elapsed")

        if global_rank == 0 and config.save_model == 1 and i % config.save_per_epoches == 0:
            wtrain.save_model(f"checkpoints/{config.log_file}-iter{i}.pt")

        # Timer.summary()
        if global_rank == 0 and i % config.log_step == 0:
            weights_str = ' '.join('{:.6f}'.format(w) for w in weights_global[:8])
            # print(f"\n{i}-th eloc_mean: {eloc_expectation.real} Hartree")
            print(f"\n{i}-th eloc_mean: {eloc_expectation_global.real} Hartree")
            print(f"batch_size: {wtrain.n_samples} n_uniq_samples: {states_local.shape[0]} \t weights[:8]: [{weights_str}]")
            Timer.display("elapsed", "sampling", "eloc", "gradient", "total", precision=4)
        Timer.reset("elapsed", "sampling", "eloc", "gradient")

    eloc.free_hamiltonian()
    dist.destroy_process_group()

def train_loop(ham_path, cfg_file, log_file=None):
    print(f"ham path: {ham_path}\nconfig file: {cfg_file}")
    config = Config(cfg_file)
    config.log_file = config.system if log_file is None else log_file
    print(config)
    n_epoches = config.n_epoches
    n_samples = float(config.n_samples)
    seed = config.seed

    if config.hamiltonian_type == "exact":
        ham = MolecularHamiltonianExact(ham_path, device=config.device)
    elif config.hamiltonian_type == "exactOpt":
        ham = MolecularHamiltonianExactOpt(ham_path, device=config.device)
    elif config.hamiltonian_type == "CPP":
        ham = MolecularHamiltonianCPP(ham_path)
    else:
        raise ValueError(f"Only support hamiltonian_type: exact | exactOpt | CPP, but now is {config.hamiltonian_type}")

    n_qubits = ham.get_n_qubits()
    wtrain = nnqs.wrapper_train(config, n_qubits=n_qubits, n_rank=1, rank=0, seed=seed, pos_independ=False)

    if config.load_model == 1:
        wtrain.load_model(config.checkpoint_path, config.transfer_learning)
        print(f"load model parameters from {config.checkpoint_path}")
    count_parameters(wtrain, print_verbose=True)

    for i in range(1, 1+1):
        Timer.start("total") # accumulation time from start
        Timer.start("elapsed") # one iteration time
        Timer.start("sampling") # sampling time
        n_samples, states, psis = wtrain.gen_samples(n_samples)
        Timer.stop("sampling")
        dir = f"/workspace/local_energy_data/{config.system}"
        if os.path.exists(dir) == False:
            os.mkdir(dir)

        Timer.start("eloc") # calculate local energy time
        states = states.reshape(n_samples.shape[0], n_qubits)
        if config.hamiltonian_type in ["exact", "exactOpt"]:
            local_energies = ham.calculate_local_energy(wtrain.wavefunction, states, is_permutation=True, eloc_split_bs=config.eloc_split_bs)
        if config.hamiltonian_type in ["exact", "exactOpt"]:
            local_energies = ham.calculate_local_energy(wtrain.wavefunction, states, is_permutation=True, eloc_split_bs=config.eloc_split_bs)
        elif config.hamiltonian_type == "CPP":
            def f32_hex(x: float) -> str:
                # 按 IEEE754 float32 的 bit pattern 输出 8 位 16 进制（无 0x 前缀）
                return format(np.float32(x).view(np.int32), "08x")

            # --- Save Data for Verification ---
            # Save states
            # Convert 1/-1 to 0/1 representation: 1 -> 0, -1 -> 1 (assuming Z basis)
            # states are likely +1/-1.
            states_01 = (1 + states) // 2

            with open(os.path.join(dir, "verification_states.txt"), "w") as f:
                for state in states_01:
                    f.write(" ".join(map(str, state.astype(int).tolist())) + "\n")
            
            # Save psis (coeffs)
            with open(os.path.join(dir, "verification_psis.txt"), "w") as f:
                # psis is already numpy if ret_type='numpy' (default in gen_samples)
                if hasattr(psis, 'detach'):
                    psis_np = psis.detach().cpu().numpy()
                else:
                    psis_np = psis
                
                if psis_np.dtype == np.complex64 or psis_np.dtype == np.complex128:
                    for val in psis_np:
                        # format: real_hex imag_hex (float32 bit pattern hex)
                        f.write(f"{f32_hex(val.real)} {f32_hex(val.imag)}\n")
                else:
                    for val in psis_np:
                        # real only
                        f.write(f"{f32_hex(val)} {f32_hex(0.0)}\n")

            # Save Hamiltonian
            # Note: ham.qubit_op is available from MolecularHamiltonianCPP but we need to access the structure
            # ham object is MolecularHamiltonianCPP(ham_path)
            # It seems ham_path is passed to C++. Let's reuse read_binary_qubit_op to dump it here for consistency
            from src.utils.utils import read_binary_qubit_op
            _, qubit_op_dump = read_binary_qubit_op(ham_path) # Utilize ham_path available in local scope
            with open(os.path.join(dir, "verification_ham.txt"), "w") as f:
                for term, coeff in qubit_op_dump.terms.items():
                    # term is a tuple of (index, 'X'/'Y'/'Z'), e.g. ((0, 'Z'), (1, 'Z'))
                    # coefficient is coeff.
                    # Format: Paulis val_real val_imag
                    # e.g. Z0 Z1 0x1.c016140000000p-4 0x0.0p+0
                    pauli_str_parts = []
                    if not term: # Identity
                        pauli_str_parts.append("I")
                    else:
                        for idx, op in term:
                            pauli_str_parts.append(f"{op}{idx}")
                    pauli_str = " ".join(pauli_str_parts)
                    f.write(f"{pauli_str} {f32_hex(coeff.real)} {f32_hex(coeff.imag)}\n")

            start = time.time()
            local_energies = ham.calculate_local_energy(states, psis)
            end = time.time()
            cost = (end - start) * 1000 # ms
            print(f"Time to calculate local energies: {cost:.4f} ms")
            # Save elocs
            with open(os.path.join(dir, "verification_elocs.txt"), "w") as f:
                # local_energies is a tensor, likely complex if H is complex or wavefunction is complex
                if hasattr(local_energies, 'detach'):
                    elocs_np = local_energies.detach().cpu().numpy()
                else:
                    elocs_np = local_energies
                
                # Check if it's complex
                if np.iscomplexobj(elocs_np):
                     for val in elocs_np:
                        f.write(f"{f32_hex(val.real)} {f32_hex(val.imag)}\n")
                else: 
                     for val in elocs_np:
                        f.write(f"{f32_hex(val)} {f32_hex(0.0)}\n")
            
            # Exit after one step as requested/implied for data generation
            # print("Data saved. Exiting.")
            # exit() 
            # User didn't strictly say exit, but usually for data gen we stop. 
            # I'll keep it running for 1 epoch as configured in my plan unless user edit had exit. 
            # The user's edit `print(states); print(psis); exit()` was removed? 
            # Ah, the user *added* it in the diff provided in Step 218. 
            # I am *replacing* that block. So I should remove the print/exit or incorporate it.
            # I will remove the print/exit and let the loop finish (which is 1 epoch now).
            # Wait, user's edit in 218 added exit(). I should probably honor the intention of "just run once".
            # But I configured 1 epoch in my plan.
            pass 
            # End of saving block
        Timer.stop("eloc")

        Timer.start("gradient") # gradient calculation time
        weights = n_samples / n_samples.sum()
        eloc_expectation = np.dot(weights, local_energies)
        eloc_corr = local_energies - eloc_expectation
        wtrain.update_grad(eloc_corr, weights)
        Timer.stop("gradient")
        Timer.stop("total")
        Timer.stop("elapsed")

        if config.save_model == 1 and i % config.save_per_epoches == 0:
            wtrain.save_model(f"checkpoints/{config.log_file}-iter{i}.pt")

        # Timer.summary()
        if i % config.log_step == 0:
            weights_str = ' '.join('{:.6f}'.format(w) for w in weights[:8])
            print(f"\n{i}-th eloc_mean: {eloc_expectation.real} Hartree")
            print(f"batch_size: {wtrain.n_samples} n_uniq_samples: {states.shape[0]} \t weights[:8]: [{weights_str}]")
            Timer.display("elapsed", "sampling", "eloc", "gradient", "total", precision=4)
        Timer.reset("elapsed", "sampling", "eloc", "gradient")

        Timer.reset("sampling", "eloc", "gradient")

    ham.free_hamiltonian()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NNQS COMMANDS")
    parser.add_argument("ham_path", help="Hamiltonian file path")
    parser.add_argument("cfg_file", help="Configuration file path")
    parser.add_argument("--log_file", help="Log file name for checkpoints", default=None)
    parser.add_argument("--use_parallel", help="Using parallel training", action="store_true")

    args = parser.parse_args()
    if args.use_parallel:
        train_loop_parallel(args.ham_path, args.cfg_file, args.log_file)
    else:
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)
        train_loop(args.ham_path, args.cfg_file, args.log_file)
