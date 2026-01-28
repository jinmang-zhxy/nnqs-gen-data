#!/usr/bin/env python                                                                                                          
# encoding: utf-8

import argparse
import os
import math

import torch
import torch.distributed as dist
import numpy as np

import src.NeuralNetworkQuantumState as nnqs
import interface.python.eloc as eloc
from src.utils.config import Config
from src.utils.utils import Timer, count_parameters

# const KEY for dataset
KEY_HAM_PATH = "ham_path"
KEY_N_ELECS = "n_elecs"
KEY_MAX_QUBITS = "max_qubits"
KEY_MOLECULES = "molecules"

# Set your train_dataset

# Example: H2 sto3g
train_dataset = {
    KEY_MAX_QUBITS: 4,
    KEY_MOLECULES: {
        "h2_sto3g_0.4": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.4/qubit_op.data", KEY_N_ELECS: 2},
        "h2_sto3g_0.5": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.5/qubit_op.data", KEY_N_ELECS: 2},
        "h2_sto3g_0.6": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.6/qubit_op.data", KEY_N_ELECS: 2},
        "h2_sto3g_0.7": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.7/qubit_op.data", KEY_N_ELECS: 2},
        "h2_sto3g_0.8": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.8/qubit_op.data", KEY_N_ELECS: 2},
    }
}

# Example: mix H2, LiH, H2O, N2
train_dataset = {
    KEY_MAX_QUBITS: 20,
    KEY_MOLECULES: {
        "h2_sto3g_0.4": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.4/qubit_op.data", KEY_N_ELECS: 2},
        "h2_sto3g_0.5": {KEY_HAM_PATH: "../molecules/thomas/h2_sto3g_0.5/qubit_op.data", KEY_N_ELECS: 2},
        "lih": {KEY_HAM_PATH: "../molecules/thomas/lih/qubit_op.data", KEY_N_ELECS: 4},
        "h2o": {KEY_HAM_PATH: "../molecules/thomas/h2o/qubit_op.data", KEY_N_ELECS: 10},
        "n2_0.75": {KEY_HAM_PATH: "../molecules/thomas/n2_0.75/qubit_op.data", KEY_N_ELECS: 14},
    }
}

# for h2o2 sto3g
def gen_pes_dataset(bonds, max_qubits, num_elecs, prefix_path):
    data_structure = {
        KEY_MAX_QUBITS: max_qubits,
        KEY_MOLECULES: {}
    }

    for bond in bonds:
        # bond_str = f"{bond:.2f}"
        bond_str = str(bond)
        data_structure[KEY_MOLECULES][bond_str] = {
            KEY_HAM_PATH: f"{prefix_path}/{bond_str}/qubit_op.data",
            KEY_N_ELECS: num_elecs
        }

    return data_structure

def gen_h2o2_pes(N=32):
    PREFIX_PATH = "/home/nas/wuyangjun/sc24_scaling_H2O2/h2o2_datasets"
    num_elecs = 18
    max_qubits = 24

    st = 0.25
    step = 0.01
    # N = 32
    ed = st + step * (N-1)
    #bonds = np.round(np.arange(st, ed, step), 2)
    bonds = np.round(np.linspace(st, ed, N), 2)
    assert len(bonds) == N

    ds = gen_pes_dataset(bonds, max_qubits, num_elecs, PREFIX_PATH)
    return ds

#train_dataset = gen_h2o2_pes(224)
train_dataset = gen_h2o2_pes(32*7)

# N2 sto3g V1
# train_dataset = {
#     KEY_MAX_QUBITS: 20,
#     KEY_MOLECULES: {
#         "n2_0.75": {KEY_HAM_PATH: "../molecules/thomas/n2_0.75/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_0.9": {KEY_HAM_PATH: "../molecules/thomas/n2_0.9/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.05": {KEY_HAM_PATH: "../molecules/thomas/n2_1.05/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.2": {KEY_HAM_PATH: "../molecules/thomas/n2_1.2/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.5": {KEY_HAM_PATH: "../molecules/thomas/n2_1.5/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.35": {KEY_HAM_PATH: "../molecules/thomas/n2_1.35/qubit_op.data", KEY_N_ELECS: 14},
#     }
# }

# # N2 sto3g V2
# train_dataset = {
#     KEY_MAX_QUBITS: 20,
#     KEY_MOLECULES: {
#         "n2_0.75": {KEY_HAM_PATH: "../molecules/thomas/n2_0.75/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.05": {KEY_HAM_PATH: "../molecules/thomas/n2_1.05/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.35": {KEY_HAM_PATH: "../molecules/thomas/n2_1.35/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.65": {KEY_HAM_PATH: "../molecules/thomas/n2_1.65/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.95": {KEY_HAM_PATH: "../molecules/thomas/n2_1.95/qubit_op.data", KEY_N_ELECS: 14},
#     }
# }

# # N2 sto3g V3
# train_dataset = {
#     KEY_MAX_QUBITS: 20,
#     KEY_MOLECULES: {
#         "n2_0.75": {KEY_HAM_PATH: "../molecules/thomas/n2_0.75/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.05": {KEY_HAM_PATH: "../molecules/thomas/n2_1.05/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.35": {KEY_HAM_PATH: "../molecules/thomas/n2_1.35/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.65": {KEY_HAM_PATH: "../molecules/thomas/n2_1.65/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_1.95": {KEY_HAM_PATH: "../molecules/thomas/n2_1.95/qubit_op.data", KEY_N_ELECS: 14},
#         "n2_2.25": {KEY_HAM_PATH: "../molecules/thomas/n2_2.25/qubit_op.data", KEY_N_ELECS: 14},
#     }
# }

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

def train_loop_parallel(train_dataset, cfg_file, log_file=None, max_qubits=-1):
    ## load config
    config = Config(cfg_file)
    config.log_file = config.system if log_file is None else log_file

    ## parallel init
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    num_train_dataset = len(train_dataset)
    assert num_train_dataset >= world_size
    num_train_data_per_rank = num_train_dataset // world_size
    (global_rank == 0) and print(f"max_qubits: {max_qubits} num_train_dataset: {num_train_dataset} num_train_data_per_rank: {num_train_data_per_rank} config_file: {cfg_file}\ntrain_dataset: {train_dataset}")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    device = ddp_setup(local_rank, device_type=config.device)
    (global_rank == 0) and print(f"[Parallel Setting] world_size: {world_size} gpu_name: {gpu_name} distributed_backend: {dist.get_backend()}")
    print(f"[Parallel Mapping] local_rank: {local_rank} global_rank: {global_rank} device: {device}")
    #dist.barrier() # just for print initial info bug here?
    # reduce for debug
    #reduce_0 = torch.arange(2, dtype=torch.int64) + 1 + 2 * global_rank
    #reduce_0 = reduce_0.cuda()
    #log_rank(global_rank, reduce_0, "reduce_0_before")
    #dist.all_reduce(reduce_0, op=dist.ReduceOp.SUM)
    #log_rank(global_rank, reduce_0, "reduce_0_after")

    (global_rank == 0) and print(config)

    ## initial model
    n_epoches = config.n_epoches
    n_samples = config.n_samples
    seed = config.seed
    close_log = (global_rank != 0)
    wtrain = nnqs.wrapper_train(config, n_qubits=max_qubits, n_rank=1, rank=0, seed=seed, close_log=close_log, pos_independ=False)

    ## load model
    if config.load_model == 1:
        wtrain.load_model(config.checkpoint_path, config.transfer_learning)
        (global_rank == 0) and print(f"load model parameters from {config.checkpoint_path}")
    (global_rank == 0) and count_parameters(wtrain, print_verbose=True)

    time_list = []
    # print(f"max_qubits: {max_qubits} num_train_dataset: {num_train_dataset}\ntrain_dataset: {train_dataset}")
    for i in range(1, n_epoches+1):
        Timer.start("total") # accumulation time from start
        Timer.start("elapsed") # one iteration time

        ## zero grads
        wtrain.zero_optimizer()

        ## training mini-batch_size = len(train_dataset)
        molecules = list(train_dataset.keys())

        # allocate task among rank
        st = num_train_data_per_rank * global_rank
        ed = num_train_data_per_rank * (global_rank + 1)
        # last rank process tail
        if global_rank == world_size - 1:
            ed = max(ed, num_train_dataset)
        molecules_local = molecules[st:ed]

        eloc_expectation_list = []
        log_rank(global_rank, molecules_local, "molecules_local")

        # for molecule, configs in train_dataset.items():
        for molecule in molecules_local:
            Timer.start("single_elapsed")
            configs = train_dataset[molecule]
            ## Re-init the specified properties: Hamiltonian, n_qubits, n_elecs
            ham_path, n_elecs = configs[KEY_HAM_PATH], configs[KEY_N_ELECS]
            Timer.start("init_ham")
            n_qubits = eloc.init_hamiltonian(ham_path)
            Timer.stop("init_ham")
            print(f"molecule: {molecule} n_qubits: {n_qubits} n_elecs: {n_elecs} hamiltonian path: {ham_path}")
            wtrain.update_n_qubits_and_n_elecs(n_qubits, n_elecs, dump_verbose=False)

            ## Sampling, calculate enenrgy expectation, and grads for the current molecule
            Timer.start("sampling") # sampling time
            n_samples, states, psis = wtrain.gen_samples(n_samples)
            Timer.stop("sampling")

            Timer.start("eloc") # calculate local energy time
            states = states.reshape(n_samples.shape[0], n_qubits)
            local_energies = eloc.calculate_local_energy(states, psis)
            Timer.stop("eloc")

            Timer.start("gradient") # gradient calculation time
            weights = n_samples / n_samples.sum()
            eloc_expectation = np.dot(weights, local_energies)
            eloc_expectation_list.append(eloc_expectation)
            eloc_corr = local_energies - eloc_expectation
            # accumulate grads among molecules (mini-batch)
            wtrain.update_grad(eloc_corr, weights, is_zero_grad=False)
            Timer.stop("gradient")
            eloc.free_hamiltonian()
            Timer.stop("single_elapsed")

            # if i % config.log_step == 0:
            #     weights_str = ' '.join('{:.6f}'.format(w) for w in weights[:8])
            #     # print(f"{i}-th eloc_mean of {molecule}: {eloc_expectation.real} Hartree")
            #     # print(f"batch_size: {wtrain.n_samples} n_uniq_samples: {states.shape[0]} \t weights[:8]: [{weights_str}]")
            #     print(f"{i}-th eloc_mean of {molecule}:")
            #     Timer.display("sampling", "eloc", "gradient", precision=4)
            #     print("")
            Timer.reset("sampling", "eloc", "gradient", "single_elapsed")

        # tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * global_rank
        # tensor = tensor.to(device)
        # log_rank(global_rank, tensor, "origin tensor")
        # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # log_rank(global_rank, tensor, "sum tensor")
        Timer.start("comm")
        ## Calculate average grads for this mini-batch and update network parameters
        avg_grad_local =  wtrain.collect_grads() / num_train_dataset
        ## mean grad among world_size
        avg_grad_global = torch.tensor(avg_grad_local, device=device)
        dist.all_reduce(avg_grad_global, op=dist.ReduceOp.SUM)
        # dist.barrier()
        # log_rank(global_rank, "avg_grad_global", "avg_grad_global")
        avg_grad_global = avg_grad_global.cpu().numpy()
        # log_rank(global_rank, "to numpy")
        # avg_grads_norm = np.sum(np.abs(avg_grad_global))
        # log_rank(global_rank, "sum")
        wtrain.set_grad_and_step(avg_grad_global)
        # log_rank(global_rank, "set grad")
        Timer.stop("elapsed")
        Timer.stop("total")
        time_list.append(Timer.get_time("elapsed"))

        if global_rank == 0 and config.save_model == 1 and i % config.save_per_epoches == 0:
            wtrain.save_model(f"checkpoints/{config.log_file}-iter{i}.pt")

        ## calculate global expectation
        # eloc_expectation = np.array(eloc_expectation_list).real.sum()
        # eloc_expectation_local = torch.tensor(eloc_expectation, device=device)
        # dist.reduce(eloc_expectation_local, dst=0, op=dist.ReduceOp.SUM)
        Timer.stop("comm")
        # eloc_expectation_global = (eloc_expectation_local / num_train_dataset).cpu().numpy()

        if global_rank == 0 and i % config.log_step == 0:
            # weights_str = ' '.join('{:.6f}'.format(w) for w in weights[:8])
            # print(f"==={i}-th=== eloc_mean of train_dataset: {eloc_expectation_global} Hartree avg_grad_norm: {avg_grads_norm}")
            Timer.display("elapsed", "total", "init_ham", "comm", precision=4)
            print("")
        Timer.reset("elapsed", "sampling", "eloc", "gradient")
    if global_rank == 0:
        print(f"time_list: {time_list}")
        print(f"time_list[10:]: {sum(time_list[10:])}")
    dist.destroy_process_group()

def train_loop(train_dataset, cfg_file, log_file=None, max_qubits=-1):
    config = Config(cfg_file)
    config.log_file = config.system if log_file is None else log_file
    print(config)
    n_epoches = config.n_epoches
    n_samples = config.n_samples
    seed = config.seed
    wtrain = nnqs.wrapper_train(config, n_qubits=max_qubits, n_rank=1, rank=0, seed=seed, pos_independ=False)

    if config.load_model == 1:
        wtrain.load_model(config.checkpoint_path, config.transfer_learning)
        print(f"load model parameters from {config.checkpoint_path}")
    count_parameters(wtrain, print_verbose=True)

    num_train_dataset = len(train_dataset)
    print(f"max_qubits: {max_qubits} num_train_dataset: {num_train_dataset}\ntrain_dataset: {train_dataset}")
    for i in range(1, n_epoches+1):
        Timer.start("total") # accumulation time from start
        Timer.start("elapsed") # one iteration time

        ## zero grads
        wtrain.zero_optimizer()

        ## training mini-batch_size = len(train_dataset)
        molecules = train_dataset.keys()
        eloc_expectation_list = []
        # for molecule, configs in train_dataset.items():
        for molecule in molecules:
            configs = train_dataset[molecule]
            ## Re-init the specified properties: Hamiltonian, n_qubits, n_elecs
            ham_path, n_elecs = configs[KEY_HAM_PATH], configs[KEY_N_ELECS]
            n_qubits = eloc.init_hamiltonian(ham_path)
            print(f"molecule: {molecule} n_qubits: {n_qubits} n_elecs: {n_elecs} hamiltonian path: {ham_path}")
            # wtrain.update_n_qubits_and_n_elecs(n_qubits, n_elecs, dump_verbose=False)
            wtrain.update_n_qubits_and_n_elecs(n_qubits, n_elecs, dump_verbose=True)

            ## Sampling, calculate enenrgy expectation, and grads for the current molecule
            Timer.start("sampling") # sampling time
            n_samples, states, psis = wtrain.gen_samples(n_samples)
            Timer.stop("sampling")

            Timer.start("eloc") # calculate local energy time
            states = states.reshape(n_samples.shape[0], n_qubits)
            local_energies = eloc.calculate_local_energy(states, psis)
            Timer.stop("eloc")

            Timer.start("gradient") # gradient calculation time
            weights = n_samples / n_samples.sum()
            eloc_expectation = np.dot(weights, local_energies)
            eloc_expectation_list.append(eloc_expectation)
            eloc_corr = local_energies - eloc_expectation
            # accumulate grads among molecules (mini-batch)
            wtrain.update_grad(eloc_corr, weights, is_zero_grad=False)
            Timer.stop("gradient")
            eloc.free_hamiltonian()

            if i % config.log_step == 0:
                weights_str = ' '.join('{:.6f}'.format(w) for w in weights[:8])
                print(f"{i}-th eloc_mean of {molecule}: {eloc_expectation.real} Hartree")
                print(f"batch_size: {wtrain.n_samples} n_uniq_samples: {states.shape[0]} \t weights[:8]: [{weights_str}]")
                Timer.display("sampling", "eloc", "gradient", precision=4)
                print("")
            Timer.reset("sampling", "eloc", "gradient")

        ## Calculate average grads for this mini-batch and update network parameters
        avg_grads =  wtrain.collect_grads() / num_train_dataset
        avg_grads_norm = np.sum(np.abs(avg_grads))
        wtrain.set_grad_and_step(avg_grads)
        Timer.stop("elapsed")
        Timer.stop("total")

        if config.save_model == 1 and i % config.save_per_epoches == 0:
            wtrain.save_model(f"checkpoints/{config.log_file}-iter{i}.pt")

        eloc_expectation = np.array(eloc_expectation_list).mean()
        if i % config.log_step == 0:
            weights_str = ' '.join('{:.6f}'.format(w) for w in weights[:8])
            print(f"==={i}-th=== eloc_mean of train_dataset: {eloc_expectation.real} Hartree avg_grad_norm: {avg_grads_norm}")
            Timer.display("elapsed", "total", precision=4)
            print("")
        Timer.reset("elapsed", "sampling", "eloc", "gradient")

    eloc.free_hamiltonian()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NNQS COMMANDS")
    parser.add_argument("cfg_file", help="Configuration file path")
    parser.add_argument("--log_file", help="Log file name for checkpoints", default=None)
    parser.add_argument("--use_parallel", help="Using parallel training", action="store_true")

    args = parser.parse_args()
    # ATTENTION: set max_qubits carefully according your training and test datasets
    if args.use_parallel:
        torch.set_num_threads(1)
        train_loop_parallel(train_dataset[KEY_MOLECULES], args.cfg_file, args.log_file, max_qubits=train_dataset[KEY_MAX_QUBITS])
    else:
        # limit torch thread
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)
        train_loop(train_dataset[KEY_MOLECULES], args.cfg_file, args.log_file, max_qubits=train_dataset[KEY_MAX_QUBITS])
