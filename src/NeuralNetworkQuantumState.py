"""
Copyright wuyangjun21@163.com 2023-02-10
"""
from datetime import datetime
import math
import time
import collections
import random
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import src.utils.complex_helper as cplx
from src.utils.utils import set_global_seed, collect_grads_impl, set_grads_impl, get_logger, MaxBatchSizeExceededError

from src.utils.GPUMemTrack import MemTracker

def create_lr_lambda(total_steps=1e5, max_lr=1e-4, warmup_ratio=5e-3, cosine_min_lr_ratio=0.1):
    warmup_steps = int(warmup_ratio * total_steps)
    print(f"total_steps: {total_steps} warmup_steps: {warmup_steps} max_lr: {max_lr}, cosine_min_lr_ratio: {cosine_min_lr_ratio}", flush=True)
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # linear warmup
            return float(current_step+1) / float(max(warmup_steps, 1)) 
        else:
            # consine decay to cosine_min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return cosine_min_lr_ratio + (1-cosine_min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265359)))
    return lr_lambda

def learning_rate(step_num, d_model, warmup_step=4000, factor=1.0):
    r"""Learning rate schedule used by LambdaLR.

    Args:
        step_num: current train step number
        d_model: model size
        warmup_step: warmup step number
        factor: scale factor
    Returns:
        lrate: learning rate
    """
    # avoid division 0
    if step_num == 0:
        step_num = 1
    lrate = (d_model**-0.5) * min(step_num**-0.5, step_num*warmup_step**-1.5)
    lrate *= factor
    return lrate

def learning_rate_v2(step_num):
    if step_num > 5000:
        return 0.5
    return 1.0

def lr_warmup_cosine(current_step, num_warmup_steps, num_training_steps, num_cycles=0.5):
    f"""Learning rate schedule with warmup+cosine."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def warmup_cosine_schedule_with_bounds(optimizer, warmup_steps=4000, total_steps=1e5, lr_max=0.001, lr_min=1e-7, num_cycles=0.5, last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < warmup_steps:
            return lr_min + (lr_max - lr_min) * float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return lr_min + (lr_max - lr_min) * max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, _lr_lambda, last_epoch, verbose=True)

class wrapper_train(nn.Module):
    r"""Wrapper of train.
    Called by Julia.

    Args:
        cfg_file: configuration file path, with yaml format
        n_qubits: number of qubits for molecular system
        seed: if seed =-1 then don't fixed random seed
        pos_independ: True -> Multi-Attn position embedding independent.
    """
    def __init__(self, config, n_qubits=6, n_rank=0, rank=0, seed=-1, close_log=False, pos_independ=False, out_device = 'cpu') -> None:
        super().__init__()
        dump_verbose = not close_log
        if dump_verbose:
            print("===========================================================")
            print("===============  NeuralNetworkQuantumState  ===============")
            print("===========================================================")
        # TODO: debug
        if seed != -1:
            set_global_seed(seed, dump_verbose=dump_verbose)

        # set logger level
        log_level = config.log_level if not close_log else 'ERROR'
        self.logger = get_logger(level=log_level)

        # model parameters
        self.out_device = out_device
        self.device = config.device
        self.model_fp = config.model_fp
        self.global_fp = torch.float32 if self.model_fp != "fp64" else torch.float64
        torch.set_default_dtype(self.global_fp)
        self.logger.info(f"set model_fp into {self.global_fp}")

        self.use_uniq_sampling_parallel = config.use_uniq_sampling_parallel
        # if self.use_uniq_sampling_parallel and self.device == 'cuda':
        #     n_gpus = torch.cuda.device_count()
        #     gpu_id = rank % n_gpus
        #     self.device = self.device + ":" + str(gpu_id)
        #     print(f"rank: {rank} device: {self.device} {n_gpus}", flush=True)
            # self.set_mpi_ranks(n_rank, rank)

        self.logger.info(f"torch_version: {torch.__version__} device: {self.device}")

        d_model = config.model.d_model
        lr = float(config.optim.lr)
        betas = tuple(config.optim.betas)
        eps = float(config.optim.eps)
        weight_decay = config.optim.weight_decay
        self.open_lr_scheduler = config.optim.open_lr_scheduler

        self.n_elecs = config.n_elecs  # electron number
        self.use_clip_grad = config.use_clip_grad

        psi_type = config.psi_type
        self.psi_type = config.psi_type
        self.logger.info(f"Using psis of {psi_type}")

        #from DecoderWaveFunctionCI import get_wavefunction
        #from DecoderWaveFunctionAmp import get_wavefunction
        if psi_type == 'complex' or psi_type == 'qk1':
            self.psi_type = 'complex'
            from src.wavefunctions.DecoderWaveFunction import get_wavefunction
        elif psi_type == 'real' or psi_type == 'qk3':
            self.psi_type = 'real'
            from src.wavefunctions.DecoderWaveFunctionReal import get_wavefunction
        elif psi_type == 'qk2':
            from src.wavefunctions.DecoderWaveFunctionCombine import get_wavefunction
        else:
            raise ValueError(f"Unsupport psi_type: {psi_type}")

        self.use_grad_accumulation = bool(config.use_grad_accumulation)
        if self.use_grad_accumulation:
            self.grad_accumulation_width = int(config.grad_accumulation_width)
            self.logger.info(f"use_grad_accumulation, grad_accumulation_width: {self.grad_accumulation_width}")

        self.n_sampling_layers = config.n_sampling_layers
        self.logger.info(f"[Sampling Tree] n_sampling_layers: {self.n_sampling_layers}")
        self.n_samples = float(config.n_samples)
        self.use_samples_recursive = config.use_samples_recursive
        if self.use_samples_recursive or self.use_uniq_sampling_parallel:
            self.n_samples_min = float(config.n_samples_min)
            self.n_samples_max = float(config.n_samples_max)
            self.n_unq_samples_min = float(config.n_unq_samples_min)
            self.n_unq_samples_max = float(config.n_unq_samples_max)
            self.n_samples_scale_factor = config.n_samples_scale_factor
            self.incr_per_epoch = config.incr_per_epoch
            self.n_samples = self.n_samples_min
            if self.use_uniq_sampling_parallel:
                self.n_unq_samples_min /= n_rank
            self.logger.info(f"Using samples recursive with n_samples_min= {self.n_samples_min}, " +
                             f"n_samples_max= {self.n_samples_max}, n_unq_samples_min= {self.n_unq_samples_min}, " +
                             f"n_unq_samples_max= {self.n_unq_samples_max}")

        self.n_epochs = 0

        # build model
        self.n_qubits = n_qubits
        n_alpha_electrons = self.n_elecs//2 if config.n_alpha_electrons is None else config.n_alpha_electrons
        n_beta_electrons = self.n_elecs//2 if config.n_beta_electrons is None else config.n_beta_electrons
        self.logger.info(f'n_qubits: {n_qubits} n_alpha_electrons: {n_alpha_electrons} n_beta_electrons: {n_beta_electrons}')
        # if system == "O2":
        #     n_alpha_electrons, n_beta_electrons = 9, 7
        # elif system == "CH2":
        #     n_alpha_electrons, n_beta_electrons = 5, 3
        # elif system == "Fe2S2":
        #     n_alpha_electrons, n_beta_electrons = 28, 18
        # elif system == "H2_F" or system == "H_HF" or system == "H_H_F":
        #     n_alpha_electrons, n_beta_electrons = 5, 4
        self.wavefunction = get_wavefunction(config, num_qubits=n_qubits, n_alpha_electrons=n_alpha_electrons, n_beta_electrons=n_beta_electrons, device=self.device, logger=self.logger)
        if self.use_uniq_sampling_parallel:
            self.wavefunction.set_mpi_ranks(n_rank, rank)
        self.wavefunction.set_seed(seed)

        # set multi-threads
        n_threads = torch.get_num_threads()
        n_interop_threads = torch.get_num_interop_threads()
        self.logger.info(f"n_threads: {n_threads}, n_interop_threads: {n_interop_threads}")

        # merge two model optims
        self.optimizer = torch.optim.AdamW([\
            {'params': self.parameters()}],
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        if self.open_lr_scheduler:
            wstep = config.optim.warmup_step  # 4k/12k
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(\
                optimizer=self.optimizer,
                lr_lambda=lambda step : learning_rate(step, d_model, warmup_step=wstep, factor=1.0),
                # lr_lambda=lambda step : learning_rate_v2(step),
                # lr_lambda=lambda step : lr_warmup_cosine(step, wstep, 100000, num_cycles=6),
                )
            # self.lr_scheduler = warmup_cosine_schedule_with_bounds(self.optimizer, warmup_steps=wstep, total_steps=1e4, lr_max=0.001, lr_min=1e-7, num_cycles=2, last_epoch=-1)
            self.logger.info(f"lr_scheduler wstep= {wstep}")

    def get_state_dict(self):
        r"""Get state dict."""
        return self.state_dict()

    def set_state_dict(self, states):
        r"""Set state dict."""
        self.load_state_dict(states)

    def save_model(self, save_path="model.pt"):
        """Save model."""
        checkpoint = {'wavefunction_state': self.wavefunction.state_dict(),
                      'optimizer_state': self.optimizer.state_dict(),
                      'lr_scheduler_state': self.lr_scheduler.state_dict() if self.open_lr_scheduler else None,
                      'n_epochs': self.n_epochs,
                      'n_samples': self.n_samples,
                      'random_state': random.getstate(),
                      'np_random_state': np.random.get_state(),
                      'torch_random_state': torch.random.get_rng_state()}
        if 'cuda' in self.device:
            checkpoint.update({'torch_cuda_random_state': torch.cuda.get_rng_state(),
                               'torch_cuda_random_state_all': torch.cuda.get_rng_state_all()})
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(checkpoint, save_path)
        self.logger.info(f'Save model into {save_path}')

    def load_model(self, load_path="model.pt", transfer_learning=False):
        """Load model."""
        checkpoint = torch.load(load_path)
        def _create_key_shape_set(state_dict):
            return set(f"{key}-{value.shape}" for key, value in state_dict.items())

        def load_intersect(model_a, model_b_checkpoint):
            # model_b_checkpoint = torch.load(model_b_load_path)
            model_a_state_dict = model_a.state_dict()
            #filtered_checkpoint = {k: v for k, v in model_b_checkpoint.items() if k in model_a_state_dict}
            filtered_checkpoint = {k: v for k, v in model_b_checkpoint.items() if k in model_a_state_dict and v.shape == model_a_state_dict[k].shape}
            #diff_set = set(model_a_state_dict.keys()) - set(model_b_checkpoint.keys())
            diff_set = _create_key_shape_set(model_a_state_dict) - _create_key_shape_set(model_b_checkpoint)
            #print(f'filtered_checkpoint: {filtered_checkpoint.keys()}', flush=True)
            # print(f'filtered_checkpoint diff set: {diff_set}', flush=True)
            self.logger.info(f'filtered_checkpoint diff set: {diff_set}')
            model_a.load_state_dict(filtered_checkpoint, strict=False)

        def copy_custom_params(model_a_ckpt, model_b, target_string="wpe", dim=0):
            """
            Copy parameters from model A to model B for all tensors containing 'wpe' in their name.
            Only copy the minimum dimension for these tensors.
            
            Args:
            - model_a (nn.Module): Source model to copy parameters from.
            - model_b (nn.Module): Target model to copy parameters to.
            """
            state_dict_a = model_a_ckpt
            state_dict_b = model_b.state_dict()

            for name, param in state_dict_a.items():
                if target_string in name and name in state_dict_b:
                    # Determine the size of the smallest dimension
                    min_dim_size = min(state_dict_b[name].shape[dim], param.shape[dim])
                    self.logger.info(f"target shape: {state_dict_b[name].shape} <- source shape: {param.shape}")
                    self.logger.info(f"load min_dim_size={min_dim_size} with dim={dim}, paramter name={name}")
                    # Copy only the minimum dimension
                    if dim == 0:
                        state_dict_b[name][:min_dim_size, ...].copy_(param[:min_dim_size, ...])
                    elif dim == 1:
                        state_dict_b[name][:, :min_dim_size].copy_(param[:, :min_dim_size])
                    else:
                        ValueError(f"Only support dim = 0 or 1")

            model_b.load_state_dict(state_dict_b)                                                                                      

        load_intersect(self.wavefunction, checkpoint['wavefunction_state'])
        copy_custom_params(checkpoint['wavefunction_state'], self.wavefunction, target_string="wpe", dim=0)
        if self.psi_type == 'complex':
            copy_custom_params(checkpoint['wavefunction_state'], self.wavefunction, target_string="phase_layers.0.layers.0.0.weight", dim=1)
        #self.wavefunction.load_state_dict(checkpoint['wavefunction_state'])
        if not transfer_learning:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.open_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
            self.n_epochs = checkpoint['n_epochs']
            self.n_samples = checkpoint['n_samples']
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['np_random_state'])
            torch.random.set_rng_state(checkpoint['torch_random_state'])
            if 'cuda' in self.device:
                torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
                torch.cuda.set_rng_state_all(checkpoint['torch_cuda_random_state_all'])
        self.logger.info(f'Load model from {load_path} \n\t\t--> train from {self.n_epochs}-th epoch')
        # frozen amplitude, just training phase
        # self.frozen_parameters()

    def collect_grads(self):
        r"""Collect grad from tensors and return a 1D numpy array."""
        return collect_grads_impl(self)

    def get_nelecs(self):
        return self.n_elecs

    def reset_model_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset_optimizer(self):
        self.optimizer.state = collections.defaultdict(dict)
        if self.open_lr_scheduler:
            self.lr_scheduler.step(0)

    def reset_model_train(self, seed=111):
        set_global_seed(seed)
        self.reset_model_parameters()
        self.reset_optimizer()
        self.logger.info(f"reset_model_train with seed={seed}")

    def set_seed(self, seed=111):
        self.wavefunction.set_seed(seed)

    def get_random_seed(self, N):
        seed_list = np.random.randint(0, 2**30, (N,))
        # seed = random.randint(0, 2 ** 32)
        return seed_list

    def set_rank(self, n_rank, rank):
        self.n_rank = n_rank
        self.rank = rank
        self.wavefunction.set_mpi_ranks(n_rank, rank)

    def set_mpi_ranks(self, n_rank, rank):
        self.rank = rank
        if self.device == "cuda":
            n_gpus = torch.cuda.device_count()
            gpu_id = self.rank % n_gpus
            torch.cuda.set_device(gpu_id)
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            self.logger.info(f"n_gpus: {n_gpus}; rank {self.rank} --> gpu id: {gpu_id}, name: {gpu_name}")
            # print(f"n_gpus: {n_gpus}; rank {self.rank} --> gpu id: {gpu_id}, name: {gpu_name}", flush=True)

    def set_n_samples(self, n_samples=1e10):
        self.n_samples = n_samples
        self.logger.info(f"set samples as {n_samples}")

    def set_n_qubits(self, n_qubits, dump_verbose=False):
        self.n_qubits = n_qubits
        if dump_verbose:
            self.logger.info(f"set n_qubits as {n_qubits}")
        self.wavefunction.update_qubit2model_permutations(self.n_qubits, dump_verbose=dump_verbose)

    def set_n_elecs(self, n_elecs, dump_verbose=False):
        self.n_elecs = n_elecs
        self.wavefunction.update_alpha_beta_electrons(n_elecs//2, n_elecs//2)
        if dump_verbose:
            self.logger.info(f"set n_elecs as {n_elecs}")

    def update_n_qubits_and_n_elecs(self, n_qubits, n_elecs, dump_verbose=False):
        self.set_n_qubits(n_qubits, dump_verbose=dump_verbose)
        self.set_n_elecs(n_elecs, dump_verbose=dump_verbose)
        # if dump_verbose:
        #     self.logger.info(f"set n_qubits={n_qubits} n_elecs={n_elecs}")

    def get_n_samples(self):
        return self.n_samples

    def update_grad(self, e_loc_corr:np.array, weights:np.array, ret_grad=False, is_zero_grad=True):
        r"""Update model grad interface."""
        if self.psi_type == 'real':
            if self.use_grad_accumulation:
                return self.cal_update_grad_real_accumulation(e_loc_corr, weights, ret_grad=ret_grad, is_zero_grad=is_zero_grad) # real
            else:
                return self.cal_update_grad_real(e_loc_corr, weights, ret_grad=ret_grad, is_zero_grad=is_zero_grad) # real
        else:
            if self.use_grad_accumulation:
                return self.cal_update_grad_cplx_accumulation(e_loc_corr, weights, ret_grad=ret_grad, is_zero_grad=is_zero_grad) # complex
            else:
                return self.cal_update_grad_cplx(e_loc_corr, weights, ret_grad=ret_grad, is_zero_grad=is_zero_grad) # complex

    def cal_get_grad(self, e_loc_corr:np.array, weights:np.array):
        r"""Update model grad interface."""
        if self.psi_type == 'real':
            return self.cal_update_grad_real(e_loc_corr, weights, ret_grad=True) # real
        else:
            return self.cal_update_grad_cplx(e_loc_corr, weights, ret_grad=True) # complex

    def gen_samples(self, n_samples, n_elecs=0, ret_type='numpy'):
        r"""Generate samples interface."""
        self.n_epochs += 1 # count training steps
        self.wavefunction.set_n_epochs(self.n_epochs)

        if self.use_uniq_sampling_parallel:
            n_samples_max = self.n_samples_max
            n_samples_scale_factor = self.n_samples_scale_factor
            if self.n_epochs % self.incr_per_epoch == 0 and self.n_samples < n_samples_max:
                self.n_samples = int(min(self.n_samples * n_samples_scale_factor, n_samples_max))
                self.logger.info(f"increasing samples into: {self.n_samples} at {self.n_epochs}-th epoch")
            n_samples = self.n_samples

        use_profile = False
        if use_profile:
            import torch.autograd.profiler as profiler
            # with profiler.profile(with_stack=True) as prof:
            with profiler.profile(with_stack=True, use_cuda=True) as prof:
                if self.psi_type == 'real':
                    n_samples, samples, psis = self.gen_samples_nade_v2_real(n_sample=n_samples, ret_type=ret_type) # real
                else:
                    n_samples, samples, psis = self.gen_samples_nade_v2_cplx(n_sample=n_samples, ret_type=ret_type) # complex
            print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10), flush=True)
            return n_samples, samples, psis

        self.mcmc = False
        # self.mcmc = True
        if self.mcmc:
            return self.gen_samples_mcmc(n_samples=n_samples)

        if self.psi_type == 'real':
            return self.gen_samples_nade_v2_real(n_sample=n_samples, ret_type=ret_type) # real
        else:
            # self.logger.info(torch.cuda.memory_summary())
            return self.gen_samples_nade_v2_cplx(n_sample=n_samples, ret_type=ret_type) # complex

    def gen_samples_mcmc(self, n_samples=100):
        from src.sampler import MCMCSampler
        # all_in = torch.Tensor([[-1., -1.,  1.,  1.],
        #             [ 1., -1., -1.,  1.],
        #             [-1.,  1.,  1., -1.],
        #             [ 1.,  1., -1., -1.]])
        # log_psi_all = torch.stack(self.wavefunction._forward_pad(all_in), -1)
        # psi_all_mod = cplx.absolute_pow2_value(cplx.exp(log_psi_all))
        # print(f"log_psi_all: {log_psi_all}")
        # print(f"psi_all_mod: {psi_all_mod}")
        with torch.no_grad():
            mcmc = MCMCSampler(self.wavefunction._forward_pad, device=self.device)
            # states = mcmc.metropolis_hastings_batch(self.n_qubits, self.n_elecs, 8192, iterations=30, num_chains=128, num_warmup=1000)
            # states = mcmc.metropolis_hastings_batch(self.n_qubits, self.n_elecs, 4096, iterations=4, num_chains=128, num_warmup=20)
            # states = mcmc.metropolis_hastings_batch(self.n_qubits, self.n_elecs, 4096*1024, iterations=30, num_chains=1024*8, num_warmup=40)
            states = mcmc.metropolis_hastings_batch(self.n_qubits, self.n_elecs, 4096/10, iterations=30, num_chains=512, num_warmup=40)
            # states = mcmc.metropolis_hastings_batch(self.n_qubits, self.n_elecs, 4096*64, iterations=30, num_chains=128*64, num_warmup=1000)
            # states = mcmc.metropolis_hastings_batch(self.n_qubits, self.n_elecs, 4, iterations=100, num_chains=2, num_warmup=2)
            unique_states, counts = torch.unique(states, dim=0, return_counts=True)
            unique_states = unique_states.type(torch.int64).reshape(-1)
            # print(f"states: {unique_states}")
            # print(f"counts: {counts}")
            self.wavefunction.set_states(unique_states, None)
        self.ln_psis = torch.stack(self.wavefunction._forward_pad(unique_states), -1)
        psis = cplx.torch_to_numpy(cplx.exp(self.ln_psis).detach())
        return counts.numpy(), unique_states.numpy(), psis

    def gen_samples_nade_v2_real(self, n_sample=100, max_unique_samples=1024, ret_type="numpy"):
        self.wavefunction.sample()
        if self.use_samples_recursive:
            outs = self.gen_samples_recursive()
        else:
            outs = self.wavefunction.forward(n_sample)
        samples = outs[0].type(torch.int64).reshape(-1)
        n_samples = outs[1].reshape(-1)
        ##psis = torch.exp(outs[3][:,0]) * outs[3][:,1]
        self.psis = outs[3]
        psis = cplx.torch_to_numpy(outs[3].detach().to(self.out_device))
        # self.logger.info(f'n_uniq: {psis.shape[0]}')
        if ret_type == "torch":
            # return n_samples, samples, psis
            return n_samples, samples, torch.exp(outs[3][:,0]), outs[3][:,1]
        else:
            #return n_samples.numpy(), samples.numpy(), psis.detach().numpy()
            return n_samples.numpy(), samples.numpy(), psis

    def gen_samples_nade_v2_cplx(self, n_sample=100, max_unique_samples=1024, ret_type="numpy"):
        self.wavefunction.sample()
        if self.use_samples_recursive:
            outs = self.gen_samples_recursive()
        else:
            if self.psi_type == 'complex':
                outs = self.wavefunction.forward(n_sample, kth=self.n_sampling_layers)
            else:
                outs = self.wavefunction.forward(n_sample)
        samples = outs[0].type(torch.int64).reshape(-1)
        n_samples = outs[1].reshape(-1)
        self.ln_psis = outs[3]
        # print(f"ln_psis: {self.ln_psis}")
        psis = cplx.torch_to_numpy(cplx.exp(outs[3]).detach().to(self.out_device))
        #psis = cplx.torch_to_numpy((outs[3]).detach())
        #self.logger.info(f'n_uniq: {psis.shape[0]}')

        # save ci and psis
        #if self.n_epochs % 1 == 0:
        #    # psis是复数，表示CI系数，samples是量子态向量，n_samples表示对应样本的出现次数
        #    #np.savez(f'h2_nnqs_ci_psis.npz', n_samples=n_samples, samples=samples, psis=psis)
        #    np.savez(f'h2_nnqs_ci_psis.npz', n_samples=n_samples, samples=samples, amps=outs[3][:,0].detach().numpy(), phases=outs[3][:,1].detach().numpy())
        #    self.logger.info("save psis")

        # return n_samples, samples, psis
        if ret_type == "torch":
            return n_samples, samples, torch.exp(outs[3][:,0]), outs[3][:,1]
        else:
            return n_samples.numpy(), samples.numpy(), psis


    def gen_samples_recursive(self, last_action=0):
        r"""Generate samples recursive, increase/decrease batch_size according unique samples."""
        action = 0  # -1 decrease samples, 0 return samples, +1 increase samples
        # torch.cuda.empty_cache()

        self.wavefunction.sample()

        try:
            if self.psi_type == 'complex':
                states, counts, probs, log_psi = self.wavefunction.forward(self.n_samples, max_batch_size=self.n_unq_samples_max, kth=self.n_sampling_layers)
            else:
                states, counts, probs, log_psi = self.wavefunction.forward(self.n_samples, max_batch_size=self.n_unq_samples_max)
            n_unq, sampling_completed = len(states), True
        except MaxBatchSizeExceededError as mbe:
            self.logger.info(f"MaxBatchSizeExceededError: {mbe.cur_uniq}")
            #n_unq, sampling_completed = self.n_unq_samples_max + 1, False
            n_unq, sampling_completed = mbe.cur_uniq, False
            action = -1

        # Only check the number of unique samples if we can still increase/decrease the number of samples used.
        if ((self.n_samples != self.n_unq_samples_min) and (self.n_samples != self.n_samples_max)) or not sampling_completed:

            # If we want more unique samples, increase the sample size.
            if n_unq < self.n_unq_samples_min and last_action >= 0:
                action = 1
                self.n_samples = int( min(self.n_samples * self.n_samples_scale_factor, self.n_samples_max) )
                self.logger.info(f"\t...{n_unq} unique samples generated --> increasing batch size to {self.n_samples/1e6:.1f}M at epoch {self.n_epochs}.")

            # If we want more fewer samples, decrease the sample size.
            elif n_unq > self.n_unq_samples_max and last_action <= 0:
            #elif n_unq > self.n_unq_samples_max*1.2 and last_action <= 0:
                action = -1
                self.n_samples = int( max(self.n_samples / self.n_samples_scale_factor, self.n_unq_samples_min) )
                self.logger.info(f"\t...{n_unq} unique samples generated --> decreasing batch size to {self.n_samples/1e6:.1f}M at epoch {self.n_epochs}.")

        if action != 0:
            if sampling_completed:
                # If sampling was completed, let's clear as much memory as we can before trying again.
                if states.shape[0] > 0:
                    log_psi = log_psi.detach()
                    del states, counts, probs, log_psi
                torch.cuda.empty_cache()
            return self.gen_samples_recursive(action)
        else:
            # if self.device[0:4] == "cuda":
            #     self.logger.info(f'torch cuda memory allocated: {torch.cuda.memory_allocated() / 1024**2} MB')
            return states, counts, probs, log_psi

    def cal_update_grad_real(self, e_loc_corr:np.array, weights:np.array, ret_grad=False, is_zero_grad=True):
        # Complex grad: 2 * <<(E_loc - E_loc_avg) * log_Psi(x)>>
        # Real grad: 2 * <<(E_loc - E_loc_avg) * log_|Psi(x)|>>
        # e_loc_corr = E_loc - E_loc_avg
        #print(f"real [1]: weights, e_loc_corr: {weights.dtype} {e_loc_corr.dtype} {e_loc_corr.real.dtype}", flush=True)
        #e_loc_corr, weights = torch.tensor(e_loc_corr.real), torch.tensor(weights)
        e_loc_corr, weights = torch.tensor(e_loc_corr.real).to(self.global_fp), torch.tensor(weights).type(self.global_fp)
        e_loc_corr, weights = e_loc_corr.to(self.psis.device), weights.to(self.psis.device)
        #print(f"real [2]: weights, e_loc_corr: {weights.dtype} {e_loc_corr.dtype}", flush=True)
        #self.psis = self.psis.to(self.device)
        #e_loc_corr = e_loc_corr.to(self.device)
        #weights = weights.to(self.device)
        exp_op = 2 * ((e_loc_corr * self.psis[:,0].abs().log()) * weights).sum()

        #print(f"exp_op: {exp_op.device} self.psis: {self.psis.device}", flush=True)
        if is_zero_grad:
            self.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()

        exp_op.backward()

        if self.use_clip_grad:
            tot_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0, norm_type=2)
            # grad = self.collect_grads()
            # print("clip cal_update_grad_real: ", tot_norm, grad[:2], np.min(np.abs(grad)), np.mean(grad), np.min(grad), np.max(grad), flush=True)
        if ret_grad:
            grad = self.collect_grads()
            return grad
        #grad = self.collect_grads()
        #grad = torch.from_numpy(grad).abs()
        #print(f"real grad type: {grad.dtype} (mean, max, min) : {grad.mean(), grad.max(), grad.min()}", flush=True)

        if not is_zero_grad: # for pretrain accumulation
            return

        self.optimizer.step()
        if self.open_lr_scheduler:
            self.lr_scheduler.step()

    def cal_update_grad_cplx(self, e_loc_corr:np.array, weights:np.array, ret_grad=False, is_zero_grad=True):
        # Complex loss: 2 * <<(E_loc - E_loc_avg) * log_Psi*(x)>>
        # Complex grad: 2 * <<(E_loc - E_loc_avg) * \delta_\theta log_Psi*(x)>>
        # e_loc_corr = E_loc - E_loc_avg
        #e_loc_corr, weights = cplx.np_to_torch(e_loc_corr), torch.tensor(weights)
        e_loc_corr, weights = cplx.np_to_torch(e_loc_corr).type(self.global_fp), torch.tensor(weights).type(self.global_fp)
        #print(f"cplx [2]: weights, e_loc_corr: {weights.dtype} {e_loc_corr.dtype}", flush=True)
        if weights.dim() < 2:
            weights = weights.unsqueeze(-1)
        
        # def _reciprocal(_z):
        #     return cplx.conj(_z) / cplx.absolute_pow2_value(_z).sum()
        # constant_reg_schedule = lambda iter : epsilon if iter < vmc_iters//2 else 0.0
        # epsilon = 0.05
        # constant_reg_schedule = lambda iter : epsilon if iter < 600 else 0.0
        # regularizer = cplx.real(cplx.scalar_mult(self.ln_psis, _reciprocal(cplx.exp(self.ln_psis)))).sum()
        # loss_regularizer = (weights * regularizer).sum()

        self.ln_psis = cplx.conj(self.ln_psis)
        e_loc_corr, weights = e_loc_corr.to(self.ln_psis.device), weights.to(self.ln_psis.device)
        exp_op = 2 * cplx.real(weights * cplx.scalar_mult(self.ln_psis, e_loc_corr)).sum(axis=0)

        # epsilon = constant_reg_schedule(self.n_epochs)
        # exp_op -= epsilon * loss_regularizer
        # print(f"epsilon: {epsilon} loss_regularizer: {loss_regularizer}", flush=True)

        if is_zero_grad:
            self.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()

        exp_op.backward()
        # grad_norm_after = np.sum(np.abs(self.collect_grads()))
        # print(f"grad_norm_before: {grad_norm_before} grad_norm_after: {grad_norm_after}", flush=True)
        # print(f"grad_norm_after: {grad_norm_after}", flush=True)

        if self.use_clip_grad:
            tot_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0, norm_type=2)
            #grad = self.collect_grads()
            #print(f"grad sum: {np.sum(grad)} mean: {np.mean(grad)} {grad[0:10]}", flush=True)
        if ret_grad:
            grad = self.collect_grads()
            return grad

        if not is_zero_grad: # for pretrain accumulation
            return
        #grad = self.collect_grads()
        #grad = torch.from_numpy(grad).abs()
        #print(f"cplx grad type: {grad.dtype} (mean, max, min) : {grad.mean(), grad.max(), grad.min()}", flush=True)

        self.optimizer.step()
        if self.open_lr_scheduler:
            self.lr_scheduler.step()

    def cal_update_grad_real_accumulation(self, e_loc_corr: np.array, weights: np.array, ret_grad=False, is_zero_grad=True):
        # Real grad: 2 * <<(E_loc - E_loc_avg) * log_|Psi(x)|>>
        # e_loc_corr = E_loc - E_loc_avg
        #print(f"real grad_accuv2 [1]: weights, e_loc_corr: {weights.dtype} {e_loc_corr.dtype} {e_loc_corr.real.dtype}", flush=True)
        #e_loc_corr, weights = torch.tensor(e_loc_corr.real), torch.tensor(weights)
        e_loc_corr, weights = torch.tensor(e_loc_corr.real).to(self.global_fp), torch.tensor(weights).type(self.global_fp)
        e_loc_corr, weights = e_loc_corr.to(self.psis.device), weights.to(self.psis.device)
        #print(f"real grad_accuv2 [2]: weights, e_loc_corr: {weights.dtype} {e_loc_corr.dtype}", flush=True)

        states = self.wavefunction.states
        grad_batch_size = self.grad_accumulation_width

        if is_zero_grad:
            self.optimizer.zero_grad()

        #e_loc_corr = e_loc_corr.to(self.device)
        #weights = weights.to(self.device)
        data_batches = zip(torch.split(states, grad_batch_size),
                           torch.split(weights, grad_batch_size),
                           torch.split(e_loc_corr, grad_batch_size))

        for states_i, weights_i, e_loc_corr_i in data_batches:
            amps_i, phases_i = self.wavefunction._forward_pad(states_i, self.wavefunction._masking)
            #amps_i = amps_i.to(self.device)
            exp_op = 2 * ((e_loc_corr_i * amps_i.abs().log()) * weights_i).sum()
            exp_op.backward()

        if self.use_clip_grad:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0, norm_type=2)

        if ret_grad:
            grad = self.collect_grads()
            return grad

        #grad = self.collect_grads()
        #grad = torch.from_numpy(grad).abs()
        #print(f"real_accu grad type: {grad.dtype} (mean, max, min) : {grad.mean(), grad.max(), grad.min()}", flush=True)

        if not is_zero_grad: # for pretrain accumulation
            return

        self.step()

    def cal_update_grad_cplx_accumulation(self, e_loc_corr:np.array, weights:np.array, ret_grad=False, is_zero_grad=True):
        # Complex loss: 2 * <<(E_loc - E_loc_avg) * log_Psi*(x)>>
        # Complex grad: 2 * <<(E_loc - E_loc_avg) * \delta_\theta log_Psi*(x)>>
        # e_loc_corr = E_loc - E_loc_avg
        #e_loc_corr, weights = cplx.np_to_torch(e_loc_corr), torch.tensor(weights)
        e_loc_corr, weights = cplx.np_to_torch(e_loc_corr).type(self.global_fp), torch.tensor(weights).type(self.global_fp)
        e_loc_corr, weights = e_loc_corr.to(self.ln_psis.device), weights.to(self.ln_psis.device)
        if weights.dim() < 2:
            weights = weights.unsqueeze(-1)

        if is_zero_grad:
            self.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()

        states = self.wavefunction.states
        nstates = states.shape[0]
        grad_batch_size = self.grad_accumulation_width
        partitions = [i * grad_batch_size for i in range(int(np.ceil(nstates/grad_batch_size)))]
        partitions.append(nstates)
        for i in range(len(partitions)-1):
            st, ed = partitions[i], partitions[i+1]
            states_i, weights_i, e_loc_corr_i = states[st:ed, ...], weights[st:ed, ...], e_loc_corr[st:ed, ...]
            amps_i, phases_i = self.wavefunction._forward_pad(states_i, self.wavefunction._masking)
            log_psis_i = torch.stack([amps_i, phases_i], -1)
            log_psis_i = cplx.conj(log_psis_i)
            exp_op = 2 * cplx.real(weights_i * cplx.scalar_mult(log_psis_i, e_loc_corr_i)).sum(axis=0)
            exp_op.backward()
            print(f"accum exp_op: {exp_op.device}")

        if self.use_clip_grad:
            tot_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0, norm_type=2)
        # grad = self.collect_grads()
        # print(f"grad sum: {np.sum(grad)} mean: {np.mean(grad)} {grad[0:4]}", flush=True)
        if ret_grad:
            grad = self.collect_grads()
            return grad

        #grad = self.collect_grads()
        #grad = torch.from_numpy(grad).abs()
        #print(f"cplx_accu grad type: {grad.dtype} (mean, max, min) : {grad.mean(), grad.max(), grad.min()}", flush=True)

        if not is_zero_grad: # for pretrain accumulation
            return

        self.step()

    def set_grad_and_step(self, model_grads, to_cuda=False):
        to_cuda = False
        if self.device[:4] == "cuda":
            to_cuda = True
        self.set_grads_(model_grads, to_cuda)
        self.step()

    def step(self):
        self.optimizer.step()
        if self.open_lr_scheduler:
            self.lr_scheduler.step()

    def zero_optimizer(self):
        self.optimizer.zero_grad()

    def set_grads_(self, model_grads, to_cuda=False):
        r"""Set new grads from 1D numpy array"""
        cuda_device = None
        if to_cuda:
            cuda_device = self.device
        set_grads_impl(self, model_grads, cuda_device)

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def frozen_parameters(self, key="amp_layers"):
        cnt = 0
        for name, param in self.named_parameters():
            if key in name:
                cnt += 1
                param.requires_grad = False
        self.logger.info(f"frozen {key} cnt: {cnt}")

def debug_simple(cfg_file="config-h4.yaml"):
    r"""Simple debug example."""
    cfgs = read_yaml(cfg_file)
    mol = cfgs['system']
    seed = int(cfgs['seed'])
    qubit_dict = {'h2': 4, 'lih': 12, 'h2o': 14, 'h10': 20, 'c2': 20, 'h2s': 22, 'c2h4o': 38, 'li2o': 30, 'c2h4o': 38, 'cna2o3': 76, 'h2_f': 64, 'h_h_f': 64, 'h_hf': 64}
    n_qubits = qubit_dict[mol.lower()]
    wtrain = wrapper_train(cfg_file, n_qubits=n_qubits, n_rank=1, rank=0, seed=seed, pos_independ=False)

    # just for sampling parallel load balancing debug
    # n_samples = 100000
    # cnts_list = []
    # n_rank = 8
    # wtrain = wrapper_train(cfg_file, n_qubits=62, n_rank=n_rank, rank=rank, seed=111, pos_independ=False, close_log=True) # H2O4S
    # for rank in range(n_rank):
    #     wtrain.set_rank(n_rank, rank)
    #     cnts, dec_inputs, final_prob = wtrain.gen_samples(n_samples)
    #     print(f"rank: {rank} cnts: {cnts.shape}")
    #     cnts_list.append(cnts.shape[0])
    # print(f"cnts_list: {cnts_list}")
    # return

    n_samples = 10000
    st = time.time()
    cnts, dec_inputs, final_prob = wtrain.gen_samples(n_samples)
    ed = time.time()
    print(f"DecoderQS: gen_samples: {(ed-st)*1000} ms")
    print("cnts: ", cnts.shape, cnts[:8])
    print("final_prob:", final_prob[:8])
    print('samples: ', dec_inputs.reshape(final_prob.shape[0], -1)[:8])
    print('psis get from train: ', final_prob[:8])

if __name__ == "__main__":
    debug_simple("config-h4.yaml")

