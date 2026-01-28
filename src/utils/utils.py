"""
Copyright wuyangjun21@163.com 2023-02-10
"""

import yaml
import random
import numpy as np
import os
import logging
from enum import Enum
from multiprocessing import cpu_count

import torch
from torch import nn
import torch.nn.functional as F

import openfermion

from time import time_ns

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Timer(metaclass=SingletonMeta):
    __timers = {}
    __clocks = {}
    def __init__(self):
        pass

    @classmethod
    def start(cls, name:str):
        cls.__clocks[name] = time_ns()

    @classmethod
    def stop(cls, name:str):
        t = time_ns()
        if not name in cls.__clocks:
            cls.__timers[name] = -1
        else:
            if not name in cls.__timers:
                cls.__timers[name] = 0.0
            cls.__timers[name] += (t - cls.__clocks[name])/1e9

    @classmethod
    def reset(cls):
        for c in cls.__timers:
            cls.__timers[c] = 0.0
    
    @classmethod
    def reset(cls, *args):
        for name in args:
            if name in cls.__timers:
                cls.__timers[name] = 0.0

    @classmethod
    def display(cls, *args, precision=4):
        out = "[Timing]\t"
        for name in args:
            if name in cls.__timers:
                out += f"{name}: {cls.__timers[name]:.{precision}f}\t"
        print(out + "seconds", flush=True)

    @classmethod
    def summary(cls):
        print(cls.__timers, flush=True)

    @classmethod
    def get_time(cls, key_name):
        return cls.__timers[key_name]

def get_logger(level=logging.INFO, name="NNQS"):
    r"""Get logger with default format."""
    if isinstance(level, int):
        level = level
    elif isinstance(level, str):
        level = getattr(logging, level.upper())
    else:
        TypeError("logging paramter: level must be int or str")
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M')
    logger = logging.getLogger(name)
    return logger

def test_logger():
    r"""Test logger."""
    logger = get_logger(level=logging.WARNING)
    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")
    logger.error("Finish", exc_info=True)

def get_proc_block_range(nele, np, i):
    r"""Get block partition of data.
    Args:
        nele: number of element, start from 0
        np: number of process.
        i: the id of process, start from 0
    Returns:
        Element range of i-th process: [st, ed)
    """
    nl = nele//np
    st = i*nl
    ed = (i+1)*nl
    if i == np-1:
        ed = nele
    return st, ed

def count_parameters(model, print_verbose=True):
    if print_verbose:
        print("---modules : parameters---\n")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if print_verbose:
            print(f"{name} : {param}")
        total_params += param
    print(f"\n--> Total trainable params: {total_params}", flush=True)
    return total_params

def collect_parameters(net, device='cpu'):
    params = list(net.parameters())
    param_values = []
    for p in params:
        param_values.append(p.data.view(-1))
    param_values = torch.cat(param_values, 0)
    if device[:4] == 'cuda':
        param_values = param_values.cpu()
    return param_values.numpy()

def set_global_seed(seed=-1, dump_verbose=True):
    r"""Set global seed."""
    if seed < 0:
        seed = random.randint(0, 2 ** 32)
    if dump_verbose:
        print("\n------------------------------------------")
        print(f"\tSetting global seed using {seed}.")
        print("------------------------------------------\n", flush=True)
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** 32))
    torch.manual_seed(random.randint(0, 2 ** 32))
    torch.cuda.manual_seed(random.randint(0, 2 ** 32))
    torch.cuda.manual_seed_all(random.randint(0, 2 ** 32))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

def set_num_threads(cpu_num=None):
    r"""Set number of threads."""
    if cpu_num is None:
        cpu_num = cpu_count() # max cpu cores
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    print(f"==[LOG]== Set num of threads into {cpu_num}")

def multinomial_arr(count, p):
    r"""Multinomial distribution sampling."""
    N = len(count)
    assert len(p) == N, "Counts and probs must have same first dimension"

    count = np.copy(count)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count

    return out

@torch.no_grad()
def multinomial_torch(_counts:torch.tensor, _probs:torch.tensor, PROB_TYPE=torch.float64, COUNT_TYPE=torch.long):
        org_dtype = torch.get_default_dtype()
        torch.set_default_dtype(PROB_TYPE)
        probs = _probs.clone().to(PROB_TYPE)
        counts = _counts.clone().to(COUNT_TYPE)
        probs_cum = torch.cumsum(probs, dim=-1)
        # probs_cum.clamp_(1e-12)
        probs_cond = torch.nan_to_num(probs / probs_cum)
        out = torch.zeros_like(_probs, dtype=COUNT_TYPE)
        for i in range(out.shape[-1]-1, 0, -1):
                binsample = torch.distributions.Binomial(counts, probs_cond[..., i]).sample().to(COUNT_TYPE)
                out[..., i] = binsample
                counts -= binsample
        out[..., 0] = counts
        torch.set_default_dtype(org_dtype)
        return out

def read_yaml(file_path="config.yaml"):
    r"""Read configurations from yaml file."""
    file = open(file_path, 'r', encoding='utf-8')
    strings = file.read()
    return yaml.safe_load(strings)

def collect_grads_impl(model):
    r"""Collect grad from tensors and return a 1D numpy array."""
    model_grads = []
    for name, parms in model.named_parameters():
        # print(parms.grad.flatten().shape, parms.grad.shape, parms.grad.nelement())
        model_grads.append(parms.grad.flatten())
    # model_grads = torch.cat(model_grads).numpy()
    # model_grads = torch.cat(model_grads).cpu().numpy()
    model_grads = torch.cat(model_grads).cpu().detach().numpy()
    return model_grads

def set_grads_impl(model, model_grads, cuda_device=None):
    r"""Set new grads from 1D numpy array, must called after backward()."""
    # copy CPU grads into cuda:i
    if not (cuda_device is None):
        model_grads = torch.from_numpy(model_grads).to(cuda_device)

    cnt = 0
    for name, params in model.named_parameters():
        # if params.grad == None:
        #     continue
        nel = params.grad.nelement()
        # print(f"type: {type(params.grad)}, {type(model_grads)}")
        #params.grad = torch.Tensor(model_grads[cnt:cnt+nel]).reshape(params.grad.shape)
        # params.grad = torch.tensor(model_grads[cnt:cnt+nel]).reshape(params.grad.shape)
        params.grad = model_grads[cnt:cnt+nel].clone().detach().requires_grad_(True).reshape(params.grad.shape)
        cnt += nel

class AmplitudeEncoding(Enum):
    AMP = 0
    LOG_AMP = 1
    PROB = 2
    LOG_PROB = 3

class InputEncoding(Enum):
    BINARY = 0
    INTEGER = 1

class NadeMasking(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

class _MaskedSoftmaxBase(nn.Module):

    def mask_input(self, x, mask, val):
        if mask is not None:
            m = mask.clone() # Don't alter original
            if m.dtype == torch.bool:
                x_ = x.masked_fill(~m.to(x.device), val)
            else:
                x_ = x.masked_fill((1 - m.to(x.device)).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

class SoftmaxLogProbAmps(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.LOG_AMP
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(x, mask, self.masked_val)
        return F.log_softmax(x_, dim=dim)

    # def forward(self, x, mask=None, dim=1):
    #     x_ = self.mask_input(x, mask, self.masked_val)
    #     return 0.5*F.log_softmax(x_, dim=dim)

    # def forward(self, x, mask=None, dim=1):
    #     x_ = self.mask_input(2*x, mask, self.masked_val)
    #     return 0.5 * F.log_softmax(x_, dim=dim)

class TanhProbAmps(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.LOG_AMP
    masked_val = float(0)

    def forward(self, x, mask=None, dim=1):
        #print(f'input_x: {x.shape}\nmask: {mask}')
        #x_ = self.mask_input(x, mask, self.masked_val)
        x_ = x
        #x_ = x / (x**2).sum(dim=-1).reshape(-1, 1) # TODO normlize
        #print(f'mask_x: {x_}\ntanh(x_): {F.tanh(x_)}')
        return torch.tanh(x_)
        #return F.tanh(x_)

class MaxBatchSizeExceededError(Exception):
    def __init__(self, cur_uniq):
        self.cur_uniq = cur_uniq

class OrbitalBlock(nn.Module):

    def __init__(self,
                 num_in=2,
                 n_hid=[],
                 num_out=4,
                 tgt_vocab_size=2, # 0/1
                 d_model=12, # embedding size
                 use_embedding=True,

                 hidden_activation=nn.ReLU,
                 bias=True,
                 batch_norm=True,
                 batch_norm_momentum=0.1,
                 out_activation=None,
                 max_batch_size=250000,
                 device=None,
                 max_transfer=0, # for transfer learning
                 logger=None,
                 ):
        super().__init__()

        self.num_in = num_in
        self.n_hid = n_hid
        self.num_out = num_out

        self.device = device
        self.max_transfer = max_transfer

        self.layers = []
        # Embedding
        self.use_embedding = use_embedding
        if use_embedding:
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_emb = nn.Embedding(64, d_model)
            num_in = d_model
            # self.layers.append(self.tgt_emb)
            print(f"using phase embedding tgt_vocab_size: {tgt_vocab_size}, d_model: {d_model}")

        layer_dims = [num_in] + n_hid + [num_out]
        logger.info(f"phase layer dims: {layer_dims}")
        for i, (n_in, n_out) in enumerate(zip(layer_dims, layer_dims[1:])):
            if batch_norm:
                l = [nn.Linear(n_in, n_out, bias=False), nn.BatchNorm1d(n_out, momentum=batch_norm_momentum)]
            else:
                l = [nn.Linear(n_in, n_out, bias=bias)]
            if (hidden_activation is not None) and i < len(layer_dims) - 2:
                l.append(hidden_activation())
            elif (out_activation is not None) and i == len(layer_dims) - 2:
                l.append(out_activation())
            l = nn.Sequential(*l)
            self.layers.append(l)

        self.max_batch_size = max_batch_size

        self.layers = nn.Sequential(*self.layers)

    def forward(self, _x):
        x = _x
        param_dtype = next(self.layers.parameters()).dtype
        x = x.type(param_dtype)

        if self.max_transfer > 0:
            # print(f"max_transfer: {self.max_transfer}")
            assert _x.shape[1] <= self.num_in
            # x = torch.zeros(_x.shape[0], self.num_in, device=self.device).type(torch.float32)
            x = torch.zeros(_x.shape[0], self.num_in, device=self.device).type(param_dtype)
            x[:, :_x.shape[1]] = _x

        if self.use_embedding:
            x = x.type(torch.int32) # -1/+1
            x = x.masked_fill(x==-1, 0)
            pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=self.device).unsqueeze(0) # shape (1, t)
            x = self.tgt_emb(x) + self.pos_emb(pos)
            # print(f"x: {x}")

        if len(x) <= self.max_batch_size:
            return self.layers(x)
        else:
            return torch.cat([self.layers(x_batch) for x_batch in torch.split(x, self.max_batch_size)])
        # return self.layers(x.clamp(min=0))

class OrbitalLUT(nn.Module):

    def __init__(self,
                 num_in=1,
                 dim_vals_in=2,
                 num_out=4,
                 out_activation=None):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.dim_vals_in = dim_vals_in

        self._dim_lut = (self.dim_vals_in ** self.num_in, self.num_out)
        self._idx_basis_vec = torch.LongTensor([self.dim_vals_in ** n for n in range(self.num_in)])

        lut = torch.randn(self._dim_lut)
        self.lut = nn.Parameter(lut, requires_grad=True)

        if out_activation is None:
            self.out_activation = out_activation()
        else:
            self.out_activation = out_activation

    @torch.no_grad()
    def _state2idx(self, s):
        return ((s.to(self._idx_basis_vec) > 0).long() * self._idx_basis_vec[:s.shape[-1]]).sum(-1)

    def forward(self, x):
        x_idx = self._state2idx(x)
        out = self.lut[x_idx]
        if self.out_activation:
            out = self.out_activation(out)
        return out

    def __repr__(self):
        str = f"ShellLUT(num_in={self.num_in}, dim_vals_in={self.dim_vals_in}, num_out={self.num_out})"
        str += f" --> lut dim = {self._dim_lut}"
        return str


def read_binary_qubit_op(filename: str = "qubit_op.data"):
    """
    """
    f = open(filename, "rb")
    identifier = f.read(8)
    # if identifier != bytes.fromhex("4026828f5c28f5c3"):  # 0x402682a9930be0df
    if np.frombuffer(identifier, dtype=np.float64) != 11.2552:
        raise ValueError("The file is not saved by QCQC.")

    n_qubits = np.frombuffer(f.read(4), dtype=np.int32)
    n_qubits = int(n_qubits)

    get_pauli_symbol = {
        0: "I",
        1: "X",
        2: "Y",
        3: "Z"
    }

    qubit_op = openfermion.QubitOperator()

    pauli_str_array = np.zeros([n_qubits], dtype=np.int32)
    chunk_size = n_qubits * 4
    coeff_bin = f.read(16)
    pauli_str_bin = f.read(chunk_size)
    while len(coeff_bin) != 0 and len(pauli_str_bin) != 0:
        assert(len(pauli_str_bin) == chunk_size)
        coeffs_array = np.frombuffer(coeff_bin, dtype=np.float64)
        pauli_str_array = np.frombuffer(pauli_str_bin, dtype=np.int32)
        coeff = coeffs_array[0] + 1.j * coeffs_array[1]
        term_list = []
        for pos, pauli_number in enumerate(pauli_str_array):
            pauli_symbol = get_pauli_symbol[pauli_number]
            if pauli_symbol != "I":
                term_list.append((pos, pauli_symbol))
        op_cur = openfermion.QubitOperator(tuple(term_list), coeff)
        qubit_op += op_cur

        coeff_bin = f.read(16)
        pauli_str_bin = f.read(chunk_size)

    return n_qubits, qubit_op

def calculate_total_memory(tensors):
    """
    Calculate the total memory size occupied by multiple PyTorch tensors.

    Args:
        tensors (list of torch.Tensor): A list of PyTorch tensors.

    Returns:
        int: The total memory size in bytes.
    """
    total_size = sum(tensor.element_size() * tensor.numel() for tensor in tensors)
    return total_size
