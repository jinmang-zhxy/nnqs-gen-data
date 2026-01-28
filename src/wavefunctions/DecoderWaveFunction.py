import time
import math
import numpy as np
from collections.abc import Iterable

import torch
from torch import nn
#from thop import profile

# select different decoder implementation
from src.networks.mingpt.model import get_decoder_amp
from src.utils.utils import set_global_seed, multinomial_arr, multinomial_torch, \
    count_parameters, NadeMasking, OrbitalBlock, OrbitalLUT, \
    MaxBatchSizeExceededError, get_logger, get_proc_block_range, SoftmaxLogProbAmps

# from src.networks.networks import ResidualMLP, DecoderOnlyTransformer
from src.wavefunctions.base import Base
import src.utils.complex_helper as cplx

def get_weight_block_partition(NP, rank, weight:np.array, n_epochs=None):
    """Get current rank assigned tasks idx: [idx_l, idx_r)"""
    w_tot = np.sum(weight)
    w_avg = w_tot / NP
    w_cumsum = np.cumsum(weight)
    # print(f"rank: {rank} weight: {weight}", flush=True)
    w_tgt_l = rank * w_avg
    w_tgt = (rank+1) * w_avg
    # idx_l = np.searchsorted(w_cumsum, w_tgt_l)+1
    idx_l = np.searchsorted(w_cumsum, w_tgt_l)
    if rank > 0:
        idx_l += 1
    idx_r = np.searchsorted(w_cumsum, w_tgt)+1
    # print(f"NP: {NP} n_epoch: {n_epochs} len: {weight.shape[0]} w_tot: {w_tot} w_avg: {w_avg}; rank: {rank} -> [{idx_l}, {idx_r}) w_tgt_l: {w_tgt_l} w_tgt: {w_tgt}", flush=True)
    return idx_l, idx_r

def cross_entropy(p, q):
    res = 0
    for i, v in enumerate(q):
        res += p[i] * torch.log(p[i]/q[i])

    return res

def get_temp(epoch, init_temp=2.0, final_temp=1.0, num_epochs=10000):
    temp = init_temp - (init_temp - final_temp) * (epoch / num_epochs)
    if epoch > num_epochs:
        return final_temp
    return temp

class DecoderWaveFunction(Base):
    def __init__(self,
                #  cfg_file,
                 config,
                 num_qubits,

                 num_lut=0,
                 n_electrons=None,
                 n_alpha_electrons=None,
                 n_beta_electrons=None,
                 masking = NadeMasking.PARTIAL,

                 amp_hidden_activation=nn.ReLU,

                 phase_hidden_size=[],
                 phase_hidden_activation=nn.ReLU,
                 phase_bias=True,

                 combined_amp_phase_blocks = False,

                 use_amp_spin_sym=True,
                 use_phase_spin_sym=True,
                 aggregate_phase=True,

                 amp_batch_norm=False,
                 phase_batch_norm=False,
                 batch_norm_momentum=1,

                 amp_activation=SoftmaxLogProbAmps,
                 phase_activation=None,

                 device=None,
                 out_device="cpu",
                 logger = None,
                 seed = -1
                 ):

        super().__init__()

        self.uniq_sampling_parallel = False
        self.n_rank = 0
        self.rank = 0

        self.seed = None
        self.n_epochs = None
        self.weight_list = []
        self.param_list = []

        # set logger level
        self.logger = logger
        if logger is None:
            log_level = config.log_level
            self.logger = get_logger(level=log_level)
        self.logger.info(f"phase_hidden_size: {phase_hidden_size}")

        self._set_is_complex_wf(True)
        self.use_amp_infer_batch = bool(config.use_amp_infer_batch)
        if self.use_amp_infer_batch:
            self.amp_infer_batch_size = int(config.amp_infer_batch_size)
            self.logger.info(f"amp_infer_batch_size: {self.amp_infer_batch_size}")

        self.use_phase_embedding = bool(config.use_phase_embedding)
        self.max_transfer = int(config.max_transfer)

        self.device = device
        self.out_device = out_device

        try:
            self.device_get_conditional_output = config.device_get_conditional_output
        except:
            self.device_get_conditional_output = None

        self.amp_batch_norm = amp_batch_norm
        self.phase_batch_norm = phase_batch_norm
        self.batch_norm_momentum = batch_norm_momentum

        self.num_qubits = self.N = num_qubits
        self.num_lut = num_lut

        self.n_tot_up = n_electrons
        self.n_alpha_up = n_alpha_electrons
        self.n_beta_up = n_beta_electrons
        self.masking = masking

        self.min_partition_samples = int(config.min_partition_samples)
        self.partition_algo = config.partition_algo

        self.drop_samples = bool(config.drop_samples)
        self.drop_samples_eps = float(config.drop_samples_eps)
        if self.drop_samples:
            self.logger.info(f"drop_samples_eps: {self.drop_samples_eps}")

        self.use_grad_accumulation = bool(config.use_grad_accumulation)

        # self.use_random_start_idx = True
        self.use_random_start_idx = False
        # self.topk_cutoff = True
        self.topk_cutoff = False
        if self.topk_cutoff:
            # self.topk_bs = 6000
            # self.topk_bs = 8000
            self.topk_bs = 30000
            self.cutoff_factor = 2
            self.logger.info(f"topk_cutoff: bs={self.topk_bs} cutoff_factor: {self.cutoff_factor}")

        if self.use_random_start_idx:
            self.logger.info(f"use random start idx: {self.use_random_start_idx}")

        self.sampling_algo = config.sampling_algo
        if self.sampling_algo == "dfs":
            self.sampling_dfs_width = int(config.sampling_dfs_width)
            self.sampling_dfs_uniq_samples_min = int(config.sampling_dfs_uniq_samples_min)
            self.logger.info(f"sampling_algo: dfs sampling_dfs_width: {self.sampling_dfs_width} sampling_dfs_uniq_samples_min: {self.sampling_dfs_uniq_samples_min}")

        # self.comb_phase = True
        self.comb_phase = False

        self.use_kv_cache = bool(config.use_kv_cache)
        if self.use_kv_cache:
            self.logger.info(f"use_kv_cache for sampling forward")

        self.use_restricted_hilbert, self._alpha_beta_restricted = False, False
        if (self.n_tot_up is not None) and ((self.n_alpha_up is None) and (self.n_beta_up is None)):
            self.use_restricted_hilbert, self._alpha_beta_restricted = True, False
            self.n_tot_down = self.N - self.n_tot_up
            self._min_n_set = min(self.n_tot_up, self.n_tot_down)
            self.logger.info(f"DecoderWaveFunction configured for f{self.n_tot_up} total electrons.")

        elif (self.n_alpha_up is not None) and (self.n_beta_up is not None):
            self.use_restricted_hilbert, self._alpha_beta_restricted = True, True

            if not isinstance(self.n_alpha_up, Iterable):
                self.n_alpha_up = [self.n_alpha_up]
            if not isinstance(self.n_beta_up, Iterable):
                self.n_beta_up = [self.n_beta_up]
            self.n_alpha_up, self.n_beta_up = np.array(self.n_alpha_up), np.array(self.n_beta_up)
            assert len(self.n_alpha_up)==len(self.n_beta_up), "Possible options for number of alpha/beta electrons do not match."
            self.n_tot_up = self.n_alpha_up[0] + self.n_beta_up[0]
            assert all( (self.n_alpha_up+self.n_beta_up)==self.n_tot_up ), "Possible options for number of alpha/beta electrons do not match."

            self.n_alpha_down = math.ceil(self.N / 2) - self.n_alpha_up
            self.n_beta_down = math.floor(self.N / 2) - self.n_beta_up
            self.n_tot_down = self.n_alpha_down + self.n_beta_down

            self._min_n_set = np.min(np.concatenate([self.n_alpha_up, self.n_beta_up, self.n_alpha_down, self.n_beta_down]))
            self.logger.info(f"DecoderWaveFunction configured for {self.n_tot_up} total electrons ({self.n_alpha_up}/{self.n_beta_up} spin up/down).")

        # "none"/"restricted"/"unrestricted"
        self.electron_conservation_type = config.electron_conservation_type.lower()
        valid_electron_conservation_types = ["none", "restricted", "unrestricted"]
        if self.electron_conservation_type not in valid_electron_conservation_types:
            ValueError("electron_conservation_type must be one of ", valid_electron_conservation_types)
        self.logger.info(f"electron_conservation_type: {self.electron_conservation_type}")

        if self.electron_conservation_type == "restricted":
            self.use_restricted_hilbert = True

        if self.use_restricted_hilbert:
            self.logger.info("Restricted hilbert spaces.")
        else:
            self._min_n_set = 0

        self.use_amp_spin_sym = use_amp_spin_sym
        self.use_phase_spin_sym = use_phase_spin_sym

        self.aggregate_phase = aggregate_phase
        self.combined_amp_phase_blocks = combined_amp_phase_blocks
        self.logger.info("Configuring spin symettries:")
        self.logger.info(f"\t--> use for amplitude = {self.use_amp_spin_sym}")
        self.logger.info(f"\t--> use for phase = {self.use_phase_spin_sym}")

        if (self.N % 2 != 0):
            raise ValueError(f"Symmetric NADE requires an even number of qubits. now n_qubits={self.N}")

        if (torch.cuda.device_count() >= 1) and (self.device[:4] == 'cuda'):
            self.logger.info(f"GPU found : model --> cuda")

        # For each shell...
        # If spin sym we need 5 outputs for |00>, |01>==|10>, |11>, |01>, |10>
        # Otherwise we need 4 outputs for |00>, |01>, |10>, |11>.
        self._n_out_amp = 5 if self.use_amp_spin_sym else 4

        # For each shell...
        # If spin sym we need 2 outputs for |00>/|11>, |01>/|10>.
        # Otherwise we need 4 outputs for |00>, |01>, |10>, |11>.
        self._n_out_phase = 3 if self.use_phase_spin_sym else 4
        self._n_out_phase = 1
        # self.phase_in = 0 # org: x1...x{n-1}
        self.phase_in = 2 # x1...xn
        # self.phase_in = 1 # x1x2->v1
        self.logger.info(f"phase_in: {self.phase_in}")

        self.amp_layers, self.phase_layers = [], []

        # phase sub-network
        n = self.N//2 - 1
        n_in = 2*n + self.phase_in
        if self.phase_in == 1:
            n_in = self.N // 2

        if self.max_transfer > 0:
            n_in = self.max_transfer
        dim_vals_in = 2
        n_in = max(1, n_in) # Make sure we have at least one input (i.e. for the first block).

        make_with_phase = self.aggregate_phase or (n == self.N // 2 - 1)
        out_activation = None

        # TODO: some phase sub-network experiments
        # phase_i = ResidualMLP(n_in, 512, self._n_out_phase, 2).to(self.device)
        # self.phase_layers.append(phase_i)
        # break
        # # phase_i = DecoderOnlyTransformer(d_model=16, nhead=4, num_layers=4, dim_feedforward=64, device=self.device).to(self.device)
        # # phase_i = DecoderOnlyTransformer(d_model=64, nhead=4, num_layers=4, dim_feedforward=256, device=self.device).to(self.device)
        # # phase_i = DecoderOnlyTransformer(d_model=128, nhead=8, num_layers=4, dim_feedforward=256, device=self.device).to(self.device)
        # phase_i = DecoderOnlyTransformer(d_model=64, nhead=8, num_layers=4, dim_feedforward=256, device=self.device).to(self.device)
        # self.phase_layers.append(phase_i)
        # break

        if make_with_phase and not self.combined_amp_phase_blocks:
            if n < num_lut:
                # Use an explicit look-up-table (lut) for this qubit.
                phase_i = OrbitalLUT(num_in=n_in,
                                        dim_vals_in=dim_vals_in,
                                        num_out=self._n_out_phase,
                                        out_activation=out_activation)
            else:
                phase_i = OrbitalBlock(num_in=n_in,
                                        n_hid=phase_hidden_size,
                                        # n_hid=[min(x, num_possible_layer_inputs(n)) for x in phase_hidden_size],
                                        num_out=self._n_out_phase,
                                        hidden_activation=phase_hidden_activation,
                                        use_embedding=self.use_phase_embedding,
                                        bias=phase_bias,
                                        batch_norm=self.phase_batch_norm,
                                        batch_norm_momentum=self.batch_norm_momentum,
                                        out_activation=out_activation,
                                        device=self.device,
                                        max_transfer=self.max_transfer,
                                        logger=self.logger).to(self.device)
            self.phase_layers.append(phase_i)

        # amplitude sub-network
        self.amp_layers, self.model_config = get_decoder_amp(config, self.num_qubits, device=self.device)
        self.amp_layers = self.amp_layers.to(self.device)
        if not self.comb_phase:
            self.phase_layers = nn.ModuleList(self.phase_layers)

        if amp_activation is not None:
            self.amplitude_activation = amp_activation()
        else:
            self.amplitude_activation = amp_activation

        if phase_activation is not None:
            self.phase_activation = phase_activation()
        else:
            self.phase_activation = phase_activation
        self.combine_amp_phase(True)

        self._idx_shell_basis_vec, self._idx_spin_basis_vec = None, None
        if self.N < 64:
            self._idx_shell_basis_vec = torch.LongTensor([2 ** n for n in range(self.N // 2)])
            self._idx_spin_basis_vec = torch.LongTensor([2 ** n for n in range(self.N)])

        self.qubit_order = config.qubit_order
        self.update_qubit2model_permutations(self.num_qubits, dump_verbose=False)
        self.logger.info("model prepared.")
    
    def update_qubit2model_permutations(self, num_qubits, dump_verbose=False):
        self.num_qubits = self.N = num_qubits
        if self.qubit_order == -1:
            self.qubit2model_permutation = torch.stack([torch.arange(self.num_qubits-2,-1,-2), torch.arange(self.num_qubits-1,-1,-2)],1).reshape(-1)
            self.qubit2model_permutation_rev = np.argsort(self.qubit2model_permutation)
            if dump_verbose:
                self.logger.info(f"Reverse sampling with qubit2model: {self.qubit2model_permutation}")

    def update_alpha_beta_electrons(self, n_alpha_electrons, n_beta_electrons):
        self.n_alpha_up = [n_alpha_electrons]
        self.n_beta_up = [n_beta_electrons]

    def sample(self, mode=True):
        self.sampling = mode

    def predict(self):
        self.sample(False)

    def combine_amp_phase(self, mode=True):
        self.combined_amp_phase = mode

    def clear_cache(self):
        pass

    def set_mpi_ranks(self, n_rank, rank):
        self.uniq_sampling_parallel = True
        self.n_rank = n_rank
        self.rank = rank
        self.logger.info(f"Using uniq sampling parallel: partition algorithm: {self.partition_algo} min_partition_samples: {self.min_partition_samples}")

    def set_seed(self, seed, real_set=False):
        self.seed = seed
        if real_set:
            set_global_seed(seed, dump_verbose=False)

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def set_states(self, states, _masking):
        self.states = states
        self._masking = _masking

    @torch.no_grad()
    def _state2idx(self, s):
        return ((s > 0).long() * self._idx_shell_basis_vec[:s.shape[-1]].to(s)).sum(-1)

    @torch.no_grad()
    def _state2id_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Map states in a batch to their respective ids.
        e.g. _state2id_batch(torch.tensor([[1, 1, -1, -1], [-1, 1, -1, 1]])) -> tensor([3, 10])
        """
        # Convert -1 to 0
        states_binary = states.clamp(min=0).to(torch.float64)
        two_powers = torch.pow(2, torch.arange(states.size(1), dtype=torch.float64, device=states.device))

        # Using matrix multiplication to get the decimal representation for each batch
        ids = torch.matmul(states_binary, two_powers)

        return ids

    def __get_x_ins(self, x, i):
        batch_size = x.shape[0]
        x_ins = []

        if i == 0:
            x_in = torch.zeros(batch_size, 1)
            x1, x2 = torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
        else:
            # x1, x2 = x[:, :2*i:2].clone(), x[:, 1:2*i:2].clone()
            # x1, x2 = x[:, :2*i:2], x[:, 1:2*i:2]
            # avoid stride copy (SW)
            x12 = x[:, :2*i].reshape(x.shape[0], i, 2).transpose(1, 2).transpose(1, 0).contiguous()
            x1, x2 = x12[0], x12[1]
            x_in = x[:, :2*i]

        x_ins.append(x_in.to(self.device))
        x_ins.append(x_in.to(self.device))

        return x_ins, x1, x2

    def __get_rand_start_idx(self):
        random_int = torch.randint(0, 4, (1,))
        return random_int

    def __get_conditional_output(self, x_ins, i, ret_phase=True, out_device=None, ret_all_amp=False, kv_caches=None, kv_idxs=None):
        r"""Given x_ins, return the output of amplitude and phase.
            x_ins[0] -> amplitude; x_ins[1] -> phase.
            Only when i == last inference, running phase inference once.
        """
        if out_device is None:
            out_device = self.out_device
        n_batches = x_ins[0].shape[0]
        ik = i
        if i == 0:
            # start dummy tag
            # x_ins[0][...] = 4
            x_ins[0][...] = self.start_idx
        else:
            # +1/-1 -> 0,1,2,3
            x_ins[0] = self._state2int_v2(x_ins[0], is_trans=False, is_disp=False).reshape(n_batches, -1)
            # pad_st = torch.full((n_batches, 1), 4.0, device=self.device)
            pad_st = torch.full((n_batches, 1), float(self.start_idx), device=self.device)
            x_ins[0] = torch.cat((pad_st, x_ins[0]), -1)

        if self.combined_amp_phase_blocks:
            if self.aggregate_phase or i == (self.N // 2 - 1):
                amp_i, phase_i = self.amp_layers[i](x_ins[0])
            else:
                amp_i = self.amp_layers[i](x_ins[0]).to(out_device)
                phase_i = torch.zeros(len(amp_i), self._n_out_phase)
        else:
            if self.use_amp_infer_batch:
                # using batch_size to infer avoid Out-Of-Memory
                BS = x_ins[0].shape[0]
                cnt = int(np.ceil(BS / self.amp_infer_batch_size))
                rem = BS % self.amp_infer_batch_size
                use_padding = False
                # print(f"x_ins: {x_ins[0].shape} BS: {BS} cnt: {cnt} rem: {rem}")
                _amp_is = []
                kv_caches_list = []
                _kv_caches, _kv_idxs = None, None
                if kv_caches is not None:
                    _kv_caches, _kv_idxs = [(torch.Tensor().to(self.device), torch.Tensor().to(self.device)) for _ in range(self.model_config.n_layer)] , None

                for ib in range(cnt):
                    st = ib*self.amp_infer_batch_size
                    ed = min(st+self.amp_infer_batch_size, BS)
                    x_batch =  x_ins[0][st:ed, :]
                    if use_padding and ib == cnt-1 and (rem != 0):
                        x_batch = torch.nn.functional.pad(x_batch, (0,0,0,self.amp_infer_batch_size-rem))
                    if kv_caches is not None:
                        ik = 0
                        if kv_idxs is not None:
                            v_st, v_ed = kv_idxs[st], kv_idxs[ed-1]
                            _kv_caches, _kv_idxs = [], kv_idxs[st:ed] - v_st
                            # print(f"st,ed: {st, ed}  v_st,v_ed: {v_st.item(), v_ed.item()}")
                            for ikv in range(len(kv_caches)):
                                _k_cache = kv_caches[ikv][0][v_st:v_ed+1].clone()
                                _v_cache = kv_caches[ikv][1][v_st:v_ed+1].clone()
                                _kv_caches.append((_k_cache, _v_cache))

                    # print(f"{i}-th ---kv_caches---")
                    _amp_i = self.amp_layers(x_batch, kv_caches=_kv_caches, kv_idxs=_kv_idxs)
                    #macs, params = profile(self.amp_layers, inputs=(x_batch, _kv_caches, _kv_idxs))
                    #print(f"amps: macs: {macs} params: {params}", flush=True)
                    # print(f"{i}-th ---origin---")
                    # _amp_i_diff = self.amp_layers(x_batch)
                    # # print(f"_amp_i: {_amp_i.shape} _amp_i_diff: {_amp_i_diff.shape}")
                    # if kv_caches is not None:
                    #     err_abs = (_amp_i - _amp_i_diff[:,-1:,:]).abs()
                    #     err_tot, err_mean, err_max = err_abs.sum().item(), err_abs.mean().item(), err_abs.max().item()
                    #     # print(f"ib: {ib} err: {err}", flush=True)
                    #     print(f"ib: {ib} err_tot, err_mean, err_max: {err_tot, err_mean, err_max}", flush=True)
                    if not ret_all_amp:
                        _amp_i = _amp_i[:, ik, :]

                    if use_padding and ib == cnt-1 and (rem != 0):
                        _amp_i = _amp_i[:rem, ...]

                    _amp_is.append(_amp_i)
                    if _kv_caches is not None:
                        kv_caches_list.append(_kv_caches)
                        # print(f"kv_caches_list: {len(kv_caches_list)} {len(_kv_caches)}")

                amp_i = torch.cat(_amp_is, dim=0).to(out_device)
                if kv_caches is not None:
                    n_batch, n_layer = len(kv_caches_list), len(kv_caches_list[0])
                    # print(f"n_batch: {n_batch} n_layer: {n_layer}")
                    _t_list = [[] for _ in range(n_layer * 2)]
                    for i_nb in range(n_batch):
                        for i_nl in range(n_layer):
                            _t_list[i_nl*2].append(kv_caches_list[i_nb][i_nl][0])
                            _t_list[i_nl*2+1].append(kv_caches_list[i_nb][i_nl][1])
                        if i_nb == n_batch - 1:
                            for i_nl in range(n_layer):
                                # _c1, _c2 = torch.cat(_t_list[i_nl*2]), torch.cat(_t_list[i_nl*2+1])
                                # print(f"_c1: {_c1.shape} _c2: {_c2.shape} {len(_t_list[i_nl*2])}")
                                kv_caches[i_nl] = (torch.cat(_t_list[i_nl*2]), torch.cat(_t_list[i_nl*2+1]))
            else:
                # full inputs infer
                amp_i = self.amp_layers(x_ins[0])
                if not ret_all_amp:
                    amp_i = amp_i[:, ik, :]

            if ret_phase and (self.aggregate_phase or i == (self.N // 2 - 1)):
                if self.aggregate_phase:
                    phase_i = self.phase_layers[i](x_ins[1]).to(out_device)
                else:
                    phase_i = self.phase_layers[0](x_ins[1]).to(out_device)
                    #macs, params = profile(self.phase_layers[0], inputs=(x_ins[1], ))
                    #print(f"phases: macs: {macs} params: {params}", flush=True)
            else:
                phase_i = None
        if ret_phase:
            return amp_i, phase_i
        else:
            return amp_i

    def __get_conditional_output_combphase(self, x_ins, i, ret_phase=True, out_device=None, ret_all_amp=False):
        r"""For the case: phase and amplitude using the same network."""
        if out_device is None:
            out_device = self.out_device
        n_batches = x_ins[0].shape[0]
        ik = i
        if i == 0:
            # start dummy tag
            x_ins[0][...] = 4
        else:
            # +1/-1 -> 0,1,2,3
            x_ins[0] = self._state2int_v2(x_ins[0], is_trans=False, is_disp=False).reshape(n_batches, -1)
            # ik = i - 1
            pad_st = torch.full((n_batches, 1), 4.0, device=self.device)
            x_ins[0] = torch.cat((pad_st, x_ins[0]), -1)

        if self.combined_amp_phase_blocks:
            if self.aggregate_phase or i == (self.N // 2 - 1):
                amp_i, phase_i = self.amp_layers[i](x_ins[0])
            else:
                amp_i = self.amp_layers[i](x_ins[0]).to(out_device)
                phase_i = torch.zeros(len(amp_i), self._n_out_phase)
        else:
            if self.use_amp_infer_batch:
                # using batch_size to infer avoid Out-Of-Memory
                BS = x_ins[0].shape[0]
                cnt = int(np.ceil(BS / self.amp_infer_batch_size))
                _amp_is, _phase_is = [], []
                for ib in range(cnt):
                    st = ib*self.amp_infer_batch_size
                    ed = min(st+self.amp_infer_batch_size, BS)
                    _amp_i, _phase_i = self.amp_layers(x_ins[0][st:ed, :], ret_phase=True)
                    if not ret_all_amp:
                        _amp_i = _amp_i[:, ik, :]
                    _amp_is.append(_amp_i)
                    _phase_is.append(_phase_i)
                amp_i = torch.cat(_amp_is, dim=0)
                phase_i = torch.cat(_phase_is, dim=0)
            else:
                # full inputs infer
                amp_i, phase_i = self.amp_layers(x_ins[0], ret_phase=True)
                if not ret_all_amp:
                    amp_i = amp_i[:, ik, :]

        return amp_i.to(out_device), phase_i.to(out_device)

    def electron_conservation_mask_restricted_all(self, states, diff_mask=None):
        r'''Construct the electron conservation restricted mask for all qubits.
        Inputs:
            states: current states
            diff_mask: if not none, check this result
        '''
        # even: 00 00 01 01
        # odd: 10 11 10 11
        n_up, n_dn = self.n_alpha_up[0], self.n_beta_up[0]

        even_sum = torch.sum(states[:, 0::2]==1, dim=-1).reshape(-1,1)
        odd_sum = torch.sum(states[:, 1::2]==1, dim=-1).reshape(-1,1)

        _mask_e = (even_sum == n_up)
        _mask_o = (odd_sum == n_dn)

        _mask_cur = _mask_e & _mask_o
        if not (diff_mask is None):
            err = (_mask_cur.type(torch.float) - diff_mask).sum()
            assert err == 0
        return _mask_cur

    def __electron_conservation_mask_restricted(self, states, i, diff_mask=None):
        r'''Construct the electron conservation restricted mask from i-th qubits.
        Inputs:
            states: current states
            i: current sampled qubit (start from 0)
            diff_mask: if not none, check this result
        '''
        # i: 01
        # i+1: 0100 0101 0110 0111 (00 01 10 11)

        # even: 00 00 01 01
        # odd: 10 11 10 11

        up_num = torch.FloatTensor([0, 1, 0, 1]).to(states.device)
        dn_num = torch.FloatTensor([0, 0, 1, 1]).to(states.device)
        n_up, n_dn = self.n_alpha_up[0], self.n_beta_up[0]

        even_sum = torch.sum(states[:, 0::2]==1, dim=-1).reshape(-1,1)
        odd_sum = torch.sum(states[:, 1::2]==1, dim=-1).reshape(-1,1)

        # generate next qubit: [n, 1] + [1, 4] => [n, 4]
        _etot = even_sum + up_num
        _otot = odd_sum + dn_num

        # _mask_e =  _etot <= n_up
        # _mask_o =  _otot <= n_dn
        _mask_e =  (_etot <= n_up) & (n_up - _etot < self.N//2 - i)
        _mask_o =  (_otot <= n_dn) & (n_dn - _otot < self.N//2 - i)

        _mask_cur = _mask_e & _mask_o
        if not (diff_mask is None):
            err = (_mask_cur.type(torch.float) - diff_mask).sum()
            assert err == 0
        return _mask_cur

    def __electron_conservation_mask_unrestricted(self, states, i, diff_mask=None):
        r'''Construct the electron conservation unrestricted mask from i-th qubits.
        Inputs:
            states: current states
            i: current sampled qubit (start from 0)
            diff_mask: if not none, check this result
        '''
        # i: 01
        # i+1: 0100 0101 0110 0111 (00 01 10 11)
        # conut 1: 0 1 1 2

        n_elecs = self.n_tot_up
        next_layer_elec_num = torch.FloatTensor([0, 1, 1, 2]).to(states.device)
        tot_sum = torch.sum(states==1, dim=-1).reshape(-1,1)

        # generate next qubit: [n, 1] + [1, 4] => [n, 4]
        _tot = tot_sum + next_layer_elec_num
        _mask_cur =  (_tot <= n_elecs) & (n_elecs - _tot < self.N - 2*i - 1)

        if not (diff_mask is None):
            err = (_mask_cur.type(torch.float) - diff_mask).sum()
            assert err == 0
        return _mask_cur

    def __get_electron_conservation_mask(self, states, i, diff_mask=None):
        if self.electron_conservation_type == "none":
            return None
        elif self.electron_conservation_type == "restricted":
            return self.__electron_conservation_mask_restricted(states, i, diff_mask=diff_mask)
        elif self.electron_conservation_type == "unrestricted":
            return self.__electron_conservation_mask_unrestricted(states, i, diff_mask=diff_mask)
        else:
            ValueError(f"Unsupported electron_conservation_type: {self.electron_conservation_type}")

    def __apply_activations(self, amp_i, phase_i, i, amp_mask=None, masking=None, close_mask=False, T=None):
        r"""Just apply activations for amplitude and phase output."""
        if masking is None:
            masking = self.masking
        # TODO: better precision ?
        # if ( masking is NadeMasking.NONE
        #     or (masking is NadeMasking.PARTIAL and i==(self.N // 2 - 1))):
        #     amp_mask = None
        if close_mask:
            amp_mask = None
        if self.amplitude_activation is not None:
            # if (amp_mask is not None):
            #     amp_mask[amp_mask.sum(-1)==0] = 1
            # T = True
            T = False
            if T:
               temp = get_temp(self.n_epochs, init_temp=2.0, final_temp=1.0, num_epochs=10000)
               amp_i = amp_i / temp

            amp_i = self.amplitude_activation(amp_i, amp_mask)

            # special case: softmax([-inf, -inf, -inf, -inf]) = nan
            # if (amp_mask is not None) and (len(self.n_alpha_up) > 1):
            if (amp_mask is not None):
                amp_i = amp_i.clone()
                amp_i[amp_mask.sum(-1)==0] = float('-inf')

        if self.phase_activation is not None:
            if self.aggregate_phase:
                phase_i = self.phase_activation(phase_i, amp_mask)
            else:
                phase_i = self.phase_activation(phase_i)
        return amp_i, phase_i

    @torch.no_grad()
    def batch_sampling_next_token_kth(self, states, counts, probs, i, partial_probs, last_uniq_samples, is_sampling=False, kv_caches=None, kv_idxs=None, masking=None):
        r"""Batch sampling for the next token (i-th), from the current `states`.
            Args:
                states: partial samples for [0, i-1] qubits, with shape (n_samples, i)
                counts: number of sampled for every state, with shape (n_samples, )
                probs: probility of every state, with shape (n_samples, )
                kv_caches/kv_idxs: for KV caches optimizations
                masking: don't care
                partial_probs: local accumulation probs
                last_uniq_samples: local root node samples
                is_sampling: whether to sampling in current iter
            Return:
                new states, counts, probs, kv_caches, kv_idxs
        """
        blockidx2spin = torch.FloatTensor([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        # Get amplitudes of states[:, 0:i]
        x_ins, x1, x2 = self.__get_x_ins(states, i)
        amp_i = self.__get_conditional_output(x_ins, i, ret_phase=False, kv_caches=kv_caches, kv_idxs=kv_idxs, out_device=self.device)
        # _amp_mask = self.__get_restricted_hilbert_mask(x1, x2, i)
        _amp_mask = self.__get_electron_conservation_mask(states, i)

        if self.electron_conservation_type != "none":
            amp_i, _ = self.__apply_activations(amp_i, None, i, _amp_mask, masking, close_mask=False)
        else:
            amp_i, _ = self.__apply_activations(amp_i, None, i, _amp_mask, masking, close_mask=True)

        amp_i = torch.where(torch.isnan(amp_i), 0, amp_i) # all is broke electron conservation

        # Sample the next states.
        #   1) Convert log_amplitudes to probabilites.
        #   2) Sample one label of the next qudit for each occurance (count) of each unique state.
        #   3) Update the states, counts and probabilites accordingly.
        #   4) Update the amplitudes and phases if we are returning the wavefunction as well.
        # probs_i = amp_i.detach().exp().pow(2)
        # if i == 0:
        #     import pdb; pdb.set_trace()
        # print(f"{i}-th: probs_sum()= {probs.sum()} counts_sum: {counts.sum()}", flush=True)
        probs_i = amp_i.detach().exp()
        # import pdb; pdb.set_trace()
        #probs_i = amp_i.detach().exp().type(torch.float64)
        # next_probs = probs.unsqueeze(1) * probs_i.to(probs)
        # if i >= 3:
        #     print(f"amp_i: {amp_i}\nstates: {states}\n_amp_mask: {_amp_mask}", flush=True)
        # print(f"partial_probs: {partial_probs.shape}", flush=True)
        partial_probs = partial_probs.reshape(-1,1) * probs_i.to(partial_probs)
        # partial_probs = partial_probs * probs_i.to(partial_probs)
        # print(f'probs_i: {probs_i}\npartial_probs: {partial_probs}', flush=True)

        if is_sampling:
            # Work around for https://github.com/numpy/numpy/issues/8317
            # probs_i_np = probs_i.cpu().numpy().astype('float64')
            next_probs = probs.unsqueeze(1) * partial_probs.to(probs).reshape(last_uniq_samples, -1)
            # next_probs = probs.unsqueeze(1) * partial_probs.to(probs).reshape(-1, 4)
            next_probs = next_probs.reshape(-1, 4)

            probs_i_np = partial_probs.cpu().numpy().astype('float64')
            probs_i_np = probs_i_np.reshape(last_uniq_samples, -1)
            new_sample_counts = torch.LongTensor(multinomial_arr(counts, probs_i_np))
            # new_sample_counts = torch.LongTensor(multinomial_torch(counts, partial_probs.reshape(last_uniq_samples, -1).cpu())) # BUG?
            new_sample_counts = new_sample_counts.reshape(-1, 4)

            new_sample_mask = (new_sample_counts > 0)
            num_new_samples_per_state = new_sample_mask.sum(1)
            # print(f"num_new_samples_per_state: {num_new_samples_per_state}")
            new_sample_next_idxs = torch.where(new_sample_counts > 0)[1]
            
            # make probs.sum() == 1
            # probs_i[~new_sample_mask] = 0
            # probs_i /= torch.sum(probs_i, dim=-1, keepdims=True)
            # next_probs = probs.unsqueeze(1) * probs_i.to(probs)
            # import pdb; pdb.set_trace()

            if self.use_kv_cache:
                kv_idxs = torch.repeat_interleave(num_new_samples_per_state).to(self.device)
                if i == self.N//2-1:
                    d1, d2, d3 = kv_caches[0][0].shape
                    param_num = d1*d2*d3*2*len(kv_caches)
                    self.logger.info(f"kv_caches: {kv_caches[0][0].shape} cache num: {param_num} Mem: {param_num * 4 / 1e6} MB")

            if i == 0:
                states = blockidx2spin[new_sample_next_idxs]
            else:
                states = torch.cat([states.repeat_interleave(num_new_samples_per_state, 0), blockidx2spin[new_sample_next_idxs]], 1)
            counts = new_sample_counts[new_sample_mask]
            # print(f"next_probs: {next_probs.shape} {new_sample_mask.shape} partial_probs: {partial_probs.shape}", flush=True)
            probs = next_probs[new_sample_mask]
            partial_probs = torch.ones(counts.shape[0]) # reinit for next sampling
        else:
            # expand: n_uniq_samples x 4
            num_uniq_samples = states.shape[0]
            new_sample_next_idxs = torch.tensor([0,1,2,3]).repeat_interleave(num_uniq_samples, 0).reshape(-1, num_uniq_samples).T.reshape(-1)
            # print(f"new_sample_next_idxs: {new_sample_next_idxs}", flush=True)
            if i == 0:
                states = blockidx2spin[new_sample_next_idxs]
            else:
                # import pdb; pdb.set_trace()
                num_new_samples_per_state = torch.zeros(num_uniq_samples, dtype=torch.int)
                num_new_samples_per_state[:] = 4
                states = torch.cat([states.repeat_interleave(num_new_samples_per_state, 0), blockidx2spin[new_sample_next_idxs]], 1)
                # filter broke electron-conservation samples
                # num_new_samples_per_state = _amp_mask.sum(dim=-1)
                # new_sample_next_idxs = torch.where(_amp_mask == True)[1]
                # states = torch.cat([states.repeat_interleave(num_new_samples_per_state, 0), blockidx2spin[new_sample_next_idxs]], 1)
                # partial_probs = partial_probs[_amp_mask]
                assert states.shape[0] == partial_probs.numel()
        return states, counts, probs, partial_probs, kv_caches, kv_idxs

    @torch.no_grad()
    def batch_sampling_next_token(self, states, counts, probs, i, kv_caches=None, kv_idxs=None, masking=None):
        r"""Batch sampling for the next token (i-th), from the current `states`.
            Args:
                states: partial samples for [0, i-1] qubits, with shape (n_samples, i)
                counts: number of sampled for every state, with shape (n_samples, )
                probs: probility of every state, with shape (n_samples, )
                kv_caches/kv_idxs: for KV caches optimizations
                masking: don't care
            Return:
                new states, counts, probs, kv_caches, kv_idxs
        """
        blockidx2spin = torch.FloatTensor([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        # Get amplitudes of states[:, 0:i]
        x_ins, x1, x2 = self.__get_x_ins(states, i)
        amp_i = self.__get_conditional_output(x_ins, i, ret_phase=False, kv_caches=kv_caches, kv_idxs=kv_idxs, out_device=self.device)
        # _amp_mask = self.__get_restricted_hilbert_mask(x1, x2, i)
        _amp_mask = self.__get_electron_conservation_mask(states, i)

        if self.electron_conservation_type != "none":
            amp_i, _ = self.__apply_activations(amp_i, None, i, _amp_mask, masking, close_mask=False)
        else:
            amp_i, _ = self.__apply_activations(amp_i, None, i, _amp_mask, masking, close_mask=True)

        # Sample the next states.
        #   1) Convert log_amplitudes to probabilites.
        #   2) Sample one label of the next qudit for each occurance (count) of each unique state.
        #   3) Update the states, counts and probabilites accordingly.
        #   4) Update the amplitudes and phases if we are returning the wavefunction as well.
        # probs_i = amp_i.detach().exp().pow(2)
        # if i == 0:
        #     import pdb; pdb.set_trace()
        # print(f"{i}-th: probs_sum()= {probs.sum()} counts_sum: {counts.sum()}", flush=True)
        probs_i = amp_i.detach().exp()
        #probs_i = amp_i.detach().exp().type(torch.float64)
        next_probs = probs.unsqueeze(1) * probs_i.to(probs)

        # Work around for https://github.com/numpy/numpy/issues/8317
        probs_i_np = probs_i.cpu().numpy().astype('float64')
        probs_i_np /= np.sum(probs_i_np, -1, keepdims=True)
        new_sample_counts = torch.LongTensor(multinomial_arr(counts, probs_i_np))

        new_sample_mask = (new_sample_counts > 0)
        num_new_samples_per_state = new_sample_mask.sum(1)
        # print(f"num_new_samples_per_state: {num_new_samples_per_state}")
        new_sample_next_idxs = torch.where(new_sample_counts > 0)[1]
        
        # make probs.sum() == 1
        # probs_i[~new_sample_mask] = 0
        # probs_i /= torch.sum(probs_i, dim=-1, keepdims=True)
        # next_probs = probs.unsqueeze(1) * probs_i.to(probs)

        if self.use_kv_cache:
            kv_idxs = torch.repeat_interleave(num_new_samples_per_state).to(self.device)
            if i == self.N//2-1:
                d1, d2, d3 = kv_caches[0][0].shape
                param_num = d1*d2*d3*2*len(kv_caches)
                self.logger.info(f"kv_caches: {kv_caches[0][0].shape} cache num: {param_num} Mem: {param_num * 4 / 1e6} MB")

        if i == 0:
            states = blockidx2spin[new_sample_next_idxs]
        else:
            states = torch.cat([states.repeat_interleave(num_new_samples_per_state, 0), blockidx2spin[new_sample_next_idxs]], 1)
        counts = new_sample_counts[new_sample_mask]
        probs = next_probs[new_sample_mask]
        return states, counts, probs, kv_caches, kv_idxs

    @torch.no_grad()
    def batch_sampling_rest_tokens(self, _states, _counts, _probs, _i, _kv_caches=None, _kv_idxs=None, masking=None, max_batch_size=None):
        r"""Batch sampling the rest tokens from _i-th token"""
        dfs_width = self.sampling_dfs_width
        nstates = _states.shape[0]
        n_block = nstates // dfs_width
        st_list, ed_list = [], []
        for j in range(n_block):
            st, ed = get_proc_block_range(nstates, n_block, j)
            st_list.append(st)
            ed_list.append(ed)
        kv_caches, kv_idxs = None, None
        final_states, final_counts, final_probs = [], [], []
        for st, ed in zip(st_list, ed_list):
            # print(f"===[st, ed): {st,ed}===")
            states, counts, probs = _states[st:ed, ...], _counts[st:ed, ...], _probs[st:ed, ...]
            if _kv_caches is not None:
                kv_idxs = _kv_idxs[st:ed] - _kv_idxs[st]
                st_kv, ed_kv = _kv_idxs[st], _kv_idxs[ed-1]+1
                kv_caches = [(t[0][st_kv:ed_kv,:], t[1][st_kv:ed_kv,:]) for t in _kv_caches]

            for i in range(_i, self.N // 2):
                # sampling next token (qubit)
                torch.cuda.nvtx.range_push(f"sampling_{i}")
                states, counts, probs, kv_caches, kv_idxs = self.batch_sampling_next_token(states, counts, probs, i, kv_caches=kv_caches, kv_idxs=kv_idxs, masking=masking)
                torch.cuda.nvtx.range_pop()

                # special case, return directly
                if states.shape[0] == 0:
                    break

                # exceed max batch size, throw exception
                if max_batch_size is not None:
                    if len(states) > max_batch_size:
                        raise MaxBatchSizeExceededError(len(states))
            final_states.append(states)
            final_counts.append(counts)
            final_probs.append(probs)
        states, counts, probs = torch.cat(final_states), torch.cat(final_counts), torch.cat(final_probs)
        return states, counts, probs

    def _drop_samples(self, counts, states, probs, drop_eps=1e-12):
        r"""drop all samples which probs < drop_eps"""
        probs_cnt = counts / counts.sum()
        drop_eps = self.drop_samples_eps
        probs_mask = probs_cnt < drop_eps
        last_probs = probs_cnt[probs_mask]
        if last_probs.shape[0] > 0:
            states = states[~probs_mask, :]
            counts = counts[~probs_mask]
            probs = probs[~probs_mask]
            # print(f"{last_probs.shape[0]} last_probs: {last_probs}", flush=True)
            print(f"n_drop: {last_probs.shape[0]}", flush=True)
        return counts, states, probs

    def _forward_sample_dfs_kth(self, batch_size, kth=1, ret_output=True, masking=None, max_batch_size=None, is_sorted_samples=False):
        r'''Generate 'batch_size' states from the underlying distribution. [BFS + DFS]
            After a few bfs sampling, then using multiple dfs-width sampling to avoid OOM in LLM (using KV Cache).
        Args:
            batch_size: initial total samples number
            ret_output: whether return results
            masking: qubit encoding format
            max_batch_size: set maximum batch size to avoid Out-Of-Memory
            is_sorted_samples: whether sort the final samples in ascending order
        Returns:
            [states, counts, probs, log_psi]
            - states: sampled unique states
            - counts: occurrence of each state
            - probs: prob. for each state
            - log_psi: log(\Psi(x)) of each state
        '''
        states = torch.zeros(1, 2, requires_grad=False)
        probs = torch.FloatTensor([1]).to(self.device)
        partial_probs = torch.FloatTensor([1]).to(self.device)
        # import pdb
        # pdb.set_trace()
        counts = torch.LongTensor([batch_size])
        states_log = []
        last_uniq_samples = states.shape[0]

        # get rand start idx
        if self.use_random_start_idx:
            self.start_idx = self.__get_rand_start_idx()
        else:
            self.start_idx = 4        
        self.logger.debug(f"start_idx: {self.start_idx}")

        over_uniq_sampling_parallel = True
        if self.uniq_sampling_parallel:
            # self.set_seed(self.seed, True) # sync rand seed over processes
            self.set_seed(self.seed+self.n_epochs, True) # sync rand seed over processes
            self.logger.info(f"rank: {self.rank} epochs: {self.n_epochs} seed: {self.seed+self.n_epochs} batch_size: {batch_size} rand: {np.random.randint(0, 2**31)}")
        else:
            # self.logger.info(f"batch_size: {batch_size}")
            # print(f"batch_size: {batch_size}", flush=True)
            pass

        # param = collect_parameters(self, self.device)
        # self.param_list.append(param)

        kv_caches, kv_idxs = None, None
        if self.use_kv_cache:
            kv_caches = [(torch.Tensor().to(self.device), torch.Tensor().to(self.device)) for _ in range(self.model_config.n_layer)]

        max_iter = self.N // 2
        i = 0
        while i < max_iter:
            if states.shape[0] == 0:
                return [states.to(self.out_device), counts, probs, None]

            states_log.append(states.shape[0])
            # Just partition once for MPI parallelization
            if self.uniq_sampling_parallel and over_uniq_sampling_parallel:
                n_states = states.shape[0]
                # TODO: bug here?
                if n_states >= self.n_rank and n_states > self.min_partition_samples:
                    counts_np = counts.numpy()
                    if self.partition_algo == "weight":
                        st, ed = get_weight_block_partition(self.n_rank, self.rank, counts_np, n_epochs=self.n_epochs)
                    else:
                        st, ed = get_proc_block_range(n_states, self.n_rank, self.rank)
                    # for debug
                    # self.weight_list.append(counts.numpy())
                    # if self.n_epochs == 10:
                    #     np.savez(f"weight-{self.rank}-{self.n_epochs}", weight_list=self.weight_list, param_list=self.param_list)
                    states, counts, probs = states[st:ed, :], counts[st:ed], probs[st:ed]
                    over_uniq_sampling_parallel = False
            # print(f"states: {states.shape} kv_idxs: {type(kv_idxs)} {kv_idxs}")
            # print(f"{i}-th states: {states.shape} kv_idxs: {kv_idxs.shape if kv_idxs is not None else type(kv_idxs) }")
            if self.sampling_algo == "dfs" and states.shape[0] > self.sampling_dfs_uniq_samples_min:
                states, counts, probs = self.batch_sampling_rest_tokens(states, counts, probs, i, _kv_caches=kv_caches, _kv_idxs=kv_idxs, masking=masking, max_batch_size=max_batch_size)
                break
            else:
                is_sampling = False
                # sampling next token (qubit)
                if (i+1) % kth == 0 or i == max_iter-1:
                    is_sampling = True

                if i % kth == 0:
                    last_uniq_samples = states.shape[0]

                states, counts, probs, partial_probs, kv_caches, kv_idxs = self.batch_sampling_next_token_kth(states, counts, probs, i, partial_probs, last_uniq_samples, is_sampling=is_sampling, kv_caches=kv_caches, kv_idxs=kv_idxs, masking=masking)
                # print(f'probs: {probs}\npartial_probs: {partial_probs}', flush=True)
                
            if self.topk_cutoff and (kth == 1 or is_sampling):
                bs = self.topk_bs
                # keep_idxs = counts.sort()[1][-2*bs:]
                keep_idxs = counts.sort()[1][-self.cutoff_factor*bs:]
                states = states[keep_idxs]
                counts = counts[keep_idxs]
                probs = probs[keep_idxs]
                partial_probs = partial_probs[keep_idxs]
                # self.logger.info(f"cut bs = {self.topk_bs}")

            # special case, return directly
            if states.shape[0] == 0:
                return [states.to(self.out_device), counts, probs, None]

            # exceed max batch size, throw exception
            if max_batch_size is not None:
                if len(states) > max_batch_size:
                    raise MaxBatchSizeExceededError(len(states))
            i = i + 1

        if self.topk_cutoff:
            bs = self.topk_bs
            keep_idxs = counts.sort()[1][-bs:]
            states = states[keep_idxs]
            counts = counts[keep_idxs]
            probs = probs[keep_idxs]

        #if max_batch_size is not None:
        #    if len(states) > max_batch_size:
        #        raise MaxBatchSizeExceededError(len(states))
        # drop samples which probs < drop_samples_eps
        if self.drop_samples:
            counts, states, probs = self._drop_samples(counts, states, probs, drop_eps=self.drop_samples_eps)

        # _mask = self._filter_all_single_occupy_mask(states)
        # counts, states, probs = counts[_mask], states[_mask], probs[_mask]

        if False:
            # counts = counts / counts.sum()
            num_save = int(counts.shape[0] * 0.8)
            print(f'num_save: {num_save} {counts.shape}', flush=True)
            # # counts, states, probs = counts[:num_save], states[:num_save, :], probs[:num_save]
            # indices = torch.randint(0, counts.shape[0], (num_save,))
            # indices = torch.randperm(counts.shape[0])[:num_save]
            _, indices = torch.topk(counts, num_save)
            # print(f'indices: {indices}', flush=True)
            counts = torch.index_select(counts, 0, indices)
            states = torch.index_select(states, 0, indices)
            # probs = torch.index_select(probs, 0, indices)

        #torch.cuda.nvtx.range_push(f"forward_pad")
        # final full states forward with gradient
        self.states, self._masking = states, masking
        if self.use_grad_accumulation:
            with torch.no_grad():
                ln_amps, phases = self._forward_pad(states, masking)
        else:
            ln_amps, phases = self._forward_pad(states, masking)
        # print(f"ln_amps: {ln_amps}\namps.exp(): {ln_amps.exp()**2}\n{probs}", flush=True)
        # import pdb; pdb.set_trace()
        # ik = self.N // 2 - 1
        # x_ins, _, _ = self.__get_x_ins(states, ik)
        # x_ins[1] =  states[:, :self.N].to(self.device)
        # _, phase_is = self.__get_conditional_output(x_ins, ik, ret_phase=True, ret_all_amp=False)
        # phases = phase_is.reshape(-1).to(self.out_device)
        # # ln_amps = probs.log().sqrt().to(self.out_device)
        # ln_amps = probs.sqrt().log().to(self.out_device)

        #torch.cuda.nvtx.range_pop()
        if kth > 1:
            psi_2, tot_counts, probs_sum = ln_amps.exp().pow(2).sum(), counts.sum(), probs.sum()
            print(f"kth: {kth} sum(|psi|^2): {psi_2}, tot_counts: {tot_counts} probs_sum: {probs_sum}", flush=True)

        # permutation sampling order (reverse order)
        if self.qubit_order == -1:
            states = states[:, self.qubit2model_permutation]

        #use_aabb_order = True
        use_aabb_order = False
        if use_aabb_order:
            states = torch.cat((states[:, 0::2], states[:, 1::2]), dim=1) # for aabb test
            self.logger.info("use_aabb_order")

        # TODO: please test more carefully
        #is_sorted_samples = True
        is_sorted_samples = False
        if is_sorted_samples:
            self.logger.info(f"Sorted samples at sampling.")
            state_idxs = self._state2id_batch(states)
            sorted_idxs = torch.argsort(state_idxs)

            states = states[sorted_idxs, :]
            counts = counts[sorted_idxs]
            probs = probs[sorted_idxs]
            ln_amps = ln_amps[sorted_idxs]
            phases = phases[sorted_idxs]

        ret = [states.to(self.out_device), counts, probs]
        #self.logger.info(f"ln_amps: {ln_amps.device} phases: {phases.device} states: {states.device}")

        if ret_output:
            if self.combined_amp_phase:
                # output = torch.stack([ln_amps, phases], -1).to(self.out_device)
                output = torch.stack([ln_amps, phases], -1)
            else:
                # output = (ln_amps.to(self.out_device), phases.to(self.out_device))
                output = (ln_amps, phases)
            ret.append(output)
        states_log.append(states.shape[0])
        self.logger.debug(f"states_size: {states_log}")

        #self.system = 'lih-org'
        #torch.save(states, f'{self.system}_states_{self.n_epochs}.pt')
        #torch.save(ln_amps.exp()*phases.cos(), f'{self.system}_amps_{self.n_epochs}.pt')
        #torch.save(ln_amps.exp()*phases.sin(), f'{self.system}_phase_{self.n_epochs}.pt')
        #torch.save(ln_amps.exp().pow(2), f'{self.system}_probs_{self.n_epochs}.pt')
        
        return ret

    def _forward_sample_dfs(self, batch_size, ret_output=True, masking=None, max_batch_size=None, is_sorted_samples=False):
        r'''Generate 'batch_size' states from the underlying distribution. [BFS + DFS]
            After a few bfs sampling, then using multiple dfs-width sampling to avoid OOM in LLM (using KV Cache).
        Args:
            batch_size: initial total samples number
            ret_output: whether return results
            masking: qubit encoding format
            max_batch_size: set maximum batch size to avoid Out-Of-Memory
            is_sorted_samples: whether sort the final samples in ascending order
        Returns:
            [states, counts, probs, log_psi]
            - states: sampled unique states
            - counts: occurrence of each state
            - probs: prob. for each state
            - log_psi: log(\Psi(x)) of each state
        '''
        states = torch.zeros(1, 2, requires_grad=False)
        probs = torch.FloatTensor([1]).to(self.device)
        # import pdb
        # pdb.set_trace()
        counts = torch.LongTensor([batch_size])
        states_log = []

        over_uniq_sampling_parallel = True
        if self.uniq_sampling_parallel:
            # self.set_seed(self.seed, True) # sync rand seed over processes
            self.set_seed(self.seed+self.n_epochs, True) # sync rand seed over processes
            self.logger.info(f"rank: {self.rank} epochs: {self.n_epochs} seed: {self.seed+self.n_epochs} batch_size: {batch_size} rand: {np.random.randint(0, 2**31)}")
        else:
            # self.logger.info(f"batch_size: {batch_size}")
            # print(f"batch_size: {batch_size}", flush=True)
            pass

        # param = collect_parameters(self, self.device)
        # self.param_list.append(param)

        kv_caches, kv_idxs = None, None
        if self.use_kv_cache:
            kv_caches = [(torch.Tensor().to(self.device), torch.Tensor().to(self.device)) for _ in range(self.model_config.n_layer)]

        for i in range(self.N // 2):
            if states.shape[0] == 0:
                return [states.to(self.out_device), counts, probs, None]

            states_log.append(states.shape[0])
            # Just partition once for MPI parallelization
            if self.uniq_sampling_parallel and over_uniq_sampling_parallel:
                n_states = states.shape[0]
                # TODO: bug here?
                if n_states >= self.n_rank and n_states > self.min_partition_samples:
                    counts_np = counts.numpy()
                    if self.partition_algo == "weight":
                        st, ed = get_weight_block_partition(self.n_rank, self.rank, counts_np, n_epochs=self.n_epochs)
                    else:
                        st, ed = get_proc_block_range(n_states, self.n_rank, self.rank)
                    # for debug
                    # self.weight_list.append(counts.numpy())
                    # if self.n_epochs == 10:
                    #     np.savez(f"weight-{self.rank}-{self.n_epochs}", weight_list=self.weight_list, param_list=self.param_list)
                    states, counts, probs = states[st:ed, :], counts[st:ed], probs[st:ed]
                    over_uniq_sampling_parallel = False
            # print(f"states: {states.shape} kv_idxs: {type(kv_idxs)} {kv_idxs}")
            # print(f"{i}-th states: {states.shape} kv_idxs: {kv_idxs.shape if kv_idxs is not None else type(kv_idxs) }")
            if self.sampling_algo == "dfs" and states.shape[0] > self.sampling_dfs_uniq_samples_min:
                states, counts, probs = self.batch_sampling_rest_tokens(states, counts, probs, i, _kv_caches=kv_caches, _kv_idxs=kv_idxs, masking=masking, max_batch_size=max_batch_size)
                break
            else:
                # sampling next token (qubit)
                torch.cuda.nvtx.range_push(f"sampling_{i}")
                states, counts, probs, kv_caches, kv_idxs = self.batch_sampling_next_token(states, counts, probs, i, kv_caches=kv_caches, kv_idxs=kv_idxs, masking=masking)
                torch.cuda.nvtx.range_pop()

            # special case, return directly
            if states.shape[0] == 0:
                return [states.to(self.out_device), counts, probs, None]

            # exceed max batch size, throw exception
            if max_batch_size is not None:
                if len(states) > max_batch_size:
                    raise MaxBatchSizeExceededError(len(states))

        #if max_batch_size is not None:
        #    if len(states) > max_batch_size:
        #        raise MaxBatchSizeExceededError(len(states))
        # drop samples which probs < drop_samples_eps
        if self.drop_samples:
            counts, states, probs = self._drop_samples(counts, states, probs, drop_eps=self.drop_samples_eps)

        # _mask = self._filter_all_single_occupy_mask(states)
        # counts, states, probs = counts[_mask], states[_mask], probs[_mask]

        if False:
            # counts = counts / counts.sum()
            num_save = int(counts.shape[0] * 0.8)
            print(f'num_save: {num_save} {counts.shape}', flush=True)
            # # counts, states, probs = counts[:num_save], states[:num_save, :], probs[:num_save]
            # indices = torch.randint(0, counts.shape[0], (num_save,))
            # indices = torch.randperm(counts.shape[0])[:num_save]
            _, indices = torch.topk(counts, num_save)
            # print(f'indices: {indices}', flush=True)
            counts = torch.index_select(counts, 0, indices)
            states = torch.index_select(states, 0, indices)
            # probs = torch.index_select(probs, 0, indices)

        #torch.cuda.nvtx.range_push(f"forward_pad")
        # final full states forward with gradient
        self.states, self._masking = states, masking
        if self.use_grad_accumulation:
            with torch.no_grad():
                ln_amps, phases = self._forward_pad(states, masking)
        else:
            ln_amps, phases = self._forward_pad(states, masking)
        # import pdb; pdb.set_trace()
        # ik = self.N // 2 - 1
        # x_ins, _, _ = self.__get_x_ins(states, ik)
        # x_ins[1] =  states[:, :self.N].to(self.device)
        # _, phase_is = self.__get_conditional_output(x_ins, ik, ret_phase=True, ret_all_amp=False)
        # phases = phase_is.reshape(-1).to(self.out_device)
        # # ln_amps = probs.log().sqrt().to(self.out_device)
        # ln_amps = probs.sqrt().log().to(self.out_device)

        #torch.cuda.nvtx.range_pop()
        # psi_2, tot_counts, probs_sum = ln_amps.exp().pow(2).sum(), counts.sum(), probs.sum()
        # print(f"psi_2: {psi_2}, tot_counts: {tot_counts} probs_sum: {probs_sum}", flush=True)
        # permutation sampling order (reverse order)
        if self.qubit_order == -1:
            states = states[:, self.qubit2model_permutation]

        #use_aabb_order = True
        use_aabb_order = False
        if use_aabb_order:
            states = torch.cat((states[:, 0::2], states[:, 1::2]), dim=1) # for aabb test
            self.logger.info("use_aabb_order")

        # TODO: please test more carefully
        #is_sorted_samples = True
        is_sorted_samples = False
        if is_sorted_samples:
            self.logger.info(f"Sorted samples at sampling.")
            state_idxs = self._state2id_batch(states)
            sorted_idxs = torch.argsort(state_idxs)

            states = states[sorted_idxs, :]
            counts = counts[sorted_idxs]
            probs = probs[sorted_idxs]
            ln_amps = ln_amps[sorted_idxs]
            phases = phases[sorted_idxs]

        ret = [states.to(self.out_device), counts, probs]
        #self.logger.info(f"ln_amps: {ln_amps.device} phases: {phases.device} states: {states.device}")

        if ret_output:
            if self.combined_amp_phase:
                output = torch.stack([ln_amps, phases], -1).to(self.out_device)
            else:
                output = (ln_amps.to(self.out_device), phases.to(self.out_device))
            ret.append(output)
        states_log.append(states.shape[0])
        self.logger.debug(f"states_size: {states_log}")

        #self.system = 'lih-org'
        #torch.save(states, f'{self.system}_states_{self.n_epochs}.pt')
        #torch.save(ln_amps.exp()*phases.cos(), f'{self.system}_ln_amps_{self.n_epochs}.pt')
        #torch.save(ln_amps.exp()*phases.sin(), f'{self.system}_phase_{self.n_epochs}.pt')
        #torch.save(ln_amps.exp().pow(2), f'{self.system}_probs_{self.n_epochs}.pt')
        
        return ret

    def _forward_pad(self, states, masking=None, permutation=False):
        r'''Given states then return amplitudes and phases.
            current states is in CPU
        '''
        x = states.reshape(-1, self.N)

        if self.qubit_order == -1 and permutation:
            x = x[:, self.qubit2model_permutation_rev]

        ik = self.N // 2 - 1
        x_ins, x1, x2 = self.__get_x_ins(x, ik)

        if self.phase_in == 2:
            x_ins[1] =  x[:, :self.N].to(self.device)
        elif self.phase_in == 1:
            x_ins[1] = self._state2int_v2(x, is_trans=False, is_disp=False).to(self.device)

        if self.comb_phase:
            ln_amp_is, phase_is = self.__get_conditional_output_combphase(x_ins, ik, ret_phase=False, ret_all_amp=True)
        else:
            # ln_amp_is, phase_is = self.__get_conditional_output(x_ins, ik, ret_phase=True, ret_all_amp=True)
            if self.device_get_conditional_output == 'cpu':
                ln_amp_is, phase_is = self.__get_conditional_output(x_ins, ik, ret_phase=True, ret_all_amp=True, out_device="cpu")
            else:
                ln_amp_is, phase_is = self.__get_conditional_output(x_ins, ik, ret_phase=True, ret_all_amp=True, out_device=self.device)

        _idxs = self._state2int_v2(x).to(ln_amp_is.device)
        ln_amps = torch.zeros(x.shape[0]).to(ln_amp_is.device)
        x = x.to(ln_amp_is.device)

        if self._n_out_phase == 1:
            phases = phase_is.reshape(-1)
        else:
            phases = torch.index_select(phase_is.reshape(-1), 0, _idxs[ik,:])

        def _get_selected_mask(idxs, n):
            selected_mask = torch.zeros(n * 4, dtype=torch.bool)
            selected_mask[idxs] = True
            selected_mask = selected_mask.reshape(-1, 4)
            return selected_mask

        for j in range(self.N//2):
            # _, _x1, _x2 = self.__get_x_ins(x, j)
            if j == 0:
                # _, _x1, _x2 = self.__get_x_ins(x, j)
                # xj = self.__get_x_ins(x, j)[0][0]
                # xj = torch.zeros(1, 1).to(ln_amp_is.device)
                xj = torch.zeros(x.shape[0], 1).to(ln_amp_is.device)
            else:
                # _x1, _x2 = x1[:, :j], x2[:, :j]
                # xj = x[:, :2*j].to(self.device)
                xj = x[:, :2*j]

            # _amp_mask = self.__get_restricted_hilbert_mask(_x1, _x2, j)
            _amp_mask = self.__get_electron_conservation_mask(xj, j)
            # if j > 0:
            #     selected_mask = _get_selected_mask(_idxs[j,:], x.shape[0])
            #     # _amp_mask = selected_mask

            ln_amp_j, _ = self.__apply_activations(ln_amp_is[:,j,:], None, j, _amp_mask, masking, close_mask=False)

            # if j > 0:
            #     _amp_mask = selected_mask
            #     selected_mask[~_amp_mask] = True # ~_amp_mask获取非电子守恒
            #     amp_j_exp = amp_j.exp().clone()
            #     amp_j_exp[~selected_mask] = 0
            #     amp_j_exp /= torch.sum(amp_j_exp, dim=-1, keepdims=True)
            #     amp_j = amp_j_exp.log()

            ln_amps += torch.index_select(ln_amp_j.reshape(-1), 0, _idxs[j,:])
        # |\Psi(x)|^2 = \Prod_{i=1}^N P(x_i|x_{i-1},x_{i-2},...,x_1)
        # |\ln\Psi(x)| = 1/2 * \Sum_{i=1}^N \ln P(x_i|x_{i-1},x_{i-2},...,x_1)
        ln_amps *= 0.5
        return ln_amps, phases

    def forward(self, x, *args, **kwargs):
        if self.sampling:
            # return self._forward_sample_dfs(x, *args, **kwargs)
            return self._forward_sample_dfs_kth(x, *args, **kwargs)
        else:
            ValueError("Current only support sampling forward")
    
    def ln_psi(self, states, *args, **kwargs):
        # return self._forward_pad(states, *args, **kwargs)
        ln_amps, phases = self._forward_pad(states, *args, **kwargs)
        ln_psis = cplx.to_complex(ln_amps, phases)
        return ln_psis

    def sampling(self, batch_size=1e12, *args, **kwargs):
        return self._forward_sample_dfs_kth(batch_size=batch_size, *args, **kwargs)

    @torch.no_grad()
    def _filter_all_single_occupy_mask(self, x, kval=-1):
        nr, nc = x.shape
        x = x.masked_fill(x==kval, 0).type(torch.int32)
        idxs = x[:, ::2] + x[:, 1::2]
        _mask = idxs <= 1
        mask = torch.ones(nr).type(torch.bool)
        for i in range(nc//2):
            mask &= _mask[:, i]
        return ~mask

    @torch.no_grad()
    def _state2int_v2(self, x, kval=-1, is_trans=True, is_disp=True):
        nr, nc = x.shape
        # x[x == kval] = 0
        x = x.masked_fill(x==kval, 0).type(torch.int32)
        idxs = x[:, ::2] + x[:, 1::2]*2

        if is_disp:
            disp = torch.arange(0, nr*4, 4).unsqueeze(-1).type(torch.int32)
            idxs += disp.to(idxs.device)

        if is_trans:
            idxs = idxs.T
        return idxs

    @torch.no_grad()
    def _state2int_idx(self, x, kval=-1):
        assert x.size(1) < 64, "Now only support N<64"
        x = x.masked_fill(x==kval, 0).type(torch.int32)
        x = (self._idx_spin_basis_vec * x).sum(-1)
        return x

def get_wavefunction(config, num_qubits=4, n_alpha_electrons=1, n_beta_electrons=1, device=None, logger=None):
    # phase_hidden_size = [512,512]
    # phase_hidden_size = [512,512,512]
    # phase_hidden_size = [512,512,512,512]
    phase_hidden_size = config.phase_hidden_size

    use_amp_spin_sym = False
    use_phase_spin_sym = False
    aggregate_phase = False

    # set_num_threads(24)

    QiankunNet = DecoderWaveFunction(\
        config, num_qubits=num_qubits, \
        n_alpha_electrons=n_alpha_electrons, n_beta_electrons=n_beta_electrons, \
        phase_hidden_size=phase_hidden_size, \
        use_amp_spin_sym=use_amp_spin_sym, use_phase_spin_sym=use_phase_spin_sym, aggregate_phase=aggregate_phase, \
        device=device, logger=logger)
    return QiankunNet

if __name__ == "__main__":
    set_global_seed(111)
    model = get_wavefunction('config-h2.yaml', num_qubits=4, n_alpha_electrons=1, n_beta_electrons=1, device='cuda') # H2
    model.sample()
    st = time.time()
    n_sample = 4000000000
    out = model.forward(n_sample)
    ed = time.time()
    print("Time sampling:", (ed-st)*1000, "ms")
    print(f'Samples: {out[0].shape}')
    print(f'==Samples==\n {out[0]}')
    print(f'cnts: {out[1][:4], torch.sum(out[1])}')
    print(f'cnts: {out[1]/torch.sum(out[1])}')
    count_parameters(model, print_verbose=False)
