import torch
from torch.distributions import Binomial
import yaml
from src.utils.utils import SingletonMeta

def get_lr_scheduler(opt, config):

    config_lr = config.optim.kwargs.lr
    total_iters = 1000
    final_lr = 1e-5
    init_lr = config_lr
    return torch.optim.lr_scheduler.LinearLR(opt,
                    start_factor = init_lr / config_lr,
                    end_factor = final_lr / config_lr,
                    total_iters = total_iters,
                    verbose=True)

@torch.no_grad()
def multinomial(_counts:torch.tensor, _probs:torch.tensor, PROB_TYPE=torch.float64, COUNT_TYPE=torch.long):
    org_dtype = torch.get_default_dtype()
    torch.set_default_dtype(PROB_TYPE)
    probs = _probs.clone().to(PROB_TYPE)
    counts = _counts.clone().to(COUNT_TYPE)
    probs_cum = torch.cumsum(probs, dim=-1)
    probs_cum.clamp_(1e-12)
    probs_cond = torch.nan_to_num(probs / probs_cum)
    out = torch.zeros_like(_probs, dtype=COUNT_TYPE)
    for i in range(out.shape[-1]-1, 0, -1):
        binsample = Binomial(counts, probs_cond[..., i]).sample().to(COUNT_TYPE)
        out[..., i] = binsample
        counts -= binsample
    out[..., 0] = counts
    torch.set_default_dtype(org_dtype)
    return out

class ConfigDict:
    def __init__(self, dictionary={}):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)

    def __str__(self, level=0):
        ret = "\t" * level + "\n"
        for key, value in self.__dict__.items():
            tabs = "  " * (level + 1)
            if isinstance(value, ConfigDict):
                ret += tabs + f"{key}: {value.__str__(level + 1)}"
            else:
                ret += tabs + f"{key}: {value}\n"
        return ret

def _update(self:ConfigDict, other:ConfigDict):
    for name, it in vars(other).items():
        if getattr(self, name, None) != None:
            me = getattr(self, name)
            if isinstance(me, ConfigDict) and isinstance(it, ConfigDict):
                _update(me, it)
                continue
        setattr(self, name, it)

class Config(metaclass=SingletonMeta):
    def __init__(self, cfg_file_path="config.yaml"):
        self.init_params()
        self.load_yaml(cfg_file_path)

    def __str__(self):
        attributes = [(name, value) for name, value in vars(self).items()]

        def _is_number(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        return '==NNQS configuration===\n' + '\n'.join(f"{name}: '{value}'" if isinstance(value, str) and not _is_number(value) else f"{name}: {value}" for name, value in attributes) + '\n===NNQS configuration===\n'

    def load_yaml(self, cfg_file_path):
        with open(cfg_file_path, 'r') as f:
            try:
                config_dict = yaml.full_load(f)
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        it = ConfigDict(value)
                        if getattr(self, key, None) is None:
                            setattr(self, key, it)
                        else:
                            me = getattr(self, key)
                            if isinstance(me, ConfigDict):
                                _update(me, it)
                            else:
                                setattr(self, key, it)
                    else:
                        setattr(self, key, value)
            except yaml.YAMLError as exc:
                print(f"Error in configuration file: {exc}")

    def init_params(self):
        #==== basic =====
        self.device = "cpu"
        self.log_step = 1
        # self.dtype = "float64"
        self.qk2_use_two_phase = False

        # self.local_energy_version = "CPP_CPU"
        self.hamiltonian_type = "CPP" # "CPP" | "exact" | "exactOpt"
        self.eloc_split_bs = 512 # used for "exactOpt" to avoid OOM
        self.save_model = 0
        self.save_per_epoches = 2000
        self.load_model = 0
        self.checkpoint_path = None
        self.transfer_learning = False
        self.log_file = None

        self.use_samples_recursive = True
        self.n_samples_min = 1e5
        self.n_samples_max = 1e12
        self.n_unq_samples_min = 1e3
        self.n_unq_samples_max = 1e6
        self.n_samples_scale_factor = 1.3
        self.n_sampling_layers = 1

        #==== system =====
        self.system = ConfigDict()
        self.system.geometry = [["H", 0., 0., 0.], ["H", 0., 0., 1.]]
        self.system.basis = "sto3g"
        self.system.run_fci = False
        self.system.run_rccsd = False
        self.system.spin = 0
        self.system.external_ham_path = None
        self.system.nelec = None

        self.n_alpha_electrons = None
        self.n_beta_electrons = None

        #==== training =====
        self.n_epoches = 100000
        self.n_samples = ConfigDict()
        # self.n_samples.min = 100000
        # self.n_samples.max = 100000000
        # self.n_samples.factor = 1.3
        # self.n_samples.step_size = 20

        #==== optimizer ====
        self.optim = ConfigDict()
        self.optim.name = 'AdamW'
        # self.optim.kwargs = ConfigDict()
        # self.optim.kwargs.lr = 3.0e-4

        # phase model for QiankunNet2
        self.phase_hidden_features = 512
        self.phase_num_blocks = 2

        # phase model for QiankunNet1
        self.phase_hidden_size = [512,512,512,512]
