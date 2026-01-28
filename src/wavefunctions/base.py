from torch import nn

class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._is_complex_wf = True

    def sampling(self, batch_size=1e12, *args, **kwargs):
        raise NotImplementedError()

    def ln_psi(self, states, *args, **kwargs):
        raise NotImplementedError()

    def psi(self, states, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, states):
        # return self.ln_psi(states) or self.psi(states)
        raise NotImplementedError()

    def is_complex_wf(self):
        return self._is_complex_wf

    def _set_is_complex_wf(self, flag: bool):
        self._is_complex_wf = flag
