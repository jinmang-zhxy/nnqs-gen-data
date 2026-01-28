import math
import torch
import src.utils.complex_helper as cplx

class MCMCSampler(torch.nn.Module):
    def __init__(self, wf=None, device=None):
        self.wf = wf
        self.device = device

    def single_update_batch_(self, n, states, iterations, num_chains, wf=None):
        rows = torch.arange(num_chains).unsqueeze(1)
        for _ in range(iterations):
            # generate candidates
            candidates = states.clone()
            # idx = torch.randint(n, size=(num_chains, 1))
            # candidates[torch.arange(num_chains).unsqueeze(1), idx] *= -1

            # swap(idx, idx+2): keep electron conservation
            # idx = torch.randint(n-2, size=(num_chains, 1), device=self.device)
            idx = torch.randint(n-2, size=(num_chains, 1))
            _tmp_val = candidates[rows, idx]
            candidates[rows, idx] = candidates[rows, idx+2]
            candidates[rows, idx+2] = _tmp_val

            # swap(idx, idx+1)
            # idx = torch.randint(n-1, size=(num_chains, 1))
            # _tmp_val = candidates[rows, idx]
            # candidates[rows, idx] = candidates[rows, idx+1]
            # candidates[rows, idx+1] = _tmp_val

            # calculate acceptance ratios
            if wf:
                # log_psi1 = wf(candidates); log_psi = wf(states)
                # p = abs2(exp(log_psi1 - log_psi))
                log_psi = torch.stack(wf(states.float()), -1)
                log_psi1 = torch.stack(wf(candidates.float()), -1)
                # print(f"log_psi: {log_psi}")
                # print(f"log_psi1: {log_psi1}")
                # acceptance_ratios = cplx.absolute_pow2_value(cplx.exp(log_psi1 - log_psi))
                acceptance_ratios = cplx.absolute_pow2_value(cplx.exp(log_psi1)) / cplx.absolute_pow2_value(cplx.exp(log_psi))
                # psi1_mod = cplx.absolute_pow2_value(cplx.exp(log_psi1))
                # psi_mod = cplx.absolute_pow2_value(cplx.exp(log_psi))
                # print(f"psi1_mod: {psi1_mod}")
                # print(f"psi_mod: {psi_mod}")
            else:
                acceptance_ratios = torch.ones(num_chains) - torch.rand(num_chains)
            # acceptance_ratios = torch.min(torch.ones_like(acceptance_ratios), acceptance_ratios)
            # print(f"acceptance_ratios: {acceptance_ratios}")
            # update states
            accept = (torch.rand(num_chains) < acceptance_ratios).unsqueeze(1)
            states[...] = torch.where(accept, candidates, states)

        return states

    def _elec_site(sele, nq, ne):
        sites_set = set()
        while True:
            sites = torch.randint(nq, (ne,)).tolist()
            for s in sites:
                sites_set.add(s)
                if len(sites_set) == ne:
                    return list(sites_set)

    def random_selection(self, nq, ne):
        assert ne % 2 == 0, "ne must be an even number"
        n_up = n_dw = ne // 2

        indices1 = torch.arange(0, nq, 2)  # Generate indices from 0 to nq (exclusive) with a step of 2
        indices2 = torch.arange(1, nq, 2)  # Generate indices from 1 to nq-1 with a step of 2

        selected_indices1 = torch.randperm(len(indices1))[:n_up]  # Randomly select n_up indices from indices1
        selected_indices2 = torch.randperm(len(indices2))[:n_dw]  # Randomly select n_dw indices from indices2

        selected_values = torch.cat([indices1[selected_indices1], indices2[selected_indices2]])
        return selected_values

    def metropolis_hastings_batch(self, nq=10, ne=4, num_samples=4096, iterations=30, num_chains=32, num_warmup=1000):
        # let num_samples is times of num_chains
        num_samples = math.ceil(num_samples / num_chains) * num_chains
        print(f"n_samples: {num_samples} iteration: {iterations} ({iterations*num_samples})", flush=True)

        # init state: (num_samples, n)
        # states = torch.randint(2, size=(num_samples, nq)) * 2 - 1
        # print(f"init states: {states}")
        # elec_sites = self._elec_site(nq, ne)
        elec_sites = self.random_selection(nq, ne)
        # states = torch.full((num_samples, nq), -1, device=self.device)
        states = torch.full((num_samples, nq), -1)
        states[:, elec_sites] = 1
        print(f"elec_sites: {elec_sites}")
        # print(f"init states: {states}")

        # warmup
        states[:num_chains, :] = self.single_update_batch_(nq, states[:num_chains, :], num_warmup, num_chains, wf=self.wf)
        # print(f"warm states: {states}")

        # MCMC
        num_samples_per_chain = num_samples // num_chains
        print(f"num_chains: {num_chains} num_samples_per_chain: {num_samples_per_chain}")
        states_chunks = states.split(num_chains)
        pre_states_chunk = states_chunks[0]
        # print(f"states_chunks: {states_chunks}")
        for states_chunk in states_chunks:
            states_chunk[...] = self.single_update_batch_(nq, pre_states_chunk, iterations, num_chains, wf=self.wf)
            pre_states_chunk = states_chunk
            # print(f"update states: {states}")

        return states

if __name__ == "__main__":
    state = metropolis_hastings(4, 5)
    print(state)
    mcmc = MCMCSampler()
    states = mcmc.metropolis_hastings_batch(10, 4, 80000, iterations=30, num_chains=10, num_warmup=1000)
    unique_states, counts = torch.unique(states, dim=0, return_counts=True)
    print(counts.shape, counts)
    # print(unique_states)
