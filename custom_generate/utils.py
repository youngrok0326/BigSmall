import torch


def ordered_stratified_resampling(weights: torch.Tensor) -> torch.Tensor:
    device = weights.device
    num_groups, num_generations = weights.shape
    probs = weights / torch.sum(weights, dim=-1, keepdim=True)
    u = (torch.arange(num_generations, device=device) + torch.rand(num_groups, num_generations, device=device)) / num_generations
    cumulative_probs = torch.cumsum(probs, dim=-1)
    cumulative_probs[:, -1] += 1e-6
    indices = torch.searchsorted(cumulative_probs, u, right=False)
    return indices.long()