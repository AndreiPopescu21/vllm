import torch
from vllm.model_executor.layers.rejection_sampler import RejectionSampler

class TopPSpeculativeSampler(RejectionSampler):
    """Rejection sampler variant that uses nucleus (top-p) sampling for recovered tokens.
    
    When a token is rejected, instead of sampling from the full recovered distribution,
    we restrict the candidates to the smallest set of tokens with cumulative probability
    at least top_p. This can help mitigate issues like overconfidence and improve the quality
    of recovered tokens.
    """
    def __init__(self, top_p: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.top_p = top_p

    def _get_recovered_probs(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,   # [batch_size, k, vocab_size]
    ) -> torch.Tensor:
        # Compute the positive difference between target and draft probabilities.
        f = torch.clamp(target_probs - draft_probs, min=self._smallest_positive_value)
        
        # Sort probabilities in descending order.
        sorted_f, sorted_indices = torch.sort(f, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_f, dim=-1)
        
        # Create a mask for tokens inside the top-p nucleus.
        mask = cumulative_probs <= self.top_p
        # Always include the first token to ensure non-empty nucleus.
        mask[..., 0] = True
        
        # Zero out tokens outside the nucleus.
        filtered_sorted_f = sorted_f * mask.to(sorted_f.dtype)
        # Renormalize to obtain a valid probability distribution.
        norm_factor = filtered_sorted_f.sum(dim=-1, keepdim=True)
        normalized_sorted_f = filtered_sorted_f / norm_factor.clamp_min(self._smallest_positive_value)
        
        # Scatter the normalized probabilities back to their original order.
        recovered_probs = torch.zeros_like(f)
        recovered_probs.scatter_(-1, sorted_indices, normalized_sorted_f)
        return recovered_probs
    