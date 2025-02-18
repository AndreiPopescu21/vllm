from vllm.model_executor.layers.rejection_sampler import RejectionSampler
import torch
from typing import Dict, Optional

class MinPSampler(RejectionSampler):
    """Apply modified rejection sampling with min_p filtering to both target and draft probabilities."""

    def __init__(self, min_p: float = 0.05, filter_value=-float('inf')):
        """Initialize the MinPRejectionSampler with a min_p value.
        
        Args:
            min_p: The minimum probability threshold. Tokens with probabilities below
                min_p * max_prob are filtered out before normalization.
        """
        super().__init__()
        self.min_p = min_p
        self.filter_value = filter_value

    def _apply_min_p_filter(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply min_p filtering to a probability distribution."""
        max_probs = torch.amax(probs, dim=-1, keepdim=True)
        mask = probs >= (self.min_p * max_probs)
        filtered_probs = probs * mask
        filtered_probs_sum = torch.sum(filtered_probs, dim=-1, keepdim=True)
        # Normalize - sum is guaranteed >0 since max_prob is included
        return filtered_probs / filtered_probs_sum

    def _get_accepted(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        """Override to apply min_p filtering before acceptance check."""
        # Apply min_p filtering to both target and draft probabilities
        target_probs = self._apply_min_p_filter(target_probs)
        draft_probs = self._apply_min_p_filter(draft_probs)
        
        # Call parent acceptance logic with filtered probabilities
        return super()._get_accepted(
            target_probs=target_probs,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            seeded_seqs=seeded_seqs,
        )
