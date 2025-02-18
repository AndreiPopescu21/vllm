import torch
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from typing import Dict, Optional

class TemperatureRejectionSampler(RejectionSampler):
    """Rejection sampler that incorporates temperature scaling into the acceptance probability.
    
    Instead of using the acceptance probability:
        min(1, q / p)
    we use:
        min(1, (q / p)^(1/temperature))
    
    This modification can help control the trade-off between fidelity to the target model
    and diversity in the generated text.
    """
    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,   # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size, device=target_probs.device)[:, None]
        token_indices = torch.arange(k, device=target_probs.device)
        
        selected_draft_probs = draft_probs[batch_indices, token_indices, draft_token_ids]
        selected_target_probs = target_probs[batch_indices, token_indices, draft_token_ids]
        
        # Generate uniform random samples for acceptance check.
        uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size, k - 1, target_probs.device)
        
        # Compute acceptance ratio with temperature scaling.
        ratio = (selected_target_probs / selected_draft_probs).clamp(min=0.0)
        if self.temperature != 1.0:
            ratio = ratio.pow(1.0 / self.temperature)
        capped_ratio = torch.minimum(ratio, torch.ones_like(ratio))
        accepted = uniform_rand < capped_ratio
        return accepted
    