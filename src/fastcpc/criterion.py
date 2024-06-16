"""CPC criterion."""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .config import CONFIG

__all__ = ["CPCCriterion"]


class CPCCriterion(nn.Module):
    """The criterion computes the CPC loss function."""

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.generator = generator

    def sample_for_prediction(self, batch: torch.Tensor, window_size: int) -> list[torch.Tensor]:
        batch_size, negative, dim = batch.shape
        batch_idx = torch.randint(
            low=0,
            high=batch_size,
            size=(CONFIG.negative_samples * window_size * batch_size,),
            device=batch.device,
            generator=self.generator,
        )
        seq_idx = torch.randint(
            low=1,
            high=negative,
            size=(CONFIG.negative_samples * window_size * batch_size,),
            device=batch.device,
            generator=self.generator,
        )
        base_idx = torch.arange(0, window_size, device=batch.device)
        base_idx = (
            base_idx.view(1, 1, window_size)
            .expand(1, CONFIG.negative_samples, window_size)
            .expand(batch_size, CONFIG.negative_samples, window_size)
        )
        seq_idx += base_idx.contiguous().view(-1)
        seq_idx = torch.remainder(seq_idx, negative)
        ext_idx = seq_idx + batch_idx * negative
        neg_ext = batch.contiguous().view(-1, dim)[ext_idx].view(batch_size, CONFIG.negative_samples, window_size, dim)

        outputs = []
        for k in range(1, CONFIG.num_predicts + 1):
            pos_seq = batch[:, k : -(CONFIG.num_predicts - k)] if k < CONFIG.num_predicts else batch[:, k:]
            pos_seq = pos_seq.view(batch_size, 1, pos_seq.size(1), dim)
            full_seq = torch.cat((pos_seq, neg_ext), dim=1)  # Add the positive sample to the first position
            outputs.append(full_seq)
        return outputs

    def forward(self, predictions: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_size, _ = latent.size()
        window_size = seq_size - CONFIG.num_predicts
        target = torch.zeros((batch_size * window_size), dtype=torch.long, device=latent.device)
        out_losses, out_acc = [], []
        for k, sampled in enumerate(self.sample_for_prediction(latent, window_size)):
            preds = (predictions[:, :, k].unsqueeze(1) * sampled).mean(dim=3)
            preds = preds.permute(0, 2, 1)
            preds = preds.contiguous().view(-1, preds.size(2))
            loss = F.cross_entropy(preds, target, reduction="mean")
            out_losses.append(loss)
            _, preds_idx = torch.max(preds, dim=1)
            out_acc.append(torch.sum(preds_idx == target).float())
        return torch.stack(out_losses), torch.stack(out_acc) / (batch_size * window_size)
