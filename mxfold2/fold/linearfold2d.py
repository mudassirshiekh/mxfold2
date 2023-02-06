from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import LengthLayer, NeuralNet


class LinearFold2D(AbstractFold):
    def __init__(self, beam_size: int = 100, **kwargs: dict[str, Any]) -> None:
        super(LinearFold2D, self).__init__(interface.LinearFoldPositional2DWrapper(beam_size=beam_size))

        n_out_paired_layers = 3
        n_out_unpaired_layers = 0
        exclude_diag = False

        self.net = NeuralNet(**kwargs, 
            n_out_paired_layers=n_out_paired_layers,
            n_out_unpaired_layers=n_out_unpaired_layers,
            exclude_diag=exclude_diag)

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })


    def make_param(self, seq: list[str]) -> list[dict[str, Any]]:
        device = next(self.parameters()).device
        score_paired: torch.Tensor
        score_unpaired: Optional[torch.Tensor]
        score_paired, score_unpaired = self.net(seq)
        B, N, _, _ = score_paired.shape

        def unpair_interval(su: torch.Tensor) -> torch.Tensor:
            su = su.view(B, 1, N)
            su = torch.bmm(torch.ones(B, N, 1).to(device), su)
            su = torch.bmm(torch.triu(su), torch.triu(torch.ones_like(su)))
            return su

        score_basepair = torch.zeros((B, N, N), device=device)
        score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
        score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
        score_mismatch_internal = score_paired[:, :, :, 1] # (B, N, N)
        score_mismatch_multi = score_paired[:, :, :, 1] # (B, N, N)
        score_mismatch_hairpin = score_paired[:, :, :, 1] # (B, N, N)
        score_unpaired = score_paired[:, :, :, 2] # (B, N, N)
        score_base_hairpin = score_unpaired
        score_base_internal = score_unpaired
        score_base_multi = score_unpaired
        score_base_external = score_unpaired

        param = [ { 
            'score_basepair': score_basepair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_mismatch_external': score_mismatch_external[i],
            'score_mismatch_hairpin': score_mismatch_hairpin[i],
            'score_mismatch_internal': score_mismatch_internal[i],
            'score_mismatch_multi': score_mismatch_multi[i],
            'score_base_hairpin': score_base_hairpin[i],
            'score_base_internal': score_base_internal[i],
            'score_base_multi': score_base_multi[i],
            'score_base_external': score_base_external[i],
            'score_hairpin_length': cast(LengthLayer, self.fc_length['score_hairpin_length']).make_param(),
            'score_bulge_length': cast(LengthLayer, self.fc_length['score_bulge_length']).make_param(),
            'score_internal_length': cast(LengthLayer, self.fc_length['score_internal_length']).make_param(),
            'score_internal_explicit': cast(LengthLayer, self.fc_length['score_internal_explicit']).make_param(),
            'score_internal_symmetry': cast(LengthLayer, self.fc_length['score_internal_symmetry']).make_param(),
            'score_internal_asymmetry': cast(LengthLayer, self.fc_length['score_internal_asymmetry']).make_param(),
            'score_helix_length': cast(LengthLayer, self.fc_length['score_helix_length']).make_param()
        } for i in range(B) ]

        return param