from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .. import interface
from .fold import AbstractFold
from .contrafold import CONTRAfold
from .zuker import ZukerFold


class CONTRAMixedFold(AbstractFold):
    def __init__(self, init_param=None, model_type: str = 'M', 
        max_helix_length: int = 30, **kwargs) -> None:
        super(CONTRAMixedFold, self).__init__(interface.CONTRAfoldMixedWrapper() if kwargs['mix_type']=='add' else interface.CONTRAfoldMixedWrapper2())
        self.turner = CONTRAfold(init_param=init_param)
        self.zuker = ZukerFold(model_type=model_type, max_helix_length=max_helix_length, **kwargs)
        self.max_helix_length = max_helix_length


    def forward(self, seq: list[str], **kwargs):
        return super().forward(seq, max_helix_length=self.max_helix_length, **kwargs)


    def make_param(self, seq: list[str], perturb: float = 0.) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ts = self.turner.make_param(seq)
        ps = self.zuker.make_param(seq, perturb)
        if perturb > 0.:
            return ( [{'turner': t, 'positional': p} for t, p in zip(ts, ps[0])],
                [{'turner': t, 'positional': p} for t, p in zip(ts, ps[1])] )
        else:
            return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]


    def make_param_on_cpu(self, param: dict[str, Any]) -> dict[str, Any]:
        param_on_cpu = { 
            'turner': {k: v.to("cpu") for k, v in param['turner'].items() },
            'positional': {k: v.to("cpu") for k, v in param['positional'].items() }
        }
        param_on_cpu = {k: self.clear_count(v) for k, v in param_on_cpu.items()}
        return param_on_cpu


    def calculate_differentiable_score(self, v: float, param: dict[str, Any], count: dict[str, Any]) -> torch.Tensor | float:
        return super().calculate_differentiable_score(v, param['positional'], count['positional'])


    def detect_device(self, param):
        return super().detect_device(param['positional'])
