from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from ..fold.fold import AbstractFold


class ShapeNLLLoss(nn.Module):
    def __init__(self, model: AbstractFold, 
            shape_model: list[nn.Module],
            perturb: float = 0., nu: float = 0.1, l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0., shape_only: bool = False,
            lwf_model: Optional[AbstractFold] = None, lwf_weight = 0.) -> None:
        super(ShapeNLLLoss, self).__init__()
        self.model = model
        self.shape_model = shape_model
        self.perturb = perturb
        self.nu = nu
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.shape_only = shape_only
        self.lwf_model = lwf_model
        self.lwf_weight = lwf_weight
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[list[int]] = None) -> torch.Tensor:
        pred: torch.Tensor
        pred_s: list[str]
        pred_bps: list[list[int]]

        if self.shape_only:
            with torch.no_grad():
                pred, pred_s, pred_bps = self.model(seq, perturb=self.perturb)

        else:
            pred, pred_s, pred_bps, param, param_without_perturb = self.model(seq, return_param=True, return_count=True, perturb=self.perturb)

            pred_params, pred_counts = [], []
            for k in sorted(param[0].keys()):
                if k.startswith('score_'):
                    pred_params.append(torch.vstack([param[i][k] for i in range(len(seq))]))
                elif k.startswith('count_'):
                    pred_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
                elif isinstance(param[0][k], dict):
                    for kk in sorted(param[0][k].keys()):
                        if kk.startswith('score_'):
                            pred_params.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))
                        elif kk.startswith('count_'):
                            pred_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))


        # predict shape reactivity
        paired = []
        for pred_bp in pred_bps:
            p = []
            for i, j in enumerate(pred_bp):
                if j==0:
                    p.append([0, 0, 1])
                elif i<j:
                    p.append([1, 0, 0])
                elif i>j:
                    p.append([0, 1, 0])
                else:
                    raise RuntimeError('unreachable')
            p = torch.tensor(p, dtype=torch.float32, requires_grad=(not self.shape_only), device=pred.device)
            paired.append(p)
        targets = [ t.to(pred.device) for t in targets ]
        nlls = self.shape_model[dataset_id](seq, paired, targets)

        if self.shape_only:
            seq_l = torch.tensor([len(s) for s in seq], device=pred.device)
            loss = nlls / seq_l

        else:
            loss = 0.
            # NOTICE: The LwF needs to be calculated before the backpropagation in the SHAPE model, due to possible memory leaks related to the returned parameters.
            seq_l = torch.tensor([len(s) for s in seq], device=pred.device)
            if self.lwf_model is not None and self.lwf_weight > 0.0:
                # FY loss for learning without forgetting (LwF)
                with torch.no_grad():
                    _, _, lwf_pairs = self.lwf_model(seq)
                lwf_ref, _, _ = self.model(seq, param=param_without_perturb, constraint=lwf_pairs, max_internal_length=None)
                loss += self.lwf_weight * (pred - lwf_ref) / seq_l

            # optimize both shape and folding models
            nlls.backward()
            grads = [ p.grad for p in paired ]
            pseudoenergy = [ self.nu*(g[:, 0]+ g[:, 1]-g[:, 2]) for g in grads ]
            #logging.debug(f"grads = {grads[0][targets[0] > -1]}")
            logging.debug(f"pseduenergy = {pseudoenergy}")

            ref: torch.Tensor
            ref_s: list[str]
            ref, ref_s, _, param, _ = self.model(seq, param=param, return_param=True, return_count=True, pseudoenergy=pseudoenergy)

            ref_counts = []
            for k in sorted(param[0].keys()):
                if k.startswith('count_'):
                    ref_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
                elif isinstance(param[0][k], dict):
                    for kk in sorted(param[0][k].keys()):
                        if kk.startswith('count_'):
                            ref_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

            # prepare backpropagation with respect to SHAPE reactivity using implicit MLE
            class ADwrapper(torch.autograd.Function):
                @staticmethod
                def forward(ctx, *input):
                    return nlls

                @staticmethod
                def backward(ctx, grad_output):
                    return tuple( grad_output * (p-r) for p, r in zip(pred_counts, ref_counts) )

            loss += ADwrapper.apply(*pred_params) / seq_l

            if self.sl_weight > 0.0:
                with torch.no_grad():
                    ref2: torch.Tensor
                    ref2, _, _ = self.turner(seq)
                loss += self.sl_weight * (ref-ref2)**2 / seq_l

        if self.shape_only:
            logging.debug(f"Loss = {loss.item()}")
        else:
            logging.debug(f"Loss = {loss.item()} = ({pred.item()} - {ref.item()})")
        logging.debug(f' seq: {seq}')
        logging.debug(f'pred: {pred_s}')
        if not self.shape_only:
            logging.debug(f' ref: {ref_s}')
        if float(loss.item())> 1e10 or torch.isnan(loss):
            logging.error(fname)
            logging.error(f"{loss.item()}, {pred.item()}, {ref.item()}")
            logging.error(seq)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss
