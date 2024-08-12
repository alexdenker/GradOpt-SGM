from typing import Optional, Any, Dict, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

from torch import Tensor

def Euler_Maruyama_sde_predictor(
    score,
    sde,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    ) -> Tuple[Tensor, Tensor]:
    
    s, _ = score(x, time_step, eval=True)
    score.zero_grad()
    s.detach() 
        
    drift, diffusion = sde.sde(x, time_step)


    x_mean = x - (drift - diffusion[:, None, None, None].pow(2)*s)*step_size
    noise = torch.sqrt(diffusion[:, None, None, None].pow(2)*step_size)*torch.randn_like(x)

    x = x_mean + noise 

    return x.detach(), x_mean.detach()


def ddim_step(
    score,
    sde, 
    x: Tensor,
    time_step,
    step_size: float,
    ) -> Tuple[Tensor, Tensor]:

    tminus1 = time_step-step_size
    tminus1 = torch.clamp(tminus1, 1e-5, 1)
    s, _ = score(x, time_step, eval=True)
    score.zero_grad()
    s.detach() 

    std_t = sde.marginal_prob_std(time_step)[:, None, None, None]
    std_tminus1 = sde.marginal_prob_std(tminus1)[:, None, None, None]
    mean_t = sde.marginal_prob_mean_scale(time_step)[:, None, None, None]
    mean_tminus1 = sde.marginal_prob_mean_scale(tminus1)[:, None, None, None]

    x0hat = (x + std_t**2*s)/mean_t

    x_mean = mean_tminus1*x0hat

    eps = - std_t*s

    noise_deterministic = std_tminus1*eps

    x = x_mean + noise_deterministic

    return x.detach(), x_mean.detach()