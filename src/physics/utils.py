

import torch 


def create_noisy_data(y, rel_noise):

    noise_level = rel_noise*torch.mean(y.abs())
    noise = torch.randn_like(y)
    y_noise = y + noise_level*noise

    return y_noise, noise_level