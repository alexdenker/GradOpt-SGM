import matplotlib.pyplot as plt 
import numpy as np 
import torch
from scipy.linalg import null_space
from gaussian_mixture_model import GaussianMixtureModel, diffuse_gmm


class InverseProblem:
    """Class of Inverse Problems. This class contains the main ingredients of the considered inverse problem.

    Attributes:
    -----
    - `A`: torch.Tensor,
            Forward operator of the inverse problem. Could be for example: `A = torch.tensor([[1.,1.],[0., 0.]])`.
    - `x_true`: torch.Tensor,
            True solution of the inverse problem. Could be for example `x_true = torch.tensor([1., 1.])`.
    - `prior`: GaussianMixtureModel,
            Prior distribution.
    - `alphas`: torch.Tensor,
            Weights of the discrepancy term (alphas[0]) and the prior term (alphas[1]).
    - `sigma`: float,
            Parameter for the diffusion process.
    """

    def __init__(self,
                 A: torch.Tensor,
                 x_true: torch.Tensor,
                 prior: GaussianMixtureModel,
                 alphas: torch.Tensor,
                 sigma: float=10.):
        self.A = A
        self.x_true = x_true
        # self.y_true = torch.matmul(self.A, self.x_true)
        self.prior = prior
        self.alphas = alphas
        self.sigma = sigma

    @property
    def y_true(self):
        return torch.matmul(self.A, self.x_true)
    

    def get_global_minimum(self, t: np.double=1e-3):
        """Get the global minimum of the loss landscape at time point `t`.

        Args:
        -----
        - `t`: np.double,
                Time point for the diffusion process.
        """
        xx, yy = np.meshgrid(np.linspace(-10,10, 1001), np.linspace(-10,10, 1001))
        pos = np.stack([xx.ravel(), yy.ravel()])
        gmm_diff_plot = diffuse_gmm(self.prior, t, self.sigma)
        r_new_plot = -np.log(gmm_diff_plot.pdf(torch.from_numpy(pos.T)))
        lossLandscapePlot = self.alphas[0]/2*torch.sum((torch.matmul(self.A, torch.from_numpy(pos).float()) - self.y_true.reshape(2, 1).repeat(1,pos.shape[1]))**2, 0) + self.alphas[1] * r_new_plot
        return torch.tensor([pos[0,torch.argmin(lossLandscapePlot)], pos[1,torch.argmin(lossLandscapePlot)]])