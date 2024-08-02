import torch 
from torch.distributions.multivariate_normal import MultivariateNormal

class GaussianMixtureModel():
    """Class of the Gaussian Mixture Model. This class contains the main ingredients of the considered prior of the inverse problem. The code is an adaptation from: https://colab.research.google.com/drive/16ZNcxNo7DJh1yZfFa2ombVd_uZqFTcQe?usp=sharing#scrollTo=Aw-koR6idhdi

    Attributes:
    ---------
    - `n_dim`: int,
            Dimension of the Gaussian Mixture Model
    - `n_components`: int,
            Number of compnents of the Gaussian Mixture Model
    - `weights`: torch.Tensor,
            Weights for individual components of the Gaussian Mixture Model.
    - `means`: list,
            List of length `n_components`. Each element is a torch.Tensor of size `n_dim` representing the mean of the corresponding component.
    - `covs`: list,
            List of length `n_components`. Each element is a torch.Tensor of size `n_dim x n_dim` representing the covariance matrix of the corresponding component.
    """
    def __init__(self, n_dim: int, n_components: int, weights: torch.Tensor, means: list, covs: list):

        self.n_components = n_components 
        self.n_dim = n_dim 

        self.weights = weights
        self.norm_weights = self.weights / torch.sum(self.weights)

        self.means = means 
        self.covs = covs
        self.precs = [torch.linalg.inv(cov) for cov in covs]

        self.components = [] 
        for i in range(self.n_components):
            self.components.append(MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]))

    def pdf(self, x):
        """ Computes the probability density function at $x$ of the Gaussian Mixture Model.

        Args:
        -----
        - `x`: torch.Tensor,
                Tensor of size `n_samples x n_dim`. Could be for example `x = np.stack([xx.ravel(), yy.ravel()]).T` with `xx, yy = np.meshgrid(np.linspace(xy_min_max[0],xy_min_max[1], 500), np.linspace(xy_min_max[0],xy_min_max[1], 500))`
        """
        component_pdf = torch.stack([torch.exp(self.components[i].log_prob(x)) for i in range(self.n_components)]).T

        weighted_compon_pdf = component_pdf * self.norm_weights

        return weighted_compon_pdf.sum(dim=1)

    def score(self, x):
        """ Compute the score of the Gaussian Mixture Model.

        Args:
        -----
        - `x`: torch.Tensor,
                Tensor of size `n_samples x n_dim`. Could be for example `x = np.stack([xx.ravel(), yy.ravel()]).T` with `xx, yy = np.meshgrid(np.linspace(xy_min_max[0],xy_min_max[1], 500), np.linspace(xy_min_max[0],xy_min_max[1], 500))`

        """
        component_pdf = torch.stack([torch.exp(self.components[i].log_prob(x)) for i in range(self.n_components)]).T
        #print(component_pdf.shape, self.norm_weights.shape)

        weighted_compon_pdf = component_pdf * self.norm_weights
        #print(weighted_compon_pdf.shape)
        participance = weighted_compon_pdf / torch.sum(weighted_compon_pdf, dim=1, keepdim=True)
        #print(participance.shape)

        scores = torch.zeros_like(x)
        for i in range(self.n_components):
            gradvec = - (x - self.means[i]) @ self.precs[i]
            scores += participance[:, i:i+1] * gradvec

        return scores

def marginal_prob_std(t, sigma):
    """ A util function which outputs the standard deviation (std) of the conditional distribution. Note that this std -> 0, when t->0. So it's not numerically stable to sample t=0 in the dataset.

    Args:
    -----
    - `t`: float,
            Time step
    - `sigma`: float,
            Standard deviation of the Gaussian noise
    """
    return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) )

def diffuse_gmm(gmm, t, sigma):
    """ Diffuse the Gaussian Mixture Model `gmm` for the given `t` and `sigma`.

    Args:
    -----
    - `gmm`: `GaussianMixtureModel`,
            Gaussian Mixture Model to be diffused
    - `t`: float,
            Time step
    - `sigma`: float,
            Standard deviation of the Gaussian noise.
    """
    lambda_t = marginal_prob_std(t, sigma)**2 # variance
    noise_cov = torch.eye(gmm.n_dim) * lambda_t
    covs_dif = [cov + noise_cov for cov in gmm.covs]
    return GaussianMixtureModel(n_dim=gmm.n_dim, 
                                n_components=gmm.n_components,
                                weights=gmm.weights, 
                                means=gmm.means, 
                                covs=covs_dif)

if __name__ == "__main__":
    import numpy as np 
    import matplotlib.pyplot as plt 

    n_dim = 2
    n_components = 5

    weights = torch.rand(n_components) 
    weights = weights / weights 
    #means = [torch.randn(n_dim)*2 for i in range(n_components)]
    
    means = [torch.tensor([2.5,2.5]), torch.tensor([2.5,0]), torch.tensor([-2.5,2.5]), 
        torch.tensor([-2.5,-2.5]), torch.tensor([-2.5,0])]
    covs = [0.1 * torch.eye(n_dim) for i in range(n_components)]

    gmm = GaussianMixtureModel(n_dim=n_dim, n_components=n_components, weights=weights, means=means, covs=covs)

    sigma = 5
    gmm = diffuse_gmm(gmm, 1, sigma)

    xx, yy = np.meshgrid(np.linspace(-5,5, 260), np.linspace(-5,5, 260))

    pos = np.stack([xx.ravel(), yy.ravel()])

    probs = gmm.pdf(torch.from_numpy(pos.T))

    print("PROBABILITIES: ", probs.shape)
    print(weights)
    fig, ax = plt.subplots(1,1)

    im = ax.pcolormesh(xx, yy, probs.reshape(xx.shape))
    fig.colorbar(im, ax=ax)
    plt.show()

    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, -torch.log(probs.reshape(xx.shape)),
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()