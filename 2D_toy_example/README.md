# 2D Toy Example

This folder contains the necessary codebase to reproduce the numerical experiments in the paper regarding the 2D inverse problem discussed in Section 4.1.

## Description of the Inverse Problem
As described in the article, we consider a two dimensional inverse problem with forward operator $`\ \mathbf{A} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\ `$ and clean measurements $\mathbf{y} = (2, 0)^\intercal.$ As a prior, we consider a Gaussian mixture model whose score can be computed analytically. The density at time step $t$ is a Gaussian mixture model with a perturbed covariance matrix $\Sigma_k^t = \Sigma_k + \frac{\sigma^{2t} -1}{2 \log \sigma} \mathbf{I}$. The diffusion process is given by the forward SDE $` d\mathbf{x}_t = \sigma^t d \mathbf{w}_t`$ with perturbation kernel $p_{t|0}(\mathbf{x}_t|\mathbf{x}_0)= \mathcal{N}(\mathbf{x}_t | \mathbf{x}_0, \frac{\sigma^{2t} -1}{2 \log \sigma} \mathbf{I})$.

We choose a constant regularisation parameter $\alpha_t = 5$ and adjust one mean of the Gaussian mixture model to the position $\mathbf{x}^* = (1, 1)^\intercal$ in order to ensure that the global minimum of the cost function $f$ with $t_\text{min}=10^{-3}$ is at $\mathbf{x}^*$.

## Evaluation of the Algorithms
We evaluate the graduated non-convexity flow (Algorithm 1) with a constant step size $\lambda_i = 1$ as well as the gradient-like method (Algorithm 2) with the adaptive smoothing schedule. The values $t_i$ are evenly spaced between $t_\text{min}$ and $t_\text{max}.$

The goal is to analyse the algorithms in terms of their convergence properties with respect to the initialisations $`\mathbf{x}_1,`$ the initial smoothing parameter $t_\text{max}$ and the iteration number. To do so, we run the algorithms with 1300 iterations for $10^{4}$ equally spaced initial points $`\mathbf{x}_1`$ on $`[-10, 10]^2`$ as well as 100 different values of $t_\text{max}\in [10^{-2}, 10]$, which are evenly distributed on a log scale. This results in a tensor with size $`[100, 10000, 1301, 2]`$, which will be stored as an instance attribute of the class `Experiments2DIP`.

## Codebase
In the following, we provide a short overview of the provided files.

- `main_2D_IP_jpynb`: Jupyter notebook script to reproduce the results. The plots are saved in the folder `plots`.
- `gaussian_mixture_model.py`: Defines the class `GaussianMixtureModel`, which contains all relevant information of the considered prior given by a Gaussian mixture model.
- `inverse_problem.py`: Defines the class `InverseProblem`, which contains all the relevant information of the considered inverse problem including the forward operator, the true solution and the prior in form of a `GaussianMixtureModel`.
- `experiments.py`: Defines the class `Experiments2DIP`, which contains the information of the numerical experiment including the inverse problem (in form of `InverseProblem`), the smoothing schedule as well as methods to plot the results.
- `optimizers.py`: Defines the classes `GradNonConvFlow` and `GradientLike`, which are subclasses `Experiments2DIP`. They contain additional information of the considered optimizer
    - `GradNonConvFlow`: Graduated Non-convexity Flow (see Algorithm 1 in the article)
    - `GradientLike`: Gradient Like Algorithm (see Algorithm 2 in the article)
