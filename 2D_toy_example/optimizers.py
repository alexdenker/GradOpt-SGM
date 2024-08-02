import numpy as np 
import torch
from gaussian_mixture_model import diffuse_gmm
from experiments import Experiments2DIP

# from joblib import Parallel, delayed
# from multiprocessing import Process
# import threading

class GradNonConvFlow(Experiments2DIP):
    """Class of the graduated non-convexity flow, which corresponds to Algorithm 1 in the article:

    Pascal Fernsel, Željko Kereta, Alexander Denker, *Convergence Properties of Score-Based Models for Linear Inverse Problems Using Graduated Optimisation*, 2024 IEEE International Workshop on Machine Learning for Signal Processing, September 22-25, 2024, London, UK

    This class contains the main ingredients of the optimization method and is a subclass of the `Experiments2DIP` class.

    Attributes:
    -----
    - `step_size`: float,
            Step size of the optimization method.
    """

    def __init__(self, step_size: float = 1, *args, **kwargs):
        self.step_size = step_size
        super().__init__(*args, **kwargs)

    def run(self):
        """Run the algorithm of the graduated non-convexity flow.
        """
        self.history_tensor = torch.zeros(self.t_initial_points.shape[0], self.x_initial_grid.shape[1], self.max_num_iter+1, self.IP.x_true.shape[0]).to(torch.float16)

        for i in range(self.t_initial_points.shape[0]):
            print('t: ', i+1, ' out of ', self.t_initial_points.shape[0])
            t_current = self.t_initial_points[i]
            if self.flag_smoothing_schedule_lin:
                smoothing_schedule = np.linspace(t_current,self.tmin, self.max_num_iter)
            else:
                smoothing_schedule = np.logspace(np.log10(t_current),np.log10(self.tmin), self.max_num_iter)
            gmm_diffs = []
            for k in range(self.max_num_iter):
                gmm_diffs.append(diffuse_gmm(self.IP.prior, smoothing_schedule[k], self.IP.sigma))
            # for j in range(self.x_initial_grid.shape[1]):
            x = torch.tensor(self.x_initial_grid).float()
            # x = torch.tensor([10, 10]).float()
            self.history_tensor[i,:,0,:] = x.T
            for k in range(self.max_num_iter):
                scores = gmm_diffs[k].score(x.T).T
                dk = smoothing_schedule[k]*(self.IP.alphas[1] * scores - self.IP.alphas[0]*torch.matmul(self.IP.A.T, torch.matmul(self.IP.A, x) - self.IP.y_true.unsqueeze(0).T)) #dk has here size: (self.IP.x_true.shape[0], self.x_initial_grid.shape[1]). The same holds for x here.
                x = x + self.step_size*dk
                self.history_tensor[i,:,k+1,:] = x.T
                # torch.save(history_tensor, 'historyTensor.pt')
            # torch.save(self.history_tensor, nameExperiment)

class GradientLike(Experiments2DIP):
    """Class of the graduated non-convexity flow, which corresponds to Algorithm 1 in the article:

    Pascal Fernsel, Željko Kereta, Alexander Denker, *Convergence Properties of Score-Based Models for Linear Inverse Problems Using Graduated Optimisation*, 2024 IEEE International Workshop on Machine Learning for Signal Processing, September 22-25, 2024, London, UK

    This class contains the main ingredients of the optimization method and is a subclass of the `Experiments2DIP` class.
    """

    def __init__(self, *args, **kwargs):
        # self.step_size = step_size
        super().__init__(*args, **kwargs)

    def run(self, alpha_Armijo: float = 1e-3, beta_Armijo: float = 9e-1):
        """Run the gradient like algorithm.
        """
        self.history_tensor = torch.zeros(self.t_initial_points.shape[0], self.x_initial_grid.shape[1], self.max_num_iter+1, self.IP.x_true.shape[0]).to(torch.float16)

        gmm_diff_tmin = diffuse_gmm(self.IP.prior, self.tmin, self.IP.sigma)

        for i in range(self.t_initial_points.shape[0]):
            print('t: ', i+1, ' out of ', self.t_initial_points.shape[0])
            t_current = self.t_initial_points[i]
            if self.flag_smoothing_schedule_lin:
                smoothing_schedule = np.linspace(t_current,self.tmin, self.max_num_iter)
            else:
                smoothing_schedule = np.logspace(np.log10(t_current),np.log10(self.tmin), self.max_num_iter)
            gmm_diffs = []
            for k in range(self.max_num_iter):
                gmm_diffs.append(diffuse_gmm(self.IP.prior, smoothing_schedule[k], self.IP.sigma))
            # for j in range(self.x_initial_grid.shape[1]):
            x = torch.tensor(self.x_initial_grid).float()
            # x = torch.tensor([10, 10]).float()
            self.history_tensor[i,:,0,:] = x.T
            for k in range(self.max_num_iter):
                scores = gmm_diffs[k].score(x.T).T
                scores_tmin = gmm_diff_tmin.score(x.T).T #scores at tmin
                tempResult = self.IP.alphas[0]*torch.matmul(self.IP.A.T, torch.matmul(self.IP.A, x) - self.IP.y_true.unsqueeze(0).T)
                dk = smoothing_schedule[k]*(self.IP.alphas[1] * scores - tempResult) #dk has here size: (self.IP.x_true.shape[0], self.x_initial_grid.shape[1]). The same holds for x here.
                grad = -self.IP.alphas[1] *scores_tmin + tempResult
                grad_dk = torch.sum(torch.mul(grad, dk), dim=0)
                mask_doStep = grad_dk < 0
                if torch.all(~mask_doStep):
                    self.history_tensor[i,:,k+1,:] = x.T
                    continue
                dk[:, ~mask_doStep] = 0 #Not needed, I think.
                # self.history_tensor[i,mask_doStep, k+1, :] = x[:, mask_doStep].T

                #Now determine the step size for all the x's based on the Armijo step size rule
                # lossLandscapePlot = self.alphas[0]/2*torch.sum((torch.matmul(self.A, torch.from_numpy(pos).float()) - self.y_true.reshape(2, 1).repeat(1,pos.shape[1]))**2, 0) + self.alphas[1] * r_new_plot
                if k==10:
                    pass
                RHS = self.IP.alphas[0]/2 * torch.sum((torch.matmul(self.IP.A, x[:, mask_doStep]) - self.IP.y_true.reshape(2, 1).repeat(1,x[:, mask_doStep].shape[1]))**2, 0) - self.IP.alphas[1] * np.log(gmm_diff_tmin.pdf(x[:, mask_doStep].T)).T
                mask_stepsizes_found = torch.zeros(dk[:, mask_doStep].shape[1], dtype=bool)
                step_sizes = torch.zeros(dk[:, mask_doStep].shape[1])
                for ell in range(10):
                    step_sizes[~mask_stepsizes_found] = torch.tensor([beta_Armijo**ell]).expand(dk[:, mask_doStep].shape[1])[~mask_stepsizes_found]
                    x_test = x[:, mask_doStep] + torch.mul(step_sizes.expand(2, step_sizes.shape[0]), dk[:, mask_doStep]) #x_test has size (2, step_sizes.shape[0])
                    loss_new = self.IP.alphas[0]/2 * torch.sum((torch.matmul(self.IP.A, x_test) - self.IP.y_true.reshape(2, 1).repeat(1,x_test.shape[1]))**2, 0) - self.IP.alphas[1] * np.log(gmm_diff_tmin.pdf(x_test.T)).T #loss_new has size (step_sizes.shape[0])
                    mask_stepsizes_found = loss_new <= RHS + alpha_Armijo*torch.mul(step_sizes, grad_dk[mask_doStep])
                    if torch.all(mask_stepsizes_found):
                        break
                x[:, mask_doStep] = x[:, mask_doStep] + torch.mul(step_sizes.expand(2, step_sizes.shape[0]), dk[:, mask_doStep])
                self.history_tensor[i,:,k+1,:] = x.T