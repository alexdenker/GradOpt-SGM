import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np 
import torch
from scipy.linalg import null_space
from gaussian_mixture_model import GaussianMixtureModel, diffuse_gmm
from inverse_problem import InverseProblem

class Experiments2DIP:
    """CLass of the experiments. This class contains the main ingredients for the numerical experiments with the 2D toy example in the article

    Pascal Fernsel, Željko Kereta, Alexander Denker, *Convergence Properties of Score-Based Models for Linear Inverse Problems Using Graduated Optimisation*, 2024 IEEE International Workshop on Machine Learning for Signal Processing, September 22-25, 2024, London, UK

    Instance Attributes:
    ---------
    - `IP`: `InverseProblem`,
            `InverseProblem` object specifying the inverse problem.
    - `x_initial_grid`: numpy.ndarray,
            Grid of initial points to start the algorithm. Shape: (2, number of initial points). It could be for example:
            `np.stack([xx.ravel(), yy.ravel()])` where
            `xx, yy = np.meshgrid(np.linspace(-10,10, 100), np.linspace(-10,10, 100))`.
    - `t_initial_points`: np.ndarray,
            Initial t's of the smoothing schedule. Could be for example np.logspace(np.log10(10),np.log10(1e-2), 100).
    - `max_num_iter`: int,
            Maximal number of iterations of the algorithm.
    - `tmin`: float,
            Minimum t of the smoothing schedule.
    - `flag_smoothing_schedule_lin`: bool,
            Flag to indicate whether the smoothing schedule is linear or logarithmic.
    - `history_tensor`: torch.Tensor,
            Tensor of shape (t_initial_points.shape[0], x_initial_grid.shape[1], max_num_iter+1, IP.x_true.shape[0]) containing the history of the iterates.
    
    Methods:
    ---------
    - `plot_convRate_globalMin()`: Plot the convergence rate to the global minimum of the cost function.
    """

    def __init__(self,
                 IP: InverseProblem,
                 x_initial_grid: np.ndarray,#shape: "(2,100)"
                 t_initial_points: np.ndarray = np.logspace(np.log10(1),np.log10(1e-3), 40),
                 max_num_iter: int = 1300,
                 tmin : float = 1e-4,
                 flag_smoothing_schedule_lin = True):
        
        self.IP = IP
        self.max_num_iter = max_num_iter
        self.x_initial_grid = x_initial_grid
        self.t_initial_points = t_initial_points
        self.tmin = tmin
        self.flag_smoothing_schedule_lin = flag_smoothing_schedule_lin
        self.history_tensor = None

    def saveTensor_convRate_globalMin(self, path: str, convergence_precision: float = 0.1, known_global_min: torch.Tensor = None):
        """Save the tensor of the rate of trajectories converging to the global minimum depending on the iteration number and t_max.

        Args:
        -----
        - `path`: str,
                Path to save the tensor.
        - `convergence_precision`: float,
                Defines the epsilon neighbourhood in order to specify the convergence of the iterates.
        - `known_global_min`: torch.Tensor,
                Known global minimum of the cost function. If `None`, the global minimum is calculated.
        """
        if known_global_min is not None:
            xmin_global = known_global_min
        else:
            xmin_global = self.IP.get_global_minimum(self.tmin)
        if self.history_tensor is None:
            raise ValueError('The history tensor is not specified.')
        fontSize = 28
        conv_rate = torch.sum(torch.le(torch.norm(self.history_tensor - xmin_global, dim=3), convergence_precision).int(), dim=1)/self.history_tensor.shape[1]
        # t_initial_points = np.logspace(np.log10(self.t_inital_minmax[0]),np.log10(self.t_inital_minmax[1]), self.num_t_initial_points)
        torch.save(conv_rate, path)

    def plot_convRate_globalMin(self, path: str, conv_rate_tensor: torch.Tensor, vminmax = [0, 1]):
        """Plot the rate of trajectories converging to the global minimum depending on the iteration number and t_max based on the tensor saved by the method `saveTensor_convRate_globalMin`.

        Args:
        -----
        - `path`: str,
                Path to save the plot.
        - `conv_rate_tensor`: torch.Tensor,
                Tensor of the rate of trajectories converging to the global minimum depending on the iteration number and t_max saved by the method `saveTensor_convRate_globalMin`.
        - `vminmax`: list,
                List of two elements specifying the minimum and maximum values of the colorbar.
        """
        fig, ax = plt.subplots(1,1)
        xx, yy = np.meshgrid(np.arange(self.max_num_iter+1), self.t_initial_points)
        ctr = ax.contour(xx, yy, conv_rate_tensor, levels=3, colors='k', alpha = 0.4)
        im = ax.pcolormesh(xx, yy, conv_rate_tensor, shading='gouraud')
        cbar = fig.colorbar(im, ax=ax, location='bottom', pad=0.1)
        im.set_clim(vmin=vminmax[0], vmax=vminmax[1])
        cbar.ax.tick_params(labelsize=32)
        ax.clabel(ctr, fontsize=28)
        plt.yscale('log')
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.locator_params(axis='x', nbins=5)
        ax.set_xlabel('Iteration', fontsize = 32, labelpad=6)
        ax.set_ylabel('$t_{max}$', fontsize = 32, labelpad=3)
        plt.gcf().set_size_inches(15, 15)
        plt.savefig(path, format="png", bbox_inches="tight")
        plt.show()

    def saveTensor_convRate_statPoints(self, path: str, convergence_precision: float = 0.1):
        """Save the tensor of the rate of trajectories converging to stationary points depending on the iteration number and t_max.

        Args:
        -----
        - `path`: str,
                Path to save the tensor.
        - `convergence_precision`: float,
                Defines the epsilon neighbourhood in order to specify the convergence of the iterates.
        """
        if self.history_tensor is None:
            raise ValueError('The history tensor is not specified.')
        fontSize = 28
        grad_field = torch.zeros(self.t_initial_points.shape[0], self.x_initial_grid.shape[1], self.max_num_iter+1, self.IP.x_true.shape[0])
        # grad = -alpha2 *scores_tmin + alpha1*torch.matmul(A.T, torch.matmul(A, xKobler) - y)
        gmm_diff_tmin = diffuse_gmm(self.IP.prior, self.tmin, self.IP.sigma)
        for i in range(self.t_initial_points.shape[0]):
            # print('t: ', i+1, ' out of ', self.t_initial_points.shape[0])
            for j in range(self.max_num_iter):
                x = self.history_tensor[i,:,j,:].T.to(torch.float32) #x has shape (self.IP.x_true.shape[0], self.x_initial_grid.shape[1])
                scores = gmm_diff_tmin.score(x.T).T #scores has shape (self.IP.x_true.shape[0], self.x_initial_grid.shape[1])
                result = -self.IP.alphas[1] *scores + self.IP.alphas[0]*torch.matmul(self.IP.A.T, torch.matmul(self.IP.A, x) - self.IP.y_true.unsqueeze(0).T)
                grad_field[i,:,j,:] = result.T
        conv_rate = torch.sum(torch.le(torch.norm(grad_field - torch.tensor([0., 0.]), dim=3), convergence_precision).int(), dim=1)/grad_field.shape[1]
        # t_initial_points = np.logspace(np.log10(self.t_inital_minmax[0]),np.log10(self.t_inital_minmax[1]), self.num_t_initial_points)
        torch.save(conv_rate, path)

    def plot_convRate_statPoints(self, path, conv_rate_tensor: torch.Tensor, vminmax = [0, 1]):
        """Plot the rate of trajectories converging to the stationary points depending on the iteration number and t_max based on the tensor saved by the method `saveTensor_convRate_localMin`.
        
        Args:
        -----
        - `path`: str,
                Path to save the plot.
        - `conv_rate_tensor`: torch.Tensor,
                Tensor of the rate of trajectories converging to the stationary points depending on the iteration number and t_max saved by the method `saveTensor_convRate_localMin`.
        - `vminmax`: list,
                List of two elements specifying the minimum and maximum values of the colorbar.
        """
        # t_initial_points = np.logspace(np.log10(self.t_inital_minmax[0]),np.log10(self.t_inital_minmax[1]), self.num_t_initial_points)
        fig, ax = plt.subplots(1,1)
        xx, yy = np.meshgrid(np.arange(self.max_num_iter+1), self.t_initial_points)
        ctr = ax.contour(xx, yy, conv_rate_tensor, levels=3, colors='k', alpha = 0.4)
        im = ax.pcolormesh(xx, yy, conv_rate_tensor, shading='gouraud')
        cbar = fig.colorbar(im, ax=ax, location='bottom', pad=0.1)
        im.set_clim(vmin=vminmax[0], vmax=vminmax[1])
        cbar.ax.tick_params(labelsize=32)
        ax.clabel(ctr, fontsize=28)
        plt.yscale('log')
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.locator_params(axis='x', nbins=5)
        ax.set_xlabel('Iteration', fontsize = 32, labelpad=6)
        ax.set_ylabel('$t_{max}$', fontsize = 32, labelpad=3)
        plt.gcf().set_size_inches(15, 15)
        plt.savefig(path, format="png", bbox_inches="tight")
        plt.show()
    
    def plot_convRate_globalMin_xyPlane(self, path, convergence_precision: float = 0.1, known_global_min: torch.Tensor = None, xy_min_max : np.array = np.array([-10, 10]), stationary_points: torch.Tensor = None, path_iterates: torch.Tensor = None):
        """Plot the Rate of trajectories (for different choices of $t_\text{max}$) converging to the global minimum depending on the initial starting point in the xy-plane. We do not need to save the tensor in advance, since we are able to calculate it quickly on the fly.

        Args:
        -----
        - `path`: str,
                Path to save the plot.
        - `convergence_precision`: float,
                Defines the epsilon neighbourhood in order to specify the convergence of the iterates.
        - `known_global_min`: torch.Tensor,
                Known global minimum of the cost function. If `None`, the global minimum is calculated. Example: torch.tensor([1., 1.])
        - `xy_min_max`: np.array,
                Array of two elements specifying the minimum and maximum values of the x and y axis. Default: np.array([-10, 10]).
        - `stationary_points`: torch.Tensor,
                Tensor of handpicked stationary points in order to plot them of shape (IP.x_true.shape[0], number of stationary points). Default: None.
        - `path_iterates`: torch.Tensor,
                Tensor of the path of iterates to plot them of shape (number of iterates, IP.x_true.shape[0]). Default: None.
        """
        if known_global_min is not None:
            xmin_global = known_global_min
        else:
            xmin_global = self.IP.get_global_minimum(self.tmin)
        if self.history_tensor is None:
            raise ValueError('The history tensor is not specified.')
        
        fontSize = 28
        # Plot loss landscape
        xx, yy = np.meshgrid(np.linspace(xy_min_max[0],xy_min_max[1], 500), np.linspace(xy_min_max[0],xy_min_max[1], 500))
        pos = np.stack([xx.ravel(), yy.ravel()])
        fig, ax = plt.subplots(1,1)
        gmm_diff_plot = diffuse_gmm(self.IP.prior, self.tmin, self.IP.sigma)
        r_new_plot = -np.log(gmm_diff_plot.pdf(torch.from_numpy(pos.T)))
        lossLandscapePlot = self.IP.alphas[0]/2*torch.sum((torch.matmul(self.IP.A, torch.from_numpy(pos).float()) - self.IP.y_true.reshape(2, 1).repeat(1,pos.shape[1]))**2, 0) + self.IP.alphas[1] * r_new_plot
        ctr = ax.contour(xx, yy, lossLandscapePlot.reshape(xx.shape), levels=25, colors='k', alpha = 0.5)
        ax.clabel(ctr, fontsize=fontSize)
        # im = ax.pcolormesh(xx, yy, lossLandscapePlot.reshape(xx.shape), alpha=0.5)
        # fig.colorbar(im, ax=ax)
        # ax.clabel(ctr)

        xy_axis_nr = np.sqrt(self.x_initial_grid.shape[1]).astype(int)
        conv_rate = torch.sum(torch.le(torch.norm(self.history_tensor[:,:,-1,:] - xmin_global, dim=2), convergence_precision).int(), dim=0)/self.history_tensor.shape[0]
        xx2 = torch.reshape(torch.Tensor(self.x_initial_grid[0,:]), (xy_axis_nr, xy_axis_nr))
        yy2 = torch.reshape(torch.Tensor(self.x_initial_grid[1,:]), (xy_axis_nr, xy_axis_nr))
        # ctr2 = ax.contour(xx2, yy2, torch.reshape(conv_rate, (xy_axis_nr, xy_axis_nr)), levels=3, colors='k', alpha = 0.3)
        im2 = ax.pcolormesh(xx2, yy2, torch.reshape(conv_rate, (xy_axis_nr, xy_axis_nr)), shading='gouraud')

        # axins = inset_axes(ax,
        #             width="100%",  
        #             height="5%",
        #             loc='lower center',
        #             borderpad=0
        #            )

        cbar = fig.colorbar(im2, ax=ax, location='bottom', pad=0.1)
        cbar.ax.tick_params(labelsize=32)
        ax.scatter(pos[0,torch.argmin(lossLandscapePlot)], pos[1,torch.argmin(lossLandscapePlot)], marker='*', c='r', s=200, label="Global Minimum")

        # if flag_plot_gradients:
        #     stationary_points = self.get_stationary_points(0.1, np.array([-10, 10]), False)
        #     ax.scatter(stationary_points[0,:], stationary_points[1,:], c='r', s=10, alpha=0.1, label="Points with $||\\nabla F(x,t_{min})|| <= 0.1$")

        if stationary_points is not None:
            ax.scatter(stationary_points[0,:], stationary_points[1,:], facecolors='none', edgecolors='r', s=60, linewidths=2, label="Local Minima")

        x1 = torch.matmul(torch.linalg.pinv(self.IP.A), self.IP.y_true)
        ns = torch.from_numpy(null_space(self.IP.A.numpy())).flatten()
        length_of_Line = 14
        ax.plot(x1 - length_of_Line*ns, x1 + length_of_Line*ns, 'k', linewidth=2.5, label="$\{x\in \mathbb{R}^2 : Ax=y\}$")

        if path_iterates is not None:
            for i in range(path_iterates.shape[0]):
                if i == 0:
                    ax.plot(path_iterates[i,:,0], path_iterates[i,:,1], c='orange', alpha=0.8, linewidth=2, label="Path of Iterates")
                    ax.scatter(path_iterates[i,0,0], path_iterates[i,0,1], c='orange', s=40, label="Start of Path", alpha=0.8)
                else:
                    ax.plot(path_iterates[i,:,0], path_iterates[i,:,1], c='orange', alpha=0.8, linewidth=2, label="")
                    ax.scatter(path_iterates[i,0,0], path_iterates[i,0,1], c='orange', s=40, label="", alpha=0.8)
            # ax.plot(path_iterates[:,0], path_iterates[:,1], c='white', alpha=0.3, linewidth=2, label="Path of Iterates")
        # ax.text(x1[0]+0.05,  x1[1]+0.05, "{x: Ax = y}")

        ax.set_xlim(xy_min_max[0], xy_min_max[1])
        ax.set_ylim(xy_min_max[0], xy_min_max[1])

        # ax.set_xlabel('Dimension x', fontsize = fontSize, labelpad=5)
        # ax.set_ylabel('Dimension y', fontsize = fontSize, labelpad=5)

        # ax.legend(loc='lower right', fontsize=24, framealpha=0.8)
        ax.legend(loc='lower right', fontsize=24)

        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.locator_params(nbins=5)
        ax.set_xlabel('', fontsize = 32, labelpad=6)
        ax.set_ylabel(' ', fontsize = 32, labelpad=7)
        plt.gcf().set_size_inches(15, 15)
        plt.savefig(path, format="png", bbox_inches="tight")
        plt.show()

        return fig, ax

    def get_stationary_points(self, precision: np.double=1e-3, xy_min_max : np.array = np.array([-10, 10]), plotting_flag = False):
        """Utility function to get the stationary points of the loss landscape.

        Args:
        -----
        - `precision`: np.double,
                Defines the epsilon neighbourhood in order to specify the points for which the norm of the gradient is smaller than `precision`.
        - `xy_min_max`: np.array,
                Array of two elements specifying the minimum and maximum values of the x and y axis. Default: np.array([-10, 10]).
        - `plotting_flag`: bool,
                Flag to indicate whether to plot the stationary points. Default: False.
        """
        num_points_xy = 2000
        xx, yy = np.meshgrid(np.linspace(xy_min_max[0],xy_min_max[1], num_points_xy), np.linspace(xy_min_max[0],xy_min_max[1], num_points_xy))
        pos = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()])).to(torch.float32)
        gmm_diff_tmin = diffuse_gmm(self.IP.prior, self.tmin, self.IP.sigma)
        scores = gmm_diff_tmin.score(pos.T).T
        grad_field_norm = torch.norm(-self.IP.alphas[1] *scores + self.IP.alphas[0]*torch.matmul(self.IP.A.T, torch.matmul(self.IP.A, pos) - self.IP.y_true.unsqueeze(0).T), dim=0)
        stationary_points = pos[:,torch.le(grad_field_norm, precision)]
        if plotting_flag:
            fig, ax = plt.subplots(1,1)
            ctr = ax.contour(xx, yy, torch.reshape(grad_field_norm, (num_points_xy, num_points_xy)), levels=3, colors='k', alpha = 0.3)
            im = ax.pcolormesh(xx, yy, torch.reshape(grad_field_norm, (num_points_xy, num_points_xy)))
            ax.scatter(stationary_points[0,:], stationary_points[1,:], c='r', s=10, label="Stationary Points")
            fig.colorbar(im, ax=ax)
            ax.clabel(ctr)
            # plt.yscale('log')
            plt.gcf().set_size_inches(15, 10)
            plt.show()
        return stationary_points