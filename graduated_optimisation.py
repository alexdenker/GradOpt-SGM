

"""
Implementation of the VarGrad loss. Change the direction of the SDE: going from 0 -> T
with x0 ~ N(0, I) and xT ~ data distribution
 
"""

import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 
import time 
import math 

import torch 
import torch.nn as nn 
import torchvision.utils as tvu

import wandb

from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src import Energy_UNetModel_full, VPSDE, SimpleTrafo, create_noisy_data
from src import EllipsesDataset

# wandb configs
entity = None 
project = None
mode = "online" #"disabled", #"online" ,
code_dir = "wandb"
dir = "wandb"


import argparse
parser = argparse.ArgumentParser(description='gradient_like')

parser.add_argument('--rel_noise', default=0.05)
parser.add_argument('--tmin', default=1e-3)
parser.add_argument("--alpha", default=10.0)
parser.add_argument("--num_steps", default=200)
parser.add_argument("--seed", default=10)
parser.add_argument("--dataset", default="aapm", choices=["aapm", "ellipses", "mnist"])
parser.add_argument("--num_angles", default=60)

parser.add_argument("--img_idx", default=0)
parser.add_argument("--use_ema", default=True)

parser.add_argument("--fixed_step_size", default=False)
parser.add_argument("--step_size", default=5e-4)
parser.add_argument("--verbose", default=False)

def main(args):


    if args.dataset == "ellipses":
        base_path = "models/ellipses"
        img_width = 128 

    elif args.dataset == "aapm":
        base_path = "models/mayo"
        img_width = 256
    else:
        raise NotImplementedError

    device = "cuda"

    cfg_optim = {
        "rel_noise": float(args.rel_noise), 
        "img_idx": int(args.img_idx),
        "num_steps": int(args.num_steps),
        "eps": float(args.tmin),
        "img_log_freq": 20,
        "alpha": float(args.alpha),
        "fixed_step_size": args.fixed_step_size,
        "step_size": float(args.step_size),
        "dataset": str(args.dataset),
        "img_width": img_width,
        "seed": int(args.seed),
        "angles": int(args.num_angles)
    }

    with open(os.path.join(base_path, "report.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)

 
    sde = VPSDE(beta_min=cfg_dict["diffusion"]["beta_min"], 
            beta_max=cfg_dict["diffusion"]["beta_max"]
            )

    model = Energy_UNetModel_full(
                marginal_prob_std=sde.marginal_prob_std,
                model_channels=cfg_dict["model"]["model_channels"],
                max_period=cfg_dict["model"]["max_period"],
                num_res_blocks=cfg_dict["model"]["num_res_blocks"],
                in_channels=cfg_dict["model"]["in_channels"],
                out_channels=cfg_dict["model"]["out_channels"],
                attention_resolutions=cfg_dict["model"]["attention_resolutions"],
                channel_mult=cfg_dict["model"]["channel_mult"])
    load_path = os.path.join(base_path,"ema_model.pt")
    print("Load model from: ",load_path)
    model.load_state_dict(torch.load(load_path))
    model.to("cuda")
    model.eval() 

    run_id = wandb.util.generate_id()
    wandb_kwargs = {
            "project": project,
            "entity": entity,
            "config": cfg_optim,
            "name":  f"{args.dataset}: gradient like (noise={args.rel_noise})",
            "mode": mode, 
            "settings": wandb.Settings(code_dir=code_dir),
            "dir": dir,
            "id": run_id
        }
    with wandb.init(**wandb_kwargs) as run:
        if args.fixed_step_size:
            save_method = "GL_fixedstep"
        else:
            save_method = "GL"
        log_dir = os.path.join("results", args.dataset, save_method, run_id)
        print("save model to ", log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        with open(os.path.join(log_dir, "config.yaml"), "w") as file:
            yaml.dump(cfg_optim, file)

        if args.dataset == "ellipses":
            val_dataset = EllipsesDataset(shape=cfg_dict["model"]["shape"], n_samples=1, normalise=True, seed=cfg_optim["img_idx"])
            x_gt = val_dataset[0].unsqueeze(0).to(device)

        elif args.dataset == "aapm":

            class MayoDataset(torch.utils.data.Dataset):
                def __init__(self, path):

                    self.X = torch.from_numpy(np.load(path)).float()

                def __len__(self):
                    return self.X.shape[0]

                def __getitem__(self, idx):
                    return self.X[idx].unsqueeze(0)
            val_dataset = MayoDataset("data/test_data.npy")

            x_gt = val_dataset[cfg_optim["img_idx"]].unsqueeze(0).to(device)

        else:
            raise NotImplementedError


        forward_op = SimpleTrafo(im_shape=[img_width,img_width], num_angles=cfg_optim["angles"])

        def forward(x):
            return forward_op.trafo(x) 

        def adjoint(y):
            return forward_op.trafo_adjoint(y) 

        def fbp(y):
            return forward_op.fbp(y) 


        with torch.no_grad():

            y = forward(x_gt)
            y_noise, noise_level = create_noisy_data(y, cfg_optim["rel_noise"])
            x_fbp = fbp(y_noise) 

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12,8))

        ax1.imshow(x_gt[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax1.set_title("x_gt")
        ax1.axis("off")

        ax2.imshow(y[0,0,:,:].detach().cpu().numpy().T)
        ax2.set_title("y")
        ax2.axis("off")

        ax3.imshow(y_noise[0,0,:,:].detach().cpu().numpy().T)
        ax3.set_title("y_noise")
        ax3.axis("off")

        ax4.imshow(x_fbp[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax4.set_title("x_fbp")
        ax4.axis("off")

        wandb.log({f"data_setup": wandb.Image(plt)})
        plt.close()
        

        #time_steps = np.linspace(1., cfg_optim['eps'], cfg_optim['num_steps'])
        time_steps = np.logspace(np.log10(1.), np.log10(cfg_optim['eps']), cfg_optim['num_steps'])
        
        time_step_eps = torch.ones(x_gt.shape[0], device=device)*cfg_optim["eps"]
        std = model.marginal_prob_std(time_step_eps)[:, None, None, None]
        mean_scale = sde.marginal_prob_mean_scale(time_step_eps)[:, None, None, None]
        snr_eps = mean_scale/std

        def f(x):
            time_step_eps = torch.ones(x.shape[0], device=device)*cfg_optim["eps"]
            energy = -model(x, time_step_eps, eval=True, only_energy=True)

            res = forward(x) - y_noise
            loss_data = 1/2*torch.sum(res**2)
            loss_reg = cfg_optim["alpha"]*energy.sum()/snr_eps
            loss = loss_data + loss_reg

            return loss.item(), loss_data.item(), loss_reg.item()


        # init_x ~ N(0,I)
        if int(args.seed) > 0:
            torch.manual_seed(int(args.seed))
        init_x = sde.prior_sampling([1, 1, img_width, img_width]).to(device)

        x = init_x

        beta0 = 0.01
        search_control_tau = 0.5
        search_control_c = 1e-3
        max_search_iter = 15

        psnr_list = []
        ssim_list = []
        
        psnr = peak_signal_noise_ratio(x_gt[0,0,:,:].cpu().numpy(), x[0,0,:,:].detach().cpu().numpy(), data_range=1.0)
        psnr_list.append(psnr)

        ssim = structural_similarity(x_gt[0,0,:,:].cpu().numpy(), x[0,0,:,:].detach().cpu().numpy(), data_range=1.0)
        ssim_list.append(ssim)

        res_dict = {}
        for i, step in tqdm(enumerate(time_steps), total=cfg_optim["num_steps"]):
            ones_vec = torch.ones(x.shape[0], device=device)
            time_step = ones_vec * step 

            score, _ = model(x, time_step, eval=True)
            std = model.marginal_prob_std(time_step)[:, None, None, None]
            mean_scale = sde.marginal_prob_mean_scale(time_step)[:, None, None, None]
            snr_t = mean_scale/std
            model.zero_grad()
            x = x.detach() 

            # mean, std = sde.marginal_prob(x, time_step) # for VESDE the mean is just x
            # x0hat = (x + std[:, None, None, None]**2*score)/mean[:,None,None,None]

            res = forward(x) - y_noise
            data_consistency_grad = adjoint(res) 

            # descent direction p 
            di = -time_step * (data_consistency_grad - cfg_optim["alpha"]*score/snr_t)

            if not cfg_optim["fixed_step_size"]:
                time_eps = torch.ones(x.shape[0], device=device)*cfg_optim["eps"]
                score0, _ = model(x, time_eps, eval=True)
                std0 = model.marginal_prob_std(time_eps)[:, None, None, None]

                model.zero_grad()
                x = x.detach() 

                with torch.no_grad():
                    # armijo step size rule

                    # find a good starting step size using Barzilai and Borwein "Two-Point Step Size Gradient Methods"
                    if i > 0:
                        diff_x = (x - last_x).detach().cpu().numpy().ravel()
                        diff_gradient = (last_gradient - di).detach().cpu().numpy().ravel() # - (g_i - g_{i-1}) = g_{i-1} - g_i

                        beta_init = 4*np.dot(diff_x, diff_gradient) / np.dot(diff_gradient, diff_gradient)
                        if beta_init == 0:
                            beta_init = beta0

                    else:
                        # use a heuristic choice for the first iteration
                        beta_init = beta0 

                    # score0(x,t) = nabla_x log p_t(x)
                    #nabla_fx = 1/noise_level**2*data_consistency_grad - cfg_optim["alpha"]*score0/snr_eps
                    nabla_fx = data_consistency_grad - cfg_optim["alpha"]*score0/snr_eps
                    
                    gradient_like_cond = np.dot(di.detach().cpu().numpy().ravel(), nabla_fx.detach().cpu().numpy().ravel()) / (np.linalg.norm(di.detach().cpu().numpy().ravel()) * np.linalg.norm(nabla_fx.detach().cpu().numpy().ravel()) )
                    wandb.log(
                    {"optim/gradient_like_cond": gradient_like_cond, "step": i}
                        ) 
                    
                    m = np.dot(di.detach().cpu().numpy().ravel(), nabla_fx.detach().cpu().numpy().ravel())
                    t = - search_control_c*m

                    if args.verbose:
                        print("< di, nabla f(x) > = ", gradient_like_cond  )
                        print("angle: ", np.arccos(gradient_like_cond)*180/np.pi)
                    
                        print("m: ", m)
                        print("t: ", t)

                    betak = beta_init
                    f_x, _, _ = f(x)
                    found_step_size = False
                    for _ in range(max_search_iter):
                        x_i = x + betak * di 
                        f_xi_delta, _, _ = f(x_i)
                        if args.verbose:
                            print("f(x) - f(x + beta*di) = ", f_x - f_xi_delta, " >  alpha * t = ", betak*t, " ?")
                        if f_x - f_xi_delta >= betak*t:
                            found_step_size = True 
                            break

                        betak = betak * search_control_tau

                    if args.verbose:
                        if found_step_size:
                            print("Found admissible step size fulfilling the armijo condition! Use: ", betak)
                        else:
                            print("Found no good step size.")
            else:
                betak = cfg_optim["step_size"]

            wandb.log(
                    {"optim/step_size": betak, "step": i}
                        ) 

            last_x = x.clone()
            last_gradient = di.clone()

            x = torch.clone(x + betak*di)
            psnr = peak_signal_noise_ratio(x_gt[0,0,:,:].cpu().numpy(), x[0,0,:,:].detach().cpu().numpy(), data_range=1.0)
            psnr_list.append(psnr)
            ssim = structural_similarity(x_gt[0,0,:,:].cpu().numpy(), x[0,0,:,:].detach().cpu().numpy(), data_range=1.0)
            ssim_list.append(ssim)

            mse_to_gt = torch.mean((x_gt - x)**2)
            loss, loss_data, loss_reg = f(x)

            if args.verbose:
                print("PSNR: ", psnr)
                print("LOSS: ", loss, loss_data, loss_reg)
            wandb.log(
                    {"optim/loss": loss, "step": i}
                        ) 
            wandb.log(
                    {"optim/loss_data": loss_data, "step": i}
                        ) 
            wandb.log(
                    {"optim/loss_reg": loss_reg, "step": i}
                        ) 
            wandb.log(
                    {"optim/psnr": psnr, "step": i}
                        ) 

            wandb.log(
                    {"optim/mse_to_gt": mse_to_gt.item(), "step": i}
                        ) 

            if i % cfg_optim["img_log_freq"] == 0:

                fig,(ax1, ax2, ax3) = plt.subplots(1,3)
                fig.suptitle("time step t= " + str(step))
                im = ax1.imshow(last_x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax1)
                ax1.axis("off")

                im = ax2.imshow(forward(x)[0,0,:,:].detach().cpu().numpy().T)
                fig.colorbar(im, ax=ax2)
                ax2.axis("off")

                im = ax3.imshow(y_noise[0,0,:,:].detach().cpu().numpy().T)
                fig.colorbar(im, ax=ax3)
                ax3.axis("off")

                wandb.log({f"reconstruction": wandb.Image(plt)})

                plt.close()
                
       
        fig,(ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,8))

        im = ax1.imshow(last_x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax1)
        ax1.set_title("Reconstruction")
        ax1.axis("off")

        im = ax2.imshow(x_gt[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax2.set_title("ground truth")
        fig.colorbar(im, ax=ax2)
        ax2.axis("off")

        im = ax3.imshow(torch.abs(last_x - x_gt)[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax3)
        ax3.set_title("| x_rec - x_gt|")
        ax3.axis("off")

        wandb.log({f"final_reco": wandb.Image(plt)})

        tvu.save_image(x_gt, os.path.join(log_dir, f"gt.png"))
        tvu.save_image(last_x, os.path.join(log_dir, f"reco.png"))

        res_dict = {
            "PSNR": float(psnr_list[-1]),
            "SSIM": float(ssim_list[-1])
        }

        with open(os.path.join(log_dir, "results.yaml"), "w") as f:
            yaml.dump(res_dict, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)