from re import X
from symbol import parameters
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

import matplotlib.animation as animation
import torch.distributions as tdist




class VariationalInference(object):
    
    def __init__(self, parameters, true_parameters, optim):
        self.parameters = parameters
        self.true_parameters = true_parameters
        self.optim = optim
        
        self.snapshots = []
        self.optim_rec = []
        self.max_elbo = -torch.inf
        self.best_epoch = None
        self.best_params = None

        self.elbo = None
        self.epoch = 0
        return 
    
    def run_parameter_optimisation(self, num_epoch, num_snapshots=100, num_records=100, num_report=10, verbose=True):
        for t in range(num_epoch):
            self.epoch += 1
            
            self.optim.zero_grad()
            self.elbo = self.elbo_fn()
            neg_elbo = -self.elbo
            neg_elbo.backward()
            self.optim.step()

            if self.elbo > self.max_elbo: 
                self.best_epoch = self.epoch
                self.max_elbo = self.elbo
                self.best_params = [param.clone().detach() for param in self.parameters]
                
            if t % (num_epoch // num_snapshots) == 0:
                snapshot = [self.epoch, [param.clone().detach() for param in self.parameters], self.elbo]
                self.snapshots.append(snapshot)

            if t % (num_epoch // num_records) == 0:
                self.optim_rec.append((self.epoch, self.elbo.detach()))

            if verbose and (t % (num_epoch // num_report)) == 0:
                self.report_optim_step()
        return

    def elbo_fn(self):
        pass
    
    def report_optim_step(self):
        pass

    def true_unnormalised_density(self, w):
        pass

    def variational_density(self, w, parameters):
        pass
        
    def plot_variational_posterior(self, N=50, levels=50, lower_lim=0, upper_lim=1, animate=False, interval=200):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        xx, yy = torch.meshgrid(torch.linspace(lower_lim, upper_lim, N), torch.linspace(lower_lim, upper_lim, N))
        w = torch.stack((xx, yy), dim=-1)

        ax = axes[0]
        z = self.true_unnormalised_density(w)
        ax.contourf(xx, yy, z, levels=levels)
        ax.contour(xx, yy, z, levels=levels, colors='k', alpha=0.5)
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")
        ax.set_title("True distribution")
        
        ax = axes[1]
        z = self.variational_density(w, self.best_params)
        contourf = ax.contourf(xx, yy, z, levels=levels)
        contour = ax.contour(xx, yy, z, levels=levels, colors='k', alpha=0.5)
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")
        ax.set_title(f"Best variational approximation\nEpoch: {self.best_epoch}, ELBO:{self.max_elbo:.3f}")

        ax = axes[2]
        epoch, parameters, elbo = self.snapshots[-1]
        z = self.variational_density(w, parameters)
        contourf = ax.contourf(xx, yy, z, levels=levels)
        contour = ax.contour(xx, yy, z, levels=levels, colors='k', alpha=0.5)
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")

        if animate:
            def animation_frame(frame_num):
                nonlocal ax, w, contour, contourf
                epoch, parameters, elbo = self.snapshots[frame_num]
                z = self.variational_density(w, parameters)
                
                for col in contourf.collections:
                    col.remove()
                contourf = ax.contourf(xx, yy, z, levels=levels)
                ax.set_title(
                    f"epoch:{epoch}, elbo:{elbo:.2f}\n"
                    f"{[list(np.around(param.clone().tolist(), 2)) for param in parameters]}"
                )
                
                for col in contour.collections:
                    col.remove()
                contour = ax.contour(xx, yy, z, levels=levels, colors='k', alpha=0.5)
                return contour, contourf

            anim = animation.FuncAnimation(fig, animation_frame, frames=len(self.snapshots), interval=interval)
            video = anim.to_html5_video()
        else:
            video = None
        
        return fig, video
    
    def plot_training_curve(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        x = np.array(self.optim_rec)
        ax.plot(x[:, 0], x[:, 1])
        ax.set_title("Training Curve")
        ax.set_ylabel("ELBO")
        ax.set_xlabel("Epoch")
        return fig