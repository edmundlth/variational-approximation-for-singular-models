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
            
            self.epoch += 1
        return

    def elbo_fn(self):
        pass
    
    def report_optim_step(self):
        pass

    def true_unnormalised_density(self, w):
        pass

    def variational_density(self, w, parameters):
        pass
    
    def log_evidence(self):
        integrand = lambda y, x: self.true_unnormalised_density(torch.tensor([x, y])).numpy()
        evidence = scipy.integrate.dblquad(integrand, 0, 1, 0, 1)
        return np.log(evidence[0])

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
        ax.set_title(f"True distribution\nlogZ={self.log_evidence():.3f}")
        
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

class RegularisedLowerIncompleteGamma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lmbda, beta):
        ctx.save_for_backward(lmbda, beta)
        return torch.igamma(lmbda, beta)
    
    @staticmethod
    def backward(ctx, grad_output):
        lmbda, beta = ctx.saved_tensors
        grad_lmbda = grad_lmbda_igamma(lmbda, beta)
        grad_beta = grad_beta_igamma(lmbda, beta)
        return grad_output * grad_lmbda, grad_output * grad_beta

class GradLambdaRegularisedLowerIncompleteGamma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lmbda, beta):
        ctx.save_for_backward(lmbda, beta)
        return grad_lmbda_igamma(lmbda, beta)
    
    @staticmethod
    def backward(ctx, grad_output):
        lmbda, beta = ctx.saved_tensors
        digamma_lmbda = torch.digamma(lmbda)
        loggamma_lmbda = torch.lgamma(lmbda)

        term1 = -torch.polygamma(1, lmbda) * torch.igamma(lmbda, beta)
        term2 = -digamma_lmbda * grad_lmbda_igamma(lmbda, beta)
        term3 = torch.exp(torch.log(grad2_lmbda_lower_incomplete_gamma(lmbda, beta)) - loggamma_lmbda)
        term4 = -torch.exp(torch.log(digamma_lmbda) - loggamma_lmbda + torch.log(grad_lmbda_lower_incomplete_gamma(lmbda, beta)))
        grad_lmbda = term1 + term2 + term3 + term4 
        grad_beta = (
            -digamma_lmbda * grad_beta_igamma(lmbda, beta) 
            + torch.exp(torch.log(grad2_lmbda_beta_lower_incomplete_gamma(lmbda, beta)) - loggamma_lmbda)
        )
        return grad_output * grad_lmbda, grad_output * grad_beta


def grad_beta_igamma(lmbda, beta):
    return torch.exp(-beta + (lmbda + 1) * torch.log(beta) - torch.lgamma(lmbda))

def grad_lmbda_igamma(lmbda, beta): 

    term = grad_lmbda_lower_incomplete_gamma(lmbda, beta)
    grad_lmbda = -torch.digamma(lmbda) * torch.igamma(lmbda, beta) + torch.exp(- torch.lgamma(lmbda)) * term
    return grad_lmbda


def grad_lmbda_lower_incomplete_gamma(lmbda, beta):
    # reference: Eq 25 for derivative of upper incomplete gamma in 
    # http://www.iaeng.org/IJAM/issues_v47/issue_3/IJAM_47_3_04.pdf
    acc = 0
    for k in range(15):
        term1 = torch.log(beta) - (1 / (lmbda - k))
        term2 = (k + lmbda) * torch.log(beta) - torch.lgamma(torch.tensor([k + 1], dtype=torch.float)) - torch.log(lmbda + k) 
        term2 = torch.exp(term2)
        if k % 2 == 1:
            term2 *= -1
        acc += term1 * term2
    return acc

def grad2_lmbda_lower_incomplete_gamma(lmbda, beta):
    # reference: Eq 25 for derivative of upper incomplete gamma in 
    # http://www.iaeng.org/IJAM/issues_v47/issue_3/IJAM_47_3_04.pdf
    acc = 0
    logbeta = torch.log(beta)
    for k in range(15):
        x = lmbda + k
        term1 = logbeta**2 / (2 * x) - logbeta / (x**2) + 1 / (x**3)

        term2 = (k + lmbda) * torch.log(beta) - torch.lgamma(torch.tensor([k + 1], dtype=torch.float)) 
        term2 = 2 * torch.exp(term2)
        if k % 2 == 1:
            term2 *= -1
        acc += term1 * term2
    return acc

def grad2_lmbda_beta_lower_incomplete_gamma(lmbda, beta):
    logbeta = torch.log(beta)
    return torch.exp(-beta + lmbda * logbeta) * logbeta