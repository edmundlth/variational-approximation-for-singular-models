import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

import matplotlib.animation as animation
import torch.distributions as tdist
from utils import (
    RegularisedLowerIncompleteGamma,
    GradLambdaRegularisedLowerIncompleteGamma,
    elbo_func_mf_gamma_trunc,
    elbo_func_mf_gamma,
    standard_form_unnormlised_density,
)

igamma = RegularisedLowerIncompleteGamma.apply
gradigamma = GradLambdaRegularisedLowerIncompleteGamma.apply


class VariationalInference(object):
    def __init__(self, parameters, true_parameters, optim):
        self.parameters = parameters
        self.true_parameters = true_parameters
        self.optim = optim

        self.max_elbo = -torch.inf
        self.best_epoch = None
        self.best_params = None

        self.elbo = self.elbo_fn()
        self.epoch = 0

        self.snapshots = [
            [self.epoch, [param.clone().detach() for param in self.parameters], self.elbo]
        ]
        self.optim_rec = []
        
        return

    def run_parameter_optimisation(
        self, num_epoch, num_snapshots=100, num_records=100, num_report=10, verbose=True
    ):
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
                snapshot = [
                    self.epoch,
                    [param.clone().detach() for param in self.parameters],
                    self.elbo,
                ]
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
        integrand = lambda y, x: self.true_unnormalised_density(
            torch.tensor([x, y])
        ).numpy()
        evidence = scipy.integrate.dblquad(integrand, 0, 1, 0, 1)
        return np.log(evidence[0])

    def plot_variational_posterior(
        self, N=50, levels=50, lower_lim=0, upper_lim=1, animate=False, interval=200
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        xx, yy = torch.meshgrid(
            torch.linspace(lower_lim, upper_lim, N),
            torch.linspace(lower_lim, upper_lim, N),
            indexing="ij",
        )
        w = torch.stack((xx, yy), dim=-1)

        ax = axes[0]
        z = self.true_unnormalised_density(w)
        ax.contourf(xx, yy, z, levels=levels)
        ax.contour(xx, yy, z, levels=levels, colors="k", alpha=0.5)
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")
        ax.set_title(f"True distribution\nlogZ={self.log_evidence():.3f}")

        ax = axes[1]
        _, parameters, elbo = self.snapshots[0]
        z = self.variational_density(w, parameters)
        contourf = ax.contourf(xx, yy, z, levels=levels)
        contour = ax.contour(xx, yy, z, levels=levels, colors="k", alpha=0.5)
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")
        ax.set_title(
            f"Initial variational approximation\n ELBO:{elbo:.3f}"
        )

        ax = axes[2]
        _, parameters, elbo = self.snapshots[-1]
        z = self.variational_density(w, parameters)
        contourf = ax.contourf(xx, yy, z, levels=levels)
        contour = ax.contour(xx, yy, z, levels=levels, colors="k", alpha=0.5)
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
                contour = ax.contour(xx, yy, z, levels=levels, colors="k", alpha=0.5)
                return contour, contourf

            anim = animation.FuncAnimation(
                fig, animation_frame, frames=len(self.snapshots), interval=interval
            )
        else:
            anim = None

        return fig, anim

    def plot_training_curve(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        x = np.array(self.optim_rec)
        ax.plot(x[:, 0], x[:, 1])
        ax.set_title("Training Curve")
        ax.set_ylabel("ELBO")
        ax.set_xlabel("Epoch")
        return fig


############################
# Child classes
############################


class GaussianFamilyOn2DGaussianPosterior(VariationalInference):
    def __init__(self, mu_0, sigma_0, lr=0.001, base_samples=100, init_mu=None, init_logsigma=None):
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.true_parameters = [mu_0, sigma_0]

        if init_mu is not None:
            self.init_mu = init_mu
        else:
            self.init_mu = torch.rand_like(self.mu_0, dtype=torch.float)

        if init_logsigma is not None:
            self.init_logsigma = init_logsigma
        else:
            self.init_logsigma = torch.log(torch.rand_like(self.mu_0, dtype=torch.float) * 2)
        self.mu = nn.Parameter(self.init_mu, requires_grad=True)
        self.logsigma = nn.Parameter(self.init_logsigma, requires_grad=True)
        
        self.parameters = [self.mu, self.logsigma]

        self.optim = torch.optim.Adam(self.parameters, lr=lr)
        self.base_samples = base_samples
        super(GaussianFamilyOn2DGaussianPosterior, self).__init__(
            self.parameters, self.true_parameters, self.optim
        )

    def elbo_fn(self):
        q = tdist.MultivariateNormal(
            loc=self.mu, covariance_matrix=torch.eye(2) * torch.exp(2 * self.logsigma)
        )
        xi = q.rsample((self.base_samples,))
        term1 = q.log_prob(xi)
        term2 = torch.sum((xi - self.mu_0) ** 2/ (2 * self.sigma_0**2), dim=1) 
        return -torch.mean(term1 + term2)

    def report_optim_step(self):
        print(
            f"Epoch {self.epoch:5d}: mu={np.around(self.mu.detach().tolist(), 2)}, "
            f"sigma={torch.exp(self.logsigma).tolist()}, elbo={self.elbo:.2f}"
        )

    def true_unnormalised_density(self, w):
        q = tdist.MultivariateNormal(
            loc=self.mu_0, covariance_matrix=torch.eye(2) * self.sigma_0
        )
        z = np.exp(q.log_prob(w))
        return z

    def variational_density(self, w, parameters):
        mu, logsigma = parameters
        q = tdist.MultivariateNormal(
            loc=mu, covariance_matrix=torch.eye(2) * torch.exp(2 * logsigma)
        )
        z = np.exp(q.log_prob(w))
        return z


class GaussianFamilyOn2DStandardForm(VariationalInference):
    def __init__(self, n, k_0, h_0, lr=0.001, base_samples=100, init_mu=None, init_logsigma=None):
        self.n = n
        self.k_0 = k_0
        self.h_0 = h_0
        self.true_parameters = [k_0, h_0]

        if init_mu is not None:
            self.init_mu = init_mu
        else:
            self.init_mu = torch.rand_like(self.k_0, dtype=torch.float)

        if init_logsigma is not None:
            self.init_logsigma = init_logsigma
        else:
            self.init_logsigma = torch.log(torch.rand_like(self.k_0, dtype=torch.float) * 2)
        self.mu = nn.Parameter(self.init_mu, requires_grad=True)
        self.logsigma = nn.Parameter(self.init_logsigma, requires_grad=True)
        self.parameters = [self.mu, self.logsigma]

        self.optim = torch.optim.Adam(self.parameters, lr=lr)
        self.base_samples = base_samples
        super(GaussianFamilyOn2DStandardForm, self).__init__(
            self.parameters, self.true_parameters, self.optim
        )

    def elbo_fn(self):
        q = tdist.MultivariateNormal(
            loc=self.mu, covariance_matrix=torch.eye(2) * torch.exp(2 * self.logsigma)
        )
        xi = q.rsample((self.base_samples,))
        term1 = q.log_prob(xi)
        term2 = torch.sum(torch.log(torch.abs(xi)) * self.h_0, dim=1)
        term3 = self.n * torch.prod(xi ** (2 * self.k_0), dim=1)
        elbo = -torch.mean(term1 - term2 + term3)
        return elbo

    def report_optim_step(self):
        print(
            f"Epoch {self.epoch:6d}: mu={np.around(self.mu.detach().tolist(), 2)}, "
            f"sigma={torch.exp(self.logsigma).tolist()}, elbo={self.elbo:.2f}"
        )

    def true_unnormalised_density(self, w):
        return standard_form_unnormlised_density(w, self.k_0, self.h_0, self.n)

    def variational_density(self, w, parameters):
        mu, logsigma = parameters
        q = tdist.MultivariateNormal(
            loc=mu, covariance_matrix=torch.eye(2) * torch.exp(2 * logsigma)
        )
        z = np.exp(q.log_prob(w))
        return z


class MeanFieldGammaOn2DStandardForm(VariationalInference):
    def __init__(
        self,
        n,
        k_0,
        lambda_0,
        lr=0.001,
        lambdas_grad=True,
        ks_grad=True,
        beta1_grad=False,
    ):

        self.n = n
        self.k_0 = k_0
        self.lambda_0 = lambda_0
        # making sure that the first lambda_0 is the RLCT
        # this makes it possible to freeze beta1=n is required.
        assert torch.min(self.lambda_0) == self.lambda_0[0]
        self.h_0 = 2 * k_0 * lambda_0 - 1
        self.true_parameters = [self.lambda_0, self.k_0, self.n]

        self.loglambdas = nn.Parameter(torch.log(lambda_0), requires_grad=lambdas_grad)
        self.logks = nn.Parameter(torch.log(k_0), requires_grad=ks_grad)
        self.logbeta1 = nn.Parameter(
            torch.log(torch.tensor([n])), requires_grad=beta1_grad
        )
        self.logbetas_rest = nn.Parameter(
            torch.log(torch.tensor([1.0])), requires_grad=True
        )
        self.parameters = [
            self.loglambdas,
            self.logks,
            self.logbeta1,
            self.logbetas_rest,
        ]

        self.optim = torch.optim.Adam(self.parameters, lr=lr)
        super(MeanFieldGammaOn2DStandardForm, self).__init__(
            self.parameters, self.true_parameters, self.optim
        )

    def elbo_fn(self):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in self.parameters]
        betas = torch.concat([beta1, betas_rest])
        return elbo_func_mf_gamma(lambdas, ks, betas, self.lambda_0, self.k_0, self.n)

    def report_optim_step(self):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in self.parameters]
        betas = torch.concat([beta1, betas_rest])
        print(
            f"Epoch {self.epoch:5d}: elbo={self.elbo:.2f}, "
            f"lambdas={list(np.around(lambdas.tolist(), 2))}, "
            f"ks={list(np.around(ks.tolist(), 2))}, "
            f"betas={list(np.around(betas.tolist(),2))}"
        )

    def true_unnormalised_density(self, w):
        return standard_form_unnormlised_density(w, self.k_0, self.h_0, self.n)

    def variational_density(self, w, parameters):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in parameters]
        betas = torch.concat([beta1, betas_rest])
        z = w ** (2 * ks * lambdas - 1) * torch.exp(-betas * (w**ks))
        normalising_const = scipy.special.gamma(lambdas) / (
            2 * ks * (betas ** (lambdas))
        )
        z = torch.prod(z / normalising_const, axis=-1)
        return z

class MeanFieldGammaOn2DStandardForm(VariationalInference):
    def __init__(
        self,
        n,
        k_0,
        lambda_0,
        lr=0.001,
        lambdas_grad=True,
        ks_grad=True,
        beta1_grad=False,
        init_params=None
    ):

        self.n = n
        self.k_0 = k_0
        self.lambda_0 = lambda_0
        # making sure that the first lambda_0 is the RLCT
        # this makes it possible to freeze beta1=n is required.
        assert torch.min(self.lambda_0) == self.lambda_0[0]
        self.h_0 = 2 * k_0 * lambda_0 - 1
        self.true_parameters = [self.lambda_0, self.k_0, self.n]
        
        if init_params is not None:
            self.init_lambdas, self.init_ks, self.init_betas = init_params
        else:
            self.init_lambdas = self.lambda_0
            self.init_ks = self.k_0
            self.init_betas = torch.tensor([n, 1.0])

        self.loglambdas = nn.Parameter(torch.log(self.init_lambdas), requires_grad=lambdas_grad)
        self.logks = nn.Parameter(torch.log(self.init_ks), requires_grad=ks_grad)
        self.logbeta1 = nn.Parameter(torch.log(self.init_betas[0:1]), requires_grad=beta1_grad)
        self.logbetas_rest = nn.Parameter(torch.log(self.init_betas[1:]), requires_grad=True)
        self.parameters = [
            self.loglambdas,
            self.logks,
            self.logbeta1,
            self.logbetas_rest,
        ]

        self.optim = torch.optim.Adam(self.parameters, lr=lr)
        super(MeanFieldGammaOn2DStandardForm, self).__init__(
            self.parameters, self.true_parameters, self.optim
        )

    def elbo_fn(self):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in self.parameters]
        betas = torch.concat([beta1, betas_rest])
        return elbo_func_mf_gamma(lambdas, ks, betas, self.lambda_0, self.k_0, self.n)

    def report_optim_step(self):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in self.parameters]
        betas = torch.concat([beta1, betas_rest])
        print(
            f"Epoch {self.epoch:5d}: elbo={self.elbo:.2f}, "
            f"lambdas={list(np.around(lambdas.tolist(), 2))}, "
            f"ks={list(np.around(ks.tolist(), 2))}, "
            f"betas={list(np.around(betas.tolist(),2))}"
        )

    def true_unnormalised_density(self, w):
        return standard_form_unnormlised_density(w, self.k_0, self.h_0, self.n)

    def variational_density(self, w, parameters):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in parameters]
        betas = torch.concat([beta1, betas_rest])
        z = w ** (2 * ks * lambdas - 1) * torch.exp(-betas * (w**ks))
        normalising_const = scipy.special.gamma(lambdas) / (
            2 * ks * (betas ** (lambdas))
        )
        z = torch.prod(z / normalising_const, axis=-1)
        return z



class MeanFieldTruncatedGammaOn2DStandardForm(MeanFieldGammaOn2DStandardForm):
    def __init__(
        self,
        n,
        k_0,
        lambda_0,
        lr=0.001,
        lambdas_grad=True,
        ks_grad=True,
        beta1_grad=False,
        init_params=None
    ):
        super(MeanFieldTruncatedGammaOn2DStandardForm, self).__init__(
            n,
            k_0,
            lambda_0,
            lr=lr,
            lambdas_grad=lambdas_grad,
            ks_grad=ks_grad,
            beta1_grad=beta1_grad,
            init_params=init_params
        )

    def elbo_fn(self):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in self.parameters]
        betas = torch.concat([beta1, betas_rest])
        return elbo_func_mf_gamma_trunc(
            lambdas, ks, betas, self.lambda_0, self.k_0, self.n
        )

    def variational_density(self, w, parameters):
        lambdas, ks, beta1, betas_rest = [torch.exp(param) for param in parameters]
        betas = torch.concat([beta1, betas_rest])
        z = w ** (2 * ks * lambdas - 1) * torch.exp(-betas * (w**ks))
        normalising_const = scipy.special.gamma(lambdas) / (
            2 * ks * (betas ** (lambdas))
        )
        z = torch.prod(z / normalising_const, axis=-1)
        return z

