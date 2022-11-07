import torch
import numpy as np
import scipy


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
    result = -torch.digamma(lmbda) + grad_lmbda_lower_incomplete_gamma(lmbda, beta)
    result *= torch.exp(-torch.lgamma(lmbda))
    return result

# def grad_lmbda_lower_incomplete_gamma(lmbda, beta):
#     # reference: Eq 25 for derivative of upper incomplete gamma in 
#     # http://www.iaeng.org/IJAM/issues_v47/issue_3/IJAM_47_3_04.pdf
#     acc = 0
#     for k in range(15):
#         term1 = torch.log(beta) / (lmbda + k)
#         term1 -= 1 / (lmbda + k)**2
        
#         term2 = torch.exp(
#             (k + lmbda) * torch.log(beta) - torch.lgamma(torch.tensor([k + 1], dtype=torch.float))
#         )
#         if k % 2 == 1:
#             term2 *= -1
#         acc += term1 * term2
#     return acc

def _integrand(t, lmbda, beta, n):
    return np.exp(-beta * t) * (beta * t)**(lmbda-1) * np.log(beta * t)**n

def _integrated1(lmbda, beta):
    return scipy.integrate.quad(_integrand, 0, 1, args=(lmbda, beta, 1))[0] * beta

_vec_integrated = np.vectorize(_integrated1)

def grad_lmbda_lower_incomplete_gamma(lmbda, beta):
    return torch.tensor(_vec_integrated(lmbda, beta))


# def grad2_lmbda_lower_incomplete_gamma(lmbda, beta):
#     # reference: Eq 25 for derivative of upper incomplete gamma in 
#     # http://www.iaeng.org/IJAM/issues_v47/issue_3/IJAM_47_3_04.pdf
#     acc = 0
#     logbeta = torch.log(beta)
#     for k in range(15):
#         x = lmbda + k
#         term1 = logbeta**2 / (2 * x) - logbeta / (x**2) + 1 / (x**3)

#         term2 = (k + lmbda) * torch.log(beta) - torch.lgamma(torch.tensor([k + 1], dtype=torch.float)) 
#         term2 = 2 * torch.exp(term2)
#         if k % 2 == 1:
#             term2 *= -1
#         acc += term1 * term2
#     return acc

def _integrated2(lmbda, beta):
    return scipy.integrate.quad(_integrand, 0, 1, args=(lmbda, beta, 2))[0] * beta

_vec_integrated2 = np.vectorize(_integrated2)

def grad2_lmbda_lower_incomplete_gamma(lmbda, beta):
    return torch.tensor(_vec_integrated2(lmbda, beta))


def grad2_lmbda_beta_lower_incomplete_gamma(lmbda, beta):
    logbeta = torch.log(beta)
    return torch.exp(-beta + lmbda * logbeta) * logbeta




def logZ_approx(k, h, n):
    lambdas = (h + 1) / (2 * k)
    rlct = np.min(lambdas)
    m = np.sum(lambdas == rlct)
    const_term = (
        scipy.special.loggamma(rlct) 
        - m * np.log(2)
    )

    if m == 2: 
        const_term -= np.sum(np.log(k))
    elif m == 1:
        i = np.argmin(lambdas)
        j = np.argmax(lambdas)
        const_term -= np.log(k[i])
        const_term -= np.log(lambdas[j] - lambdas[i])

    leading_terms = -rlct * np.log(n) + (m -1) * np.log(np.log(n)) + const_term
    return leading_terms

igamma = RegularisedLowerIncompleteGamma.apply
gradigamma = GradLambdaRegularisedLowerIncompleteGamma.apply # lambda x, y: torch.digamma(x) #
def elbo_func_mf_gamma_trunc(lambdas, ks, betas, lambda_0, k_0, n):
    r = k_0 / ks
    iglambdas_betas = igamma(lambdas, betas)
    logbetas = torch.log(betas)
    term1 = n * torch.exp(torch.sum(
        -r * logbetas
        + torch.lgamma(lambdas + r) - torch.lgamma(lambdas)
        + torch.log(igamma(lambdas + r, betas)) - torch.log(iglambdas_betas)
    ))

    term2 = torch.sum(
        torch.log(2 * ks) + lambdas * logbetas 
        - torch.lgamma(lambdas) - torch.log(iglambdas_betas)
        - lambdas * (igamma(lambdas + r, betas) / iglambdas_betas)
        + (lambdas - r * lambda_0) * (gradigamma(lambdas, betas) / iglambdas_betas - logbetas)
    )
    return -term1 - term2


def elbo_func_mf_gamma(lambdas, ks, betas, lambda_0, k_0, n):
    r = k_0 / ks
    logbetas = torch.log(betas)    
    term1 = n * torch.exp(torch.sum(
        -r * logbetas 
        +  torch.lgamma(lambdas + r) - torch.lgamma(lambdas)
    ))
    
    term2 = torch.sum(
        torch.log(2 * ks) + lambdas * logbetas 
        - torch.lgamma(lambdas) - lambdas 
        + (lambdas - r * lambda_0) * (torch.digamma(lambdas) - logbetas)
    )
    return -term1 - term2


def standard_form_unnormlised_density(w, k, h, n):
    return torch.abs(torch.prod(w**h, dim=-1)) * torch.exp(-n * torch.prod(w ** (2 * k), dim=-1))