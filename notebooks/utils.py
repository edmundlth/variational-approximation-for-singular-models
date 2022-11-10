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
        grad_lmbda = grad2_lmbda_igamma(lmbda, beta)
        grad_beta = grad_beta_lmbda_igamma(lmbda, beta)
        return grad_output * grad_lmbda, grad_output * grad_beta

def grad_beta_lmbda_igamma(lmbda, beta):
    return grad_beta_igamma(lmbda, beta) * (torch.log(beta) - torch.digamma(lmbda))

def grad_beta_igamma(lmbda, beta):
    return torch.exp(
        -beta + (lmbda - 1) * torch.log(beta) - torch.lgamma(lmbda)
    )

D1_ORD6_STENCIL = torch.tensor([-1.0/60, 3.0/20, -3.0/4, 0.0, 3.0/4, -3.0/20, 1.0/60])
D2_ORD6_STENCIL = torch.tensor([1.0/90, -3.0/20, 3.0/2, -49.0/18, 3.0/2, -3.0/20, 1.0/90])
EPSILON = 1e-3
DELTAS = torch.arange(-3, 4, step=1, dtype=torch.float) * EPSILON
def grad_lmbda_igamma(lmbda, beta):
    xs = DELTAS.repeat((lmbda.shape[0], 1)).T + lmbda
    return torch.sum(torch.igamma(xs, beta).T * (D1_ORD6_STENCIL / EPSILON), axis=1)

# def grad_lmbda_igamma(lmbda, beta): 
#     result = -torch.digamma(lmbda) + grad_lmbda_lower_incomplete_gamma(lmbda, beta)
#     result *= torch.exp(-torch.lgamma(lmbda))
#     return result

def grad2_lmbda_igamma(lmbda, beta):
    xs = DELTAS.repeat((lmbda.shape[0], 1)).T + lmbda
    return torch.sum(torch.igamma(xs, beta).T * (D2_ORD6_STENCIL / EPSILON**2), axis=1)

# def grad2_lmbda_igamma(lmbda, beta):
#     digamma_lmbda = torch.digamma(lmbda)
#     loggamma_lmbda = torch.lgamma(lmbda)

#     term1 = -torch.polygamma(1, lmbda) * torch.igamma(lmbda, beta)
#     term2 = -digamma_lmbda * grad_lmbda_igamma(lmbda, beta)
#     term3 = torch.exp(torch.log(grad2_lmbda_lower_incomplete_gamma(lmbda, beta)) - loggamma_lmbda)
#     term4 = -torch.exp(torch.log(digamma_lmbda) - loggamma_lmbda + torch.log(grad_lmbda_lower_incomplete_gamma(lmbda, beta)))
#     return term1 + term2 + term3 + term4 


def grad_lmbda_lower_incomplete_gamma(lmbda, beta):
    mask = beta < lmbda + 20
    result = torch.zeros_like(lmbda)
    result[~mask] = torch.digamma(lmbda[~mask]) * torch.exp(torch.lgamma(lmbda[~mask]))
    if torch.all(mask):
        return result
    # reference: Eq 25 for derivative of upper incomplete gamma in 
    # http://www.iaeng.org/IJAM/issues_v47/issue_3/IJAM_47_3_04.pdf
    l = lmbda[mask]
    b = beta[mask]
    # acc = 0
    for k in range(15):
        term1 = torch.log(b) / (l + k)
        term1 -= 1 / (l + k)**2
        
        term2 = torch.exp(
            (k + l) * torch.log(b) - torch.lgamma(torch.tensor([k + 1], dtype=torch.float))
        )
        if k % 2 == 1:
            term2 *= -1
        result[mask] += term1 * term2
    return result

# def _integrand(t, lmbda, beta, n):
#     return np.exp(-beta * t) * (beta * t)**(lmbda-1) * np.log(beta * t)**n

# def _integrated1(lmbda, beta):
#     return scipy.integrate.quad(_integrand, 0, 1, args=(lmbda, beta, 1))[0] * beta

# _vec_integrated = np.vectorize(_integrated1)

# def grad_lmbda_lower_incomplete_gamma(lmbda, beta):
#     return torch.tensor(_vec_integrated(lmbda, beta))


def grad2_lmbda_lower_incomplete_gamma(lmbda, beta):
    mask = beta < lmbda + 20
    result = torch.zeros_like(lmbda)
    result[~mask] = (torch.polygamma(1, lmbda[~mask]) + torch.digamma(lmbda[~mask])) * torch.exp(torch.lgamma(lmbda[~mask]))
    if torch.all(mask):
        return result
    
    # reference: Eq 25 for derivative of upper incomplete gamma in 
    # http://www.iaeng.org/IJAM/issues_v47/issue_3/IJAM_47_3_04.pdf
    # acc = 0
    logbeta = torch.log(beta[mask])
    l = lmbda[mask]
    for k in range(100):
        x = l + k
        term1 = logbeta**2 / (2 * x) - logbeta / (x**2) + 1 / (x**3)

        term2 = (k + l) * logbeta - torch.lgamma(torch.tensor([k + 1], dtype=torch.float)) 
        term2 = 2 * torch.exp(term2)
        if k % 2 == 1:
            term2 *= -1
        result[mask] += term1 * term2
    return result

# def _integrated2(lmbda, beta):
#     return scipy.integrate.quad(_integrand, 0, 1, args=(lmbda, beta, 2))[0] * beta

# _vec_integrated2 = np.vectorize(_integrated2)

# def grad2_lmbda_lower_incomplete_gamma(lmbda, beta):
#     return torch.tensor(_vec_integrated2(lmbda, beta))


def grad2_lmbda_beta_lower_incomplete_gamma(lmbda, beta):
    logbeta = torch.log(beta)
    return torch.exp(-beta + lmbda * logbeta) * logbeta




def logZ_approx(k, h, n):
    lambdas = (h + 1) / (2 * k)
    rlct = np.min(lambdas)
    m = np.sum(lambdas == rlct)
    const_term = (
        scipy.special.loggamma(rlct) 
        - np.sum(np.log(2 * k))
        - scipy.special.loggamma(m)
    )
    index = np.argsort(lambdas)[m:]
    const_term -= np.sum(np.log(2 * k[index]) + np.log(lambdas[index] - rlct))
    leading_terms = -rlct * np.log(n) + (m -1) * np.log(np.log(n)) + const_term
    return leading_terms

igamma = RegularisedLowerIncompleteGamma.apply # torch.igamma
gradigamma = GradLambdaRegularisedLowerIncompleteGamma.apply # lambda x, y: torch.digamma(x) #
def elbo_func_mf_gamma_trunc(lambdas, ks, betas, lambda_0, k_0, n, ignore_term=False):
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
    )
    # this term is the problematic term that involves derivatives of incomplete gamma function. 
    # when lambdas and ks matches the true parameters, this should be zero, 
    # but even there, it would generically has non-zero gradient. 
    if not ignore_term:
        term2 += (lambdas - r * lambda_0) * (gradigamma(lambdas, betas) / iglambdas_betas - logbetas)
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