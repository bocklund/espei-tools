# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform, triang
from espei.utils import rv_zero



def get_prior(p, prior):
        if prior == 'normal':
            # TODO: control scale hyperparameter better. Here we take the chain_std_deviation*5 as the prior std_deviation
            rv_instance = norm(loc=p, scale=np.abs(0.1*p*5))
        elif prior == 'halfnormal':
            # TODO: control scale hyperparameter better. Here we take the chain_std_deviation*5 as the prior std_deviation
            rv_instance = norm(loc=p, scale=np.abs(0.1*p*5/2.0))
        elif prior == 'uniform':
            # TODO: control scale hyperparameter manually here, later we can update
            distance_frac = 1.0
            # hyperparameter is the distance to low, e.g.
            # hyperparameter=3.0 :  low:=p-(3.0*p), high:=p+(3.0*p)
            diff = abs(distance_frac*p)
            rv_instance = uniform(loc=p-diff, scale=2*diff)
        elif prior == 'triangular':
            distance_frac = 1.0
            # hyperparameter is the distance to low, e.g.
            # hyperparameter=3.0 :  low:=p-(3.0*p), high:=p+(3.0*p)
            diff = abs(distance_frac*p)
            # the maximum (controlled by c) is always in the center here
            rv_instance = triang(loc=p-diff, scale=2*diff, c=0.5)
        elif prior == 'halftriangular':
            distance_frac = 0.5 
            # hyperparameter is the distance to low, e.g.
            # hyperparameter=3.0 :  low:=p-(3.0*p), high:=p+(3.0*p)
            diff = abs(distance_frac*p)
            # the maximum (controlled by c) is always in the center here
            rv_instance = triang(loc=p-diff, scale=2*diff, c=0.5)
        elif prior == 'zero':
            rv_instance = rv_zero()
        else:
            raise ValueError('Invalid prior ({}) specified. Specify one of "normal", "uniform", "triangular", "zero"'.format(prior))
        return rv_instance
def plot_prior(param_value):
    x_grid = np.linspace(param_value-abs(param_value)*1.5, param_value+abs(param_value)*1.5, 10000)
    plt.plot(x_grid, norm(scale=abs(param_value/10), loc=param_value).pdf(x_grid), label='initialization')
    for prior in ('normal', 'halfnormal', 'uniform', 'triangular', 'halftriangular'):
        plt.plot(x_grid, get_prior(param_value, prior).pdf(x_grid), label=prior+' prior')
    plt.title('Initial value: {}'.format(param_value))
    plt.ylabel('probability density')
    plt.legend()
    plt.show()
    
def plot_log_prior(param_value):
    x_grid = np.linspace(param_value-abs(param_value)*1.5, param_value+abs(param_value)*1.5, 10000)
    # span the region of 99% of the initial values
    iv = norm(scale=abs(param_value/10), loc=param_value).interval(0.99)
    plt.axvspan(iv[0], iv[1], alpha=0.3, label='99% of initial values')
    for prior in ('normal', 'halfnormal', 'uniform', 'triangular', 'halftriangular'):
        plt.plot(x_grid, get_prior(param_value, prior).logpdf(x_grid), label=prior+' prior')
    plt.title('Initial value: {}'.format(param_value))
    plt.ylabel('log probability density')
    plt.legend()
    plt.show()
    
def get_prior_same_oom(p, prior):
        if prior == 'normal':
            # TODO: control scale hyperparameter better. Here we take the chain_std_deviation*5 as the prior std_deviation
            rv_instance = norm(loc=p, scale=np.abs(p/3))
        elif prior == 'uniform':
            # TODO: control scale hyperparameter manually here, later we can update
            pair = (p*0.1, p*10)
            rv_instance = uniform(loc=min(pair), scale=max(pair))
        elif prior == 'triangular':
            pair = (p*0.1, p*10)
            # the maximum (controlled by c) is always in the center here
            rv_instance = triang(loc=min(pair), scale=max(pair), c=0.5)
        elif prior == 'zero':
            rv_instance = rv_zero()
        else:
            raise ValueError('Invalid prior ({}) specified. Specify one of "normal", "uniform", "triangular", "zero"'.format(prior))
        return rv_instance
def plot_prior_same_oom(param_value):
    x_grid = np.linspace(param_value-abs(param_value)*1.5, param_value+abs(param_value)*1.5, 10000)
    plt.plot(x_grid, norm(scale=abs(param_value/10), loc=param_value).pdf(x_grid), label='initialization')
    for prior in ('normal', 'uniform', 'triangular'):
        plt.plot(x_grid, get_prior_same_oom(param_value, prior).pdf(x_grid), label=prior+' prior')
    plt.title('Initial value: {}'.format(param_value))
    plt.ylabel('probability density')
    plt.legend()
    plt.show()
    
def plot_log_prior_same_oom(param_value):
    x_grid = np.linspace(param_value-abs(param_value)*1.5, param_value+abs(param_value)*1.5, 10000)
    # span the region of 99% of the initial values
    iv = norm(scale=abs(param_value/10), loc=param_value).interval(0.99)
    plt.axvspan(iv[0], iv[1], alpha=0.3, label='99% of initial values')
    for prior in ('normal', 'uniform', 'triangular'):
        plt.plot(x_grid, get_prior_same_oom(param_value, prior).logpdf(x_grid), label=prior+' prior')
    plt.title('Initial value: {}'.format(param_value))
    plt.ylabel('log probability density')
    plt.legend()
    plt.show()
   
if __name__ == '__main__':
    plot_prior(10000)
    plot_log_prior(10000)
