#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from stack import *
from scipy.optimize import minimize
from jax import vmap
import numpy

# from jax.config import config
# config.update("jax_enable_x64", True)


def trans_stack(epsilon, thickness, config=None, wl=None,eps_fun=None):
    t = []

    def _t(pars):
        lambda0, eps = pars
        # for lambda0, eps in zip(wl, epsilon):
        config["layers"]["unknown"]["epsilon"] = eps
        config["layers"]["unknown"]["thickness"] = thickness
        config["wave"]["lambda0"] = lambda0
        
        if eps_fun is not None:
            a,b = eps_fun
            config["layers"]["layer subs"]["epsilon"] = np.interp(lambda0, a,b)
            # config["layers"]["layer subs"]["epsilon"] = eps_fun(lambda0)
        # 
        # layers, wave = config["layers"], config["wave"]
        # eps = [d["epsilon"] for d in layers.values()]
        # eps = [e if not callable(e) else e(lambda0) for e in eps]
        # for l,e in zip(config["layers"],eps):
        #     config["layers"][l]["epsilon"] =e
            
        
        
        t, gamma = get_coeffs_stack(config)
        return t
        # t.append(out)

    pars = wl, epsilon
    t = vmap(_t)(pars)
    return np.array(t)


@jit
def fmini(x, config=None, t_exp=None, weights=(1, 1), wl=None,eps_fun=None, stats=False):
    # epsilon = (x[0] + 1j * x[1])*ones
    nl = len(wl)
    thickness = x[-1]
    x_ = x[:-1]
    epsilon = x_[:nl] + 1j * x_[nl:]
    # freq = 1 / wl
    
    config["layers"]["unknown"]["thickness"] = thickness

    layers, wave = config["layers"], config["wave"]
    
    thicknesses = [d["thickness"] for d in layers.values() if "thickness" in d.keys()]

    hphase = sum(thicknesses)
    gamma = 2 * pi / wl
    phasor = np.exp(-1j * gamma * hphase)
    t_exp1 = t_exp*phasor

    t_model = trans_stack(epsilon, thickness, config=config, wl=wl,eps_fun=eps_fun)
    mse_func = np.mean(np.abs(t_exp1 - t_model) ** 2)# / np.mean(np.abs(t_exp1) ** 2)
    # epsmax,epsmin = np.max((epsilon)),np.min((epsilon))
    # deps= np.abs(epsmax-epsmin)
    #
    # no = np.max(np.array([deps,1e-3]))
    # mse_grad = np.mean(np.abs(np.gradient(epsilon) ) ** 2)/ no**2
    mse_grad = np.mean(np.abs(np.gradient(epsilon)) ** 2) / np.mean(
        np.abs(epsilon) ** 2
    )  # * np.mean(np.abs( freq/ epsilon)**2)
    mse = weights[0] * mse_func + weights[1] * mse_grad

    if stats:
        return mse_func, mse_grad
    else:
        return mse


jac_ = grad(fmini)

jit_gfunc = jit(jac_)


def jac(x, **kwargs):
    return numpy.float64(jit_gfunc(x, **kwargs))
