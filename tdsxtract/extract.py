#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from pltstyle import *
from scipy.constants import c
from optim import *
import scipy.signal as signal
import sys
from scipy.interpolate import interp1d


def smooth(raw_data, cutoff=0.05):
    # First, design the Buterworth filter
    N = 3  # Filter order
    B, A = signal.butter(N, cutoff, output="ba")
    smooth_data = signal.filtfilt(B, A, raw_data)
    return smooth_data


def get_epsilon_estimate(Vref, Vsamp, time):
    refMax = np.argmax(
        Vref
    )  #%Find the peak of the TD signals to calculate time delay then the bulk refractive index
    sampMax = np.argmax(Vsamp)
    t0 = time[refMax]
    t1 = time[sampMax]
    eps_delay = (
        (t1 - t0) * c / (h * 1e-6) + 1
    ) ** 2  # Good guess for bulk refractive index here
    return eps_delay


#######  FILES  ###################

film = False
film = True
saveplt = False

case = "film"
# case = "meta"

subsname = "si" if case == "meta" else "sapp"

if not film:
    if case == "meta":
        ref_file = "Hanchi_Ref3_11609.txt"
        samp_file = "SuperPolishedSilicon_FacingAnt_11613.txt"
    else:
        ref_file = "HanchiRef_11603.txt"
        samp_file = "Hanchi_Standard_Sapp_PolishedSideFirst_11604.txt"

else:
    if case == "meta":
        ref_file = "Ref_11615.txt"
        samp_file = "Nano_FacingAnt_11616.txt"
    else:
        # ref_file = "Hanchi_Ref2_11606.txt"
        # samp_file = "Hanchi_Standard_Sapp_UnpolishedFirst_11605.txt"
        ref_file = "Hanchi_Ref2_11606.txt"
        samp_file = "Hanchi_FilmFirst_11607.txt"


## load
ref_data = numpy.loadtxt(ref_file, skiprows=13)
samp_data = numpy.loadtxt(samp_file, skiprows=13)

## check if times are the same
assert np.all(ref_data[:, 0] == samp_data[:, 0])

## thicknesses
if not film:
    thickness_subs = 0.0
    if case == "meta":
        thickness_film = 500.0
    else:
        thickness_film = 430.0
else:

    if case == "meta":
        thickness_subs = 500.0
        thickness_film = 4.5 + 0.4
    else:
        thickness_subs = 430.0
        thickness_film = 0.4

# h = thickness_subs
h = thickness_film


# hphase = thickness_subs
hphase = thickness_film + thickness_subs


delay = ref_data[:, 0]

# 2*delay since 2 paths of the laser are shifted on the delay stage
time = 2 * delay * 1e-6 / c
dt = time[1] - time[0]
fs = 1 / dt

Vref = ref_data[:, 1]
Vsamp = samp_data[:, 1]


eps_delay = get_epsilon_estimate(Vref, Vsamp, time)
# plt.plot(delay, V)

Tref_ = np.fft.rfft(Vref)
Tsamp_ = np.fft.rfft(Vsamp)

Tsamp = Tsamp_ / Tref_

N = len(Tsamp)

fs = 1 / np.gradient(time)
freq = fs[0:N] * np.linspace(0, 1 / 2, N)


archname = "film_sapp.npz"
arch = np.load(archname)
epsilon_load = arch["epsilon"]
epsilon_smooth_load = arch["epsilon_smooth"]
epsilon_mean_load = arch["epsilon_mean"]
freqs_load = arch["freqs"]

#
# def eps_sub_func(wl):
#     return np.interp(wl, c / freqs_load, epsilon_smooth_load)


if film:
    if case == "meta":
        eps_substrate = 12.389 - 0.0715j  ## silicon
    else:
        eps_substrate = 9.40630913-0.03536057j#03651004j  ## sapphire
else:
    eps_substrate = 1.0 - 0.0j


wl_load = c / freqs_load * 1e6

eps_array = [wl_load, np.flipud(epsilon_smooth_load)]
eps_array = [wl_load, np.flipud(epsilon_load)]
eps_array = [wl_load, np.ones_like(wl_load)*epsilon_mean_load]
# eps_array = None

##############################################################################
#################  OPTIMIZATION  #############################################
##############################################################################

force_passive = False
weight = 0.9
delta_h = 0.001  # tolerance on thickness (to be optimized)

tand = 3
eps_re0 = 1 if film else eps_delay
eps_im0 = -0
eps_re_min, eps_re_max = 1, 500

##############################################################################
##############################################################################
##############################################################################


if __name__ == "__main__":

    freqmax = 3e12
    freqmin = 0.26e12
    freqmin_THz, freqmax_THz = freqmin * 1e-12, freqmax * 1e-12
    # freq =  freq[1:]

    imax = np.argmin(abs(freq - freqmax))
    imin = np.argmin(abs(freq - freqmin))
    # imin = 1

    freqs = freq[imin:imax]

    # t_exp = Tsubs[imin:imax]
    t_exp = Tsamp[imin:imax]

    # f =interp1d(freqs,t_exp,kind="cubic")
    # freqs_ = np.linspace(freqs[0],freqs[-1],1000)
    # t_exp = f(freqs_)
    # t_exp = np.interp(freqs_,freqs,t_exp)

    # freqs = freqs_

    freqs_THz = freqs * 1e-12

    wl = c / freqs * 1e6
    nl = len(wl)
    ones = np.ones_like(wl)

    gamma = 2 * pi / wl
    phasor = np.exp(-1j * gamma * hphase)
    t_experiment = t_exp  # * phasor

    # plt.clf()

    def plot_trans(t_experiment, ax, icol=0, labs=None):

        ax[0].plot(
            freqs_THz,
            np.abs(t_experiment) ** 2,
            "-",
            alpha=1,
            c=mpl_colors[icol],
            label=labs[0],
        )
        # ax[0].plot(freqs_THz, np.abs(t_experiment) ** 2, label="simu", c=mpl_colors[0])

        ax[1].plot(
            freqs_THz,
            (np.angle(t_experiment)) * 180 / pi,
            "-",
            alpha=1,
            c=mpl_colors[icol],
            label=labs[1],
        )
        # ax[1].plot(freqs_THz, (np.angle(t)) * 180 / pi, label="simu", c=mpl_colors[1])

        ax[0].set_ylabel(r"amplitude (a.u.)")
        # ax[0].set_xlabel(r"frequency $f$ (THz)")
        ax[0].set_ylim(0, 1)
        ax[1].set_ylabel(r"phase (degree)")
        ax[1].set_xlabel(r"frequency $f$ (THz)")
        plt.xlim([freqmin_THz, freqmax_THz])

        # plt.ylim([0, 1])
        plt.tight_layout()

    if saveplt:

        fig = plt.gcf()
        ax = fig.get_axes()
        if not film or len(ax) == 0:
            fig, ax = plt.subplots(2, 1, figsize=(4, 3), sharex=True)

        icol = 1 if film else 0
        if case == "meta":
            label = (
                "silicon substrate + BST metasurface" if film else "silicon substrate"
            )
        else:
            label = (
                "sapphire substrate + BST thin film" if film else "sapphire substrate"
            )

        plot_trans(t_experiment, ax, icol, labs=[label, None])
        ax[0].legend(fontsize=6)

        name = f"{subsname}_measT"
        if film:
            name = "film_" + name

        if saveplt:
            plt.savefig(f"./fig/{name}.pdf")

    layers = OrderedDict(
        {
            "superstrate": {"epsilon": 1.0, "mu": 1.0},
            "unknown": {"epsilon": 1.0, "mu": 1.0, "thickness": thickness_film},
            "layer subs": {
                "epsilon": eps_substrate,
                "mu": 1.0,
                "thickness": thickness_subs,
            },
            "substrate": {"epsilon": 1.0, "mu": 1.0},
        }
    )

    # layers = OrderedDict(
    #     {
    #         "superstrate": {"epsilon": 1.0, "mu": 1.0},
    #         "unknown": {"epsilon": 1.0, "mu": 1.0, "thickness": thickness_subs},
    #         "substrate": {"epsilon": 1.0, "mu": 1.0},
    #     }
    # )
    wave = {
        "lambda0": 1,
        "theta0": 0.0 * pi,
        "phi0": 0 * pi,
        "psi0": 0 * pi,
    }

    config = dict(layers=layers, wave=wave)

    passiv = 1 - float(force_passive)
    eps_im_min, eps_im_max = -tand * eps_re_max, passiv * tand * eps_re_max

    hmin, hmax = (1 - delta_h) * h, (1 + delta_h) * h

    initial_guess = numpy.hstack([ones * eps_re0, ones * eps_im0, h])
    initial_guess = numpy.float64(initial_guess)
    bounds = [(float(eps_re_min), float(eps_re_max)) for i in range(nl)]
    bounds += [(float(eps_im_min), float(eps_im_max)) for i in range(nl)]
    bounds += [(float(hmin), float(hmax))]
    bounds = numpy.float64(bounds)

    weights = (float(weight), float(1 - weight))

    fmini_opt = lambda x: fmini(
        x, config=config, weights=weights, t_exp=t_experiment, wl=wl, eps_fun=eps_array,
    )

    # def fmini_opt(x):
    #     # print(x)
    #     # out =  fmini(
    #     #     x, config=config, weights=weights, t_exp=t_experiment, wl=wl, eps_fun=eps_array,
    #     # )
    #     out = sum(x)
    #     # print(out)
    #     return out

    # eps_sub_func = lambda wl: 22. - 0 * 1j
    #
    # @jit
    # def test(x,wl, eps_array=None):
    #     if eps_array is not None:
    #         a,b = eps_array
    #         out = np.interp(wl, a,b)
    #     else:
    #         out = 12
    #     return out + sum(x)
    #
    # test(initial_guess,wl[-1]*1e-6,eps_array=eps_array)
    # test(initial_guess,wl[12])
    # xsaxas

    jac_opt = lambda x: jac(
        x, config=config, weights=weights, t_exp=t_experiment, wl=wl, eps_fun=eps_array,
    )
    # options = {"maxiter": 5000, "disp": True}

    options = {
        "disp": True,
        "maxcor": 250,
        "ftol": 1e-16,
        "gtol": 1e-16,
        "eps": 1e-11,
        "maxfun": 15000,
        "maxiter": 15000,
        "iprint": 1,
        "maxls": 200,
        "finite_diff_rel_step": None,
    }
    # options.update(options_solver)

    jacobian = jac_opt
    # jacobian = None
    # bounds=None

    opt = minimize(
        fmini_opt,
        initial_guess,
        bounds=bounds,
        tol=1e-16,
        options=options,
        jac=jacobian,
        method="L-BFGS-B",
    )


    # epsilon = (x[0] + 1j * x[1])*ones
    hopt = opt.x[-1]
    _epsopt = opt.x[:-1]
    epsilon_opt = _epsopt[:nl] + 1j * _epsopt[nl:]
    eps_av = complex(np.mean(epsilon_opt))

    cutoff = 7 / nl

    eps_smooth = smooth(epsilon_opt.real, cutoff) + 1j * smooth(
        epsilon_opt.imag, cutoff
    )

    hphase = hopt + thickness_subs
    phasor = np.exp(-1j * gamma * hphase)
    t_experiment *= phasor

    # epsilon_opt = (1.0 +1*0j)* ones
    # epsilon_opt = eps_sapphire * ones

    plt.close("all")

    fig, ax = plt.subplots(1, 2, figsize=(5, 3))
    ax[0].plot(
        freqs_THz,
        epsilon_opt.real,
        "s",
        label="Re",
        alpha=0.3,
        lw=0,
        c=mpl_colors[0],
        mew=0,
    )
    ax[1].plot(
        freqs_THz,
        epsilon_opt.imag,
        "s",
        label="Im",
        alpha=0.3,
        lw=0,
        c=mpl_colors[1],
        mew=0,
    )

    fig.suptitle("permittivity")
    ax[0].set_ylabel(r"$\varepsilon'$")
    ax[0].set_xlabel(r"frequency $f$ (THz)")
    ax[1].set_ylabel(r"$\varepsilon''$")
    ax[1].set_xlabel(r"frequency $f$ (THz)")
    ax[0].plot(
        freqs_THz, eps_smooth.real, "-", label="Re", alpha=1, lw=0.5, c=mpl_colors[0]
    )

    ax[1].plot(
        freqs_THz, eps_smooth.imag, "-", label="Im", alpha=1, lw=0.5, c=mpl_colors[1]
    )
    ax[0].set_xlim([freqmin_THz, freqmax_THz])
    ax[1].set_xlim([freqmin_THz, freqmax_THz])

    yrat_re, yrat_im = 0.2, 0.2
    ax[0].set_ylim(
        epsilon_opt.real.min() * (1 - yrat_re), epsilon_opt.real.max() * (1 + yrat_re)
    )
    ax[1].set_ylim(
        epsilon_opt.imag.min() * (1 + yrat_im),
        epsilon_opt.imag.max() * (1 + np.sign(epsilon_opt.imag.max()) * yrat_im),
    )
    # plt.ylim([0, 1])
    plt.tight_layout()
    plt.pause(0.1)

    name = f"{subsname}_extracted_eps"
    if film:
        name = "film_" + name
    if saveplt:
        plt.savefig(f"./fig/{name}.pdf")

    tand = epsilon_opt.imag / epsilon_opt.real
    tand_smooth = smooth(tand, cutoff)
    #
    # fig, ax = plt.subplots(1, figsize=(3, 3))
    # plt.plot(
    #     freqs_THz, tand, "s", label="raw", alpha=0.3, lw=0, c=mpl_colors[2], mew=0,
    # )
    # plt.plot(freqs_THz, tand_smooth, "--", label="smooth", c=mpl_colors[2])
    #
    # ax.set_ylabel(r"$\tan\,\delta$")
    # ax.set_xlabel(r"frequency $f$ (THz)")
    # ax.set_xlim([freqmin_THz, freqmax_THz])
    # plt.tight_layout()
    #
    # name = f"{subsname}_extracted_tand"
    # if film:
    #     name = "film_" + name
    # if saveplt:
    #     plt.savefig(f"./fig/{name}.pdf")
    #
    ### ---------------

    epsil = epsilon_opt
    epsilon_av = eps_av * ones
    # epsil = eps_smooth

    t = trans_stack(epsil, hopt, config=config, wl=wl)
    t_smooth = trans_stack(eps_smooth, hopt, config=config, wl=wl)
    t_mean = trans_stack(epsilon_av, hopt, config=config, wl=wl)

    fig, ax = plt.subplots(2, 1, figsize=(4, 3.5), sharex=True)
    ax[0].plot(
        freqs_THz,
        np.abs(t_experiment) ** 2,
        "-s",
        alpha=0.2,
        lw=2,
        c=mpl_colors[0],
        label="measured",
    )
    ax[0].plot(freqs_THz, np.abs(t) ** 2, label="simulation", c=mpl_colors[0])
    ax[0].plot(
        freqs_THz,
        np.abs(t_smooth) ** 2,
        "--",
        label="simulation smooth",
        c=mpl_colors[0],
    )
    ax[0].plot(
        freqs_THz, np.abs(t_mean) ** 2, "--", label="simulation mean", c=mpl_colors[2],
    )
    ax[1].plot(
        freqs_THz,
        (np.angle(t_experiment)) * 180 / pi,
        "-",
        alpha=0.2,
        lw=2,
        c=mpl_colors[1],
        label="measured",
    )
    ax[1].plot(freqs_THz, (np.angle(t)) * 180 / pi, label="simulation", c=mpl_colors[1])
    ax[1].plot(
        freqs_THz,
        (np.angle(t_smooth)) * 180 / pi,
        "--",
        label="simulation  smooth",
        c=mpl_colors[3],
    )
    ax[1].plot(
        freqs_THz,
        (np.angle(t_mean)) * 180 / pi,
        "--",
        label="simulation  mean",
        c=mpl_colors[1],
    )
    ax[0].set_ylabel(r"amplitude (a.u.)")
    # ax[0].set_xlabel(r"frequency $f$ (THz)")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylabel(r"phase (degree)")
    ax[1].set_xlabel(r"frequency $f$ (THz)")
    ax[1].set_ylim(-180, 300)
    plt.xlim([freqmin_THz, freqmax_THz])

    # plt.ylim([0, 1])
    plt.tight_layout()
    ax[0].legend(fontsize=6)
    ax[1].legend(fontsize=6)

    name = f"{subsname}_checkT"
    if film:
        name = "film_" + name
    if saveplt:
        plt.savefig(f"./fig/{name}.pdf")

    archname = f"{case}_bst" if film else f"{case}_{subsname}"
    np.savez(
        archname,
        epsilon=epsilon_opt,
        epsilon_smooth=eps_smooth,
        epsilon_mean=eps_av,
        freqs=freqs,
        t_experiment=t_experiment,
    )
    
    data = np.vstack((freqs/1e12,eps_smooth.real,eps_smooth.imag)).T
    
    import numpy as npo
    
    npo.savetxt("epsilon_BST.csv", data, fmt="%f",delimiter=",", header="frequency (THz), Re epsilon, Imag epsilon")

    # ### test dispersive
    #
    # def eps_disp(l):
    #     return 1+l


#
# plt.clf()
# plt.plot(t_experiment.real[:10],t_experiment.imag[:10],"o-")
#

annot = lambda ax, s: ax.text(
    0.05, 0.8, f"error = {np.mean(s):.3e}", fontsize=5, transform=ax.transAxes
)


fig, ax = plt.subplots(3, sharex=True)
delta_t = np.abs(t_experiment - t)
ax[0].plot(freqs_THz, delta_t)

annot(ax[0], delta_t)

delta_t_smooth = np.abs(t_experiment - t_smooth)
ax[1].plot(freqs_THz, delta_t_smooth)
ax[1].set_xticks([], minor=True)
annot(ax[1], delta_t_smooth)

delta_t_mean = np.abs(t_experiment - t_mean)
ax[2].plot(freqs_THz, delta_t_mean)
ax[2].set_xticks([], minor=True)
annot(ax[2], delta_t_mean)


# fig, ax =plt.subplots(3)
# ax[0].plot(freqs_THz, np.abs(1-t/t_experiment ))
# ax[2].plot(freqs_THz, np.abs(1-t_smooth/t_experiment ))
# ax[3].plot(freqs_THz, np.abs(1-t_mean/t_experiment ))
#

# plt.figure()
# plt.plot(freqs_THz, epsilon_opt)
