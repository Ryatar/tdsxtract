#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""
Extraction example
==================

Tutorial.
"""


import matplotlib.pyplot as plt
import numpy as npo

from tdsxtract import *

testdir = "./"


sample_thickness = 500e-6

###############################################################################
# First we load the reference and sample signals

pos_ref, v_ref = load(f"{testdir}/data/reference.txt")
pos_samp, v_samp = load(f"{testdir}/data/sample.txt")
assert npo.all(pos_ref == pos_samp)
position = pos_ref * 1e-6

###############################################################################
# We convert the position to time delay and plot the data

time = pos2time(position)
plt.figure()
plt.plot(time * 1e12, v_ref, label="reference")
plt.plot(time * 1e12, v_samp, label="sample")
plt.xlabel("time (ps)")
plt.ylabel("transmitted signal (mV)")
plt.legend()
plt.tight_layout()

###############################################################################
# By looking at the first peak shift, we can get a rough estimate of  the
# permittivity

eps_guess = get_epsilon_estimate(v_ref, v_samp, time, sample_thickness)
print(eps_guess)

###############################################################################
# We now switch to the frequency domain by computing the Fourier transform

freqs_ref, fft_ref = fft(time, v_ref)
freqs_samp, fft_samp = fft(time, v_samp)
freqs_THz = freqs_ref * 1e-12

plt.figure()
plt.plot(freqs_THz, npo.abs(fft_ref), label="reference")
plt.plot(freqs_THz, npo.abs(fft_samp), label="sample")
plt.xlabel("frequency (THz)")
plt.ylabel("transmitted signal amplitude (mV)")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(freqs_THz, npo.unwrap(npo.angle(fft_ref)) * 180 / npo.pi, label="reference")
plt.plot(freqs_THz, npo.unwrap(npo.angle(fft_samp)) * 180 / npo.pi, label="sample")
plt.xlabel("frequency (THz)")
plt.ylabel("transmitted signal phase (degrees)")
plt.legend()
plt.tight_layout()

###############################################################################
# Let's calculate the transmission coefficient

transmission = fft_samp / fft_ref

freqs_THz, imin, imax = restrict(freqs_THz, 0.1, 2.5)
transmission = transmission[imin:imax]

plt.figure()
plt.plot(freqs_THz, npo.abs(transmission))
plt.xlabel("frequency (THz)")
plt.ylabel("transmission amplitude")
plt.tight_layout()

plt.figure()
plt.plot(freqs_THz, (npo.angle(transmission)) * 180 / npo.pi)
plt.xlabel("frequency (THz)")
plt.ylabel("transmission phase (phase)")
plt.tight_layout()

###############################################################################
# To describe our sample we use a Sample object

sample = Sample(
    {
        "unknown": {"epsilon": None, "mu": 1.0, "thickness": sample_thickness},
    }
)

###############################################################################
# We ar now ready to perform the extraction

wavelengths = c / freqs_THz * 1e-12

epsilon_opt, h_opt, opt = extract(
    sample,
    wavelengths,
    transmission,
    eps_re_min=1,
    eps_re_max=100,
    eps_im_min=-10,
    eps_im_max=10,
    epsilon_initial_guess=eps_guess,
)


eps_smooth = smooth(epsilon_opt)

plt.figure()
plt.plot(freqs_THz, epsilon_opt.real, "o", label="raw")
plt.plot(freqs_THz, eps_smooth.real, label="smoothed")
plt.xlabel("frequency (THz)")
plt.ylabel(r"Re $\varepsilon$")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(freqs_THz, epsilon_opt.imag, "o", label="raw")
plt.plot(freqs_THz, eps_smooth.imag, label="smoothed")
plt.xlabel("frequency (THz)")
plt.ylabel(r"Im $\varepsilon$")
plt.legend()
plt.tight_layout()


t_model = sample_transmission(
    epsilon_opt, h_opt, sample=sample, wavelengths=wavelengths
)

t_model_smooth = sample_transmission(
    eps_smooth, h_opt, sample=sample, wavelengths=wavelengths
)


gamma = 2 * np.pi / wavelengths
thickness_tot = sum([k["thickness"] for lay, k in sample.items()])
phasor = np.exp(-1j * gamma * thickness_tot)

fig, ax = plt.subplots()
ax.plot(
    freqs_THz,
    npo.abs(transmission) ** 2,
    "s",
    label="meas.",
    alpha=0.3,
    lw=0,
    c="#656e7e",
    mew=0,
)
ax.plot(
    freqs_THz,
    npo.abs(t_model) ** 2,
    "-",
    label="sim. raw",
    alpha=0.3,
    lw=1,
    c="#656e7e",
    mew=0,
)
ax.plot(
    freqs_THz,
    npo.abs(t_model_smooth) ** 2,
    "-",
    label="sim. smooth",
    alpha=1,
    lw=1,
    c="#AE383E",
)

ax.set_ylim(0, 1)
ax.set_xlabel("frequency (THz)")
ax.legend()
ax.set_ylabel(r"amplitude")
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(
    freqs_THz,
    (npo.angle(phasor * transmission)) * 180 / pi,
    "s",
    label="meas.",
    alpha=0.3,
    lw=0,
    c="#656e7e",
    mew=0,
)
ax.plot(
    freqs_THz,
    (npo.angle(t_model)) * 180 / pi,
    "-",
    label="sim. raw",
    alpha=0.3,
    lw=1,
    c="#656e7e",
    mew=0,
)
ax.plot(
    freqs_THz,
    (npo.angle(t_model_smooth)) * 180 / pi,
    "-",
    label="sim. smooth",
    alpha=1,
    lw=1,
    c="#AE383E",
)
ax.set_xlabel("frequency (THz)")
ax.legend()
ax.set_ylabel(r"phase")
plt.tight_layout()
