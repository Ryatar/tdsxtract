#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from scipy.constants import c

import scipy.signal as signal


def pos2time(position):
    """Converts stage position to time shift.

    Parameters
    ----------
    position : array
        The position of the stage.

    Returns
    -------
    array
        Corresponding time shift.
        
    Notes
    -----
    2*delay since 2 paths of the laser are shifted on the delay stage

    """
    time = 2 * delay * 1e-6 / c
    return time


def smooth(data, cutoff=0.05, order=3):
    """Data smoothing.

    Parameters
    ----------
    data : array
        The data to smooth.
    cutoff : float
        Filter cutoff frequency (the default is 0.05).
    order : int
        Filter order (the default is 3).

    Returns
    -------
    array
        Smoothed data.

    """
    # Buterworth filter
    B, A = signal.butter(order, cutoff, output="ba")
    smooth_data = signal.filtfilt(B, A, data)
    return smooth_data


def get_epsilon_estimate(voltage_reference, voltage_sample, time, sample_thickness):
    """Get a rough estimate of the permittivity value.

    Parameters
    ----------
    voltage_reference : array
        Reference voltage.
    voltage_sample : array
        Sample voltage.
    time : array
        Time steps.
    sample_thickness : float
        Thickness of the sample (in microns).

    Returns
    -------
    float
        Permittivity estimate.

    """

    # Find the peak of the TD signals to calculate time delay
    # then the bulk refractive index
    ref_max = np.argmax(voltage_reference)
    samp_max = np.argmax(voltage_sample)
    t0 = time[ref_max]
    t1 = time[samp_max]
    # Good guess for bulk refractive index here
    epsilon_estimate = ((t1 - t0) * c / (sample_thickness * 1e-6) + 1) ** 2
    return epsilon_estimate
