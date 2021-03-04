#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as npo


def load(filename):
    """Load measurement data.

    Parameters
    ----------
    filename : str
        Name of the file to load.

    Returns
    -------
    tuple (position, voltage)
        position of the delay satge (in microns) and signal amplitude (in V).

    """
    data = npo.loadtxt(filename, skiprows=13)
    position = data[:, 0]
    voltage = data[:, 1]
    return position, voltage
