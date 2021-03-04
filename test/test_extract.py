#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from tdsxtract.io import *
import numpy as npo

pos_ref, v_ref = load("./data/reference.txt")
pos_samp, v_samp = load("./data/sample.txt")

assert npo.all(pos_ref == pos_samp)
