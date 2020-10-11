# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility function for smoothing data using a median filter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
from third_party.robust_mean import robust_mean


def tmod(t, p, e):
    tmodn = (t % p) - (e % p)
    tmodn = tmodn + p * (tmodn <= -0.5 * p) - p * (tmodn >= 0.5 * p)
    return(tmodn)

def get_overlap(min1, max1, min2, max2):
    return(max(0, min(max1, max2) - max(min1, min2)))

def new_binning(time, flux, period, num_bins, t_min, t_max):
  t = time.copy()
  
  bins_left_edge, step = np.linspace(
      t_min, t_max, num=num_bins, endpoint=False, retstep=True)

  bin_width = step.copy()
  hbw = bin_width / 2
  
  bins_center = bins_left_edge + 0.5 * bin_width

  cadence = 0.4904 / 24 #converts hours to days
  hc = cadence / 2
  hbw = bin_width / 2
  
  f = np.zeros(num_bins)
  v = np.ones(num_bins)
  for i, b in enumerate(bins_center):
    #time from bin center
    t_c = tmod(t, period, b)
    
    #find which points are within the bin
    bin_mask = np.where(abs(t_c) <= hbw + hc)
    in_bin = t_c[bin_mask]
    f_x = flux[bin_mask]
    
    if len(f_x) == 0:
        v[i] = 0.0
        continue

    if len(f_x) == 1:
        f[i] = f_x[0]
        continue
    
    #calculate the robust mean to remove outliers
    _, _, mask = robust_mean.robust_mean(f_x, 3)
    
    #remove outliers
    f_x = f_x[mask]
    in_bin = in_bin[mask]
    
    if len(f_x) == 0:
        v[i] = 0.0
        continue

    #get the weight of each time point within the bin
    weight = [get_overlap(-hbw, hbw, in_bin[j] - hc, in_bin[j] + hc) / bin_width
              for j in range(len(in_bin))]
    # TODO: don't ignore nans.
    bin_flux = np.nansum(weight * f_x) / np.nansum(weight)
    f[i] = bin_flux

  return f, v
