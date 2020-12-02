# Copyright 2018 Liang Yu.
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

"""Functions for reading TESS data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import glob
import os

import numpy as np
import astropy 

from lctools.hdf5lc import HDFLightCurve 

'''
def tess_filenames(tic, base_dir):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing Kepler data.
      sector: Int, sector number of data.
      cam: Int, camera number of data.
      ccd: Int, CCD number of data.
      injected: Bool, whether target also has a light curve with injected planets.
      injected_dir: Directory containing light curves with injected transits.
      check_existence: If True, only return filenames corresponding to files that
          exist.

    Returns:
      filename for given TIC.
    """
    fitsfile = "tess*-%.16d-*-cr_llc.fits.gz" % int(tic)
    file_names = glob.glob(os.path.join(base_dir, fitsfile))
    if len(file_names) != 1:
        raise ValueError(f'found {len(file_names)} files for {tic}: {file_names}')
    filename, = file_names

    return filename
'''

def read_tess_light_curve(filename, flux_key):
    """Reads time and flux measurements for a Kepler target star.

    Args:
      filename: str name of fits file containing light curve.
      flux_key: Key of fits column containing flux.

    Returns:
      time: Numpy array; the time values of the light curve.
      flux: Numpy array corresponding to the time array.
    """
    lc = HDFLightCurve(filename)
    lc.load_from_file(label='all', ap=-1, rawmagkey='RawMagnitude', dmagkey='KSPMagnitude')
    time = lc.data['jd'].astype(float)
    if flux_key == 'RawMagnitude':
        mag = lc.data['rlc'].astype(float)
    elif flux_key == 'KSPMagnitude':
        mag = lc.data['ltflc'].astype(float)
    quality = lc.data['flag']
    quality_flag = quality==0
    # Remove outliers
    time = time[quality_flag]
    mag = mag[quality_flag]
    # Remove NaN flux values.
    valid_indices = np.where(np.isfinite(mag))
    time = time[valid_indices]
    mag = mag[valid_indices]
    flux = 10.**(-0.4*(mag - np.nanmedian(mag)))
    return time,flux 


