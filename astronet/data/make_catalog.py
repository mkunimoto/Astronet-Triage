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

"""Functions for creating catalog of TESS TCEs, containing TIC IDs, BLS information, stellar params, sector, camera and
CCD number."""

import numpy as np
import pandas as pd
import os
import tsig
from tsig import catalog
from qlp.util.gaia import GaiaCatalog
from tsig.spacecraft import Spacecraft
from tsig.spacecraft.geometry import LevineModel
from tsig.mission import MissionProfile
import multiprocessing
import argparse
import logging
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_worker_processes",
    type=int,
    default=1,
    help="Number of subprocesses for processing the TCEs in parallel.")

parser.add_argument(
    '--input',
    nargs='+',
    help='CSV file(s) containing the TCE table(s) for training. Must contain '
         'columns: TIC, TCE planet number, final disposition',
    required=True)

parser.add_argument(
    "--base_dir",
    type=str,
    default='/pdo/users/yuliang/',
    help="Directory where TCE lists are located, and where the output will be saved.")

parser.add_argument(
    "--out_name",
    type=str,
    default='tces.csv',
    help="Name of output file.")

parser.add_argument(
    "--camera",
    type=int,
    default=-1,
    help="the camera number for the tce list, if applicable")

parser.add_argument(
    "--ccd",
    type=int,
    default=-1,
    help="the ccd number for the tce list, if applicable")



def star_query(tic):
    """

    :param tic:  TIC of the target star. May be an int or a possibly zero-
          padded string.
    :param ra: RA of target star. Float.
    :param dec: Dec of target star. Float.

    :return: dict containing stellar parameters.
    """

    field_list = ["id", "mass", "ra", "dec", "rad", "e_rad", "teff", "e_teff", "logg", "e_logg", "tmag", "e_tmag"]
    result, _ = tsig_catalog.query_by_id(list(np.array(tic).astype(int).astype(str)), ",".join(field_list))

    dtype = [(field_list[k], float) for k in xrange(len(field_list))]
    t = np.array(result, dtype=dtype)
    starparam = {}
    starparam["mass"] = np.array(t[:]["mass"])
    starparam["rad"] = np.array(t[:]["rad"])
    starparam["e_rad"] = np.array(t[:]["e_rad"])
    starparam["teff"] = np.array(t[:]["teff"])
    starparam["e_teff"] = np.array(t[:]["e_teff"])
    starparam["logg"] = np.array(t[:]["logg"])
    starparam["e_logg"] = np.array(t[:]["e_logg"])
    starparam["tmag"] = np.array(t[:]["tmag"])
    starparam["e_tmag"] = np.array(t[:]["e_tmag"])
    starparam["ra"] = np.array(t[:]["ra"])
    starparam["dec"] = np.array(t[:]["dec"])
    return starparam


def bls_params(tic, sector, cam, ccd, base_dir='/pdo/qlp-data/'):
#def bls_params(tic, sector, cam, ccd, base_dir='/pdo/qlp-data/injection/'):
    """

    :param tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
    :param sector: Int, sector number of data.
    :param cam: Int, camera number of data.
    :param ccd: Int, CCD number of data.
    :param base_dir: Base directory containing BLS files.
    :return: dataframe containing BLS information on significant peaks.
    """
    #filename = os.path.join(base_dir, 'sector-' + str(sector), 'ffi', 'cam' + str(cam), 'ccd' + str(ccd), 'BLS',
    #                        str(tic)+'.blsanal')
    filename = os.path.join('bls', str(tic)+'.blsanal')
    #print (filename)
    peaks = pd.read_table(filename, delimiter=' ', header=0, escapechar='#', dtype=float)
    is_tce  = True
    if (peaks['BLS_SignaltoPinknoise_1_0'].iloc[0] > 9) and (peaks['BLS_Npointsintransit_1_0'].iloc[0] > 5):
    #if (peaks['BLS_SignaltoPinknoise_1_0'].iloc[0] > 7) and (peaks['BLS_SignaltoPinknoise_1_0'].iloc[0]<9) and (peaks['BLS_Npointsintransit_1_0'].iloc[0] > 5):
        if (peaks['BLS_OOTmag_1_0'].iloc[0] < 12) and (peaks['BLS_SN_1_0'].iloc[0] > 5):
            is_tce = True
        elif (peaks['BLS_OOTmag_1_0'].iloc[0] >= 12) and (peaks['BLS_SN_1_0'].iloc[0] > 9):
            is_tce = True
        else:
            is_tce = False
    else:
        is_tce = False

    return peaks, is_tce


def _process_tce(tce_table):
    """
    Find camera, ccd number and stellar params of target given TIC and sector, using catalogs under /scratch/tmp
    :param tce_table: Pandas dataframe containing TIC ID, RA and Dec
    :param sector: Int, sector number of data
    :return: tce with stellar params, camera and ccd columns filled
    """

    if FLAGS.num_worker_processes > 1:
        current = multiprocessing.current_process()

    total = len(tce_table)
    tce_table['camera'] = 0
    tce_table['ccd'] = 0
    tce_table['star_rad'] = np.nan
    tce_table['star_mass'] = np.nan
    tce_table['teff'] = np.nan
    tce_table['logg'] = np.nan
    tce_table['SN'] = np.nan
    tce_table['Qingress'] = np.nan

    starparam = star_query(tce_table['tic_id'])
    tce_table['star_rad'] = starparam['rad']
    tce_table['star_mass'] = starparam['mass']
    tce_table['teff'] = starparam['teff']
    tce_table['logg'] = starparam['logg']
    tce_table['RA'] = starparam['ra']
    tce_table['Dec'] = starparam['dec']
    tce_table['Tmag'] = starparam['tmag']
    #runit through everything
    #tce_table = tce_table[tce_table.Tmag<=12]

    cnt = 0
    for index, tce in tce_table.iterrows():
        if FLAGS.num_worker_processes == 1:
            if cnt % 10 == 0:
                print 'Processed %s/%s TCEs' % (cnt, total)
        else:
            if cnt % 10 == 0:
                logger.info('Process %s: processing TCE %s/%s ' %(current.name, cnt, total))


        #if FLAGS.camera==-1 or FLAGS.ccd==-1:
        #    sc_ra, sc_dec, sc_roll = MissionProfile.pointing_to_rdr("sector%d" % tce['Sectors'], "tess_profile.cfg")
        #    sc.set_pointing(sc_ra, sc_dec, sc_roll)
        #    for cam_id in xrange(1, 5):
        #        x, y, ccd_n = model.celestial_to_pixel(
        #            starparam['ra'], starparam['dec'], sc_ra, sc_dec, sc_roll,
        #            sc.get_cam_geometry(cam_id),
        #            sc.get_ccd_geometries(cam_id), spp)
        #        if ccd_n[0]:
        #            tce_table.camera.loc[index] = cam_id
        #            tce_table.ccd.loc[index] = ccd_n[0]
        #else:
        tce_table.camera[index]=FLAGS.camera
        tce_table.ccd[index]=FLAGS.ccd

        try:
            bls, is_tce = bls_params(tce['tic_id'], tce['Sectors'], tce_table.camera.loc[index], tce_table.ccd.loc[index])
        except (IOError, pd.errors.EmptyDataError):
            print 'Skipped %s. BLS file does not exist.' % tce['tic_id']
            continue

        if is_tce:
            tce_table.Epoc.loc[index] = bls['BLS_Tc_1_0'].iloc[0]
            tce_table.Period.loc[index] = bls['BLS_Period_1_0'].iloc[0]
            tce_table.Duration.loc[index] = bls['BLS_Qtran_1_0'].iloc[0] * bls['BLS_Period_1_0'].iloc[0] * 24
            tce_table['Transit_Depth'].loc[index] = bls['BLS_Depth_1_0'].iloc[0] * 1e6
            tce_table.SN.loc[index] = bls['BLS_SignaltoPinknoise_1_0'].iloc[0]
            tce_table.Qingress.loc[index] = bls['BLS_Qingress_1_0'].iloc[0]
        cnt+=1
    tce_table = tce_table[np.isfinite(tce_table['Period'])]
    return tce_table


def parallelize(data):
    partitions = FLAGS.num_worker_processes
    data_split = np.array_split(data, partitions)

    pool = multiprocessing.Pool(processes=partitions)
    df = pd.concat(pool.map(_process_tce, data_split))
    pool.close()
    pool.join()

    return df


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    print 'Reading TIC'
    tsig_catalog = catalog.TIC()
    print 'Reading Gaia catalog'
    gaia_catalog = GaiaCatalog()

    #tce_table_names = [os.path.join(FLAGS.base_dir, csv) for csv in FLAGS.input]
    tce_table_names = [csv for csv in FLAGS.input]

    tce_table = pd.DataFrame()
    for table in tce_table_names:
        tces = pd.read_csv(table, header=0, usecols=[1,2,3,4,5,6,8,10,12,14,16])
        tce_table = pd.concat([tce_table, tces], ignore_index=True)

    if FLAGS.num_worker_processes == 1:
        tce_table = _process_tce(tce_table)
    else:
        logger = multiprocessing.get_logger()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        logger.info('Process started')
        tce_table = parallelize(tce_table)
    tce_table.to_csv(FLAGS.out_name)
