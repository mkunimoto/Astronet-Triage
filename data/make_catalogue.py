import numpy as np
import pandas as pd
import os
import multiprocessing
import argparse
import logging
import sys

from qlp.util.dataio import readtableline
from tsig import catalog

parser = argparse.ArgumentParser()
parser.add_argument('---num_worker_processes', type=int, default=1, help='Number of subprocesses for processing the TCEs in parallel')
parser.add_argument('--input', type=str, help='CSV file containing the TCE table for astronet. Must contain: tic_id,Tmag,Period,Epoc,Transit_depth,Duration,Sectors,Camera,CCD', required=True)
parser.add_argument('--base_dir', type=str, help='Directry where TCE lists are located, and where the output will be saved', required=True)
parser.add_argument('--output', type=str, default='tces.csv', help='Name of output file')

def star_query(tic):
    field_list = ["id", "mass", "ra", "dec", "rad", "tmag", "teff", "logg"]
    result,_ = c.query_by_id(list(np.array(tic).astype(int).astype(str)), ",".join(field_list))
    dtype = [(field_list[k], float) for k in xrange(len(field_list))]
    t = np.array(result, dtype=dtype)
    star_params = {}
    for item in field_list:
        star_params[item] = np.array(t[:][item])
    return star_params

def initial_triage(tic, sector, cam, ccd, base_dir='/pdo/qlp-data/'):
    filename = os.path.join(base_dir, 'sector-'+str(sector), 'ffi', 'cam'+str(cam), 'ccd'+str(ccd), 'BLS', str(tic)+'.blsanal')
    bls = readtableline(filename)
    is_tce = True
    if (bls['SignaltoPinknoise'] > 9) and (bls['Npointsintransit'] > 5):
        if (bls['OOTmag'] < 12) and (bls['SN'] > 5):
            is_tce = True
        elif (bls['OOTmag'] >= 12) and (bls['SN'] > 9):
            is_tce = True
        else:
            is_tce = False
    else:
        is_tce = False
    return bls, is_tce

def _process_tce(tce_table):
    if FLAGS.num_worker_processes > 1:
        current = multiprocessing.current_process()
    total = len(tce_table)
    starparam = star_query(tce_table['tic_id'])
    tce_table['star_rad'] = starparam['rad']
    tce_table['star_mass'] = starparam['mass']
    tce_table['teff'] = starparam['teff']
    tce_table['logg'] = starparam['logg']
    tce_table['RA'] = starparam['ra']
    tce_table['Dec'] = starparam['dec']
    tce_table['Tmag'] = starparam['tmag']
    tce_table['SN'] = np.nan
    tce_table['Qingress'] = np.nan
    cnt = 0
    for index, tce in tce_table.iterrows():
        if FLAGS.num_worker_processes == 1:
            if cnt % 10 == 0:
                print 'Processed %s/%s TCEs' % (cnt, total)
        else:
            if cnt % 10 == 0:
                logger.info('Process %s: processing TCE %s/%s ' % (current.name, cnt, total))
        try:
            sector = int(tce['Sectors'])
            cam = int(tce['Camera'])
            ccd = int(tce['CCD'])
            bls, is_tce = initial_triage(int(tce['tic_id']), sector, cam, ccd)
            if is_tce:
                tce_table.Epoc.loc[index] = bls['Tc']
                tce_table.Period.loc[index] = bls['Period']
                tce_table.Duration.loc[index] = bls['Qtran']*bls['Period']
                tce_table.Transit_Depth.loc[index] = bls['Depth']
                tce_table.SN.loc[index] = bls['SignaltoPinknoise']
                tce_table.Qingress.loc[index] = bls['Qingress']
            cnt += 1
        except:
            continue
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
    c = catalog.TIC()
    tce_table = pd.read_csv(os.path.join(FLAGS.base_dir, FLAGS.input))
    if FLAGS.num_worker_processes == 1:
        tce_table = _process_tce(tce_table)
    else:
        logger = multiprocess.get_logger()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        logger.info('Process started')
        tce_table = parallelize(tce_table)
    tce_table.to_csv(os.path.join(FLAGS.base_dir, FLAGS.output))
