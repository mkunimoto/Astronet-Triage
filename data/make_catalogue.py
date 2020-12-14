import multiprocessing
import numpy as np
import pandas as pd
import optparse
import os
import logging
import sys

from qlp.util.dataio import readtableline
from tsig import catalog

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_worker_processes",
    type=int,
    default=1,
    help="Number of subprocesses for processing the TCEs in parallel.")

parser.add_argument(
    '--input',
    type=str,
    help='CSV file(s) containing the TCE table(s) for training. Must contain columns: tic_id,Tmag,Period,Epoc,Transit_Depth,Duration,Sectors',
    required=True)

parser.add_argument(
    "--base_dir",
    type=str,
    default='/pdo/users/mkuni/',
    help="Directory where TCE lists are located, and where the output will be saved.")

parser.add_argument(
    "--output",
    type=str,
    default='tces.csv',
    help="Name of output file.")

def star_query(ticid):
    star_params = {}
    field_list = ["id", "mass", "rad", "tmag"]
    result,_ = c.query_by_id(ticid, ",".join(field_list))
    dtype = [(field_list[k], float) for k in xrange(len(field_list))]
    t = np.array(result, dtype=dtype)
    for item in field_list:
        star_params[item] = t[:][item][0]
    return star_params

def initial_triage(tic, bls_dir):
    filename = os.path.join(bls_dir, %i.blsanal' % tic)
    bls = readtableline(filename)
    is_tce = True
    if bls['SignaltoPinknoise'] < 9:
        is_tce = False
    if bls['Npointsintransit'] < 5:
        is_tce = False
    if bls['Depth'] > 0.5:
        is_tce = False
    if bls['OOTmag'] <= 12:
        if bls['SN'] < 5:
            is_tce = False
    else:
        if bls['SN'] < 9:
            is_tce = False
    return bls, is_tce

def process_tce(tce_table):
    if nprocs > 1:
        current = multiprocessing.current_process()
    total = len(tce_table)
    tce_table['star_rad'] = np.nan
    tce_table['star_mass'] = np.nan
    cnt = 0
    for index, tce in tce_table.iterrows():
        cnt += 1
        if nprocs == 1:
            if cnt % 10 == 0:
                print 'Processed %s/%s TCEs' % (cnt, total)
        else:
            if cnt % 10 == 0:
                logger.info('Process %s: processing TCE %s/%s ' % (current.name, cnt, total))
        tic = int(tce['tic_id'])
        try:
            sector=int(tce['Sectors'])
            cam=int(tce['Camera'])
            ccd=int(tce['CCD'])
            bls_dir = '/pdo/qlp-data/sector-%i/ffi/cam%i/ccd%i/BLS' % (sector, cam, ccd)
            bls, is_tce = initial_triage(tic, bls_dir)
            if is_tce:
                star_params = star_query(tic)
                tce_table.Epoc.loc[index] = bls['Tc']
                tce_table.Period.loc[index] = bls['Period']
                tce_table.Transit_Depth.loc[index] = bls['Depth']
                tce_table.Duration.loc[index] = bls['Qtran']*bls['Period']
                tce_table.star_rad.loc[index] = star_params['rad']
                tce_table.star_mass.loc[index] = star_params['mass']
                tce_table.Tmag.loc[index] = star_params['tmag']
        except:
            continue
    tce_table = tce_table[np.isfinite(tce_table['Period'])]
    return tce_table

def parallelize(data):
    partitions = nprocs
    data_split = np.array_split(data, partitions)
    pool = multiprocessing.Pool(processes=partitions)
    df = pd.concat(pool.map(process_tce, data_split))
    pool.close()
    pool.join()
    return df

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    c = catalog.TIC()
    tce_table = pd.read_csv(os.path.join(FLAGS.base_dir, FLAGS.input))
    if FLAGS.num_worker_processes == 1:
        tce_table = process_tce(tce_table)
    else:
        logger = multiprocessing.get_logger()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        logger.info('Process started')
        tce_table = parallelize(tce_table)
    tce_table.to_csv(os.path.join(FLAGS.base_dir, FLAGS.output))
    

