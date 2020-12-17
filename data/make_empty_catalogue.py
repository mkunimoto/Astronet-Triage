import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir', type=str, help='Directory where TCE lists are located', required=True)
parser.add_argument('--input', type=str, help='txt file containing the TIC IDs of TCEs that will go into CSV table', required=True)
parser.add_argument('--save_dir', type=str, help='Directory where CSV file will be generated', required=True)
parser.add_argument('--sector', type=int, default=1, help='the sector for which this catalog is generating')
parser.add_argument('--cam', type=int, default=1, help='the cam for which this catalog is generating')
parser.add_argument('--ccd', type=int, default=1, help='the ccd for which this catalog is generating')
parser.add_argument('--output', help='the name of the output CSV table')

FLAGS, unparsed = parser.parse_known_args()

columns = ['tic_id', 'RA', 'Dec', 'Tmag', 'Period', 'Epoc', 'Transit_Depth', 'Duration', 'Sectors', 'Camera', 'CCD']
tics = np.loadtxt(os.path.join(FLAGS.base_dir, FLAGS.input), dtype=int)
tces = pd.DataFrame(columns=columns)
tces['tic_id'] = tics
tces['Sectors'] = FLAGS.sector
tces['Camera'] = FLAGS.cam
tces['CCD'] = FLAGS.ccd
if FLAGS.output is not None:
    fname = FLAGS.output
else:
    fname = 's%i_cam%i_ccd%i.csv' % (FLAGS.sector, FLAGS.cam, FLAGS.ccd)
tces.to_csv(os.path.join(FLAGS.save_dir, fname), index=False)
