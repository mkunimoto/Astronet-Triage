import numpy as np
import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base_dir",
    type=str,
    default='/pdo/users/mkuni',
    help="Directory where TCE lists are located.")

parser.add_argument(
    '--input',
    type=str,
    help='txt file containing the TIC IDs of TCEs that will go into CSV table.',
    required=True)

parser.add_argument(
    '--output',
    type=str,
    help='name of output CSV table',
    required=True)

parser.add_argument(
    "--save_dir",
    type=str,
    default='/pdo/users/mkuni',
    help="Directory where CSV file will be generated.")

parser.add_argument(
    "--sector",
    type=int,
    help='sector number')

parser.add_argument(
    "--cam",
    type=int,
    help="camera number")

parser.add_argument(
    "--ccd",
    type=int,
    help="ccd number")

FLAGS, unparsed = parser.parse_known_args()

columns = ['tic_id','Tmag','Period','Epoc','Transit_Depth','Duration','Sectors','Camera','CCD']
tics = np.loadtxt(FLAGS.input, dtype=int)
lc_tces = pd.DataFrame(columns=columns)
lc_tces['tic_id'] = tics
lc_tces['Sectors'] = FLAGS.sector
lc_tces['Camera'] = FLAGS.cam
lc_tces['CCD'] = FLAGS.ccd
lc_tces.to_csv(os.path.join(FLAGS.save_dir, FLAGS.output),index=False)