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

"""Script to download additional data with astroquery."""

from absl import app
from astroquery import mast
import numpy as np
import pandas as pd


def download_data(tic_id, ext_catalog):
    catalog_data = mast.Catalogs.query_object('TIC {}'.format(tic_id), catalog="TIC", radius='0.1s')
    for i, r_id in enumerate(catalog_data['ID']):
        if r_id == str(tic_id):
            ext_catalog.at[tic_id, 'objType'] = catalog_data['objType'][i]


def download_all_data():
    tces_file = '../tce_bls_instar.csv'
    tce_table = pd.read_csv(tces_file, header=0).set_index('tic_id')

    try:
        ext_catalog = pd.read_csv('../ext_mast_data.csv', header=0).set_index('tic_id')
        print('Loaded existing catalog with {} records'.format(len(ext_catalog.index)))
    except:
        ext_catalog = pd.DataFrame(
            {
                'objType': np.array([], dtype=np.str),
                'tic_id': np.array([], dtype=np.int64),
            }
        ).set_index('tic_id')
        print('Created new catalog')

    n = len(tce_table)
    print('...', end='')
    for i, tic_id in enumerate(tce_table.index):
        if tic_id in ext_catalog.index:
            continue
        download_data(tic_id, ext_catalog)
        print('\rDone {}/{}'.format(i, n), end='')
        if i % 500 == 0:
            ext_catalog.to_csv('../ext_mast_data.csv')
            
    print('\nAll done.')        

    ext_catalog.to_csv('../ext_mast_data.csv')
    
    
def main(_):
    download_all_data()


if __name__ == "__main__":
    app.run(main)
