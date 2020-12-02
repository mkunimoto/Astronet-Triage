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
r"""Script to preprocesses data from the Kepler space telescope.

This script produces training, validation and test sets of labeled Kepler
Threshold Crossing Events (TCEs). A TCE is a detected periodic event on a
particular Kepler target star that may or may not be a transiting planet. Each
TCE in the output contains local and global views of its light curve; auxiliary
features such as period and duration; and a label indicating whether the TCE is
consistent with being a transiting planet. The data sets produced by this script
can be used to train and evaluate models that classify Kepler TCEs.

The input TCEs and their associated labels are specified by the DR24 TCE Table,
which can be downloaded in CSV format from the NASA Exoplanet Archive at:

  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce

The downloaded CSV file should contain at least the following column names:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  tce_period: Orbital period of the detected event, in days.
  tce_time0bk: The time corresponding to the center of the first detected
      traisit in Barycentric Julian Day (BJD) minus a constant offset of
      2,454,833.0 days.
  tce_duration: Duration of the detected transit, in hours.
  av_training_set: Autovetter training set label; one of PC (planet candidate),
      AFP (astrophysical false positive), NTP (non-transiting phenomenon),
      UNK (unknown).

The Kepler light curves can be downloaded from the Mikulski Archive for Space
Telescopes (MAST) at:

  http://archive.stsci.edu/pub/kepler/lightcurves.

The Kepler data is assumed to reside in a directory with the same structure as
the MAST archive. Specifically, the file names for a particular Kepler target
star should have the following format:

    .../${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

where:
  kep_id is the Kepler id left-padded with zeros to length 9;
  quarter_prefix is the file name quarter prefix;
  type is one of "llc" (long cadence light curve) or "slc" (short cadence light
    curve).

The output TFRecord file contains one serialized tensorflow.train.Example
protocol buffer for each TCE in the input CSV file. Each Example contains the
following light curve representations:
  global_view: Vector of length 2001; the Global View of the TCE.
  local_view: Vector of length 201; the Local View of the TCE.

In addition, each Example contains the value of each column in the input TCE CSV
file. Some of these features may be useful as auxiliary features to the model.
The columns include:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  av_training_set: Autovetter training set label.
  tce_period: Orbital period of the detected event, in days.
  ...
"""
import argparse
import multiprocessing
import os
import sys

from absl import logging
from absl import app
import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.preprocess import preprocess


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_tce_csv_file",
    type=str,
    required=True,
    help="CSV file containing the TESS TCE table. Must contain "
    "columns: row_id, tic_id, toi_id, Period, Duration, "
    "Epoc (t0).")

parser.add_argument(
    "--tess_data_dir",
    type=str,
    required=True,
    help="Base folder containing TESS data.")

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory in which to save the output.")

parser.add_argument(
    "--num_shards",
    type=int,
    default=20,
    help="Number of file shards to divide the training set into.")



def _set_float_feature(ex, tce, name, value):
  """Sets the value of a float feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  if isinstance(value, np.ndarray):
    value = value.reshape((-1,))
  values = [float(v) for v in value]
  if any(np.isnan(values)):
    raise ValueError(f'NaNs in {name} for {tce.tic_id}')
  ex.features.feature[name].float_list.value.extend(values)


def _set_bytes_feature(ex, name, value):
  """Sets the value of a bytes feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].bytes_list.value.extend([str(v).encode("latin-1") for v in value])


def _set_int64_feature(ex, name, value):
  """Sets the value of an int64 feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].int64_list.value.extend([int(v) for v in value])


def _process_tce(tce, bkspace=None):
  orig_time, orig_flux = preprocess.read_and_process_light_curve(
      tce.tic_id, FLAGS.tess_data_dir, 'RawMagnitude')
  ex = tf.train.Example()

  detrended_time, detrended_flux, _ = preprocess.detrend_and_filter(
      tce.tic_id, orig_time, orig_flux, tce.Period, tce.Epoc, tce.Duration, bkspace)
  time, flux, fold_num = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, tce.Period, tce.Epoc)
    
  global_view, global_std, global_mask, _ = preprocess.global_view(
      tce.tic_id, time, flux, tce.Period)
  local_view, local_std, local_mask, _ = preprocess.local_view(
      tce.tic_id, time, flux, tce.Period, tce.Duration)
  secondary_view, secondary_std, secondary_mask, _ = preprocess.secondary_view(
      tce.tic_id, time, flux, tce.Period, tce.Duration)

  sample_segments_view = preprocess.sample_segments_view(tce.tic_id, time, flux, fold_num, tce.Period)

  _set_float_feature(ex, tce, 'global_view', global_view)
  _set_float_feature(ex, tce, 'global_std', global_std)
  _set_float_feature(ex, tce, 'global_mask', global_mask)
  _set_float_feature(ex, tce, 'local_view', local_view)
  _set_float_feature(ex, tce, 'local_std', local_std)
  _set_float_feature(ex, tce, 'local_mask', local_mask)
  _set_float_feature(ex, tce, 'secondary_view', secondary_view)
  _set_float_feature(ex, tce, 'secondary_std', secondary_std)
  _set_float_feature(ex, tce, 'secondary_mask', secondary_mask)

  _set_float_feature(ex, tce, 'sample_segments_view', sample_segments_view)

  _set_float_feature(ex, tce, 'n_folds', [max(fold_num) if len(fold_num) else 0])
  _set_float_feature(ex, tce, 'n_points', [len(fold_num) if len(fold_num) else 0])

  time, flux, fold_num = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, tce.Period * 2, tce.Epoc - tce.Period / 2)
  global_view, _, _, _ = preprocess.global_view(tce.tic_id, time, flux, tce.Period * 2)
  _set_float_feature(ex, tce, 'global_view_double_period', global_view)

  time, flux, fold_num = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, tce.Period / 2, tce.Epoc)
  global_view, _, _, _ = preprocess.global_view(tce.tic_id, time, flux, tce.Period / 2)
  _set_float_feature(ex, tce, 'global_view_half_period', global_view)

  for bkspace_f in [0.3, 0.7, 1.5, 5.0]:
    time, flux, _ = preprocess.detrend_and_filter(
        tce.tic_id, orig_time, orig_flux, tce.Period, tce.Epoc, tce.Duration, bkspace_f)
    time, flux, fold_num = preprocess.phase_fold_and_sort_light_curve(time, flux, tce.Period, tce.Epoc)
    global_view, _, global_mask, _ = preprocess.global_view(tce.tic_id, time, flux, tce.Period)
    _set_float_feature(ex, tce, f'global_view_{bkspace_f}', global_view)
    _set_float_feature(ex, tce, f'global_view_{bkspace_f}_mask', global_mask)


  for col_name, value in tce.items():
    if col_name.lower() in ('tic_id', 'tic id', 'epoc', 'sectors') or col_name.startswith('disp_'):
        _set_int64_feature(ex, col_name, [int(value)])
    else:
        f_val = float(value)
        if np.isnan(f_val):
            _set_int64_feature(ex, f'{col_name}_present', [0])
            f_val = 0
        else:
            _set_int64_feature(ex, f'{col_name}_present', [1])
        _set_float_feature(ex, tce, col_name, [f_val])

  return ex


def _process_file_shard(tce_table, file_name):
  process_name = multiprocessing.current_process().name
  shard_name = os.path.basename(file_name)
  shard_size = len(tce_table)
    
  existing = {}
  try:
    tfr = tf.data.TFRecordDataset(file_name)
    for record in tfr:
      ex_str = record.numpy()
      ex = tf.train.Example.FromString(ex_str)
      existing[ex.features.feature["tic_id"].int64_list.value[0]] = ex_str
  except:
    pass

  with tf.io.TFRecordWriter(file_name) as writer:
    num_processed = 0
    num_skipped = 0
    num_existing = 0
    print("", end='')
    for _, tce in tce_table.iterrows():
      num_processed += 1
      print("\r                                      ", end="")
      print(f"\r[{num_processed}/{shard_size}] {tce['tic_id']}", end="")

      if int(tce["tic_id"]) in existing:
        print(" exists", end="")
        sys.stdout.flush()
        writer.write(existing[int(tce["tic_id"])])
        num_existing += 1
        continue

      try:
        print(" processing", end="")
        sys.stdout.flush()
        example = _process_tce(tce)
      except Exception as e:
        print(f" *** error: {e}")
        num_skipped += 1
        continue

      print(" writing                   ", end="")
      sys.stdout.flush()
      writer.write(example.SerializeToString())

  num_new = num_processed - num_skipped - num_existing
  print(f"\r{shard_name}: {num_processed}/{shard_size} {num_new} new {num_skipped} bad            ")


def main(_):
    tf.io.gfile.makedirs(FLAGS.output_dir)

    tce_table = pd.read_csv(
        FLAGS.input_tce_csv_file,
        header=0,
        dtype={'tic_id': str})

    num_tces = len(tce_table)
    logging.info("Read %d TCEs", num_tces)

    # Further split training TCEs into file shards.
    file_shards = []  # List of (tce_table_shard, file_name).
    boundaries = np.linspace(
        0, len(tce_table), FLAGS.num_shards + 1).astype(np.int)
    for i in range(FLAGS.num_shards):
      start = boundaries[i]
      end = boundaries[i + 1]
      file_shards.append((
          start,
          end,
          os.path.join(FLAGS.output_dir, "%.5d-of-%.5d" % (i, FLAGS.num_shards))
      ))

    logging.info("Processing %d total file shards", len(file_shards))
    for start, end, file_shard in file_shards:
        _process_file_shard(tce_table[start:end], file_shard)
    logging.info("Finished processing %d total file shards", len(file_shards))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
