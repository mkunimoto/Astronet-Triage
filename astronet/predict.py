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

"""Generates predictions using a trained model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from absl import app
import numpy as np
import tensorflow as tf
import pandas as pd

from astronet.ops import dataset_ops
from astronet.util import config_util


parser = argparse.ArgumentParser()


parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Directory containing a model checkpoint.")

parser.add_argument(
    "--data_files",
    type=str,
    required=True,
    help="Comma-separated list of file patterns matching the TFRecord files.")

parser.add_argument(
    "--output_file",
    type=str,
    default='',
    help="Name of file in which predictions will be saved.")


def predict():
    model = tf.saved_model.load(FLAGS.model_dir)
    config = config_util.load_config(FLAGS.model_dir)

    ds = dataset_ops.build_dataset(
        file_pattern=FLAGS.data_files,
        input_config=config['inputs'],
        batch_size=1,
        include_labels=False,
        reverse_time_series_prob=0,
        shuffle_filenames=False,
        repeat=1,
        use_tpu=False,
        one_hot_labels=(config['hparams']['output_dim'] > 1),
        include_identifiers=True)
    
    if config.hparams.output_dim > 1:
      if config.inputs.labels_are_columns:
        label_index = {i:k for i, k in enumerate(config.inputs.label_columns)}
      else:
        label_index = {v:k for k, v in config.inputs.label_map.items()}
    else:
      print('Binary prediction threshold: {} (orientative)'.format(
              config.hparams.prediction_threshold))
      label_index = {0: 'PC_prob'}

    print('0 records', end='')
    series = []
    for features, identifiers in ds:
      preds = model(features)

      row = {}
      for k, v in identifiers.items():
        row[k] = v.numpy().item()
      for i, p in enumerate(preds.numpy()[0]):
        row[label_index[i]] = p

      series.append(row)
      print('\r{} records'.format(len(series)), end='')

    results = pd.DataFrame.from_dict(series)
    
    if FLAGS.output_file:
      with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
        results.to_csv(f)
        
    return results, config


def main(_):
    predict()


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)