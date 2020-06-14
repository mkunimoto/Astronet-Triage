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

"""Script for training an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import sys

from absl import app
from absl import logging

import tensorflow as tf

from astronet import models
from astronet.ops import dataset_ops
from astronet.util import config_util

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
    required=True,
    help="Name of the model and training configuration.")

parser.add_argument(
    "--train_files",
    type=str,
    required=True,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the training dataset.")

parser.add_argument(
    "--eval_files",
    type=str,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the validation dataset.")

parser.add_argument(
    "--model_dir",
    type=str,
    default="",
    help="Directory for model checkpoints and summaries.")

parser.add_argument(
    "--train_steps",
    type=int,
    default=12000,
    help="Total number of steps to train the model for.")

parser.add_argument(
    "--train_epochs",
    type=int,
    default=1,
    help="Total number of epochs to train the model for.")

parser.add_argument(
    "--shuffle_buffer_size",
    type=int,
    default=15000,
    help="Size of the shuffle buffer for the training dataset.")


def train(model, config):
  if FLAGS.model_dir:
    dir_name = "{}/{}_{}_{}".format(
        FLAGS.model_dir,
        FLAGS.model,
        FLAGS.config_name,
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    config_util.log_and_save_config(config, dir_name)

  ds = dataset_ops.build_dataset(
      file_pattern=FLAGS.train_files,
      input_config=config['inputs'],
      batch_size=config['hparams']['batch_size'],
      include_labels=True,
      reverse_time_series_prob=0,
      shuffle_filenames=True,
      shuffle_values_buffer=FLAGS.shuffle_buffer_size,
      repeat=None,
      use_tpu=False,
      one_hot_labels=(config['hparams']['output_dim'] > 1))

  if FLAGS.eval_files:
    eval_ds = dataset_ops.build_dataset(
        file_pattern=FLAGS.eval_files,
        input_config=config['inputs'],
        batch_size=config['hparams']['batch_size'],
        include_labels=True,
        reverse_time_series_prob=0,
        shuffle_filenames=False,
        repeat=1,
        use_tpu=False,
        one_hot_labels=(config['hparams']['output_dim'] > 1))
  else:
    eval_ds = None

  assert config['hparams']['optimizer'] == 'adam'
  if config.hparams.learning_rate_schedule:
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=config.hparams.learning_rate_steps,
        values=config.hparams.learning_rate_value)
  else:
    lr = config['hparams']['learning_rate']
    beta_1 = 1.0 - config['hparams']['one_minus_adam_beta_1']
    beta_2 = 1.0 - config['hparams']['one_minus_adam_beta_2']
    epsilon = config['hparams']['adam_epsilon']
  optimizer=tf.keras.optimizers.Adam(
      learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

  loss = tf.keras.losses.BinaryCrossentropy()

  if config['hparams']['output_dim'] == 1:
    metrics = [
      tf.keras.metrics.Recall(name='r', thresholds=config['hparams']['prediction_threshold']),
      tf.keras.metrics.Precision(name='p', thresholds=config['hparams']['prediction_threshold']),
    ]
  else:
    metrics = [
        tf.keras.metrics.Recall(
            name='r',
            class_id=config.inputs.primary_class,
            thresholds=config['hparams']['prediction_threshold'],
        ),
        tf.keras.metrics.Precision(
            name='p',
            class_id=config.inputs.primary_class,
            thresholds=config['hparams']['prediction_threshold'],
        ),
    ]
    
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  history = model.fit(
      ds,
      epochs=FLAGS.train_epochs,
      steps_per_epoch=FLAGS.train_steps,
      validation_data=eval_ds)

  if FLAGS.model_dir:
    tf.saved_model.save(model, export_dir=dir_name)
    print("Model saved:\n    {}\n".format(dir_name))

  return history


def main(_):
  config = models.get_model_config(FLAGS.model, FLAGS.config_name)

  model_class = models.get_model_class(FLAGS.model) 
  model = model_class(config)
    
  train(model, config)
    
  # TODO: Add TB callback


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
