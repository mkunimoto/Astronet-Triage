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

"""Configurations for model building, training and evaluation.

Available configurations:
  * base: One time series feature per input example. Default is "global_view".
  * local_global: Two time series features per input example.
      - A "global" view of the entire orbital period.
      - A "local" zoomed-in view of the transit event.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def base():
  """Returns the base config for model building, training and evaluation."""
  return {
      # Configuration for reading input features and labels.
      "inputs": {
          "labels_are_columns": False,
          "label_columns": [],
          
          # Feature specifications.
          "features": {
              "global_view": {
                  "length": 201,
                  "is_time_series": True,
              },
          },

          # Name of the feature containing training labels.
          "label_feature": "Disposition",

          # Label string to integer id.
          "label_map": {
              "PC": 1,  # Planet Candidate.
              "EB": 1,
              "J": 0,  # "Junk"
              "V": 0,  # Variable star
              "IS": 0, # Instrumental noise
              "O": 0, # Other
          },
          
          "primary_class": 1,
      },
      # Hyperparameters for building and training the model.
      "hparams": {
          # Number of output dimensions (predictions) for the classification
          # task. If >= 2 then a softmax output layer is used. If equal to 1
          # then a sigmoid output layer is used.
          "output_dim": 1,

          # Fully connected layers before the logits layer.
          "num_pre_logits_hidden_layers": 0,
          "pre_logits_hidden_layer_size": 0,
          "pre_logits_dropout_rate": 0.0,

          # Number of examples per training batch.
          "batch_size": 64,

          # Learning rate parameters.
          "learning_rate": 1e-5,

          # Optimizer for training the model.
          "optimizer": "adam",

          # If not None, gradient norms will be clipped to this value.
          "clip_gradient_norm": None,
      }
  }

def local_global():
  """Base configuration for a CNN model with separate local/global views."""
  config = base()

  # Override the model features to be local_view and global_view time series.
  config["inputs"]["features"] = {
      "local_view": {
          "length": 61,
          "is_time_series": True,
      },
      "global_view": {
          "length": 201,
          "is_time_series": True,
      },
  }

  config["hparams"]["time_series_hidden"] = {
      "local_view": {
          "cnn_num_blocks": 2,
          "cnn_block_size": 2,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 5,
          "convolution_padding": "same",
          "pool_size": 7,
          "pool_strides": 2,
      },
      "global_view": {
          "cnn_num_blocks": 5,
          "cnn_block_size": 2,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 5,
          "convolution_padding": "same",
          "pool_size": 5,
          "pool_strides": 2,
      },
  }

  config["hparams"]["prediction_threshold"] = 0.5

  config["hparams"]["learning_rate"] = 1e-3
  config["hparams"]["learning_rate_schedule"] = False
  config["hparams"]["one_minus_adam_beta_1"] = 0.1
  config["hparams"]["one_minus_adam_beta_2"] = 0.001
  config["hparams"]["adam_epsilon"] = 1e-7

  config["hparams"]["use_batch_norm"] = False
  config["hparams"]["pre_logits_dropout_rate"] = 0.0

  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 512

  config["tune_params"] = [
      {
          'parameter': 'use_batch_norm', 'type': 'CATEGORICAL',
          'categorical_value_spec' : {'values': ['True', 'False']}},
      {
          'parameter': 'prediction_threshold', 'type': 'DOUBLE',
          'double_value_spec' : {'min_value': 0.1, 'max_value': 0.5}},
      {
          'parameter': 'learning_rate', 'type': 'DOUBLE',
          'double_value_spec' : {'min_value': 1e-6, 'max_value': 1e-3},
          'scale_type': 'UNIT_LOG_SCALE'},
      {
          'parameter': 'one_minus_adam_beta_1', 'type': 'DOUBLE',
          'double_value_spec' : {'min_value': 1e-2, 'max_value': 0.9},
          'scale_type': 'UNIT_LOG_SCALE'},
      {
          'parameter': 'one_minus_adam_beta_2', 'type': 'DOUBLE',
          'double_value_spec' : {'min_value': 1e-4, 'max_value': 0.9},
          'scale_type': 'UNIT_LOG_SCALE'},
      {
          'parameter': 'adam_epsilon', 'type': 'DOUBLE',
          'double_value_spec' : {'min_value': 1e-8, 'max_value': 1e-5},
          'scale_type': 'UNIT_LOG_SCALE'},
      {
          'parameter': 'batch_size', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 8, 'max_value' : 128}},
      {
          'parameter': 'num_pre_logits_hidden_layers', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 4}},
      {
          'parameter': 'pre_logits_hidden_layer_size', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 32, 'max_value' : 1024}},
      {
          'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
          'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.4}},
      {
          'parameter': 'cnn_block_filter_factor', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
      {
          'parameter': 'cnn_block_size', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 4}},
      {
          'parameter': 'cnn_initial_num_filters', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 4, 'max_value' : 32}},
      {
          'parameter': 'cnn_kernel_size', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
      {
          'parameter': 'cnn_num_blocks_global', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
      {
          'parameter': 'cnn_num_blocks_local', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
      {
          'parameter': 'pool_size_global', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
      {
          'parameter': 'pool_size_local', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
      {
          'parameter': 'pool_strides', 'type' : 'INTEGER',
          'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}}
  ]
    
  return config


def local_global_new():
  config = local_global()

  config["inputs"]["labels_are_columns"] = True
  config["inputs"]["label_columns"] = ["disp_E", "disp_N", "disp_J", "disp_S", "disp_B"]
  config["inputs"]["primary_class"] = 0
  config["inputs"]["example_weighting"] = True

  config["inputs"]["features"]["secondary_view"] = {
      "length": 61,
      "is_time_series": True,
  }

  config["hparams"]["time_series_hidden"]["secondary_view"] = {
      "cnn_num_blocks": 2,
      "cnn_block_size": 2,
      "cnn_initial_num_filters": 16,
      "cnn_block_filter_factor": 2,
      "cnn_kernel_size": 5,
      "convolution_padding": "same",
      "pool_size": 7,
      "pool_strides": 2,
  }

  config["hparams"]["output_dim"] = 5

  config["inputs"]["features"]["Epoc"] = {
      "is_time_series": False,
      "mean": 1496.751505,
      "std": 154.114326,
      "length": 1,
      "has_nans": False,
  }

  config["inputs"]["features"]["Period"] = {
      "is_time_series": False,
      "mean": 8.071377,
      "std": 11.233816,
      "length": 1,
      "has_nans": False,
  }

  config["inputs"]["features"]["Duration"] = {
      "is_time_series": False,
      "mean": 0.196459,
      "std": 0.172065,
      "length": 1,
      "has_nans": False,
  }

  config["inputs"]["features"]["Transit_Depth"] = {
      "is_time_series": False,
      "mean": 3.847200e+05,
      "std": 3.220359e+07,
      "length": 1,
      "has_nans": False,
  }

  config["inputs"]["features"]["Tmag"] = {
      "is_time_series": False,
      "mean": 10.162480,
      "std": 1.225660,
      "length": 1,
      "has_nans": False,
  }

  config["inputs"]["features"]["star_mass"] = {
      "is_time_series": False,
      "mean": 1.382456,
      "std": 0.387535,
      "length": 1,
      "has_nans": True,
  }

  config["inputs"]["features"]["star_rad"] = {
      "is_time_series": False,
      "mean": 11.881122,
      "std": 19.495874,
      "length": 1,
      "has_nans": True,
  }

  return config


def local_global_new_tuned():
  config = local_global_new()

  # studies/and_yet_another_2_local_global_new
  config['hparams'] = {'adam_epsilon': 2.5037055725611666e-07,
     'batch_size': 83,
     'clip_gradient_norm': None,
     'learning_rate': 5.203528044134961e-06,
     'learning_rate_schedule': False,
     'num_pre_logits_hidden_layers': 4,
     'one_minus_adam_beta_1': 0.16168028483420177,
     'one_minus_adam_beta_2': 0.022674419033475692,
     'optimizer': 'adam',
     'output_dim': 5,
     'pre_logits_dropout_rate': 0.1690298097832756,
     'pre_logits_hidden_layer_size': 482,
     'prediction_threshold': 0.2152499407880693,
     'time_series_hidden': {'global_view': {'cnn_block_filter_factor': 2,
                                            'cnn_block_size': 1,
                                            'cnn_initial_num_filters': 17,
                                            'cnn_kernel_size': 2,
                                            'cnn_num_blocks': 3,
                                            'convolution_padding': 'same',
                                            'pool_size': 5,
                                            'pool_strides': 1},
                            'local_view': {'cnn_block_filter_factor': 2,
                                           'cnn_block_size': 1,
                                           'cnn_initial_num_filters': 17,
                                           'cnn_kernel_size': 2,
                                           'cnn_num_blocks': 3,
                                           'convolution_padding': 'same',
                                           'pool_size': 6,
                                           'pool_strides': 1},
                            'secondary_view': {'cnn_block_filter_factor': 2,
                                               'cnn_block_size': 1,
                                               'cnn_initial_num_filters': 17,
                                               'cnn_kernel_size': 2,
                                               'cnn_num_blocks': 3,
                                               'convolution_padding': 'same',
                                               'pool_size': 6,
                                               'pool_strides': 1}},
     'use_batch_norm': False}

  return config



def multiview():
  config = local_global_new_tuned()

  config['hparams']['time_series_hidden']['twice_local'] = config['hparams']['time_series_hidden']['local_view']
  config['hparams']['time_series_hidden']['half_local'] = config['hparams']['time_series_hidden']['local_view']

  return config
