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

"""A TensorFlow model for identifying exoplanets in astrophysical light curves.

AstroModel is a concrete base class for models that identify exoplanets in
astrophysical light curves. This class implements a simple fully-connected neural network model that can be extended by subclasses.

The general framework for AstroModel and its subclasses is as follows:

  * Model inputs:
     - Zero or more time_series_features (e.g. astrophysical light curves)
     - Zero or more aux_features (e.g. orbital period, transit duration)

  * Labels:
     - An integer feature with 2 or more values (eg. 0 = Not Planet, 1 = Planet)

  * Model outputs:
     - The predicted probabilities for each label

  * Architecture:

                         predictions
                              ^
                              |
                           logits
                              ^
                              |
                   (pre_logits_hidden_layers)
                              ^
                              |
                       pre_logits_concat
                              ^
                              |
                        (concatenate)
                ^                           ^
                |                           |
     (time_series_hidden_layers)    (aux_hidden_layers)
                ^                           ^
                |                           |
       time_series_features           aux_features


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import tensorflow as tf


class AstroModel(tf.keras.Model):

  def __init__(self, config):
    super(AstroModel, self).__init__()

    self.ts_blocks = self._create_ts_blocks(config)
    self.aux_block = self._create_aux_block(config)

    self.final = [
      tf.keras.layers.Concatenate()
    ]

    for i in range(config['hparams']['num_pre_logits_hidden_layers']):
      self.final.append(
          tf.keras.layers.Dense(
              units=config['hparams']['pre_logits_hidden_layer_size'],
              activation='relu'))
      if config['hparams']['use_batch_norm']:
        self.final.append(tf.keras.layers.BatchNormalization())
      self.final.append(
          tf.keras.layers.Dropout(
              config['hparams']['pre_logits_dropout_rate']))

    if config['hparams']['output_dim'] == 1:
      self.final.append(
          tf.keras.layers.Dense(
              units=config['hparams']['output_dim'],
              activation='sigmoid'))
    else:
      self.final.append(
          tf.keras.layers.Dense(
              units=config['hparams']['output_dim'],
              activation='sigmoid'))
        

  def _create_ts_blocks(self, config):
      raise NotImplementedError()

  def _create_aux_block(self, config):
      raise NotImplementedError()

  def _apply_block(self, block, input_, training):
    y = input_
    for layer in block:
      y = layer(y, training=training)
    return y

  def call(self, inputs, training=None):
    ts_inputs = {}
    aux_inputs = {}
    for k, v in inputs.items():
        if k in ('global_view', 'local_view', 'secondary_view'):
            ts_inputs[k] = v
        else:
            aux_inputs[k] = v
    y = [
      self._apply_block(
          self.ts_blocks[k], v, training) for k, v in ts_inputs.items()]
    y.extend(aux_inputs.values())
    y = self._apply_block(self.final, y, training)
    
    return y

