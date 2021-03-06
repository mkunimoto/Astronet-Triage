# Copyright 2020 The TESS team.
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

"""Script for tuning an AstroNet model.

Required PIP packages:

  google-cloud
  google-cloud-storage
  google-api-python-client
  requests
  tf
"""

import argparse
import datetime
import json
import logging
import time
import os
import pprint
import sys

from absl import app
from absl import logging as absl_logging
from apiclient import errors
from google.cloud import storage
from googleapiclient import discovery
from google_auth_oauthlib import flow

from astronet import train
from astronet import models
from astronet.util import config_util
from astronet.util import configdict


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
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
    "--train_steps",
    type=int,
    default=12000,
    help="Total number of steps to train the model for.")

parser.add_argument(
    "--train_epochs",
    type=int,
    default=1,
    help="Leave this set to 1.")

parser.add_argument(
    "--shuffle_buffer_size",
    type=int,
    default=15000,
    help="Size of the shuffle buffer for the training dataset.")

parser.add_argument(
    "--client_secrets",
    type=str,
    required=True,
    help="OAuth secrets file, see https://github.com/googleapis/"
    "google-api-python-client/blob/master/docs/client-secrets.md.")

parser.add_argument(
    "--study_id",
    type=str,
    default="new_tess_lab_{}".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
    help="Unique identifier string for the study.")

parser.add_argument(
    "--tune_trials",
    type=int,
    default=1000,
    help="Total number of trials to tunr the model for.")

parser.add_argument(
    "--ensemble_count",
    type=int,
    default=3,
    help="Model ensemble size.")


REGION = 'us-central1'

CLOUD_PROJECT_ID = os.environ["CLOUD_PROJECT_ID"]


def study_parent():
  return 'projects/{}/locations/{}'.format(CLOUD_PROJECT_ID, REGION)


def study_id():
  return '{}_{}'.format(FLAGS.study_id, FLAGS.config_name)


def study_name():
  return '{}/studies/{}'.format(study_parent(), study_id())


def trial_parent():
  return study_name()


def trial_name(trial_id):
  return '{}/trials/{}'.format(study_name(), trial_id)


def operation_name(operation_id):
  return 'projects/{}/locations/{}/operations/{}'.format(
      CLOUD_PROJECT_ID, REGION, operation_id)


def study_config(config):
  metrics = [
      {'metric': 'r', 'goal': 'MAXIMIZE'},
      {'metric': 'p', 'goal': 'MAXIMIZE'}
  ]

  return {
      'study_config': {
          'algorithm' : 'ALGORITHM_UNSPECIFIED',
          'parameters' : config['tune_params'],
          'metrics' : metrics,
          'max_trial_count' : FLAGS.tune_trials,
      }
  }


def initialize_client():
  appflow = flow.InstalledAppFlow.from_client_secrets_file(
      FLAGS.client_secrets,
      scopes=['https://www.googleapis.com/auth/cloud-platform'])
  appflow.run_console()
  credentials = appflow.credentials

  client = storage.Client(CLOUD_PROJECT_ID)
  bucket = client.get_bucket('caip-optimizer-alpha-public')
  blob = bucket.get_blob('api/alpha-ml_google_rest_v1.json')
  service = blob.download_as_string()
  return discovery.build_from_document(
      service=service, credentials=credentials)


def create_study(client, study):
  req = client.projects().locations().studies().create(
      parent=study_parent(), studyId=study_id(), body=study)
  try:
    return req.execute()
  except errors.HttpError as e: 
    if e.resp.status != 409:  # Study already exists
      raise e
    

# FIXME
def map_param(hparams, param):
  name = param['parameter']
  if name in (
      'learning_rate', 'pre_logits_dropout_rate', 'prediction_threshold',
      'one_minus_adam_beta_1', 'one_minus_adam_beta_2', 'adam_epsilon'):
    hparams[name] = param['floatValue']
  elif name in (
    'use_batch_norm',):
    hparams[name] = (param['stringValue'].lower() == 'true')
  elif name in (
    'batch_size', 'num_pre_logits_hidden_layers', 'pre_logits_hidden_layer_size'):
    hparams[name] = int(param['intValue'])
  elif name in (
    'cnn_block_filter_factor', 'cnn_block_size', 'cnn_initial_num_filters',
    'cnn_kernel_size', 'pool_strides'):      
    hparams['time_series_hidden']['local_view'][name] = int(param['intValue'])
    hparams['time_series_hidden']['secondary_view'][name] = int(param['intValue'])
    hparams['time_series_hidden']['global_view'][name] = int(param['intValue'])
  elif name in ('cnn_num_blocks_global', 'pool_size_global'):
    hparams['time_series_hidden']['global_view'][name[:-7]] = int(param['intValue'])
  elif name in ('cnn_num_blocks_local', 'pool_size_local'):
    hparams['time_series_hidden']['local_view'][name[:-6]] = int(param['intValue'])
    hparams['time_series_hidden']['secondary_view'][name[:-6]] = int(param['intValue'])
  else:
    raise ValueError('param missing from tune.map_param' + str(param))


def execute_trial(trial_id, params, model_class, config, ensemble_count):
  print(("=========== Start Trial: [{0}] =============").format(trial_id))
  for param in params:
    map_param(config['hparams'], param)
  
  ensemble_val_r = []
  ensemble_val_p = []
  for _ in range(ensemble_count):
    model = model_class(config)
    history = train.train(model, config).history

    val_r = history['val_r'][-1]
    val_p = history['val_p'][-1]

    if val_r < 0.001:
      val_r = 0.001
    if val_p < 0.001:
      val_p = 0.001
        
    ensemble_val_r.append(val_r)
    ensemble_val_p.append(val_p)

    # Only ensemble promising models.
    if (val_r < 0.5) or (val_p < 0.2):
      break

  # Select metric with poorest val_r.
  selected = 0
  for i, (val_r, val_p) in enumerate(zip(ensemble_val_r, ensemble_val_p)):
    if val_r < ensemble_val_r[selected]:
      selected = i
  val_r = ensemble_val_r[selected]
  val_p = ensemble_val_p[selected]

  metric_r = {'metric': 'r', 'value': float(val_r)}
  metric_p = {'metric': 'p', 'value': float(val_p)}
  measurement = {'step_count': 1, 'metrics': [metric_r, metric_p]}
  return measurement


def tune(client, model_class, config, ensemble_count):
  client_id = 'client1'
  suggestion_count_per_request =  1
  max_trial_id_to_stop = FLAGS.tune_trials

  trial_id = 0
  while trial_id < max_trial_id_to_stop:
    resp = client.projects().locations().studies().trials().suggest(
        parent=trial_parent(), 
        body={
            'client_id': client_id,
            'suggestion_count': suggestion_count_per_request}
    ).execute()
    op_id = resp['name'].split('/')[-1]

    # Poll the suggestion long-running operations.
    get_op = client.projects().locations().operations().get(
        name=operation_name(op_id))
    while True:
      operation = get_op.execute()
      if 'done' in operation and operation['done']:
        break
      time.sleep(1)

    for suggested_trial in operation['response']['trials']:
      trial_id = int(suggested_trial['name'].split('/')[-1])

      # Featch the suggested trials.
      trial = client.projects().locations().studies().trials().get(
          name=trial_name(trial_id)
      ).execute()
      if trial['state'] in ['COMPLETED', 'INFEASIBLE']:
        continue
    
      try:
        measurement = execute_trial(
            trial_id, trial['parameters'], model_class, config, ensemble_count)
        feasible = True
      except Exception as e:
        print(type(e), e)
        measurement = None
        feasible = False
        infeasible_reason = str(e)

      if feasible:
        client.projects().locations().studies().trials().addMeasurement(
            name=trial_name(trial_id), 
            body={'measurement': measurement}
        ).execute()
        client.projects().locations().studies().trials().complete(
          name=trial_name(trial_id)
        ).execute()
      else:
        client.projects().locations().studies().trials().complete(
          name=trial_name(trial_id),
          body={'trialInfeasible': True, 'infeasibleReason': infeasible_reason}
        ).execute()        


def main(_):
  config = models.get_model_config(FLAGS.model, FLAGS.config_name)

  model_class = models.get_model_class(FLAGS.model) 
  study = study_config(config)

  client = initialize_client()
  print('Study:')
  pprint.pprint(create_study(client, study))

  tune(client, model_class, config, FLAGS.ensemble_count)
    
  print('All done. Study name:', study_name())


if __name__ == "__main__":
  logger = logging.getLogger().setLevel(logging.WARNING)
  absl_logging.set_verbosity(absl_logging.WARNING)

  FLAGS, unparsed = parser.parse_known_args()
  train.FLAGS = FLAGS
  train.FLAGS.model_dir = ""
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
