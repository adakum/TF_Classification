from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys,os

import tensorflow as tf

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/data/adkuma/Ranking/Model/',
    help='Base directory for the model.')

parser.add_argument(
    '--dropout', type=str, default=0.2,
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='regressor',
    help="Valid model types: {'classifier', 'regressor'}.")

parser.add_argument(
    '--train_epochs', type=int, default=10, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=32, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='train_data.tsv',
    help='Path to the training data.')

parser.add_argument('--test_data', type=str, default='test_data.tsv', help='Path to the test data.')

parser.add_argument('--agi_dim', type=int, default=0, help='Dimesion of AGI Vector')

parser.add_argument('--rs_dim', type=int, default=99, help='Dimesion of AGI Vector')

parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning Rate')

parser.add_argument(
    '--predict', type=bool, default=False, help='Number of examples per batch.')

parser.add_argument(
    '--use_weight', type=bool, default=True, help='Use weight per sample')


def AGI_model_columns(agi_dim):

  total_cols = 2*agi_dim
  agi_cols = []
  for i in range(total_cols):
    agi_cols = agi_cols + [tf.feature_column.numeric_column("agi_feature_{}".format(i))]    

  # agi_cols = agi_cols  + [tf.feature_column.numeric_column("weight_column")]
  return agi_cols


def RS_model_columns(dim):
  total_cols = dim
  rs_cols = []
  for i in range(total_cols):
    rs_cols = rs_cols + [tf.feature_column.numeric_column("rs_feature_{}".format(i))]      
  return rs_cols

def build_estimator(model_dir, model_type, agi_dim=0, rs_dim=0, use_weight=False):
  
  # all  columns names 
  model_columns = []
  agi_columns = []
  rs_columns = []

  # if agi columns present
  if agi_dim > 0:
    agi_columns = AGI_model_columns(agi_dim)
    model_columns = model_columns + agi_columns

  # if RS features present
  if rs_dim > 0:
    rs_columns = RS_model_columns(rs_dim)
    model_columns = model_columns + rs_columns


  # hidden layers 
  hidden_units = [100]

  # if classification task 
  if model_type == "classification":
    # if weight column present
    if use_weight == True:
      
      return tf.estimator.DNNClassifier(
        model_dir = model_dir,
        feature_columns = model_columns,
        weight_column = "weight",
        hidden_units = hidden_units
        ,config = tf.estimator.RunConfig(save_checkpoints_steps = 5000, keep_checkpoint_max = 15)
        ,activation_fn = tf.nn.tanh
        )
      # , session_config = tf.ConfigProto(log_device_placement=True

    else :
      
      return tf.estimator.DNNClassifier(
        model_dir = model_dir,
        feature_columns = model_columns,
        hidden_units = hidden_units,
        dropout = FLAGS.dropout
        )

  elif model_type == "regression":
    
    return tf.estimator.DNNRegressor(
      model_dir = model_dir,
      feature_columns = model_columns,
      # weight_column = "weight_column",
      hidden_units= hidden_units,
      dropout= FLAGS.dropout
      )


def input_fn(data_file, num_epochs, shuffle, batch_size, agi_dim=0, rs_dim=0, weight=True):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  if weight == True:
    _CSV_COLUMNS = ["query", "question"] + ["label"] + ["weight"]
  else :
    _CSV_COLUMNS = ["query", "question"] + ["label"] 

  
  total_features  = 0 
  if agi_dim > 0:
    _CSV_COLUMNS =  _CSV_COLUMNS + [ "agi_feature_{}".format(i) for i in range(2*agi_dim)] 
    total_features = total_features + 2*agi_dim

  if rs_dim > 0:
    _CSV_COLUMNS =  _CSV_COLUMNS + [ "rs_feature_{}".format(i) for i in range(rs_dim)] 
    total_features = total_features + rs_dim

  # query + question + label + weight + features ....  
  if weight :
    _CSV_COLUMN_DEFAULTS = [[""], [""]] + [[0.0]] + [[0.0]] + [[0.0] for i in range(total_features)]
  else :
    _CSV_COLUMN_DEFAULTS = [[""], [""]] + [[0.0]] + [[0.0]] + [[0.0] for i in range(total_features)]

  def parse_csv(value):
    print('Parsing', data_file)
    columns  = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim="\t")
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('label')
    query  = features.pop('query')
    question = features.pop('question')
    print("Done Parsing")
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size = 1024)

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels


def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type, FLAGS.agi_dim, FLAGS.rs_dim, FLAGS.use_weight)

  if FLAGS.predict == True:
    # evaluate : 
    results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, FLAGS.epochs_per_eval, False, FLAGS.batch_size, FLAGS.agi_dim, FLAGS.rs_dim, FLAGS.use_weight))
    for key in sorted(results):
      print("{0}: {1}".format(key, results[key]))

    predictions = list(model.predict(input_fn=lambda: input_fn(FLAGS.test_data, FLAGS.epochs_per_eval, False, FLAGS.batch_size, FLAGS.agi_dim, FLAGS.rs_dim, FLAGS.use_weight)))
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  else :
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
      
      print("Starting Epoch {}".format(n))
      print('-'*60)
      
      model.train(input_fn=lambda: input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size, FLAGS.agi_dim, FLAGS.rs_dim, FLAGS.use_weight))

      print("Evaluation Starting !")
      results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size, FLAGS.agi_dim, FLAGS.rs_dim, FLAGS.use_weight))

      x = model.predict(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size, FLAGS.agi_dim, FLAGS.rs_dim, FLAGS.use_weight))
      print(list(x))
      # Display evaluation metrics
      print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
      print('-' * 60)

      for key in sorted(results):
        print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
