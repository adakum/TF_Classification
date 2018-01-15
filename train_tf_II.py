from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys, os, time

import tensorflow as tf

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument(
    '--model_dir', type=str, default='/data/adkuma/Ranking/Model_II/',
    help='Base directory for the model.')

parser.add_argument(
    '--dropout', type=str, default=0.2,
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=10, help='Number of training epochs.')


parser.add_argument(
    '--steps_per_eval', type=int, default=1000, help='The number of training epochs to run between evaluations.')


parser.add_argument(
    '--batch_size', type=int, default=256, help='Number of examples per batch.')

parser.add_argument(
    '--optimizer', type=str, default="adam", help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='sampleModel.tsv', help='Path to the training data.')

parser.add_argument('--test_data', type=str, default='sampleModel.tsv', help='Path to the test data.')

# parser.add_argument('--agi_dim', type=int, default=100, help='Dimesion of AGI Vector')

parser.add_argument('--rs_dim', type=int, default=99, help='RS Features')

parser.add_argument('--total_features', type=int, default=99, help='Total Features')

parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning Rate')

parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle Training Dataset')

parser.add_argument('--bnorm_before_activation', type=int, default=1, help='Batch Norm before non linear layer')

parser.add_argument('--n_classes', type=int, default=2, help='Num Classes')

parser.add_argument('--just_evaluate', type=int, default=0, help='Num Classes')

parser.add_argument('--hidden_units', type=str, default="50,25", help='Hidden units')

def train_input_fn(data_file, num_epochs, shuffle, batch_size, total_features=0, weight=True):
  """Generate an input function for the Estimator."""


  _CSV_COLUMNS = ["query", "question"] + ["label"] + ["weight"]
  _CSV_COLUMNS =  _CSV_COLUMNS + [ "rs_feature_{}".format(i) for i in range(total_features)] 
  # query + question + label + weight + features ....  
  _CSV_COLUMN_DEFAULTS = [[""], [""]] + [[0]] + [[0.0]] + [[0.0] for i in range(total_features)]

  def parse_csv(value):
    print('Parsing', data_file)
    columns  = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim="\t")
    features = dict(zip(_CSV_COLUMNS, columns))
    labels   = features.pop('label')
    query    = features.pop('query')
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

def my_model(features, labels, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # get the weight column
    weights = features[params['weight']]

    # hidden units 
    count = 0
    for units in params['hidden_units']:
        # activation None 
        net = tf.layers.dense(net, units=units, activation=None , name="dense_"+str(count))
        count = count + 1
        
        if params["bnorm_before_activation"]==1:
          # add batch normalization
          net = tf.layers.batch_normalization(net, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
          # add activation
          net = tf.nn.relu(net, 'relu')
        else:
          # add activation
          net = tf.nn.relu(net, 'relu')
          # add batch normalization
          net = tf.layers.batch_normalization(net, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))


    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None, name="dense_logits")

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
            'weight': weights,
            # 'labels' : tf.ones([10]),
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    one_hot_labels = tf.one_hot(labels, params['n_classes'], 1, 0)
    
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits = logits, weights = weights, reduction = tf.losses.Reduction.MEAN)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op',
                                   weights = weights)
    # AUC calculation
    auc = tf.metrics.auc(  labels = labels,
                           predictions = tf.nn.softmax(logits)[:,1],
                           name='auc_op',
                           weights = weights,
                           num_thresholds=10000
                           )
    # positive class mean
    pos_class_mean = tf.metrics.mean(values = tf.nn.softmax(logits)[:,1],
                                     weights = weights)


    w_sum = tf.metrics.mean(weights)
    metrics = {'accuracy': accuracy, 'weight' : w_sum, 'AUC':auc, 'pos_class_prob_mean':pos_class_mean}

    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('AUC', auc[1])
    tf.summary.scalar('pos_class_prob_mean', pos_class_mean[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    if FLAGS.optimizer =="adam":
      # Option : Try changing epsilon, as mentioned
      optimizer = tf.train.AdamOptimizer()
    elif FLAGS.optimizer == "adagrad":
      optimizer = tf.train.AdagradOptimizer()
    elif FLAGS.optimizer == "sgd":
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op  = optimizer.minimize(loss, global_step = tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):

    print("Goint to Empty Model DIR in 5 secs ")
    time.sleep(5)
    
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True) 
    
    # write arguments in ModelDir : 
    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)
      print("Directory Created {}".format(FLAGS.model_dir))

    with open(os.path.join(FLAGS.model_dir,"arguments"), "w") as writer:
      for key,val in vars(FLAGS).items():
        writer.write("--{0}={1}\n".format(key,val))

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for i in range(FLAGS.total_features):
        my_feature_columns.append(tf.feature_column.numeric_column("rs_feature_{}".format(i)))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn = my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [int(x) for x in FLAGS.hidden_units.split(',')],
            'n_classes': FLAGS.n_classes,
            'learning_rate' : FLAGS.learning_rate,
            'weight' : "weight",
            'bnorm_before_activation' : FLAGS.bnorm_before_activation
        },
        model_dir=FLAGS.model_dir)


    if FLAGS.just_evaluate==0:
    # Train the Model.
      while True:
        classifier.train(input_fn= lambda: train_input_fn(FLAGS.train_data, FLAGS.train_epochs, FLAGS.shuffle, FLAGS.batch_size,  total_features=FLAGS.total_features, weight=True)
          , steps = FLAGS.steps_per_eval)
        eval_result = classifier.evaluate(input_fn = lambda: train_input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size, total_features = FLAGS.total_features, weight = True))
              
      print("Done Training !!")
    
    
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    predictions = classifier.predict(input_fn = lambda: train_input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size, total_features = FLAGS.total_features, weight = True))

    print("Done Prediction !!")

    for pred_dict in (predictions):
        class_id    = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        weight      = pred_dict['weight']
        # labels      = pred_dict['labels']

        print("classId {0} probability {1}  weights {2}".format(class_id, probability, weight))


def main_(argv):
  features, labels = train_input_fn(FLAGS.train_data, FLAGS.train_epochs, FLAGS.shuffle, FLAGS.batch_size, total_features=FLAGS.total_features, weight=True)
  sess = tf.Session()
  x, y = sess.run([features, labels])
  print(x)
  print(y)
  print("weight {}".format(x["weight"]))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    print("DO VERIFY")
    print("-"*60)
    for key, value in vars(FLAGS).items():
      print("key {0} value {1}".format(key, value))
    print("-"*60)
    time.sleep(10)
    tf.app.run(main)