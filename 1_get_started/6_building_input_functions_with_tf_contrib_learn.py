"""
TensorFlow > Develop > GET STARTED > Building Input Functions with tf.contrib.learn
https://www.tensorflow.org/get_started/input_fn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# 1. Importing the housing data
COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age',
           'dis', 'tax', 'ptratio', 'medv']
FEATURES = ['crim', 'zn', 'indus', 'nox', 'rm',
            'age', 'dis', 'tax', 'ptratio']
LABEL = 'medv'

train_set = pd.read_csv('./../dataset/boston/boston_train.csv', skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv('./../dataset/boston/boston_test.csv', skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv('./../dataset/boston/boston_predict.csv', skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

# 2. Defining FeatureColumns and creating the regressor
feature_cols = [tf.contrib.layers.real_valued_column(k)
                for k in FEATURES]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir='./restored_model/6')

# 3. Building the input_fn
def input_fn(data_set):
    # Preprocess your data here...
    # nothing

    # ... then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    feature_cols = {k: tf.constant(data_set[k].values)
                    for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels

# 4. Training the regressor
regressor.fit(input_fn=lambda: input_fn(train_set), steps=5000)

# 5. Evaluating the model
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev['loss']
print('Loss: {0:f}'.format(loss_score))

# 6. Making predictions
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list an dpring predictions
predictions = list(itertools.islice(y, 6))
print('Predictions: {}'.format(str(predictions)))