import tensorflow as tf

# Encoding sparse columns
eye_color = tf.contrib.layers.sparse_column_with_keys(
    column_name='eye_color', keys=['blue', 'brown', 'green'])
education = tf.contrib.layers.sparse_column_with_hash_bucket(
    'education', hash_bucket_size=1000)

# Feature crosses
sport = tf.contrib.layers.sparse_column_with_hash_bucket(
    'sport', hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket(
    'city', hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column(
    [sport, city], hash_bucket_size=int(1e4))

# Continuous columns
age = tf.contrib.layers.real_valued_column('age')

# Bucketization
age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries =[18, 25, 30, 45, 45, 50, 55, 60, 65])

# Linear estimators
e = tf.contrib.learn.LinearClassifier(feature_columns=[],
                                      model_dir='YOUR_MODEL_DIR')
e.fit(input_fn='input_fn_train', steps=200)
results = e.evaluate(input_fn='input_fn_test', steps=1)
for key in sorted(results):
    print('%s: %s' % (key, results[key]))

e = tf.contrib.learn.DNNLinearConbinedClassifier(
    model_dir="YOUR_MODEL_DIR",
    linear_feature_columns=[],
    dnn_feature_columns=[],
    dnn_hidden_units=[100, 50])