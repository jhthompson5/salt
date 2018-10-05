import tensorflow as tf

def conv_nn_model(features,labels,mode):
    input_layer_images = tf.reshape(features,(-1,104,104,1))
    input_layer_images = tf.cast(input_layer_images,tf.float32)
    labels = tf.reshape(labels,(-1,10816))
    conv1 = tf.layers.conv2d(
        inputs = input_layer_images,
        filters = 64,
        kernel_size = [10,10],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
#   pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2, padding='same')
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=256,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool3 = tf.layers.average_pooling2d(inputs=conv3,pool_size=[2,2],strides=2)
 
    pool3_flat = tf.reshape(pool3,(-1,26*26*256))

    logits = tf.layers.dense(inputs=pool3_flat, units=10816, activation=tf.nn.relu)

    logits_round = tf.round(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits_round)

    loss = tf.losses.sigmoid_cross_entropy(logits=logits,multi_class_labels=labels)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=logits_round)
            }
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(    
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)