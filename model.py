import tensorflow as tf

def conv_nn_model(features,labels,mode):
    input_layer_images = tf.reshape(features,(-1,128,128,1),name='reshape1')
    input_layer_images = tf.cast(input_layer_images,tf.float32)
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels,(-1,10201),name='reshape2')
    conv1 = tf.layers.conv2d(
        inputs = input_layer_images,
        filters = 64,
        kernel_size = [20,20],
        padding='same',
        activation=tf.nn.relu
    )
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[20,20],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )

    conv3 = tf.layers.conv2d(
        inputs = pool1,
        filters = 128,
        kernel_size = [10,10],
        padding='same',
        activation=tf.nn.relu
    )
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters= 128,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    ) 
    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=[2,2],
        strides=2
    )

    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    )
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters= 256,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    )
    pool3 = tf.layers.average_pooling2d(
        inputs=conv6,
        pool_size=[2,2],
        strides=2
        )

    conv7 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters= 512,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool4 = tf.layers.average_pooling2d(
        inputs=conv8,
        pool_size=[2,2],
        strides=2
        )
    
    midconv = tf.layers.conv2d(
        inputs=pool4,
        filters = 1024,
        kernel_size=[2,2],
        padding='same',
        activation=tf.nn.relu
    )

    tconv1 = tf.layers.conv2d_transpose(
        inputs=midconv,
        filters=1024,
        kernel_size=[2,2],
        strides=2
    )
    conv9 = tf.layers.conv2d(
        inputs=tconv1,
        filters= 512,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    ) 
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters= 512,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    )
    tconv2 = tf.layers.conv2d_transpose(
        inputs=conv10,
        filters=512,
        kernel_size=[2,2],
        strides=2
    )
    conv11 = tf.layers.conv2d(
        inputs=tconv2,
        filters= 256,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    ) 
    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters= 256,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    )
    tconv3 = tf.layers.conv2d_transpose(
        inputs=conv12,
        filters=256,
        kernel_size=[2,2],
        strides=2
    )
    conv13 = tf.layers.conv2d(
        inputs=tconv3,
        filters= 128,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    ) 
    conv14 = tf.layers.conv2d(
        inputs=conv13,
        filters= 128,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    )
    tconv4 = tf.layers.conv2d_transpose(
        inputs=conv14,
        filters=128,
        kernel_size=[2,2],
        strides=2
    )
    conv15 = tf.layers.conv2d(
        inputs=tconv4,
        filters= 64,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    ) 
    conv16 = tf.layers.conv2d(
        inputs=conv15,
        filters= 64,
        kernel_size=[10,10],
        padding="same",
        activation=tf.nn.relu
    )
    convOut = tf.layers.conv2d(
        inputs=conv16,
        filters= 1,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    conv_flat = tf.reshape(convOut,(-1,16384))

    logits = tf.layers.dense(
        inputs=conv_flat,
        units = 10201,
    )
    

    '''  tf.layers.conv2d(
        inputs=conv16,
        filters = 1,
        kernel_size=1,
        padding='same',
        activation=tf.nn.relu
    ) '''
 
#    pool2_flat = tf.reshape(pool2,(-1,26*26*32))

#   logits = tf.layers.dense(inputs=pool2_flat, units=10816, activation=tf.nn.relu)

#    logits_round = tf.round(logits)
    #print("output:",tf.shape(logits),"labels:",tf.shape(labels))

    #logits_flat = tf.reshape(logits,(-1,16384))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

    loss = tf.losses.sigmoid_cross_entropy(logits=logits,multi_class_labels=labels)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=logits)
            }
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(    
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)