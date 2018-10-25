import tensorflow as tf
from unet_utils import Recorder


tf.logging.set_verbosity(tf.logging.INFO)

def conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations

    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(flags),
        name="upsample_{}".format(name))


def make_unet(X, training, flags=None):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X / 127.5 - 1
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    #tf.summary.histograms('conv1',conv1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


def IoU(logit,y):

       y_pred_flat = tf.layers.flatten(logit)
       y_true_flat = tf.layers.flatten(y)

       intersection = 2*tf.reduce_sum(y_pred_flat * y_true_flat, axis=1) + 1e-7

       denominator = tf.reduce_sum(y_pred_flat, axis=1)+tf.reduce_sum(y_true_flat, axis=1) + 1e-7

       loss = tf.reduce_mean(intersection/denominator)
       
       return loss



def unet(features,labels,mode,params):


      layer9_up = make_unet(features,training= mode==tf.estimator.ModeKeys.TRAIN,flags=0.1)

      tf.summary.image('PREDICTED_MASK',layer9_up)
      tf.summary.image('INPUT_IMAGE', features)

      #print 'Features shape %s, logits shape %s, labels shape %s'%(features.shape,layer9_up.shape,labels.shape)

      loss = IoU(layer9_up, labels)
      #tf.summary.scalar('IOU_LOSS', loss)


      mean_iou, update_op_iou = tf.metrics.mean_iou(labels=labels,predictions=layer9_up, num_classes=1, name='acc_op')
      tf.summary.scalar('MEAN_IOU', mean_iou)

      optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])


      global_step = tf.train.get_global_step()

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss, global_step = global_step)

      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




def main():
    


    rec = Recorder()
    train_outpath='../train.tfrecords'

    configuration = tf.estimator.RunConfig(
                                   model_dir = './checkpoints/',
                                   keep_checkpoint_max=3,
                                   save_checkpoints_secs = 100,
                                   log_step_count_steps=10)  # set the frequency of logging steps for loss function

    classifier = tf.estimator.Estimator(model_fn = unet, params = {'learning_rate' : 0.001}, config=configuration)

    classifier.train(input_fn = lambda:rec.imgs_input_fn(train_outpath,height=128,width=128,shuffle=True,repeat_count=-1), steps=500)
    #print classifier


def test():
    features = tf.placeholder(tf.float32,[None,128,128,3])
    labels = tf.placeholder(tf.float32,[None,128,128,1])
    mode = tf.estimator.ModeKeys.TRAIN
    params = {'learning_rate':0.001}

    model = unet(features,labels,mode,params)
    print model
    
if __name__=='__main__':
    main()
    #test()




