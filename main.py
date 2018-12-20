#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

BATCH_SIZE = 2
LEARNING_RATE = 0.0002 
KEEP_PROB = 0.5

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_image_input = 'image_input:0'
    vgg_keep_prob = 'keep_prob:0'
    vgg_layer3_out = 'layer3_out:0'
    vgg_layer4_out = 'layer4_out:0'
    vgg_layer7_out = 'layer7_out:0'

    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_image_input)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out)
    
    return image_input, keep_prob, layer3, layer4, layer7
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    with tf.name_scope('input'):
      with tf.name_scope('layer3'):
        pool3_1x1 = conv1x1(vgg_layer3_out, num_classes,kernel_initializer, kernel_regularizer)
        variable_summaries(pool3_1x1)
      with tf.name_scope('layer4'):      
        pool4_1x1 = conv1x1(vgg_layer4_out, num_classes,kernel_initializer, kernel_regularizer)
        variable_summaries(pool4_1x1)
      with tf.name_scope('layer7'):
        pool7_1x1 = conv1x1(vgg_layer7_out, num_classes,kernel_initializer, kernel_regularizer)
        variable_summaries(pool7_1x1)
    
    # inference
    
    with tf.name_scope('output'):
      with tf.name_scope('deconvolution'):      
        deconv7 = tf.layers.conv2d_transpose(pool7_1x1, num_classes, 
                                             kernel_size=4, strides=2, padding='same',
                                             kernel_initializer=kernel_initializer, 
                                             kernel_regularizer=kernel_regularizer)
      with tf.name_scope('fuse1'):
        fuse1 = tf.add(deconv7, pool4_1x1)
        deconv_fuse1 = tf.layers.conv2d_transpose(fuse1, num_classes, 
                                                  kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=kernel_initializer,
                                                  kernel_regularizer=kernel_regularizer)
      with tf.name_scope('fuse2'):
        fuse2 = tf.add(deconv_fuse1, pool3_1x1)
        out = tf.layers.conv2d_transpose(fuse2, num_classes, 
                                         kernel_size=16, strides=8, padding='same',
                                         kernel_initializer=kernel_initializer, 
                                         kernel_regularizer=kernel_regularizer)
    
    return out
#tests.test_layers(layers)


def conv1x1 (layer_name, num_classes,kernel_initializer,kernel_regularizer):
    
    return tf.layers.conv2d(layer_name, num_classes,kernel_size=1,padding='same'
                            ,kernel_initializer=kernel_initializer
                            ,kernel_regularizer=kernel_regularizer)
  

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fn_logits")
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label)
    
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name="fn_loss")

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, name="fn_train_op")
    
    return logits, train_op, cross_entropy_loss
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    sess.run(tf.global_variables_initializer())
    
    loss_per_epoch = []
    
    for epoch in range(epochs):
      
        losses, i = [], 0
        
        for images, labels in get_batches_fn(batch_size):
          
            i += 1
            
            feed_dict = {input_image: images, 
                         correct_label: labels, 
                         keep_prob: KEEP_PROB, 
                         learning_rate: LEARNING_RATE}
            
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            losses.append(loss)
        
        training_loss = sum(losses) / len(losses)
        
        loss_per_epoch.append(training_loss)
        
        print("EPOCH: %d of %d, LOSS: %.5f" % (epoch+1, epochs, training_loss))
        
    return loss_per_epoch
#tests.test_train_nn(train_nn)
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def run():
  
    tf.logging.set_verbosity(tf.logging.INFO)
    
    
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './data/runs'
    video_dir = './data/test_video'
    
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('data_dir', sess.graph)
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        
        out = layers(layer3, layer4, layer7, num_classes)
        
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)
        
        logits, train_op, cross_entropy_loss = optimize(out, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        
        epochs = 100
        batch_size = BATCH_SIZE
        
        loss_per_epoch = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)     
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)        
     
        saver.save(sess, './data/trained_model')
        # Save the variables to disk.
        save_path = saver.save(sess, "./data/trained_model.ckpt")
        
        print("Model saved in : %s" % save_path)

        # OPTIONAL: Apply the trained model to a video
        
        path_to_video = './data/test_video/'
 
        if len(sys.argv) == 2:
          path_to_video = sys.argv[1]
        print(path_to_video)
        files = os.listdir(path_to_video)
        for name in files:
          filename, file_extension = os.path.splitext(name)
          ExtractVideo(video_dir, path_to_video + name, path_to_video + "out_" + filename + ".avi", sess, logits, keep_prob, image_shape, image_input)    
          print(name, " >>> out_" + filename + ".avi" )
 
        writer.close()
        sess.close()
        


if __name__ == '__main__':
    run()

