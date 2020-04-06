import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import pdb
import numpy as np
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
#this code modified by lss
def _variable_on_cpu(name, shape, para):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
        name: name of the variable
        shape: list of ints
        para: parameter for initializer
    
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        if name == 'weights':
            # initializer = tf.truncated_normal_initializer(stddev=para, dtype=dtype)
            initializer = tf.contrib.layers.xavier_initializer(seed=1)
        else:
            initializer = tf.constant_initializer(para)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
      name,
      shape,
      stddev)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
def conv(x, kernel_height, kernel_width, num_kernels, stride_y, stride_x, name,
         reuse=False, padding='SAME'):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    print(x.get_shape())

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_height, kernel_width,
                                               input_channels, num_kernels], 1e-1)
        biases = _variable_on_cpu('biases', [num_kernels], 0.0)

        # Apply convolution function
        conv = convolve(x, weights)

        # Add biases
        bias = tf.nn.bias_add(conv, biases)

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, is_training, name, reuse=False,
       relu=True, batch_norm=False):
    with tf.variable_scope(name, reuse=reuse) as scope:

        # Create tf variable for the weights and biases
        # weights = _variable_with_weight_decay('weights', [num_in, num_out], 1e-1, wd)
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)
        biases = _variable_on_cpu('biases', [num_out], 1.0)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if batch_norm:
            # Adds a Batch Normalization layer
            act = tf.contrib.layers.batch_norm(act, center=True, scale=True,
                                               trainable=True, is_training=is_training,
                                               reuse=reuse, scope=scope)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pooling(x, kernel_height, kernel_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

class VGG16:
    # Build the AlexNet model
    IMAGE_SIZE = 224    # input images size

    def __init__(self,num_classes):

        # Parse input arguments into class variables
        self.wd = 0.00005
        self.NUM_CLASSES = num_classes
        self.reuse=False
        self.WEIGHTS_PATH = 'MVFSL-TC/vgg16-ImageNet.npy'
        self.batch_norm=True
        # self.vgg16(self.X, is_training = self.is_training, reuse = self.reuse)

    def __call__(self, input_image, is_training, keep_prob):
        with tf.variable_scope('g', reuse=self.reuse):

        # 1st Layer: Conv_1-2 (w ReLu) -> Pool
            conv1_1 = conv(input_image, 3, 3, 64, 1, 1, name='conv1_1', reuse=self.reuse)
            conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name='conv1_2', reuse=self.reuse)
            pool1 = max_pooling(conv1_2, 2, 2, 2, 2, name='pool1')

        # 2nd Layer: Conv_1-2 (w ReLu) -> Pool
            conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', reuse=self.reuse)
            conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', reuse=self.reuse)
            pool2 = max_pooling(conv2_2, 2, 2, 2, 2, name='pool2')
        # with tf.variable_scope('g_2', reuse=self.reuse):       
        # 3rd Layer: Conv_1-3 (w ReLu) -> Pool
            conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', reuse=self.reuse)
            conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', reuse=self.reuse)
            conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', reuse=self.reuse)
            pool3 = max_pooling(conv3_3, 2, 2, 2, 2, name='pool3')

        # 4th Layer: Conv_1-3 (w ReLu) -> Pool
        with tf.variable_scope('g_2', reuse=self.reuse):
            conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', reuse=self.reuse)
            conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', reuse=self.reuse)
            conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', reuse=self.reuse)
            pool4 = max_pooling(conv4_3, 2, 2, 2, 2, name='pool4')

        # 5th Layer: Conv_1-3 (w ReLu) -> Pool
            conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', reuse=self.reuse)
            conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', reuse=self.reuse)
            conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', reuse=self.reuse)
            # pool5 = max_pooling(conv5_3, 2, 2, 2, 2, name='pool5')
            # conv5_3= tf.nn.l2_normalize(conv5_3,3,epsilon=1e-12, name=None)
            print(conv5_3.get_shape())         
        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_2')
        return conv5_3
    
    def load_initial_weights(self, session):

        not_load_layers = ['fc7','fc6','fc8']
        if self.WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            print(op_name)

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in not_load_layers:
                with tf.variable_scope('g/'+op_name, reuse=True):
                    data = weights_dict[op_name]
                    # Biases
                    var = tf.get_variable('biases')
                    session.run(var.assign(data['biases']))
                    # Weights
                    var = tf.get_variable('weights')
                    session.run(var.assign(data['weights']))

        print('Loading the weights is Done.')
class VGG16_ingre:
    # Build the AlexNet model
    IMAGE_SIZE = 224    # input images size

    def __init__(self,num_classes):

        # Parse input arguments into class variables
        self.wd = 0.00005
        self.NUM_CLASSES = num_classes
        self.reuse=False
        self.WEIGHTS_PATH = 'MVFSL-TC/mynet_ingre-101.npy'
        self.batch_norm=True

    def __call__(self, input_image, is_training, keep_prob):
        with tf.variable_scope('g_ingre', reuse=self.reuse):

        # 1st Layer: Conv_1-2 (w ReLu) -> Pool
            conv1_1 = conv(input_image, 3, 3, 64, 1, 1, name='conv1_1', reuse=self.reuse)
            conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name='conv1_2', reuse=self.reuse)
            pool1 = max_pooling(conv1_2, 2, 2, 2, 2, name='pool1')

        # 2nd Layer: Conv_1-2 (w ReLu) -> Pool
            conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', reuse=self.reuse)
            conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', reuse=self.reuse)
            pool2 = max_pooling(conv2_2, 2, 2, 2, 2, name='pool2')
               
        # 3rd Layer: Conv_1-3 (w ReLu) -> Pool
            conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', reuse=self.reuse)
            conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', reuse=self.reuse)
            conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', reuse=self.reuse)
            pool3 = max_pooling(conv3_3, 2, 2, 2, 2, name='pool3')

        # 4th Layer: Conv_1-3 (w ReLu) -> Pool
            conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', reuse=self.reuse)
            conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', reuse=self.reuse)
            conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', reuse=self.reuse)
            pool4 = max_pooling(conv4_3, 2, 2, 2, 2, name='pool4')

        # 5th Layer: Conv_1-3 (w ReLu) -> Pool
            conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', reuse=self.reuse)
            conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', reuse=self.reuse)
            conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', reuse=self.reuse)
            # pool5 = max_pooling(conv5_3, 2, 2, 2, 2, name='pool5')
            # conv5_3= tf.nn.l2_normalize(conv5_3,3,epsilon=1e-12, name=None)
            print(conv5_3.get_shape())
        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_ingre/conv5_3')
        self.variables2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_ingre/conv5_2')
        self.variables3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_ingre/conv5_1')
        self.variables4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_ingre/conv4_1')
        self.variables5 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_ingre/conv4_2')
        self.variables6 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_ingre/conv4_3')
        return conv5_3
    def load_initial_weights(self, session):

        not_load_layers = ['fc7','fc6','fc8_tune']
        if self.WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='latin1',allow_pickle=True).item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            print(op_name)
            if op_name not in not_load_layers:
                with tf.variable_scope('g_ingre/'+op_name, reuse=True):
                    data = weights_dict[op_name]
                    # print(data)
                    # Biases
                    var = tf.get_variable('biases')
                    session.run(var.assign(data['biases']))
                    # Weights
                    var = tf.get_variable('weights')
                    session.run(var.assign(data['weights']))

        print('Loading the weights is Done.') 
   
class RelationModule:
    def __init__(self, layer_sizes):
        self.reuse = False
        self.layer_sizes = layer_sizes

    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        this module use to implement relstion module
        """

        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        with tf.variable_scope('RelationModule', reuse=self.reuse):
            with tf.variable_scope('conv_layers'):

                with tf.variable_scope('RelationModule_conv1'):
                    # pdb.set_trace()
                    g_conv1_encoder = tf.layers.conv2d(image_input, 64, [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv1_encoder = leaky_relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv1_encoder = max_pool(g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)

                with tf.variable_scope('RelationModule_conv2'):
                    g_conv2_encoder = tf.layers.conv2d(g_conv1_encoder, 64, [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv2_encoder = leaky_relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv2_encoder = max_pool(g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv2_encoder = tf.nn.dropout(g_conv2_encoder, keep_prob=keep_prob)
                


                with tf.variable_scope('fully_connected_relu'):
                    # pdb.set_trace()
                    g_fc1_encoder = tf.contrib.layers.flatten(g_conv2_encoder)
                    g_fc1_encoder = tf.contrib.layers.fully_connected(g_fc1_encoder, 8, trainable=True, scope='fc_relu')

                with tf.variable_scope('fully_connected_sigmoid'):
                    g_fc2_encoder = tf.contrib.layers.fully_connected(g_fc1_encoder, 1, activation_fn=tf.nn.sigmoid,
                                                                      trainable=True, scope='fc_sigmoid')

            g_conv_encoder = g_fc2_encoder
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RelationModule')
        return g_conv_encoder


class AttentionalClassify:
    def __init__(self):
        self.reuse = False

    def __call__(self, similarities, support_set_y, name, training=False):
        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size, 1]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :param name: The name of the op to appear on tf graph
        :param training: Flag indicating training or evaluation stage (True/False)
        :return: Softmax pdf
        """
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification',
                                                                                   reuse=self.reuse):
            pdb.set_trace()
            softmax_similarities = tf.nn.softmax(similarities)
            preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds


class DistanceNetwork:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
        with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
            eps = 1e-10
            similarities = []
            for support_image in tf.unstack(support_set, axis=0):
                sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
                support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
                dot_product = tf.matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
                dot_product = tf.squeeze(dot_product, [1, ])
                cosine_similarity = dot_product * support_magnitude
                similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1, values=similarities)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
        self.reuse = True
        return similarities




# class MatchingNetwork:
class RelationNetwork:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob,
                 batch_size=100, num_channels=1, is_training=False, learning_rate=0.001, rotate_flag=False, fce=False,
                 num_classes_per_set=5,
                 num_samples_per_class=5):
        self.batch_size = batch_size
        self.fce = fce
        self.g = VGG16(num_classes=132)
        self.g_ingre=VGG16_ingre(num_classes=132)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.RelationModule = RelationModule(layer_sizes=[64, 64, 64, 64])
        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.target_image = target_image
        self.target_label = target_label
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.k = None
        self.rotate_flag = rotate_flag
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate
    def loss(self):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        with tf.name_scope("losses"):
            [num_classes, spc] = self.support_set_labels.get_shape().as_list()
            self.support_set_labels = tf.reshape(self.support_set_labels, shape=(num_classes * spc,))
            self.support_set_labels = tf.one_hot(self.support_set_labels, self.num_classes_per_set)  # one hot encode
            encoded_images = []
            [num_classes, spc, h, w, c] = self.support_set_images.get_shape().as_list()            
            self.support_set_images = tf.reshape(self.support_set_images, shape=(num_classes * spc, h, w, c))
            # self.support_set_images_ingre = tf.transpose(self.support_set_images, [0,3,1,2])
            red, green, blue = tf.split(self.support_set_images,3,3)
            bgr = tf.concat([blue,green,red],3)
            print("out encoded_images_ingre", bgr.get_shape().as_list())
            # pdb.set_trace()
            encoded_images = self.g(input_image=bgr, is_training=self.is_training,
                                    keep_prob=self.keep_prob)
            encoded_images_ingre = self.g_ingre(input_image=bgr, is_training=self.is_training,
                                    keep_prob=self.keep_prob)         
            encoded_images_concat=  tf.concat([encoded_images,encoded_images_ingre],3)
            print("out encoded_images_concat", encoded_images_concat.get_shape().as_list())
            target_image = self.target_image
            [num, h, w, c] = self.target_image.get_shape().as_list()  
            print(type(target_image))
            red_t, green_t, blue_t = tf.split(target_image,3,3)
            bgr_t = tf.concat( [blue_t,green_t,red_t],3)            
            gen_encode = self.g(input_image=bgr_t, is_training=self.is_training, keep_prob=self.keep_prob)
            gen_encode_ingre = self.g_ingre(input_image=bgr_t, is_training=self.is_training, keep_prob=self.keep_prob)
            gen_encode_concat=  tf.concat([gen_encode,gen_encode_ingre],3)
            concat_encoded_images = tf.concat([tf.stack([encoded_images_concat] * target_image.shape[0]), tf.stack([gen_encode_concat] * num_classes, axis=1)], 4)
            print("out shape11", concat_encoded_images.get_shape().as_list())
            # pdb.set_trace()
            # encoded_images.append(gen_encode)
            [num_query, num_classes, dim_1, dim_2, dim_3] = concat_encoded_images.get_shape().as_list()
            concat_encoded_images = tf.reshape(concat_encoded_images, [-1, dim_1, dim_2, dim_3])
            similarities = self.RelationModule(concat_encoded_images, training=self.is_training, keep_prob=self.keep_prob)
            print("out shape111", similarities.get_shape().as_list())
            similarities = tf.reshape(similarities, [num_query, num_classes])
            softmax_similarities = tf.nn.softmax(similarities)
            self.similarities = similarities
            preds = tf.squeeze(tf.matmul(similarities, self.support_set_labels))
            self.preds = preds
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label, tf.int64))
            self.prediction = tf.argmax(preds, 1)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            targets = tf.one_hot(self.target_label, self.num_classes_per_set)
            mean_square_error_loss = tf.reduce_mean(tf.square((preds-1)*targets + preds*(1-targets)))

            # tf.add_to_collection('similarities', similarities)
            tf.add_to_collection('mean square error losses', mean_square_error_loss)
            tf.add_to_collection('accuracy', accuracy)

        return {
            # self.similarities: tf.add_n(tf.get_collection('similarities'), name='similarities'),
            self.classify: tf.add_n(tf.get_collection('mean square error losses'), name='total_classification_loss'),
            self.dn: tf.add_n(tf.get_collection('accuracy'), name='accuracy')
        }

    def train(self, losses):
        c_opt = tf.train.AdamOptimizer(beta1=0.9, learning_rate=self.learning_rate)
        c_opt2 = tf.train.AdamOptimizer(beta1=0.9, learning_rate=self.learning_rate*5)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):  # Needed for correct batch norm usage
            if self.fce:
                train_variables = self.lstm.variables + self.g.variables + self.fce_f.variables + self.RelationModule.variables
            else:
                train_variables =self.g.variables+self.g_ingre.variables+self.g_ingre.variables2+self.g_ingre.variables3+self.g_ingre.variables4+self.g_ingre.variables5+self.g_ingre.variables6
                train_variables2 =  self.RelationModule.variables
            c_error_opt_op1= c_opt.minimize(losses[self.classify],var_list=train_variables)
            c_error_opt_op2 = c_opt2.minimize(losses[self.classify],var_list=train_variables2)
            c_error_opt_op=tf.group(c_error_opt_op1,c_error_opt_op2)
        return c_error_opt_op
    def init_train(self):
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        summary = tf.summary.merge_all()
        return summary, losses, c_error_opt_op
