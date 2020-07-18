import numpy as np
import tensorflow as tf

class DecoderType:
    BestPath = 0
    WordBeamSearch = 1
    BeamSearch = 2

class CNNBlock(tf.keras.layers.Layer):
    """ Create CNN layers and return output of these layers """
    def __init__(self,):
        super(CNNBlock, self).__init__()
        self.kernel1 = self.kernel()
        self.kernel2 = self.kernel([5, 5, 64, 128])
        self.kernel3 = self.kernel([3, 3, 128, 128])
        self.kernel4 = self.kernel([3, 3, 128, 256])
        self.kernel5 = self.kernel([3, 3, 256, 256])
        self.kernel6 = self.kernel([3, 3, 256, 512])
        self.kernel7 = self.kernel([3, 3, 512, 512])

    def kernel(self, shape=[5, 5, 1, 64], stddev=0.1):
        return tf.Variable(
            tf.random.truncated_normal(shape, stddev=stddev))

    def conv_pool(self, X, kernel):
        conv = tf.nn.conv2d(
            X, kernel, padding='SAME', strides=(1, 1, 1, 1))
        learelu = tf.nn.leaky_relu(conv, alpha=0.01)
        pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

    def call(self, input_tensor, training=False):
        x = tf.expand_dims(input=input_tensor, axis=3)

        # First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
        with tf.name_scope('Conv_Pool_1'):
            conv = tf.nn.conv2d(
                x, self.kernel1, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Second Layer: Conv (5x5) + Pool (1x2) - Output size: 400 x 16 x 128
        with tf.name_scope('Conv_Pool_2'):
            conv = tf.nn.conv2d(
                pool, self.kernel2, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Third Layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 8 x 128
        with tf.name_scope('Conv_Pool_BN_3'):
            conv = tf.nn.conv2d(
                pool, self.kernel3, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Fourth Layer: Conv (3x3) - Output size: 200 x 8 x 256
        with tf.name_scope('Conv_4'):
            conv = tf.nn.conv2d(
                pool, self.kernel4, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)

        # Fifth Layer: Conv (3x3) + Pool(2x2) - Output size: 100 x 4 x 256
        with tf.name_scope('Conv_Pool_5'):
            conv = tf.nn.conv2d(
                learelu, self.kernel5, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Sixth Layer: Conv (3x3) + Pool(1x2) + Simple Batch Norm - Output size: 100 x 2 x 512
        with tf.name_scope('Conv_Pool_BN_6'):
            conv = tf.nn.conv2d(
                pool, self.kernel6, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Seventh Layer: Conv (3x3) + Pool (1x2) - Output size: 100 x 1 x 512
        with tf.name_scope('Conv_Pool_7'):
            conv = tf.nn.conv2d(
                pool, self.kernel7, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')
        return pool

class Squeeze(tf.keras.layers.Layer):
    def __init__(self,):
        super(Squeeze, self).__init__()
    
    def call(self, input_tensor, training=False):
        x =  tf.squeeze(input_tensor, axis=[2])
        return x

class ProjectChar(tf.keras.layers.Layer):
    def __init__(self, charList):
        super(ProjectChar, self).__init__()
        self.charList = charList
        self.numHidden = 512
        self.kernel = tf.Variable(tf.random.truncated_normal(
            [1, 1, self.numHidden * 2, len(self.charList) + 1], stddev=0.1))

    def call(self, input_tensor, training=False):
        x = tf.expand_dims(input_tensor, 2)
        x = tf.nn.atrous_conv2d(value=x, filters=self.kernel, rate=1, padding='SAME')
        x = tf.squeeze(x, axis=[2])
        return tf.transpose(x, [1, 0, 2])

class RNNBlock(tf.keras.layers.Layer):
    def __init__(self, charList):
        super(RNNBlock, self).__init__()
        self.numHidden = 512
        self.rnn_cell = [tf.keras.layers.LSTMCell(self.numHidden, use_bias=False) for _ in range(2)]
        self.stacked = tf.keras.layers.StackedRNNCells(self.rnn_cell)
        self.stacked = tf.keras.layers.RNN(self.stacked, return_sequences=True)
        self.squeeze = Squeeze()
        self.project_char = ProjectChar(charList) 
    
    def call(self, input_tensor, training=False):
        x = self.squeeze(input_tensor)
        x = tf.keras.layers.Bidirectional(self.stacked)(x)
        return self.project_char(x)


class Model(object):
    batchSize = 32 # 50
    imgSize = (800, 64)
    maxTextLen = 100

    def __init__(self, charList, decoderType):
        self.charList = charList
        self.cnn_pass = CNNBlock()
        self.rnn_pass = RNNBlock(charList)
        self.decoderType = decoderType

    def get_model(self,):
        inputs = tf.keras.Input(shape=self.imgSize)
        cnn_output = self.cnn_pass(inputs)
        outputs = self.rnn_pass(cnn_output)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def decoderToText(self, ctcOutput):
        """ Extract texts from output of CTC decoder """
        # Contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(Model.batchSize)]
        # Word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(Model.batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)
        # TF decoders: label strings are contained in sparse tensor
        else:
            # Ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]
            # Go over all indices and save mapping: batch -> values
            idxDict = {b : [] for b in range(Model.batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)
        # Map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

def ctc_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.nn.ctc_loss(y_true, y_pred, 
        label_length=[Model.maxTextLen]*Model.batchSize, logit_length=[Model.maxTextLen]*Model.batchSize,
        blank_index=-1)) 
    