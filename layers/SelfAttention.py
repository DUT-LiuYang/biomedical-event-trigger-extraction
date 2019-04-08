from keras import initializers, constraints, regularizers
from keras.engine import Layer
import keras.backend as K
from keras.layers import Dot
import numpy as np


class SelfAttention(Layer):
    def __init__(self,
                 # W_regularizer=None, b_regularizer=None,
                 # W_constraint=None, b_constraint=None,
                 tri_att=True,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.tri_att = tri_att
        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.b_regularizer = regularizers.get(b_regularizer)
        #
        # self.W_constraint = constraints.get(W_constraint)
        # self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.sen_len = 125
        self.dim = 200
        super(SelfAttention, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def call(self, x, mask=None):

        x1, x2 = x

        matrix = Dot(axes=-1)([x1, x2])
        # matrix = K.tanh(matrix)

        # matrix = K.exp(matrix)
        # print(np.shape(matrix))

        if mask is not None:
            mask_tensor = K.cast(mask[0], K.floatx())  # (?, sen_len)
            mask_tensor = K.expand_dims(mask_tensor)  # (?, sen_len, 1)
            mask_tensor = K.permute_dimensions(mask_tensor, (0, 2, 1))  # (?, 1, sen_len)
            mask_tensor = (K.ones_like(mask_tensor) - mask_tensor) * 100  # (?, 1, sen_len)
            matrix = matrix - mask_tensor  # (?, sen_len, sen_len)
            # matrix = matrix - K.eye(125) * 100

        matrix = K.softmax(matrix)

        if self.tri_att:
            res = Dot(axes=[2, 1])([matrix, x2])
        else:
            res = Dot(axes=[1, 1])([matrix, x1])
        # matrix = K.permute_dimensions(matrix, (0, 2, 1))

        return [res, matrix]

    def compute_output_shape(self, input_shape):
        return [input_shape[1], [input_shape[0][0], input_shape[0][1], input_shape[0][1]]]
