from keras import backend as K
from keras.engine import Layer


class GatedLayer(Layer):
    """
    Layer for gated. Fuse the inputs[1] and inputs[2] according to the gate calculated with inputs[0]
    """
    def __init__(self, use_bias=False, **kwargs):
        super(GatedLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.kernel1 = self.add_weight(name='kernel1',
                                       shape=[dim, 1],
                                       initializer='glorot_uniform')

        dim_2 = input_shape[1][-1]
        dim_3 = input_shape[2][-1]
        self.kernel2 = self.add_weight(name='kernel2',
                                       shape=[dim_2, dim_3],
                                       initializer='glorot_uniform')

        self.built = True

    def call(self, inputs, mask=None):

        key, v1, v2 = inputs

        v1 = K.dot(key, self.kernel2)  # (?, sen_len, dim3)
        v1 = K.tanh(v1)  # (?, sen_len, dim3)

        gate = K.dot(key, self.kernel1)  # (?, sen_len, 1)
        gate = K.sigmoid(gate)  # (?, sen_len, 1)
        # gate = K.repeat_elements(gate, 200, axis=2)
        res = gate * v1 + (1 - gate) * v2  # (?, sen_len, dim3)

        return res  # (?, sen_len, dim3)

    def compute_output_shape(self, input_shape):
        return input_shape[2]
