import keras.backend as K
from keras.layers import Concatenate, Multiply, Lambda, Dot, Add
from keras.activations import softmax

from layers.SimilarityMatrix import SimilarityMatrix


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = Dot(axes=-1)([input_1, input_2])

    w_att = Lambda(lambda x: softmax(x, axis=1),
                   output_shape=unchanged_shape)(attention)

    in_aligned = Dot(axes=1)([w_att, input_1])

    return in_aligned


def minus_soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = SimilarityMatrix()([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    in1_aligned = Dot(axes=1)([w_att_1, input_1])

    return in1_aligned


def subtract(input_1, input_2):
    """subtract element-wise"""
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    """Get multiplication and subtraction then concatenate results"""
    mult = Multiply()([input_1, input_2])
    sub = subtract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    """Apply layers to input then concatenate result"""
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


