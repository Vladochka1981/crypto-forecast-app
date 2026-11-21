"""Кастомный слой attention для загрузки Attention-GRU."""
import tensorflow as tf
from keras.layers import Layer
from keras.saving import register_keras_serializable


@register_keras_serializable(package="crypto")
class ArticleStyleTemporalAttention(Layer):
    def __init__(self, attn_units=64, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = attn_units

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(d, self.attn_units),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True)
        self.b = self.add_weight(name="b", shape=(self.attn_units,),
                                 initializer="zeros", trainable=True)
        self.v = self.add_weight(name="v", shape=(self.attn_units, 1),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b)
        score = tf.tensordot(score, self.v, axes=[[2], [0]])
        alpha = tf.nn.softmax(score, axis=1)
        return x * alpha, tf.squeeze(alpha, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"attn_units": self.attn_units})
        return config