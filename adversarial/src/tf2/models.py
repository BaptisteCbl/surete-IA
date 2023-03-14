from keras.engine import base_layer
import tensorflow as tf
from keras import Model
from keras.layers import Input, Conv2D, Dense, Flatten


def create_small_CNN():
    input = Input(shape=(28, 28, 1))
    x = Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu')(input)
    x = Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x_1 = Dense(100, activation='relu')(x)
    output = Dense(10)(x)
    return Model(inputs=input, outputs=output)


def create_small_CNN_conf():
    input = Input(shape=(28, 28, 1))
    x = Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu')(input)
    x = Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x_1 = Dense(100, activation='relu')(x)
    output_1 = Dense(10)(x)
    x_2 = Dense(100, activation='relu')(x)
    output_2 = Dense(1)(x)
    outputs = [output_1, output_2]
    return Model(inputs=input, outputs=outputs)


class NoiseLayer(base_layer.BaseRandomLayer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.seed = seed
        self.alpha = self.add_weight(
            "alpha",
            initializer=tf.keras.initializers.Constant(
                value=0.25
            )
        )

    def call(self, inputs):
        std = tf.math.reduce_std(inputs)
        noise = self._random_generator.random_normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=std,
            dtype=inputs.dtype,
        )
        return inputs + self.alpha*noise


def create_small_CNN_noise():
    input = Input(shape=(28, 28, 1))
    x = NoiseLayer()(input)
    x = Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = NoiseLayer()(x)
    x = Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = NoiseLayer()(x)
    x_1 = Dense(100, activation='relu')(x)
    output = Dense(10)(x)
    return Model(inputs=input, outputs=output)