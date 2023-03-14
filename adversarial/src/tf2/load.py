import tensorflow as tf
import tensorflow_datasets as tfds
from easydict import EasyDict

def ld_fashion_mnist(batch_size=128):
    """Load training and test data."""

    def augment_mirror(x):
        return tf.image.random_flip_left_right(x)

    def augment_shift(x, w=4):
        y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    def one_hot(y):
        return tf.one_hot(y, 10)

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 127.5
        image -= 1.0
        return image, label

    dataset, info = tfds.load("fashion_mnist", with_info=True, as_supervised=True)
    fashion_mnist_train, fashion_mnist_test = dataset["train"], dataset["test"]

    fashion_mnist_train = fashion_mnist_train.map(
        lambda x, y: (augment_mirror(augment_shift(x)), y)
    )
    fashion_mnist_train = fashion_mnist_train.map(convert_types).shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)
    fashion_mnist_test = fashion_mnist_test.map(convert_types).batch(batch_size)

    return EasyDict(train=fashion_mnist_train, test=fashion_mnist_test)


def ld_cifar10():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 127.5
        image -= 1.0
        return image, label

    dataset, info = tfds.load("cifar10", with_info=True, as_supervised=True)

    def augment_mirror(x):
        return tf.image.random_flip_left_right(x)

    def augment_shift(x, w=4):
        y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    cifar10_train, cifar10_test = dataset["train"], dataset["test"]
    # Augmentation helps a lot in CIFAR10
    cifar10_train = cifar10_train.map(
        lambda x, y: (augment_mirror(augment_shift(x)), y)
    )
    cifar10_train = cifar10_train.map(convert_types).shuffle(10000).batch(128)
    cifar10_test = cifar10_test.map(convert_types).batch(128)

    return EasyDict(train=cifar10_train, test=cifar10_test)