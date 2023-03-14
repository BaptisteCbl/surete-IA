import math
import numpy as np
import tensorflow as tf
from keras import Model

from src.tf2.load import ld_fashion_mnist
from src.tf2.models import create_small_CNN, create_small_CNN_conf, create_small_noise
from src.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from src.tf2.attacks.fast_gradient_method import fast_gradient_method


def run_fashion_conf(*args, **kwargs):
    # Load training and test data
    data = ld_fashion_mnist(batch_size=kwargs['batch_size'])
    model = create_small_CNN_conf()
    loss_conf = tf.losses.BinaryCrossentropy(from_logits=True)
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    lmbda = kwargs['lmbda']

    # Metrics to track the different accuracies.
    train_loss = tf.metrics.Mean(name="train_loss")
    train_class_loss = tf.metrics.Mean(name="train_class_loss")
    train_conf_loss = tf.metrics.Mean(name="train_conf_loss")

    test_conf_clean = tf.metrics.Mean(name="test_conf_clean")
    test_conf_fgsm = tf.metrics.Mean(name="test_conf_fgsm")
    test_conf_pgd = tf.metrics.Mean(name="test_conf_pgd")

    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

    def conf_loss(labels=None, logits=None):
        p, c = logits
        ones = tf.ones_like(c)
        p_p = tf.einsum('...i,...j->...j', c, p) + tf.einsum('...i,...j->...j', ones-c, tf.one_hot(labels, 10))
        l_t = loss_object(labels, p_p)
        l_c = loss_conf(ones, c)
        train_class_loss(l_t)
        train_conf_loss(l_c)
        return l_t + lmbda*l_c

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = conf_loss(labels=y, logits=model(x))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    # Train model with adversarial training
    for epoch in range(kwargs['nb_epochs']):
        # keras like display of progress
        progress_bar_train = tf.keras.utils.Progbar(60000)
        for (x, y) in data.train:
            if kwargs['adv_train']:
                # Replace clean example with adversarial example for adversarial training
                p, c = model(x)
                x = projected_gradient_descent(model, x, kwargs['eps'], 0.01, 40, np.inf, loss_fn=conf_loss, y=tf.argmax(p, 1))
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result()), ("loss_class", train_class_loss.result()), ("loss_conf", train_conf_loss.result())])

    # Evaluate on clean and adversarial data
    progress_bar_test = tf.keras.utils.Progbar(10000)
    clean_all_histogram = tf.zeros(10)
    clean_correct_histogram = tf.zeros(10)
    fgm_all_histogram = tf.zeros(10)
    fgm_correct_histogram = tf.zeros(10)
    pgd_all_histogram = tf.zeros(10)
    pgd_correct_histogram = tf.zeros(10)

    def add_confs(all_hist, correct_hist, y, p, c):
        correct = tf.cast(y == tf.argmax(p, 1), tf.float32)
        disc_c = tf.cast(tf.math.floor(tf.reshape(c, tf.size(c))*10), tf.int32)
        one_hot = tf.one_hot(disc_c, 10)
        all_to_add = tf.math.reduce_sum(one_hot, axis=0)
        correct_to_add = tf.einsum('ij,i->j', one_hot, correct)
        all_hist += all_to_add
        correct_hist += correct_to_add
        return all_hist, correct_hist
        
    for x, y in data.test:
        p, c = model(x)
        c = tf.math.sigmoid(c)
        test_acc_clean(y, p)
        test_conf_clean(c)
        clean_all_histogram, clean_correct_histogram = add_confs(y, p, c)

        x_fgm = fast_gradient_method(model, x, kwargs['eps'], np.inf, loss_fn=conf_loss, y=tf.argmax(p, 1))
        p_fgm, c = model(x_fgm)
        c = tf.math.sigmoid(c)
        test_acc_fgsm(y, p_fgm)
        test_conf_fgsm(c)
        fgm_all_histogram, fgm_correct_histogram = add_confs(y, p_fgm, c)

        x_pgd = projected_gradient_descent(model, x, kwargs['eps'], 0.01, 40, np.inf, loss_fn=conf_loss, y=tf.argmax(p, 1))
        p_pgd, c = model(x_pgd)
        c = tf.math.sigmoid(c)
        test_acc_pgd(y, p_pgd)
        test_conf_pgd(c)
        pgd_all_histogram, pgd_correct_histogram = add_confs(y, p_pgd, c)

        progress_bar_test.add(x.shape[0])


    def fn(elem):
        if tf.math.is_nan(elem):
            elem = 0
        return elem
    clean_hist = tf.map_fn(fn, clean_correct_histogram/clean_all_histogram)*100
    fgm_hist = tf.map_fn(fn, fgm_correct_histogram/fgm_all_histogram)*100
    pgd_hist = tf.map_fn(fn, pgd_correct_histogram/pgd_all_histogram)*100

    print(
        "test acc on clean examples (%): {:.3f} - test conf on clean examples (%): {:.3f}".format(
            test_acc_clean.result() * 100,
            test_conf_clean.result() * 100,
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f} - test conf on on FGM adversarial examples (%): {:.3f}".format(
            test_acc_fgsm.result() * 100,
            test_conf_fgsm.result() * 100,
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f} - test conf on on PGD adversarial examples (%): {:.3f}".format(
            test_acc_pgd.result() * 100,
            test_conf_pgd.result() * 100,
        )
    )

    plt.figure()
    plt.bar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], clean_hist, width=0.1, color='dodgerblue')
    plt.xlabel("Confidence")
    plt.ylabel("Correctly classified (%)")
    plt.show()

    plt.figure()
    plt.bar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fgm_hist, width=0.1, color='fuchsia')
    plt.xlabel("Confidence")
    plt.ylabel("Correctly classified (%)")
    plt.show()

    plt.figure()
    plt.bar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], pgd_hist, width=0.1, color='orange')
    plt.xlabel("Confidence")
    plt.ylabel("Correctly classified (%)")
    plt.show()


def run_fashion_noise(*args, **kwargs):
    # Load training and test data
    data = ld_fashion_mnist(batch_size=kwargs['batch_size'])
    if kwargs['noise']:
        model = create_small_CNN_noise()
    else:
        model = create_small_CNN()
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    # Metrics to track the different accuracies.
    train_loss = tf.metrics.Mean(name="train_loss")

    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

    if kwargs['adv_train']:
        adv_batch_size = math.floor(kwargs['batch_size']*kwargs['adv_train_coeff'])
    else:
        adv_batch_size = 0

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_object(y, model(x))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    # Train model with adversarial training
    for epoch in range(kwargs['nb_epochs']):
        # keras like display of progress
        progress_bar_train = tf.keras.utils.Progbar(60000)
        for (x, y) in data.train:
            if adv_batch_size:
                # Replace clean example with adversarial example for adversarial training
                x_adv = projected_gradient_descent(model, x[:adv_batch_size, ...], kwargs['eps'], 0.01, 40, np.inf)
                x_clean = x[adv_batch_size:, ...]
                x = tf.concat([x_adv, x_clean], 0)
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])

    # Evaluate on clean and adversarial data
    progress_bar_test = tf.keras.utils.Progbar(10000)
    for x, y in data.test:
        p = model(x)
        test_acc_clean(y, p)

        x_fgm = fast_gradient_method(model, x, kwargs['eps'], np.inf)
        p_fgm = model(x_fgm)
        test_acc_fgsm(y, p_fgm)

        x_pgd = projected_gradient_descent(model, x, kwargs['eps'], 0.01, 40, np.inf)
        p_pgd = model(x_pgd)
        test_acc_pgd(y, p_pgd)

        progress_bar_test.add(x.shape[0])

    print(
        "test acc on clean examples (%): {:.3f}".format(
            test_acc_clean.result() * 100,
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            test_acc_fgsm.result() * 100,
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            test_acc_pgd.result() * 100,
        )
    )