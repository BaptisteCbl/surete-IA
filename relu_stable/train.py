"""Trains a model, saving checkpoints along the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import shutil
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange

import logging
import sys
import time

from datasubset import DataSubset
import models.MNIST_improved_ia
import models.MNIST_naive_ia
from pgd_attack import LinfPGDAttack

class arguments:
    def add_dataset(self, dataset):
        self.dataset = dataset
    def add_model(self, model):
        self.model = model
    def add_train_param(self, param):
        self.train_param = param
    def add_attack(self, attack):
        self.attack = attack
    def add_eval_param(self,param):
        self.eval_param = param
    def add_eval_adv_param(self,param):
        self.eval_adv_param = param

with open('config.json') as config_file:
    config = json.load(config_file)
if os.path.exists('job_parameters.json'):
    with open('job_parameters.json') as config_file:
        param_config = json.load(config_file)
    for k in param_config.keys():
        assert k in config.keys()
    config.update(param_config)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

# Training parameters
data_set = config['data_set']
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
num_eval_steps = config['num_eval_steps']
dataset_size = config['num_training_examples']
batch_size = config['training_batch_size']
eval_during_training = config['eval_during_training']
adv_training = config['adversarial_training']
w_l1 = config["w_l1"]
w_rsloss = config["w_rsloss"]

# Eval parameters
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

# Output directory
model_dir = config['model_dir']

# Setting up the training data
if data_set == "fashion_mnist":
    mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
else:
    mnist = input_data.read_data_sets('data/mnist', one_hot=False)

mnist_train = DataSubset(mnist.train.images,
                         mnist.train.labels,
                         dataset_size)
global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the model
if config["estimation_method"] == 'improved_ia':
    model = models.MNIST_improved_ia.Model(config)
elif config["estimation_method"] == 'naive_ia':
    model = models.MNIST_naive_ia.Model(config)
else:
    print("Defaulting to Naive IA for ReLU bound estimation")
    model = models.MNIST_naive_ia.Model(config)

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent + \
                                                    w_l1 * model.l1_loss + \
                                                    w_rsloss * model.rsloss,
                                                    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'],
                       config['incremental'])

# Set up eval adversary in the case of an incremental training schedule
eval_attack = LinfPGDAttack(model,
                       config['eval_epsilon'],
                       40,
                       config['eval_epsilon']/10.0,
                       config['random_start'],
                       config['loss_func'])

args = arguments()
args.add_dataset("dataset="+config["data_set"])
args.add_model("estimation_method="+config["estimation_method"])
args.add_train_param("randomseed="+str(config['random_seed'])+", max_num_training_steps="+str(config['max_num_training_steps'])+", num_output_steps="+str(config['num_output_steps'])+", num_summary_steps="+str(config['num_summary_steps'])+", num_checkpoint_steps="+str(config['num_checkpoint_steps'])+", num_eval_steps="+str(config['num_eval_steps'])+", dataset_size="+str(config['num_training_examples'])+", batch_size="+str(config['training_batch_size'])+", eval_during_training="+str(config['eval_during_training'])+", adv_training="+str(config['adversarial_training'])+", w_l1="+str(config["w_l1"])+", w_rsloss="+str(config["w_rsloss"]))
args.add_attack(("LinfPGDAttack(eps="+str(config['epsilon'])+", k="+str(config['k'])+", a="+str(config['a'])+", random_start="+str(config['random_start'])+", loss_func="+str(config['loss_func'])+", incremental="+str(config['incremental'])+")") if adv_training else "None")
args.add_eval_adv_param(("LinfPGDAttack(eps="+str(config['eval_epsilon'])+", k="+"40"+", a="+str(config['eval_epsilon']/10.0)+", random_start="+str(config['random_start'])+", loss_func="+str(config['loss_func'])+", incremental="+"false"+")") if adv_training&eval_during_training else "None")


trainedmodelsdir = os.getcwd() + "\\trained_models\\"
if not os.path.exists(trainedmodelsdir):
    os.mkdir(trainedmodelsdir)

trainedmodelsdir = trainedmodelsdir +"\\"+ data_set
if not os.path.exists(trainedmodelsdir):
    os.mkdir(trainedmodelsdir)

i = 0
tmodeldir = trainedmodelsdir+ "\\trained_model" + str(i)
while os.path.exists(tmodeldir):
    i+=1
    tmodeldir = trainedmodelsdir+ "\\trained_model" + str(i)
os.mkdir(tmodeldir)    

# Setting up the Tensorboard and checkpoint outputs
ckptdir = tmodeldir + "\\ckpt"
if not os.path.exists(ckptdir):
    os.makedirs(ckptdir)

# Keep track of accuracies in Tensorboard
saver = tf.train.Saver(max_to_keep=3)

shutil.copy('config.json', model_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    log_train = tmodeldir + "\\log_train.csv"
    logger_train = logging.getLogger("train")
    logger_train.setLevel(logging.INFO)
    for handler in logger_train.handlers[:]:
        logger_train.removeHandler(handler)
    handler_train = logging.FileHandler(log_train)
    logger_train.addHandler(handler_train)

    logger_train.info("# " + args.dataset)
    logger_train.info("# " + args.train_param)
    if adv_training:
        logger_train.info("# " + args.attack)
    if eval_during_training:
        logger_train.info("# " + args.eval_adv_param)

    #logger.info("Epoch, Train loss,Train acc, Time, Time elapsed, Train loss adv,Train acc adv, unstablerelus1, unstablerelus2, unstablerelus3, ur1loss, ur2loss, ur3loss")
    logger_train.info("Epoch, Train loss,Train acc, Time, Time elapsed")

    if eval_during_training:
        log_eval = tmodeldir + "\\log_eval.csv"
        logger_eval = logging.getLogger("eval")
        logger_eval.setLevel(logging.INFO)
        handler_eval = logging.FileHandler(log_eval)
        logger_eval.addHandler(handler_eval)

        logger_eval.info("# " + args.dataset)
        logger_eval.info("# " + args.train_param)
        if adv_training:
            logger_eval.info("# " + args.attack)
        if eval_during_training:
            logger_eval.info("# " + args.eval_adv_param)
        logger_eval.info("Epoch, nat acc, adv acc, nat loss, adv loss, unstablerelus1, unstablerelus2, unstablerelus3, ur1loss, ur2loss, ur3loss")
    

    sess.run(tf.compat.v1.global_variables_initializer())
    training_time = 0.0
    total_time = 0

    # Main training loop
    for ii in range(max_num_training_steps + 1):
        start_time = time.time()
        x_batch, y_batch = mnist_train.get_next_batch(batch_size,
                                                      multiple_passes=True)

        # Compute Adversarial Perturbations
        start = timer()
        if adv_training:
            x_batch_adv = attack.perturb(x_batch, y_batch, sess, ii/max_num_training_steps)
        else:
            x_batch_adv = x_batch
        end = timer()
        training_time += end - start

        nat_dict = {model.x_input: x_batch,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv,
                    model.x_input_natural: x_batch,
                    model.y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            nat_loss = sess.run(model.xent, feed_dict=nat_dict)
            print('    Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training nat loss {:.4}'.format(nat_loss))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                
                logger_train.info("{},{:.4f},{:.4f},{:.2f},{:.2f}".format(
                    ii,
                    nat_loss,
                    100 * nat_acc,
                    training_time,
                    total_time
                    )
                )  
                total_time += training_time
                training_time = 0.0
       

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save( sess,
                        os.path.join(ckptdir, 'checkpoint'),
                        global_step=global_step)

        # Evaluate
        if eval_during_training and ii % num_eval_steps == 0 and ii > 0:
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
            total_xent_nat = 0.
            total_xent_adv = 0.
            total_corr_nat = 0
            total_corr_adv = 0
            tot_unstable1 = 0
            tot_unstable2 = 0
            tot_unstable3 = 0
            tot_unstable1l = 0
            tot_unstable2l = 0
            tot_unstable3l = 0

            for ibatch in trange(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)

                x_batch_eval = mnist.test.images[bstart:bend, :]
                y_batch_eval = mnist.test.labels[bstart:bend]

                dict_nat_eval = {model.x_input: x_batch_eval,
                               model.x_input_natural: x_batch_eval,
                               model.y_input: y_batch_eval}

                x_batch_eval_adv = eval_attack.perturb(x_batch_eval, y_batch_eval, sess)

                dict_adv_eval = {model.x_input: x_batch_eval_adv,
                               model.x_input_natural: x_batch_eval,
                               model.y_input: y_batch_eval}

                cur_corr_nat, cur_xent_nat = sess.run(
                                              [model.num_correct,model.xent],
                                              feed_dict = dict_nat_eval)
                cur_corr_adv, cur_xent_adv = sess.run(
                                              [model.num_correct,model.xent],
                                              feed_dict = dict_adv_eval)
                un1, un2, un3 = \
                sess.run([model.unstable1, model.unstable2, \
                          model.unstable3],
                          feed_dict = dict_nat_eval)
                un1l, un2l, un3l = \
                sess.run([model.un1loss, model.un2loss, \
                          model.un3loss],
                          feed_dict = dict_nat_eval)
                tot_unstable1 += np.sum(un1)
                tot_unstable2 += np.sum(un2)
                tot_unstable3 += np.sum(un3)
                tot_unstable1l += w_rsloss * un1l
                tot_unstable2l += w_rsloss * un2l
                tot_unstable3l += w_rsloss * un3l

                total_xent_nat += cur_xent_nat
                total_xent_adv += cur_xent_adv
                total_corr_nat += cur_corr_nat
                total_corr_adv += cur_corr_adv

            avg_un1 = tot_unstable1 / num_eval_examples
            avg_un2 = tot_unstable2 / num_eval_examples
            avg_un3 = tot_unstable3 / num_eval_examples
            avg_un1l = tot_unstable1l / num_eval_examples
            avg_un2l = tot_unstable2l / num_eval_examples
            avg_un3l = tot_unstable3l / num_eval_examples

            avg_xent_nat = total_xent_nat / num_eval_examples
            avg_xent_adv = total_xent_adv / num_eval_examples
            acc_nat = total_corr_nat / num_eval_examples
            acc_adv = total_corr_adv / num_eval_examples

            print('Eval at {}:'.format(ii))
            print('  natural: {:.2f}%'.format(100 * acc_nat))
            print('  adversarial: {:.2f}%'.format(100 * acc_adv))
            print('  avg nat loss: {:.4f}'.format(avg_xent_nat))
            print('  avg adv loss: {:.4f}'.format(avg_xent_adv))
            print('  unstablerelus1, unstablerelus2, unstablerelus3: {}, {}, {}'.format(avg_un1,
                avg_un2, avg_un3))
            print('  ur1loss, ur2loss, ur3loss: {}, {}, {}'.format(avg_un1l,
                avg_un2l, avg_un3l))

            logger_eval.info("{},{:.2f},{:.2f},{:.4f},{:.4f},{},{},{},{},{},{}".format(
                    ii,
                    100 * acc_nat,
                    100 * acc_adv,
                    avg_xent_nat,
                    avg_xent_adv,
                    avg_un1,
                    avg_un2,
                    avg_un3,
                    avg_un1l,
                    avg_un2l,
                    avg_un3l,
                    )
                )

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        end = timer()
        training_time += end - start
