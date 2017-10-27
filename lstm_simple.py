#! /usr/bin/env python
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import argparse
import sys
import tensorflow as tf
import sklearn
import time
import os
import shutil
from datetime import datetime
from terminaltables import AsciiTable
try:
    import cPickle as pickle # python2
except ImportError:
    import pickle

from keras.layers import Input, Dense, concatenate
from keras.layers.core import RepeatVector, Flatten, Reshape, Dropout
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model, Sequential
from keras.datasets import mnist
from keras import regularizers
from keras import initializers
from keras import backend as K
from keras import optimizers
from keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard,
                             History, ReduceLROnPlateau, CSVLogger)
from animation_simple import Animation

class Raw():
    pass

class Paths():
    pass

class Dataset():
    def __init__(self):
        self.train = Paths()
        self.test = Paths()
        self.val = Paths()

class Data():
    abs_full = None
    abs = Dataset()
    rel = Dataset()
    raw = Raw()

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='data/circles/coordinates2')
parser.add_argument('-m', '--model-path', type=str, default='./models/default_model')
parser.add_argument('-p', '--patience', type=int, default=2)
parser.add_argument('-lm', '--load-model-path', type=str, default='')
parser.add_argument('-sm', '--save-model-path', type=str, default='./best_model.h5')

parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('-ss', '--step-size', type=int, default=1)
parser.add_argument('-ipl', '--input-point-len', type=int, default=30)
parser.add_argument('-u', '--lstm-units', type=int, default=128)
parser.add_argument('-dr1', '--dropout-input', type=float, default=0.2)
parser.add_argument('-dr3', '--dropout-output', type=float, default=0.3)
parser.add_argument('-lr', '--learning-rate', type=float, default=0.0002)
parser.add_argument('-lo', '--loss', type=str, default='mean_absolute_error')
parser.add_argument('-gc', '--gradient-clip', type=float, default=10.0)
parser.add_argument('-opt', '--optimizer', type=str, default='rmsprop')
parser.add_argument('-bs', '--batch-size', type=int, default=64)
parser.add_argument('-pl', '--test-pred-len', type=int, default=40)

parser.add_argument('-tsp', '--test-split', type=float, default=0.1)
parser.add_argument('-vsp', '--val-split', type=float, default=0.1)
parser.add_argument('-tpl', '--train-pred-len', type=int, default=5)
parser.add_argument('-v', '--verbose', type=int, default=1)
parser.add_argument('-hc', '--hyperopti-count', type=int, default=100)

parser.add_argument('-no-center', '--no-center', action='store_true')
parser.add_argument('-na', '--no-animation', action='store_true')
parser.add_argument('-scatter', '--scatter', action='store_true')
parser.add_argument('-hyperopti', '--hyperopti', action='store_true')
args = parser.parse_args()

assert args.data != '', 'need data'

###############
# Functions
###############

def print_dataset_sizes(train_split, data):
    table_data = [
            ['', 'train', 'test', 'val'],
            ['splits', train_split, args.test_split, args.val_split],
            ['points', len(data.raw.train), len(data.raw.test),
             len(data.raw.val)]
    ]
    table = AsciiTable(table_data)
    print(table.table)

def print_data_shape(dataset):
    table_data = [['shapes', 'x', 'y']]

    p = lambda s, d: table_data.append([s, d.x.shape, d.y.shape])
    p('train:', dataset.train)
    p('test:', dataset.test)
    p('val:', dataset.val)

    table = AsciiTable(table_data)
    print(table.table)

def print_model_params():
    params = [['epochs', args.epochs]]
    params.append(['batch size', args.batch_size])
    params.append(['lstm units', args.lstm_units])
    params.append(['input point len', args.input_point_len])
    params.append(['gradient clip', args.gradient_clip])
    params.append(['train pred len', args.train_pred_len])
    params.append(['test pred len', args.test_pred_len])
    params.append(['dropout input', args.dropout_input])
    params.append(['learning rate', args.learning_rate])
    params.append(['optimizer', args.optimizer])

    table = AsciiTable(params)
    table.inner_heading_row_border = False
    print(table.table)

def undo_standardization(data, dataset):
    dataset = np.copy(dataset)
    dataset *= data.std_rel
    dataset += data.mean_rel
    return dataset

def sliding_window(data, size=5, step_size=1):
    steps = range(0, len(data) - (size - 1), step_size)
    slided = np.zeros((len(steps), size) + data.shape[1:])
    for i, pos in enumerate(steps):
        slided[i] = data[pos:pos + size]
    return slided

def reshape_paths(paths, flat=True):
    if flat:
        return paths.reshape(paths.shape[0], paths.shape[1] * 2)
    else:
        return paths.reshape(paths.shape[0], paths.shape[1] // 2, 2)

def abs_to_rel_paths(paths):
    for i, abs_path in enumerate(paths):
        for j, coord in enumerate(abs_path):
            if j > 0:
                paths[i][j - 1] = paths[i][j] - paths[i][j - 1]
    return paths[:, :-1]

def abs_to_rel_dataset(dataset):
    dataset = np.copy(dataset)
    for i, coord in enumerate(dataset):
        if i > 0:
            dataset[i - 1] = dataset[i] - dataset[i - 1]
    return dataset[:-1]

def get_data():
    dataframe = pandas.read_csv(args.data, sep=' ', engine='python',
                                header=None)
    dataframe = dataframe[dataframe[0] >= 0]
    dataframe = dataframe[dataframe[1] >= 0]
    abs_dataset = dataframe.as_matrix()

    return abs_dataset

def gen_train_data(data, dataset, rel_coords, pred_len, center=True,
                   flat_x=False, flat_y=False):
    """
    Stamps out the input vectors and the labels
    First stamps input and labels together, they converts to rel coords if
    needed, then seperates into X and y
    """
    dataset = np.copy(dataset)

    size = args.input_point_len + pred_len
    x_and_y = sliding_window(dataset, size, 1)

    if rel_coords:
        x_and_y = abs_to_rel_paths(x_and_y)
        if center:
            x_and_y -= data.mean_rel
            x_and_y /= data.std_rel
        divider = args.input_point_len - 1
    else:
        divider = args.input_point_len

    X = x_and_y[:, :divider, :]
    y = x_and_y[:, divider:, :]

    if flat_x:
        X = reshape_paths(X, flat=True)
    if flat_y:
        y = reshape_paths(y, flat=True)

    return X, y

def get_x_y():
    """
    Returns a dict containing X, y for both inputs for the train, test and val sets
    """
    data = Data()
    data.abs_full = get_data()
    data.rel_full = abs_to_rel_dataset(data.abs_full)
    # add std of relative data
    data.std_rel = np.std(data.rel_full, axis=0)
    data.std_abs = np.std(data.abs_full, axis=0)
    data.mean_rel = np.mean(data.rel_full, axis=0)
    data.mean_abs = np.mean(data.abs_full, axis=0)

    train_split = 1 - (args.test_split + args.val_split)
    train_len = int(len(data.abs_full) * train_split)
    test_len = int(len(data.abs_full) * args.test_split)

    data.raw.train = data.abs_full[0:train_len]
    data.raw.test = data.abs_full[train_len:train_len + test_len]
    # take the val data from the end to make the animation more useful
    data.raw.val = data.abs_full[train_len + test_len:]

    data.abs.train.x, data.abs.train.y = gen_train_data(data, data.raw.train, False,
                                                        args.train_pred_len,
                                                        flat_y=True)
    data.abs.test.x, data.abs.test.y = gen_train_data(data, data.raw.test, False,
                                                      args.test_pred_len)
    data.abs.val.x, data.abs.val.y = gen_train_data(data, data.raw.val, False,
                                                    args.train_pred_len,
                                                    flat_y=True)
    data.rel.train.x, data.rel.train.y = gen_train_data(data, data.raw.train, True,
                                                        args.train_pred_len,
                                                        center=not args.no_center,
                                                        flat_y=True)
    data.rel.test.x, data.rel.test.y = gen_train_data(data, data.raw.test, True,
                                                      args.test_pred_len,
                                                      center=not args.no_center)
    data.rel.val.x, data.rel.val.y = gen_train_data(data, data.raw.val, True,
                                                    args.train_pred_len,
                                                    center=not args.no_center,
                                                    flat_y=True)
    if args.verbose is not 0:
        print_dataset_sizes(train_split, data)
    return data


def hyperopti():
    def write_files(path, parameters, model, history, lowest_loss_idx,
                    delta, num_run):
        # save model
        model.save(path + 'final_model.h5')
        shutil.copyfile(args.save_model_path, path + 'best_model.h5')

        # config file
        f = open(path + 'config', 'w')
        f.write('data' + '\t' + args.data + '\n')
        f.write('test split' + '\t' + str(args.test_split) + '\n')
        f.write('input point len' + '\t' + str(args.input_point_len) + '\n')
        f.write('loss' + '\t' + str(history['loss'][lowest_loss_idx]) + '\n')
        f.write('val_loss' + '\t' + str(history['val_loss'][lowest_loss_idx]) + '\n')
        for key in parameters.keys():
            f.write(key + '\t' + str(getattr(args, key)) + '\n')
        f.write('epochs' + '\t' + str(args.epochs) + '\n')
        f.write('time' + '\t' + str(int(delta)) + 's\n')
        f.close()

        # losses text file
        f = open(path + 'losses', 'w')
        f.write('loss\tval_loss\n')
        for idx in range(len(history['loss'])):
            f.write(str(history['loss'][idx]) + '\t')
            f.write(str(history['val_loss'][idx]) + '\n')
        f.close()

        # write graphs
        lines = [e[:4] + '= ' + str(getattr(args, e))[:5] for e in parameters.keys()]
        text_str = '\n'.join(lines)

        plt.cla()
        plt.title('run ' + str(num_run))
        plt.text(0.8, 0.65, text_str, transform=plt.gca().transAxes)
        plt.axhline(y=-50.0, color='black', linestyle='--', linewidth=1)
        plt.plot(history['loss'], label='loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend(loc='upper left')
        plt.savefig(path + 'plot' + str(num_run) + '.png')


    parameters = {
            # 'lstm_units': lambda: 2**int(np.random.uniform(1, 9)),
            # 'input_point_len': lambda: 2**int(np.random.uniform(2, 6)),
            'dropout': lambda: np.random.randint(2, 7) * 0.1,
    }
    args.verbose = 0
    keys = list(parameters.keys()) + ['loss', 'val_loss', 'epochs', 'time']
    print('\t'.join(keys))

    # needed for the results dir name
    format = '%Y-%m-%d_%H:%M:%S'
    current_time = datetime.fromtimestamp(time.time()).strftime(format)
    for i in range(args.hyperopti_count):
        for key, func in parameters.items():
            setattr(args, key, func())

        values = [getattr(args, key) for key in parameters.keys()]
        values_str = '\t'.join([str(val) for val in values])
        print(values_str, end='')
        sys.stdout.flush()

        start = time.time()
        history, model = run()
        history = history.history
        delta = time.time() - start
        lowest_loss_idx = np.argmin(history['val_loss'])
        loss = "{0:.6f}".format(history['loss'][lowest_loss_idx])
        val_loss = "{0:.6f}".format(history['val_loss'][lowest_loss_idx])

        # print results
        results = [loss, val_loss]
        results.append(len(history['loss']))
        results.append(str(int(delta)) + 's')
        results_str = '\t' + '\t'.join([str(res) for res in results])
        print (results_str)
        sys.stdout.flush()

        # directory for the hyperopt results
        main_dirname = __file__.split('/')[-1] + '-' + current_time
        path = './hyperopt/' + main_dirname + '/'
        path += '_'.join(['run', str(i), loss, val_loss]) + '/'
        os.makedirs(path)
        write_files(path, parameters, model, history, lowest_loss_idx, delta, i)

def run():
    data = get_x_y()
    dataset = data.rel

    ##########
    # Model
    #########

    if not args.load_model_path:
        if args.hyperopti:
            # clear the backend to start from scratch
            K.clear_session()

        callbacks = [History(),
                     ModelCheckpoint(args.save_model_path, monitor='val_loss',
                                     save_best_only=True, verbose=0),
                     EarlyStopping(monitor='val_loss', patience=6),
                     CSVLogger('history.log')]

        input_point_len = args.input_point_len - 1

        model = Sequential()
        model.add(Dropout(args.dropout_input, input_shape=(input_point_len, 2)))
        model.add(LSTM(args.lstm_units))
        model.add(Dense(2 * args.train_pred_len))

        if args.optimizer == 'adam':
            optimizer = optimizers.Adam(clipvalue=args.gradient_clip)
        elif args.optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=args.learning_rate,
                                           clipvalue=args.gradient_clip)
        elif args.optimizer == 'nadam':
            optimizer = optimizers.Naam(clipvalue=args.gradient_clip)
        model.compile(optimizer=optimizer, loss=args.loss)

        if args.verbose is not 0:
            print_data_shape(dataset)
            print_model_params()
            print(model.summary())
            print()

        history = model.fit(dataset.train.x, dataset.train.y,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  shuffle=True,
                  validation_data=(dataset.val.x, dataset.val.y),
                  callbacks=callbacks,
                  verbose=args.verbose)

    #############
    # Prediction
    #############

    if not args.hyperopti:
        print('\nprediction start')
        if args.load_model_path:
            model = load_model(args.load_model_path)
        else:
            model = load_model(args.save_model_path)

        # prediction setup
        test_size = len(dataset.test.x)
        pred_len = args.train_pred_len
        preds = np.zeros((test_size, args.test_pred_len, 2))
        x = np.copy(dataset.test.x)
        # predict all trajectories at once
        for i in range(0, args.test_pred_len, pred_len):
            pred_points = model.predict_on_batch(x) # (test_size, pred_len * 2)
            pred_points = pred_points.reshape((test_size, pred_len, 2))
            preds[:, i:i + pred_len] = pred_points
            # appending prediction to next input
            x[:, :-pred_len] = x[:, pred_len:]
            x[:, -pred_len:] = pred_points
            print(str(i) + '/' + str(args.test_pred_len))

        print('prediction finished\n')

        ####################
        # Prediction error calculation
        ####################

        # compute elementwise squared error of the relative data
        se = (preds - data.rel.test.y) ** 2
        print('data: ' + args.data.split('/')[-2])
        print('mse avg: ' + "{0:.4f}".format(np.average(se)))

        #############
        # Animation
        #############

        if not args.no_animation:
            Animation(data.abs.test.x,
                      data.abs.test.y,
                      data.rel.test.y,
                      data.mean_rel,
                      data.std_rel,
                      preds)

    else:
        return history, model

if args.hyperopti:
    hyperopti()
else:
    run()