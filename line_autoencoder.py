#! /usr/bin/env python
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import argparse
import tensorflow as tf
try:
    import cPickle as pickle # python2
except ImportError:
    import pickle

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from animation_autoencoder import Animation

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='data/rectangles/coordinates2')
parser.add_argument('-m', '--model_path', type=str, default='./models/default_model')
parser.add_argument('-p', '--patience', type=int, default=1)
parser.add_argument('-lm', '--load_model_path', type=str, default='')
parser.add_argument('-sm', '--save_model_path', type=str, default='./latest_model.hdf5')
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-ss', '--step_size', type=int, default=1)
parser.add_argument('-ldim', '--latent_dims', type=int, default=2)
parser.add_argument('-lo', '--loss', type=str, default='mean_absolute_error')
parser.add_argument('-opt', '--optimizer', type=str, default='adam')
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('-il', '--input_len', type=int, default=40)
parser.add_argument('-tsp', '--test_split', type=float, default=0.15)
parser.add_argument('-na', '--no_animation', action='store_true')
parser.add_argument('-lsplot', '--latent_spaces_plots', action='store_true')
args = parser.parse_args()

callbacks = [EarlyStopping(monitor='val_loss', patience=args.patience,
                           verbose=0),
             ModelCheckpoint(args.save_model_path, monitor='val_loss',
                             save_best_only=True, verbose=0)
             ]

scatter_fig = plt.figure()
scatter_ax = scatter_fig.add_subplot(1,1,1)
onclick_fig = plt.figure()
onclick_ax = onclick_fig.add_subplot(1,1,1)

def reshape_paths(paths, flat=True):
    if flat:
        return paths.reshape(paths.shape[0], paths.shape[1] * 2)
    else:
        return paths.reshape(paths.shape[0], paths.shape[1] // 2, 2)

def generate_training_data(dataset, args):
    """
    Stamps out the input vectors
    """
    steps = range(0, len(dataset) - args.input_len, args.step_size)
    X = np.zeros((len(steps), args.input_len, 2), dtype=np.float32)
    for i, pos in enumerate(steps):
        X[i] = dataset[pos:pos + args.input_len]
    return X

def get_data(args):
    """
    Returns (abs_dataset, rel_dataset) for absolute and relative data
    """
    if not args.data:
        print('Please specify data file')
        exit()

    dataframe = pandas.read_csv(args.data, sep=' ', engine='python', header=None)
    abs_dataset = dataframe.as_matrix()

    abs_paths = generate_training_data(abs_dataset, args)
    rel_paths = np.zeros((abs_paths.shape[0], abs_paths.shape[1] -1, 2))
    for i, path in enumerate(abs_paths):
        for j, coor in enumerate(abs_paths[i]):
            if j > 0:
                rel_paths[i][j - 1] = abs_paths[i][j] - abs_paths[i][j - 1]

    abs_paths = reshape_paths(abs_paths, flat=True)
    rel_paths = reshape_paths(rel_paths, flat=True)
    return abs_paths, rel_paths

def rel_to_abs_paths(abs_paths, decoded, train_len, args):
    """
    Convert relative paths to absolute coords for the animation
    """
    decoded = reshape_paths(decoded, flat=False)
    abs_paths = reshape_paths(np.copy(abs_paths), flat=False)
    abs_paths_test = abs_paths[train_len:]
    # copy for the cumsum
    abs_paths_test[:, 1:, :] = decoded
    abs_paths_test = np.cumsum(abs_paths_test, axis=1)
    # wrap around
    return abs_paths_test % 1.0

def visualize_latent_space(latent_spaces, decoder, args):
    """
    Prints scatter plot of the latent space.
    """
    def on_click(event):
        """
        Left click: show real size
        Right click: resize
        """
        print('button pressed', event.button, event.xdata, event.ydata)
        if event.xdata is not None:
            data = np.array([[event.xdata, event.ydata],])
            decoded = decoder.predict(data)
            decoded = reshape_paths(decoded, flat=False)
            decoded = np.cumsum(decoded, axis=1)

            onclick_ax.clear()
            onclick_ax.set_title('Decoded latent space', fontsize=12)
            if event.button == 1:
                # left click
                onclick_ax.set_xlim([-0.5, 0.5])
                onclick_ax.set_ylim([-0.5, 0.5])
            else:
                onclick_ax.autoscale(True)
            onclick_ax.scatter(decoded.T[0], decoded.T[1])

    plt.ion()
    latent_spaces = latent_spaces.T
    print('variances: ', np.var(latent_spaces[0]), np.var(latent_spaces[1]))

    # only plotting the first 2 dims
    scatter_ax.scatter(latent_spaces[0], latent_spaces[1])
    scatter_fig.canvas.mpl_connect('button_press_event', on_click)
    plt.pause(0.01)

##############
# script start
##############

abs_paths, rel_paths = get_data(args)
train_len = int(len(abs_paths) * (1 - args.test_split))

effective_input_len = (args.input_len - 1) * 2
x_train, x_test = rel_paths[:train_len], rel_paths[train_len:]

input_path = Input(shape=(effective_input_len,))
encoded = Dense(78, activation='softsign',
                name='encode1')(input_path)
encoded = Dense(12, activation='softsign',
                name='encode2')(encoded)
encoded = Dense(args.latent_dims, activation='softsign',
                name='latent')(encoded)
decoded = Dense(12, activation='softsign',
                name='decode1')(encoded)
decoded = Dense(78, activation='softsign',
                name='decode2')(decoded)
decoded = Dense(effective_input_len, activation='softsign',
                name='fully-decoded')(decoded)
autoencoder = Model(input_path, decoded)

print(autoencoder.summary())

encoder = Model(input_path, encoded)

encoded_input = Input(shape=(args.latent_dims,))
decoder_layer = autoencoder.layers[-1]
ael = autoencoder.layers
decoder = Model(encoded_input, ael[-1](ael[-2](ael[-3](encoded_input))))

autoencoder.compile(optimizer=args.optimizer, loss=args.loss)
autoencoder.fit(x_train, x_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks)

# test decoding to look at the latent space
latent_spaces = encoder.predict(x_test)
decoded_paths = decoder.predict(latent_spaces)

visualize_latent_space(latent_spaces, decoder, args)

if not args.no_animation:
    decoded_paths_coor = rel_to_abs_paths(abs_paths, decoded_paths,
                                          train_len, args)
    # start the animation
    Animation(reshape_paths(abs_paths, flat=False),
            train_len,
            latent_spaces,
            decoded_paths_coor,
            scatter_ax)
