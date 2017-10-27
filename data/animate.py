#! /usr/bin/env python2
from __future__ import print_function, division
from basic_animation import Animation
import numpy as np
import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='many_circles/coordinates2')
parser.add_argument('-p', '--points', type=int, default='30')
args = parser.parse_args()

def get_data():
    dataframe = pandas.read_csv(args.data, sep=' ', engine='python',
                                header=None)
    dataframe = dataframe[dataframe[0] >= 0]
    dataframe = dataframe[dataframe[1] >= 0]
    abs_dataset = dataframe.as_matrix()

    return abs_dataset

def sliding_window(data, size=5, step_size=1):
    steps = range(0, len(data) - (size - 1), step_size)
    slided = np.zeros((len(steps), size) + data.shape[1:])
    for i, pos in enumerate(steps):
        slided[i] = data[pos:pos + size]
    return slided

data = get_data()
data = sliding_window(data, args.points, 1)
Animation(data)
