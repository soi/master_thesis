# Prediction of human touch input
This is the repository of the final version of my thesis' source code and text, which goes by the title "Self-supervised prediction of sensory-motor trajectories with deep neural nets". It contains the four models mentioned in the thesis.

## Getting started
I have tested all scripts with Python 2.7 but they should word with Python 3 as well. To start learning with the default parameters simply run one of the files, e.g.:

```
python2 simple_lstm.py
```

or

```
python2 lstm_mdn2d.py
```

## Use different data or parameters
All the datasets can be found in the folder `data`. Be sure to use the `coordinates2` file of each folder as they contain the cleaned up touch points. The preprocessing scripts can be found in the data folder as well. The parameters are usually the same for all scripts.

To use a different dataset use the `-d` flag with the full path:

```
python2 lstm_mdn2d.py -d data/horizontal_lines/coordinates2
```

To use a previously trained model from the folder `results` use the `-lm` flag with the full path together with the corresponding dataset

```
python2 lstm_mdn2d.py -d data/horizontal_lines/coordinates2 -lm results/lstm_mdn2d/horizontal_lines.h5
```

There are many other parameters and to see them all use `--help`:

```
python2 lstm_mdn2d.py --help
```

