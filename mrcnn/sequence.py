import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from mrcnn import utils

class MRCNNSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [xx[idx * self.batch_size:(idx + 1) * self.batch_size] for xx in self.x]
        batch_y = [yy[idx * self.batch_size:(idx + 1) * self.batch_size] for yy in self.y]

        return batch_x, batch_y
