#!/usr/bin/env python
'''
This is the second experiment.
'''

import tensorflow as tf
import numpy as np
from os.path import join,isdir,isfile
from os import mkdir
from json import dump
from dataset_2 import dataset,labels
from models import models




output_folder = 'results'
output_file = join(output_folder,'results.csv')
input_sizes = [i for i in range(10,51,10)]
