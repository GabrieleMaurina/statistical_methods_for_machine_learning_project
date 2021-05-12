#!/usr/bin/env python
'''
This script analyzes the entire fruits folder and saves numpy arrays ready to be
used by a machine learning model.
'''





from dataset import dataset





for i in range(10,51,10):
    dataset(i)
