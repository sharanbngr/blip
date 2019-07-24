# -*- coding: utf-8 -*-
import numpy as np
import random

'''
Code to set (and hopefully keep) random state variables for the purpose of reproducability.
'''

def SetRandomState(seed):
    np.random.seed(seed)
    random.seed(seed)
    randst = np.random.RandomState(seed)
    return randst


