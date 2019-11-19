import numpy as np
import pandas as pd

from lm import *

def summary(object):
    if isinstance(object, lm):
        pass
    else:
        print('Input object is not of type lm, cannot output summary.')
        raise TypeError


def predict(object):
    pass