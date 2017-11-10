#!/usr/bimn/env python

import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np


'''
Range of model
Aspect Ratio: 1 - 8  diameter/thickness   Temperature  573 - 1473 K     Shape: 0 cylinder  1 square

To generate a correction factor run the program and change the values of the parameters

The value outputted can be multiplied by the value of thermal diffusivity calculated from
Parker's equation using the half rise time

'''
#PS I LOVE COMPUTER SCIENCE and Mathematics!



pickle_in = open('r_forest_regressor.pickle','rb')
clf = pickle.load(pickle_in)



aspect = 2 #diameter/thickness
temperature = 512 #[K]
shape = 0# 0 cylinder, 1 Rectange,

def correction_factor(aspect, temperature, shape):

	data = np.asarray((aspect,temperature,shape))

	calc_over_actual = clf.predict(data)
	factor = 1 / calc_over_actual
	print(factor)
	return factor


correction = correction_factor(aspect, temperature, shape)
