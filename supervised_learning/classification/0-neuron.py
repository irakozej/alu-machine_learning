#!/usr/bin/env python3

import numpy as np

class neuron:

	def __init__(self, nx):
		if  type(nx) is not  int:
 			raise TypeError('nx must be an interger')
                if nx <1:
                        raise TypeError('nx must be a positive integer')
                self.W = np.random.randn(1, nx)
                self.b = 0
                self.A = 0
