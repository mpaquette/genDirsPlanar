#!/usr/bin/env python

import numpy as np
from gs import angleVec3

import sys

if __name__ == "__main__":
	# loading data
	scheme = np.genfromtxt(sys.argv[1])

	print('\n\nshape: {}   (should be Nx9)'.format(scheme.shape))

	n1 = np.linalg.norm(scheme[:,:3], axis=1)
	n2 = np.linalg.norm(scheme[:,3:6], axis=1)
	n3 = np.linalg.norm(scheme[:,6:], axis=1)
	n = np.concatenate((n1[:,None], n2[:,None], n3[:,None]), axis=1)
	print('\n\nVector norms:')
	for i in range(n.shape[0]):
		print(n[i])

	print('\n\nB scalling norms:')
	for i in range(n.shape[0]):
		print(n[i]**2)

	print('\n\nAngle between first and second')
	for i in range(scheme.shape[0]):
		print((180/np.pi)*angleVec3(scheme[i,:3], scheme[i,3:6]))
