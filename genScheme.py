#!/usr/bin/env python
from genDirs import scheme_2s_1s_linPlan

import sys

if __name__ == "__main__":
	print('N1 N2 Nlow b1 b2 blow outfilename')

	N = [int(sys.argv[1]), int(sys.argv[2])]
	N_lowlow = [int(sys.argv[3])]
	bvalues = [int(sys.argv[4]), int(sys.argv[5])]
	bvalues_lowlow = [int(sys.argv[6])]
	filenameout = sys.argv[7]

	scheme_2s_1s_linPlan(N, N_lowlow, bvalues, bvalues_lowlow, filenameout, display=False)
