import numpy as np
from gs import findLinIndepRandomRot, gramSchmidt3
from sphHist import plotScatter3
from scilpy.samplingscheme.gen_scheme import gen_scheme
from scilpy.samplingscheme.optimize_scheme import swap_sampling_eddy


def savingLinPlan(g1,g2,scale,filename):
	g1 = g1*scale[:, None]
	g2 = g2*scale[:, None]
	g3 = np.zeros_like(g1)
	grad = np.concatenate((g1,g2,g3), axis=1)
	np.savetxt(filename, grad)


def scheme_2s_1s_linPlan(N, N_lowlow, bvalues, bvalues_lowlow, filenameout, display=False):
	## Generate 2 shell of N[0] and N[1] directions with Caruyer MS scheme for linear encoding
	## Use same np.sum(N) directions as normals for Planar encoding
	## scheme order: first b linear, first b planar, second b linear, second b planar
	## Generate independantly N_lowlow direction for low-low b-shell
	## Interleave uniformely low-low b directions
	## Using bruteforce heuristic for EDDY optimization

	# Conversion to list, NOT type testing
	if type(N) is np.ndarray:
		N = N.tolist()
	elif type(N) is tuple:
		N = list(N)

	if type(N_lowlow) is np.ndarray:
		N_lowlow = N_lowlow.tolist()
	elif type(N_lowlow) is tuple:
		N_lowlow = list(N_lowlow)
	elif type(N_lowlow) is list:
		pass
	else:
		# assumes single number
		N_lowlow = [N_lowlow]

	if type(bvalues) is np.ndarray:
		bvalues = bvalues.tolist()
	elif type(bvalues) is tuple:
		bvalues = list(bvalues)

	if type(bvalues_lowlow) is np.ndarray:
		bvalues.append(bvalues_lowlow[0])
	elif type(bvalues_lowlow) is tuple:
		bvalues.append(bvalues_lowlow[0])
	elif type(bvalues_lowlow) is list:
		bvalues.append(bvalues_lowlow[0])
	else:
		# assumes single number
		bvalues.append(bvalues_lowlow)


	dirs_lowlow, _ = gen_scheme(N_lowlow)
	# main directions
	dirs_all, shell_all = gen_scheme(N)
	# optimize for EDDY (flipping hemisphere heuristic)
	dirs_all, shell_all = swap_sampling_eddy(dirs_all, shell_all, verbose = 0)
	# split shells
	dirs_low = dirs_all[shell_all==0]
	dirs_high = dirs_all[shell_all==1]

	if display:
		# Look at uniformity
		plotScatter3(dirs_lowlow, mirror=True, title='quasi-B0 with mirror directions (2x {})'.format(N_lowlow[0]))
		plotScatter3(dirs_low, mirror=True, title='low shell with mirror directions (2x {})'.format(N[0]))
		plotScatter3(dirs_high, mirror=True, title='high shell with mirror directions (2x {})'.format(N[1]))
		plotScatter3(dirs_all, mirror=True, title='both shell with mirror directions (2x {})'.format(np.sum(N)))

	# for each direction, generate 2 orthogonal direction on the perpendicular plane
	planar_dirs_low = np.zeros((dirs_low.shape[0], 2, 3))
	for idx in range(dirs_low.shape[0]):
		# current directions
		v = dirs_low[idx]
		# generate initial non colinear vectors, v1 is v
		v1, v2, v3 = findLinIndepRandomRot(v)
		# generate 3 orthonormal vectors, u1 is v normalized
		u1, u2, u3 = gramSchmidt3(v1, v2, v3)
		planar_dirs_low[idx, 0] = u2
		planar_dirs_low[idx, 1] = u3

	# for each direction, generate 2 orthogonal direction on the perpendicular plane
	planar_dirs_high = np.zeros((dirs_high.shape[0], 2, 3))
	for idx in range(dirs_high.shape[0]):
		# current directions
		v = dirs_high[idx]
		# generate initial non colinear vectors, v1 is v
		v1, v2, v3 = findLinIndepRandomRot(v)
		# generate 3 orthonormal vectors, u1 is v normalized
		u1, u2, u3 = gramSchmidt3(v1, v2, v3)
		planar_dirs_high[idx, 0] = u2
		planar_dirs_high[idx, 1] = u3

	if display:
		plotScatter3(np.concatenate((planar_dirs_low[:,0], planar_dirs_low[:,1]), axis=0), mirror=True, title='planar low with mirror directions (4x {})'.format(N[0]))
		plotScatter3(np.concatenate((planar_dirs_high[:,0], planar_dirs_high[:,1]), axis=0), mirror=True, title='planar high with mirror directions (4x {})'.format(N[1]))
		plotScatter3(np.concatenate((planar_dirs_low[:,0], planar_dirs_low[:,1],planar_dirs_high[:,0], planar_dirs_high[:,1]), axis=0), mirror=True, title='planar all with mirror directions (4x {})'.format(np.sum(N)))

	bvalues = np.array(bvalues, dtype=np.float)
	scalling = np.sqrt(bvalues/bvalues.max())
	print('requested b-values are {}'.format(bvalues))
	print('using sqrt(b) scalling, the vector norms will be {}'.format(scalling))

	## scheme order: low b linear, low b planar, high b linear, high b planar
	g1tmp = np.concatenate((dirs_low, planar_dirs_low[:,0], dirs_high, planar_dirs_high[:,0]))
	scalling1tmp = np.concatenate((scalling[0]*np.ones(dirs_low.shape[0]),scalling[0]*np.ones(dirs_low.shape[0]),scalling[1]*np.ones(dirs_high.shape[0]),scalling[1]*np.ones(dirs_high.shape[0])),axis=0)
	g2tmp = np.concatenate((dirs_low, planar_dirs_low[:,1], dirs_high, planar_dirs_high[:,1]))
	# scalling2tmp = np.concatenate((scalling[0]*np.ones(dirs_low.shape[0]),scalling[0]*np.ones(dirs_low.shape[0]),scalling[1]*np.ones(dirs_high.shape[0]),scalling[1]*np.ones(dirs_high.shape[0])),axis=0)


	interleave_every = int(np.floor((2*np.sum(N)-1)/float(N_lowlow[0])))
	print('{} quasi-B0 to interleave in {} dirs :: 1 every {}'.format(N_lowlow[0], 2*np.sum(N), interleave_every))

	g1 = np.zeros((g1tmp.shape[0]+N_lowlow[0],3))
	scalling1 = np.zeros((scalling1tmp.shape[0]+N_lowlow[0]))
	g2 = np.zeros((g2tmp.shape[0]+N_lowlow[0],3))
	# scalling2 = np.zeros((scalling2tmp.shape[0]+N_lowlow[0]))
	for i in range(N_lowlow[0]):
		g1[i*interleave_every+i:(i+1)*interleave_every+i] = g1tmp[i*interleave_every:(i+1)*interleave_every]
		scalling1[i*interleave_every+i:(i+1)*interleave_every+i] = scalling1tmp[i*interleave_every:(i+1)*interleave_every]
		g1[(i+1)*interleave_every+i] = dirs_lowlow[i]
		scalling1[(i+1)*interleave_every+i] = scalling[2]
		g2[i*interleave_every+i:(i+1)*interleave_every+i] = g2tmp[i*interleave_every:(i+1)*interleave_every]
		# scalling2[i*interleave_every+i:(i+1)*interleave_every+i] = scalling2tmp[i*interleave_every:(i+1)*interleave_every]
		g2[(i+1)*interleave_every+i] = dirs_lowlow[i]
		# scalling2[(i+1)*interleave_every+i] = scalling[2]
	g1[(i+1)*interleave_every+i+1:] = g1tmp[(i+1)*interleave_every:]
	g2[(i+1)*interleave_every+i+1:] = g2tmp[(i+1)*interleave_every:]
	scalling1[(i+1)*interleave_every+i+1:] = scalling1tmp[(i+1)*interleave_every:]
	# scalling2[(i+1)*interleave_every+i+1:] = scalling2tmp[(i+1)*interleave_every:]


	savingLinPlan(g1,g2,scalling1,filenameout)



