import numpy as np
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.viz import fvtk
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D


def genDirs(N):
	init_dirs = np.random.randn(N, 3)
	init_dirs /= np.linalg.norm(init_dirs, axis=1)[:,None]
	init_hemi = HemiSphere(xyz=init_dirs)
	dirs_hemi,_ = disperse_charges(init_hemi, iters=1000)
	pts = dirs_hemi.vertices
	# shifting to z+ hemi
	pts = pts*np.sign(pts[:,2])[:,None]
	return pts


def mirrorDirs(pts):
	return np.concatenate((pts,-pts), axis=0)


def sphericalHist(directions, N_bins_half):
	# generating directions for the histogram
	pts = genDirs(N_bins_half)
	# shifting to z+ hemi
	dirs = directions*np.sign(directions[:,2])[:,None]

	# histogram init
	odf = np.zeros(pts.shape[0])

	# slow loop to fill bins
	for idx in range(dirs.shape[0]):
		b = np.argmin(np.linalg.norm(pts-dirs[idx],axis=1))
		odf[b] += 1

	return odf, pts


def displaySphericalHist(odf, pts, minmax = False):
	# assumes pts and odf are hemisphere
	fullsphere = HemiSphere(xyz=pts).mirror()
	fullodf = np.concatenate((odf, odf), axis=0)

	r = fvtk.ren()
	if minmax:
		a = fvtk.sphere_funcs(fullodf - fullodf.min(), fullsphere)
	else:
		a = fvtk.sphere_funcs(fullodf, fullsphere)
	fvtk.add(r,a)
	fvtk.show(r)


def scat(p):
    return [p[:,i] for i in range(p.shape[1])]


def plotScatter3(pts, mirror=True, title=''):
	fig = pl.figure()
	ax = fig.add_subplot(111, projection='3d')
	if mirror:
		points = mirrorDirs(pts)
	else:
		points = pts
	ax.scatter3D(*scat(points))
	pl.title(title)
	pl.show()
