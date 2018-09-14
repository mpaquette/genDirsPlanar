import numpy as np
from gs import findLinIndep, findLinIndepRandomRot, gramSchmidt3, checkOrtho3
from sphHist import genDirs, mirrorDirs, sphericalHist, displaySphericalHist, plotScatter3


# Orthogonality test
N = 100
res = []
print('Testing orthogonalization scheme with {} random init'.format(N))
for i in range(N):
	v = np.random.randn(3)
	v1, v2, v3 = findLinIndep(v)
	u1, u2, u3 = gramSchmidt3(v1, v2, v3)
	res.append(checkOrtho3(u1, u2, u3))
print('All passed? {}'.format(np.all(np.array(res))))


# Distribution of vector without any specific treatment
N = 100
res = []
print('Looking at coverage with {} points'.format(N))
pts = mirrorDirs(genDirs(N//2))
for i in range(N):
	v = pts[i]
	v1, v2, v3 = findLinIndep(v)
	u1, u2, u3 = gramSchmidt3(v1, v2, v3)
	res.append(u2)
	res.append(u3)
res = np.array(res)
plotScatter3(pts, mirror=False, title='u1')
plotScatter3(res, mirror=False, title='u23')
plotScatter3(res, mirror=True, title='u23 mir')
odf, pts = sphericalHist(res, np.round(N/5))
displaySphericalHist(odf, pts)

# Orthogonality test with random orientation
N = 100
res = []
print('Testing orthogonalization scheme 2 with {} random init'.format(N))
for i in range(N):
	v = np.random.randn(3)
	v1, v2, v3 = findLinIndepRandomRot(v)
	u1, u2, u3 = gramSchmidt3(v1, v2, v3)
	res.append(checkOrtho3(u1, u2, u3))
print('All passed? {}'.format(np.all(np.array(res))))


# Distribution of vector with specific treatment
N = 100
res = []
print('Looking at coverage with {} points'.format(N))
pts = mirrorDirs(genDirs(N//2))
for i in range(N):
	v = pts[i]
	v1, v2, v3 = findLinIndepRandomRot(v)
	u1, u2, u3 = gramSchmidt3(v1, v2, v3)
	res.append(u2)
	res.append(u3)
res = np.array(res)
plotScatter3(pts, mirror=False, title='u1')
plotScatter3(res, mirror=False, title='u23')
plotScatter3(res, mirror=True, title='u23 mir')
odf, pts = sphericalHist(res, np.round(N/5))
displaySphericalHist(odf, pts)

