import os, shutil
from minlpe import minlpe


debug = False
folder = 'tmp'+os.sep+'minlpe'
if os.path.exists(folder+'1'): shutil.rmtree(folder+'1')
xs, ys, _ = minlpe(lambda x: x**2, [[-5, 5]], folder=folder+'1', maxSampleSize=20, seed=0, samplerTimeConstraint=None, debug=debug)
print('1D sampling')
print('xs =', xs.reshape(-1).tolist())
print('ys =', ys.reshape(-1).tolist())
print()

# continue calculation
xs, ys, _ = minlpe(lambda x: x**2, [[-5, 5]], folder=folder+'1', maxSampleSize=25, seed=0, samplerTimeConstraint=None, debug=debug)
print('1D sampling. Continue')
print('xs =', xs.reshape(-1).tolist())
print('ys =', ys.reshape(-1).tolist())
print()

if os.path.exists(folder+'2'): shutil.rmtree(folder+'2')
xs, ys, _ = minlpe(lambda x: x[0]**2-x[1]**3, [[-5, 5], [-5, 5]], folder=folder+'2', maxSampleSize=50, seed=0, samplerTimeConstraint=None, debug=debug)
print('2D sampling')
print('xs =', xs.tolist())
print('ys =', ys.reshape(-1).tolist())




