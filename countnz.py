"""Look at the tracefile and count the number of iterations.

Useful when the output of ESPEI cannot be seen and you want to know how
many iterations have elapsed.
"""

infile = 'trace.npy'

import numpy as np

trace_array = np.load(infile)
nz = np.nonzero(np.all(trace_array != 0, axis=-1))
s = trace_array.shape
iterations = trace_array[nz].reshape(s[0], -1, s[2]).shape[1]
print('Iterations: {}'.format(iterations))
