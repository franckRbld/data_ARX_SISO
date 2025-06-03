from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/63901077/arx-models-in-gekko

na = 1  # Number of A coefficients
nb = 1  # Number of B coefficients
'''
ny = 1  # Number of outputs
nu = 1  # Number of inputs
'''

# A (na x ny)
# actual A,B,C values are from 5 h data
A = np.array([[0.960187147]])
# B (ny x (nb x nu))
B = np.array([[[-0.000361506092]]])
C = np.array([0.565842747871903])

# create parameter dictionary
# parameter dictionary p['a'], p['b'], p['c']
# a (coefficients for a polynomial, na x ny)
# b (coefficients for b polynomial, ny x (nb x nu))
# c (coefficients for output bias, ny)
p = {'a': A, 'b': B, 'c': C}

m = GEKKO(remote=False)
y, u = m.arx(p)

# load inputs
#tf = 719 # final time
u1 = np.append(np.ones(500), np.zeros(500), 0)
u2 = np.append(u1, np.ones(500), 0)
u3 = np.append(u2, np.zeros(500), 0)
u4 = np.append(u3, np.ones(500), 0)
u5 = np.append(u4, np.zeros(1001), 0)
u[0].value = u5


cv = y[0]
mv = u[0]
cv.value = 14.2
#y[0].value = 14.2

m.time = np.linspace(start=0, stop=u5.shape[0] - 1, num=u5.shape[0])
m.options.imode = 4
m.options.nodes = 2
#m.options.SOLVER = 1
# simulate
m.solve(disp=False)


plt.subplot(2, 1, 1)
plt.plot(m.time, u[0], 'r--', label='u[0]')
plt.plot(m.time, mv.value, 'r--', label='MV')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(m.time, y[0], 'b-', label='y[0]')
plt.plot(m.time, cv.value, 'b-', label='CV')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.legend()
plt.show()
