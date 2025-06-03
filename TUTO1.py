# Référence : https://apmonitor.com/dde/index.php/Main/AutoRegressive

"""
ARX Model
An autoregressive exogenous input (ARX) model is a combination of an AR model and an X model, and it is represented by
the following equation:
y(t)=c+a1y(t−1)+a2y(t−2)+…+apy(t−p)+b1u(t−1)+b2u(t−2)+…+bqu(t−q)+e(t)y(t)=c+a1y(t−1)+a2y(t−2)+…+apy(t−p)+b1u(t−1)+b2u(t−2)+…+bqu(t−q)+e(t)

ARX time series models are a linear representation of a dynamic system in discrete time. Putting a model into ARX form
is the basis for many methods in process dynamics and control analysis. Below is the time series model with
a single input and single output with k as an index that refers to the time step.
yk+1=na∑i=1aiyk−i+1+nb∑i=1biuk−i+1yk+1=∑i=1naaiyk−i+1+∑i=1nbbiuk−i+1
With na=3, nb=2, nu=1, and ny=1 the time series model is:
yk+1=a1yk+a2yk−1+a3yk−2+b1uk+b2uk−1yk+1=a1yk+a2yk−1+a3yk−2+b1uk+b2uk−1

The time-delay between in the input and output allows the model to take into account the fact that the input and output
of a system may not be perfectly synchronized in time. There may also be multiple inputs and
multiple outputs such as when na=1, nb=1, nu=2, and ny=2.
y1k+1=a1,1y1k+b1,1u1k+b1,2u2ky1k+1=a1,1y1k+b1,1u1k+b1,2u2k
y2k+1=a1,2y2k+b2,1u1k+b2,2u2ky2k+1=a1,2y2k+b2,1u1k+b2,2u2k

Time series models are used for identification and advanced control. It has been in use in the process industries such
as chemical plants and oil refineries since the 1980s.
Model predictive controllers rely on dynamic models of the process, most often linear empirical models obtained by
system identification.
Below is an overview of how to simulate and identify with ARX models using Python Gekko.
There is also a Graphical User Interface (GUI) to identify models with the BYU PRISM Seeq SysID Open-Source Package.
"""

import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

'''
na = 2 # Number of A coefficients
nb = 1 # Number of B coefficients
ny = 2 # Number of outputs
nu = 2 # Number of inputs
'''

# A (na x ny)
A = np.array([[0.36788, 0.36788], [0.223, -0.136]])
# B (ny x (nb x nu))
B1 = np.array([0.63212, 0.18964]).T
B2 = np.array([0.31606, 1.26420]).T
B = np.array([[B1], [B2]])
C = np.array([0, 0])

# create parameter dictionary
# parameter dictionary p['a'], p['b'], p['c']
# a (coefficients for a polynomial, na x ny)
# b (coefficients for b polynomial, ny x (nb x nu))
# c (coefficients for output bias, ny)
p = {'a': A, 'b': B, 'c': C}

# Create GEKKO model
m = GEKKO(remote=False)

# Build GEKKO ARX model
y, u = m.arx(p)

# load inputs
tf = 20                 # final time
u1 = np.zeros(100 + 1)  # tf + 1)
u2 = u1.copy()
u1[20:] = 3.0
u2[40:] = 5.0
u1[:] += 10.0
u2[:] += 10.0

u[0].value = u1.copy()
u[1].value = u2.copy()

#Test
#'''
y[0].value = 5 * 0
y[1].value = 5 * 0
#'''

m.time = np.linspace(start=0, stop=tf, num=u1.shape[0])
m.options.imode = 4
m.options.nodes = 2
m.solve(disp=False)

t1 = np.array(m.time, dtype=float)
t2 = np.array(u, dtype=float).T
t3 = np.array(y, dtype=float).T
y_1, p_1, k_1 = m.sysid(t=t1, u=t2, y=t3, na=2, nb=2, pred='meas', shift='calc')  # 'meas' for ARX regression form, explicit solution

print('Vérification:\t', max(abs(y_1[:, 0] - y[0].value)))
print('Vérification:\t', max(abs(y_1[:, 1] - y[1].value)))

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(m.time, u[0].value, 'r-', label=r'$u_0$')
plt.plot(m.time, u[1].value, 'b--', label=r'$u_1$')
plt.ylabel('MV')
plt.legend(loc='best')
plt.tight_layout()
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.subplot(3, 1, 2)
plt.plot(m.time, y[0].value, 'r:', label=r'$y_0$')
plt.plot(m.time, y[1].value, 'b.-', label=r'$y_1$')
plt.ylabel('CV')
plt.xlabel('Time (sec)')
plt.legend(loc='best')
plt.tight_layout()
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.subplot(3, 1, 3)
plt.plot(m.time, y_1[:, 0], 'r:', label=r'$y_0$')
plt.plot(m.time, y_1[:, 1], 'b.-', label=r'$y_1$')
plt.ylabel('m.sysid CV')
plt.xlabel('Time (sec)')
plt.legend(loc='best')
plt.tight_layout()
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.show()

