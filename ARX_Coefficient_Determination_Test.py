import numpy as np
import pandas as pd
from gekko import GEKKO
from TUTO1_ARX_V4_svg_test_PID_debogage import resolution_ARX_SISO
import matplotlib.pyplot as plt

def myARX(vFlux, vTemp, coefA, coefB, coefC):
    c_Ai_Bi_C = np.zeros(shape=(vFlux.shape[0] - 1, coefA + coefB + coefC), dtype=float)
    v_V = np.zeros(shape=(vFlux.shape[0]), dtype=float)
    v_V[:] = vTemp[:] - vTemp[0]
    k = int(0)
    for i in range(coefA):
        c_Ai_Bi_C[k:, i] = v_V[0:-k - 1].T.copy()
        k += 1
    k = int(0)
    for i in range(coefB):
        c_Ai_Bi_C[k:, i + coefA] = vFlux[0:-k - 1].T.copy()
        k += 1
    if CoefficientC == 1:
        c_Ai_Bi_C[:, coefA + coefB] = 1

    mCoefficient_Y = np.zeros(shape=(vFlux.shape[0] - 1, 1), dtype=float)
    mCoefficient_Y[:, 0] = v_V[1:].T.copy()
    '''
    print('(AT.A)\n', matriceCoefficient_Ai_Bi.T.dot(matriceCoefficient_Ai_Bi))
    print('np.diag(AT.A)\n', np.diag(matriceCoefficient_Ai_Bi.T.dot(matriceCoefficient_Ai_Bi)))
    print('\n')
    '''

    c_Solution = (np.linalg.inv(
        c_Ai_Bi_C.T.dot(c_Ai_Bi_C) + 0 * np.eye(c_Ai_Bi_C.shape[1])).
                             dot(c_Ai_Bi_C.T).dot(mCoefficient_Y))

    return c_Solution

decimals = 30
data = pd.read_excel('data.xlsx').values
data = np.round(data, decimals=decimals)
#print(data.shape)

dataTime = data[:, 0]
dataFlux = data[:, 1]
dataTemp = data[:, 2]

CoefficientA = 2
CoefficientB = 2
CoefficientC = 1

matriceCoefficient_Ai_Bi = np.zeros(shape=(data.shape[0] - 1, CoefficientA + CoefficientB + CoefficientC), dtype=float)
inputVector = np.zeros(shape=(data.shape[0]), dtype=float)
inputVector[:] = dataTemp[:] - dataTemp[0]

k = int(0)
for i in range(CoefficientA):
    matriceCoefficient_Ai_Bi[k:, i] = inputVector[0:-k-1].T.copy()
    k += 1
k = int(0)
for i in range(CoefficientB):
    matriceCoefficient_Ai_Bi[k:, i + CoefficientA] = dataFlux[0:-k-1].T.copy()
    k += 1
if CoefficientC == 1:
    matriceCoefficient_Ai_Bi[:, CoefficientA + CoefficientB] = 1

matriceCoefficient_Y = np.zeros(shape=(data.shape[0] - 1, 1), dtype=float)
matriceCoefficient_Y[:, 0] = inputVector[1:].T.copy()


myRound = 3
np.set_printoptions(precision=myRound)
print('(AT.A)\n', matriceCoefficient_Ai_Bi.T.dot(matriceCoefficient_Ai_Bi))
print('np.diag(AT.A)\n', np.diag(matriceCoefficient_Ai_Bi.T.dot(matriceCoefficient_Ai_Bi)))
print('\n')
np.set_printoptions(precision=None)


Coefficient_Solution1 = (
    np.linalg.inv(matriceCoefficient_Ai_Bi.T.dot(matriceCoefficient_Ai_Bi) + 1e-13 * np.eye(matriceCoefficient_Ai_Bi.shape[1]))
    .dot(matriceCoefficient_Ai_Bi.T).dot(matriceCoefficient_Y))

Coefficient_Solution2 = np.linalg.lstsq(matriceCoefficient_Ai_Bi, matriceCoefficient_Y, rcond=None)[0]

Coefficient_Solution3 = myARX(
                            vFlux=dataFlux,
                            vTemp=dataTemp,
                            coefA=CoefficientA,
                            coefB=CoefficientB,
                            coefC=CoefficientC
                                )


myDict_Coefficient_Solution = {}
myDict_Coefficient_Solution['Coefficient_Solution1'] = Coefficient_Solution1
myDict_Coefficient_Solution['Coefficient_Solution2'] = Coefficient_Solution2
myDict_Coefficient_Solution['Coefficient_Solution3'] = Coefficient_Solution3

'''
for keys, value in myDict_Coefficient_Solution.items():
    print(keys)
    print(value)
    print('\n')
'''

'''
for i in range(CoefficientA):
    print('Coefficient_A' + str(i + 1) + ' = ' + str(Coefficient_Solution1[i, 0]))
for i in range(CoefficientB):
    print('Coefficient_B' + str(i + 1) + ' = ' + str(Coefficient_Solution1[i + CoefficientA, 0]))
if CoefficientC == 1:
    print('Coefficient_C = ' + str(Coefficient_Solution1[CoefficientA + CoefficientB, 0]))
print('\n')

for i in range(CoefficientA):
    print('Coefficient_A' + str(i + 1) + ' = ' + str(Coefficient_Solution2[i, 0]))
for i in range(CoefficientB):
    print('Coefficient_B' + str(i + 1) + ' = ' + str(Coefficient_Solution2[i + CoefficientA, 0]))
if CoefficientC == 1:
    print('Coefficient_C = ' + str(Coefficient_Solution2[CoefficientA + CoefficientB, 0]))
print('\n')

for i in range(CoefficientA):
    print('Coefficient_A' + str(i + 1) + ' = ' + str(Coefficient_Solution3[i, 0]))
for i in range(CoefficientB):
    print('Coefficient_B' + str(i + 1) + ' = ' + str(Coefficient_Solution3[i + CoefficientA, 0]))
if CoefficientC == 1:
    print('Coefficient_C = ' + str(Coefficient_Solution3[CoefficientA + CoefficientB, 0]))
print('\n')
'''

m = GEKKO(remote=False)
y_1, p_1, k_1 = m.sysid(t=dataTime, u=dataFlux, y=dataTemp, na=CoefficientA, nb=CoefficientB, nk=0, shift='calc', pred='meas')        # 'meas' for ARX regression form, explicit solution
print('ARX GEKKO')
for keys, value in p_1.items():
    print(keys, value)
print('\n')

myCoefA = Coefficient_Solution1[:CoefficientA]
myCoefB = Coefficient_Solution1[CoefficientA:CoefficientA+CoefficientB]
myCoefC = Coefficient_Solution1[CoefficientA+CoefficientB:CoefficientA+CoefficientB+CoefficientC]
print('ARX myARX')
print('a', myCoefA)
print('b', myCoefB)
print('c', myCoefC)

myOffset = dataTemp[0:myCoefA.shape[0]].reshape(myCoefA.shape[0], 1)
dataTemp_ARX = resolution_ARX_SISO(A=myCoefA, B=myCoefB, C=myCoefC, vU=dataFlux.reshape(dataFlux.shape[0], 1), offsetY=myOffset)
dataTemp_ARX_ = resolution_ARX_SISO(A=p_1['a'], B=p_1['b'], C=p_1['c'], vU=dataFlux.reshape(dataFlux.shape[0], 1), offsetY=myOffset)

plt.figure(num='Figure 1: ARX', figsize=(8, 6), tight_layout=True)
plt.plot(dataTime, dataTemp, label=r'$y_{Reference}$', lw=1, ls='-')
#plt.plot(dataTime, dataTemp_ARX.flatten(), label=r'$y_{ARX}$', lw=1, ls='--')
plt.plot(dataTime, y_1.flatten(), label=r'$y_{ARX_(GEKKO)}$', lw=1, ls='--', marker='x', markersize=0.5, color='black')
plt.plot(dataTime, dataTemp_ARX_.flatten(), label=r'$y_{ARX(GEKKO)(HOMEMADE)}$', lw=1, ls='--', marker='x', markersize=0.5, color='black')
plt.xlabel('Time (sec)')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5)
'''
plt.annotate(text='ARX\n'
             + 'a' + str(myCoefA) + '\n'
             + 'b' + str(myCoefB) + '\n'
             + 'c' + str(myCoefC) + '\n', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=5, ha='left', va='top',
             alpha=1.0, bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', linewidth=0.5))
'''
ax1 = plt.gca().twinx()  # Set a second y-axis
ax1.set_ylabel('Delta_Température [°C]')  # Label for the second y-axis
#ax1.plot(dataTime, dataTemp_ARX.flatten() - dataTemp, label=r'$y_{ARX} - y_{Reference}$', color='green', ls='-', markersize=0.5, marker='x', lw=1)
ax1.plot(dataTime, y_1.flatten() - dataTemp, label=r'$y_{ARX(GEKKO)} - y_{Reference}$', ls='-', markersize=0.5, marker='x', lw=1)
ax1.plot(dataTime, dataTemp_ARX_.flatten() - dataTemp, label=r'$y_{ARX(GEKKO)(HOMEMADE)} - y_{Reference}$', ls='-', markersize=0.5, marker='x', lw=1)
ax1.legend(loc='lower right')
plt.show()