from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and parse into columns
url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)
t = data['Time']
u = data[['H1', 'H2']].values
y = data[['T1', 'T2']].values
#y = data[['T1']].values

'''
myList = (0, 1, 3, 4)
test1 = pd.DataFrame()
for i in range(data.shape[1]):
    test1[data.columns[i]] = data[data.columns[i]].values
test2 = data.copy()
test3 = pd.DataFrame()
for i in myList:
    test3[data.columns[i]] = data[data.columns[i]].values
test3_ = test3.values
'''

m = GEKKO()

# system identification
na = 6  # output coefficients
nb = 6  # input coefficients


def funcCriteria(nA, nB, nValue, vSortie, vSortieARX, Choice=None):
    if Choice is None:
        Choice = 1

    match Choice:
        case 1:
            """
            AIC (Akaike’s Information Criteria) cf. F-Ident-Practical_aspects page35 (Semestre 6)
            """
            return nValue * np.log(
                1 / nValue * np.sum((vSortie.flatten() - vSortieARX.flatten()) ** 2) +
                2 * (nA + nB + 1) +
                nValue * np.log(2 * np.pi)
                + 1
                            )
        case 2:
            """
            SMY2
            """
            return np.sum((vSortie.flatten() - vSortieARX.flatten()) ** 2)


def map_sysid(t, u, y, na, nb, pred):
    optimal = np.inf
    i_opt = 1
    j_opt = 1
    k = 0
    vCritere_1 = []
    vCritere_2 = []
    m_map = GEKKO()
    #flag = False
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            #if flag:
            #    break
            yp, p, K = m_map.sysid(t, u, y, i, j, pred=pred, diaglevel=0)
            Criteria_1 = funcCriteria(nA=i, nB=j, nValue=u.shape[0], vSortie=y, vSortieARX=yp, Choice=1)
            Criteria_2 = funcCriteria(nA=i, nB=j, nValue=u.shape[0], vSortie=y, vSortieARX=yp, Choice=2)
            vCritere_1.append(Criteria_1)
            vCritere_2.append(Criteria_2)
            if Criteria_1 * 0 + Criteria_2 * 1 < optimal:
                optimal = Criteria_1.copy()
                i_opt = i
                j_opt = j
            #if optimal < 1e-7:      #Replace the AIC criteria logic: on sort dès que possible un meilleur compromis
            #    flag = True
            k += 1
    yp, p, K = m_map.sysid(t, u, y, i_opt, j_opt, pred=pred)
    return i_opt, j_opt, yp, p, K


i_opt, j_opt, yp_opt, p_opt, K_opt = map_sysid(t, u, y, na, nb, pred='meas')
yp, p, K = m.sysid(t, u, y, na, nb, pred='meas')

print('-------------------------------------\tp_opt')
for keys, value in p_opt.items():
    print(keys, value)
print('-------------------------------------\tp')
for keys, value in p.items():
    print(keys, value)

plt.figure(figsize=(10, 7), tight_layout=True)
plt.subplot(3, 1, 1)
plt.plot(t, u, label=r'$Heater_1$')
plt.legend([r'$Heater_1$', r'$Heater_2$'])
plt.ylabel('Heaters')
plt.xlabel('Time (sec)')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.subplot(3, 1, 2)
for i in range(y.shape[1]):
    plt.plot(t, y[:, i], label=r'$T1_{meas}$')
    plt.plot(t, yp[:, i], '--', marker='o', markersize=2, label=r'$T1_{arx gekko}$')
    plt.plot(t, yp_opt[:, i], '--', marker='*', markersize=2, label=r'$T1_{arx gekko}opt$')
#plt.legend([r'$T1_{meas}$', r'$T1_{pred}$', r'$T1_{pred}opt$'])
plt.legend()
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (sec)')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.subplot(3, 1, 3)
for i in range(y.shape[1]):
    plt.plot(t, y[:, i] - yp[:, i], '--', lw=1, label=r'$T_{meas} - T_{arx gekko}$', marker='o', markersize=1)
    plt.plot(t, y[:, i] - yp_opt[:, i], '--', lw=1, label=r'$T_{meas} - T_{arx gekko}opt$$')
plt.legend()
plt.ylabel('Ecart Temperature (°C)')
plt.xlabel('Time (sec)')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)

plt.savefig('TUTO3_test.png', dpi=300)
plt.show()
