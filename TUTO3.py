from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and parse into columns
url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)
t = data['Time']
u = data[['H1', 'H2']].values
#u = data[['H1']].values
y = data[['T1', 'T2']].values
y = data[['T1']].values

m = GEKKO()

# system identification
na = 6  # output coefficients
nb = 6  # input coefficients


def resolution_ARX_SISO(A, B, C, vU, offsetY):
    #New version 18_04_2025
    
    vY = np.zeros(shape=(len(vU), A.shape[1]), dtype=float)
    #vY[0] = offsetY[0]#np.array(offsetY, dtype=float)

    try:
        A = A[:, 0]                 #.copy()
        B = B[0, :, 0]              #.copy()
        C = C                       #.copy()
    except:
        A = A.copy()
        B = B.copy()
        C = C.copy()

    n_na     = A.shape[0]
    n_nb     = B.shape[0]

    starter = max(n_na, n_nb)

    vY[0:starter, :] = offsetY    #for k in range(n_Output):    #    vY[0, k] += offsetY[k]

    for i in range(starter, vU.shape[0]):                                   # Balayage de chaque valeur temporelle
    #for i in range(1, len(vU)):
        #print('\n')
        for j in range(len(A)):
            if i - j > 0:
                vY[i, 0] += A[j] * vY[i - j - 1, 0]
                #print(A[j])
                #print(i - j - 1)
                #print(vY[i - j - 1])
        for j in range(len(B)):
            if i - j > 0:
                vY[i, 0] += B[j] * vU[i - j - 1, 0]
                #print(B[j])
                #print(i - j - 1)
                #print(vU[i - j - 1])
        if C is None or not C.all():
            pass
        else:
            vY[i] += C[0]

    #vY = np.array(vY, dtype=float) + offsetY
    return vY


def resolution_ARX_MIMO(A, B, C, vU, offsetY):
    #New version 18_04_2025

    #Input parameters
    #B[0] = n output
    #B[1] = nb
    #B[2] = n input
    #A[0] = na
    #A[1] = n output

    n_Output = A.shape[1]
    n_Input  = B.shape[2]
    n_na     = A.shape[0]
    n_nb     = B.shape[1]

    vY = np.zeros(shape=(vU.shape[0], n_Output), dtype=float)
    #vY[0, :] = offsetY    #for k in range(n_Output):    #    vY[0, k] += offsetY[k]

    starter = max(n_na, n_nb)

    vY[0:starter, :] = offsetY    #for k in range(n_Output):    #    vY[0, k] += offsetY[k]

    for i in range(starter, vU.shape[0]):                                   # Balayage de chaque valeur temporelle
    #for i in range(1, vU.shape[0]):                                         # Balayage de chaque valeur temporelle
        for k in range(n_Output):                                           # Pour chaque grandeur de sortie

            for l in range(n_Input):                                        # Pour chaque grandeur d'entrée
                for j in range(n_nb):                                       # nb
                    if i - j > 0:
                        #print('B', B[k, j, l])
                        vY[i, k] += B[k, j, l] * vU[i - j - 1, l]

            for j in range(n_na):                                           # nb
                if i - j > 0:
                    #print('A', A[j, k])
                    vY[i, k] += A[j, k] * vY[i - j - 1, k]

            if not C.size:
                pass
            else:
                #print('C', C[k])
                vY[i, k] += C[k]

    #for k in range(n_Output):
    #    vY[:, k] += offsetY[k]

    return vY


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


def map_sysid(t, u, y, na, nb, pred):           #Methode ARX (system identification)
    optimal = np.inf
    i_opt = 1
    j_opt = 1
    k = 0
    vCritere_1 = []
    vCritere_2 = []
    m_map = GEKKO()
    namin = 1
    nbmin = 1
    for i in range(namin, namin + na + 1):
        for j in range(nbmin, nbmin + nb + 1):
            #print('i', i, '\tj', j)
            yp, p, K = m_map.sysid(t=t, u=u, y=y, na=i, nb=j, nk=0, shift='calc', pred=pred)
            Criteria_1 = funcCriteria(nA=i, nB=j, nValue=u.shape[0], vSortie=y, vSortieARX=yp, Choice=1)
            Criteria_2 = funcCriteria(nA=i, nB=j, nValue=u.shape[0], vSortie=y, vSortieARX=yp, Choice=2)
            vCritere_1.append(Criteria_1)
            vCritere_2.append(Criteria_2)
            if Criteria_1 * 1 + Criteria_2 * 0 < optimal:
                optimal = Criteria_1.copy()
                i_opt = i
                j_opt = j
            k += 1
    yp, p, K = m_map.sysid(t=t, u=u, y=y, na=i_opt, nb=j_opt, nk=0, shift='calc', pred=pred)
    return i_opt, j_opt, yp, p, K


i_opt, j_opt, yp_opt, p_opt, K_opt = map_sysid(t, u, y, na, nb, pred='meas')
print('p_opt')
print('i_opt ', i_opt, '\tj_opt ', j_opt)
for keys, value in p_opt.items():
    print(keys, value)


liste1 = []
liste2 = []
for a, b in p_opt.items():
    liste1.append(a)
    liste2.append(b)
yp_Homemade1 = resolution_ARX_SISO(A=liste2[0], B=liste2[1], C=liste2[2], vU=u, offsetY=y[0, :])
yp_Homemade2 = resolution_ARX_MIMO(A=liste2[0], B=liste2[1], C=liste2[2], vU=u, offsetY=y[0, :])

yp, p, K = m.sysid(t, u, y, na=i_opt, nb=i_opt, nk=0, pred='meas')
print('p')
for keys, value in p.items():
    print(keys, value)

plt.figure(figsize=(10, 8), tight_layout=True)
plt.subplot(3, 1, 1)
plt.plot(t, u, label=r'$Heater_1$')
plt.legend([r'$Heater_1$', r'$Heater_2$'])
plt.ylabel('Heaters')
plt.xlabel('Time (sec)')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.subplot(3, 1, 2)
for i in range(y.shape[1]):
    plt.plot(t, y[:, i], label=r'$T_{meas}$')
    plt.plot(t, yp_opt[:, i], '--', lw=1, marker='o', markersize=2, label=r'$T_{arx gekko}$')
    #plt.plot(t, yp_Homemade1[:, i], '--', lw=1, marker='x', markersize=2, label=r'$T_{arx HM1}$')
    plt.plot(t, yp_Homemade2[:, i], '--', lw=1, marker='*', markersize=2, label=r'$T_{arx HM2}$')
plt.legend()
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (sec)')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
plt.subplot(3, 1, 3)
for i in range(y.shape[1]):
    plt.plot(t, y[:, i] - yp_opt[:, i], '--', lw=1, label=r'$T_{meas} - T_{arx gekko}$', marker='o', markersize=1)
    #plt.plot(t, y[:, i] - yp_Homemade1[:, i], '--', lw=1, label=r'$T_{meas} - T_{arx SISO}$')
    plt.plot(t, y[:, i] - yp_Homemade2[:, i], '--', lw=1, label=r'$T_{meas} - T_{arx MIMO}$')
    #plt.plot(t, yp_opt[:, i] - yp_Homemade1[:, i], '--', lw=1, label=r'$T_{arx gekko} - T_{arx SISO}$')
plt.legend()
plt.ylabel('Ecart Temperature (°C)')
plt.xlabel('Time (sec)')
plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)

plt.savefig('TUTO3.png', dpi=300)
plt.show()
