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
import pandas as pd

# Référence : https://stackoverflow.com/questions/63901077/arx-models-in-gekko

def funcCriteria(nA, nB, nValue, vSortie, vSortieARX, Choice=1):

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
            SMY2 (least squares)
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
    namin = int(1)   #1
    nbmin = int(1)   #1

    for i in range(namin, na + 1):
        for j in range(nbmin, nb + 1):
            #if flag:
            #    break
            yp, p, K = m_map.sysid(t, u, y, i, j, pred=pred)
            Criteria_1 = funcCriteria(nA=i, nB=j, nValue=u.shape[0], vSortie=y, vSortieARX=yp, Choice=1)
            Criteria_2 = funcCriteria(nA=i, nB=j, nValue=u.shape[0], vSortie=y, vSortieARX=yp, Choice=2)
            vCritere_1.append(Criteria_1)
            vCritere_2.append(Criteria_2)
            if Criteria_1 * 1 + Criteria_2 * 0 < optimal:
                optimal = Criteria_1.copy()
                i_opt = i
                j_opt = j
            #if optimal < 1e-7:      #Replace the AIC criteria logic: on sort dès que possible un meilleur compromis
            #    flag = True
            k += 1
    yp, p, K = m_map.sysid(t, u, y, i_opt, j_opt, pred=pred)
    return i_opt, j_opt, yp, p, K


def resolution_ARX_SISO(A, B, C, vU, offsetY):
    # New version 18_04_2025

    vY = np.zeros(shape=(len(vU), A.shape[1]), dtype=float)
    # vY[0] = offsetY[0]#np.array(offsetY, dtype=float)

    try:
        A = A[:, 0]  # .copy()
        B = B[0, :, 0]  # .copy()
        C = C  # .copy()
    except:
        A = A.copy()
        B = B.copy()
        C = C.copy()

    n_na = A.shape[0]
    n_nb = B.shape[0]

    starter = max(n_na, n_nb)

    vY[0:starter, :] = offsetY  # for k in range(n_Output):    #    vY[0, k] += offsetY[k]

    for i in range(starter, vU.shape[0]):  # Balayage de chaque valeur temporelle
        # for i in range(1, len(vU)):
        # print('\n')
        for j in range(len(A)):
            if i - j > 0:
                vY[i, 0] += A[j] * vY[i - j - 1, 0]
                # print(A[j])
                # print(i - j - 1)
                # print(vY[i - j - 1])
        for j in range(len(B)):
            if i - j > 0:
                vY[i, 0] += B[j] * vU[i - j - 1, 0]
                # print(B[j])
                # print(i - j - 1)
                # print(vU[i - j - 1])
        if C is None or not C.all():
            pass
        else:
            vY[i] += C[0]

    # vY = np.array(vY, dtype=float) + offsetY
    return vY


def resolution_ARX_MISO(A, B, C, vU, offsetY):
    # New version 18_04_2025

    # Input parameters
    # B[0] = n output
    # B[1] = nb
    # B[2] = n input
    # A[0] = na
    # A[1] = n output

    n_Output = A.shape[1]
    n_Input = B.shape[2]
    n_na = A.shape[0]
    n_nb = B.shape[1]

    vY = np.zeros(shape=(vU.shape[0], n_Output), dtype=float)
    starter = max(n_na, n_nb)

    try:
        vY[0:starter, :] = offsetY[0:starter, :]  # for k in range(n_Output):    #    vY[0, k] += offsetY[k]
    except:
        vY[0:starter, :] = offsetY[0]

    for i in range(n_Output):                                                       # Pour chaque grandeur de sortie
        for j in range(starter, vU.shape[0]):                                       # Balayage de chaque valeur temporelle

            for k in range(0, n_na):  # nb
                # print('A', A[j, k])
                vY[j, i] += A[k, i] * vY[j - 1 - k, i]

            for iu in range(0, n_Input):  # Pour chaque grandeur d'entrée
                for k in range(0, n_nb):  # nb
                    # print('B', B[k, j, l])
                    vY[j, i] += B[i, k, iu] * vU[j - 1 - k, iu]

            if not C.size:
                pass
            else:
                # print('C', C[k])
                vY[j, i] += C[i]

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


def GEKKO_solve_sysid(u, y, alpha, beta, gamma):
    ypred = np.zeros(shape=(u.shape[0], alpha.shape[1]), dtype=float)
    # Predict using prior model values

    ny = alpha.shape[1]
    na = alpha.shape[0]
    n = u.shape[0]
    nb = beta.shape[1]
    nk = 0
    nbk = nb + nk
    nu = beta.shape[0]
    m = max(na, nbk)

    ypred[0:m, :] = y[0:m, :]
    for i in range(ny):
        #ypred[0:m, i] = y[0:m, i]
        for j in range(m, n):
            for k in range(na):
                ypred[j][i] += alpha[k][i] * ypred[j - 1 - k][i]
            for iu in range(nu):
                for k in range(nbk):
                    ypred[j][i] += beta[i][k][iu] * u[j - 1 - k][iu]
            ypred[j][i] += gamma[i]

    #print(np.allclose(ypred, y, rtol=1e-03))

    return ypred


def calculation():

    '''
    na = 2 # Number of A coefficients
    nb = 2 # Number of B coefficients
    ny = 1 # Number of outputs
    nu = 1 # Number of inputs
    '''

    # create parameter dictionary
    # parameter dictionary p['a'], p['b'], p['c']
    # a (coefficients for a polynomial, na x ny)
    # b (coefficients for b polynomial, ny x (nb x nu))
    # c (coefficients for output bias, ny)
    '''
    A = np.array([[1.880096113423550000], [-0.880117210451247000]], dtype=float)
    B = np.array([[[0.002694864492691150], [-0.002691167450352280]]], dtype=float)      # B (ny x (nb x nu))
    C = np.array([-0.010954860173862600], dtype=float)
    p = {'a': A, 'b': B, 'c': C}
    print('ARX homemade')
    for keys, value in p.items():
        print(keys, value)
    print('\n')
    '''

    # -------------------------------------------- load inputs Stabilisation
    df = pd.read_excel(io='data.xlsx', sheet_name='Tabelle1', header=0)
    df_time = df[df.columns[0]][:].values
    df_flux = df[df.columns[1]][:].values
    df_temp = df[df.columns[2]][:].values

    '''
    # -------------------------------------------- Préparation dataset
    df_dtime = df_time[1] - df_time[0]
    n_value = 2
    for i in range(n_value):
        df_time = np.append(df_time, [df_time[-1] + df_dtime], axis=0)
    #df_time = np.linspace(start=df_time[0], stop=df_time[-1], num=df_time.shape[0])

    df_flux = np.insert(df_flux, [0], np.zeros((n_value)))
    df_temp = np.insert(df_temp, [0], df_temp[0] * np.ones((n_value)))

    df_temp_offset = df[df.columns[2]][0]
    '''

    # -------------------------------------------- Reformatage
    df_time = df_time.reshape(df_time.shape[0], 1)
    df_flux = df_flux.reshape(df_flux.shape[0], 1)
    df_temp = df_temp.reshape(df_temp.shape[0], 1)

    na_ini = 5
    nb_ini = 5

    i_opt, j_opt, yARX_opt, p_opt, K_opt = map_sysid(t=df_time, u=df_flux, y=df_temp, na=na_ini, nb=nb_ini, pred='meas')
    print('\n')
    print('ARX optimal')
    for keys, value in p_opt.items():
        print(keys, value)
    print('\n')

    myOffsetY  = df_temp[0]
    myOffsetY_ = df_temp[0:p_opt['a'].shape[0]]

    yARX_homemade1 = resolution_ARX_SISO(A=p_opt['a'], B=p_opt['b'], C=p_opt['c'], vU=df_flux, offsetY=myOffsetY)
    yARX_homemade2 = resolution_ARX_SISO(A=p_opt['a'], B=p_opt['b'], C=p_opt['c'], vU=df_flux, offsetY=myOffsetY_)
    yARX_homemade3 = resolution_ARX_MISO(A=p_opt['a'], B=p_opt['b'], C=p_opt['c'], vU=df_flux, offsetY=myOffsetY)
    yARX_homemade4 = resolution_ARX_MISO(A=p_opt['a'], B=p_opt['b'], C=p_opt['c'], vU=df_flux, offsetY=myOffsetY_)
    yARX_homemade5 = resolution_ARX_MIMO(A=p_opt['a'], B=p_opt['b'], C=p_opt['c'], vU=df_flux, offsetY=myOffsetY)
    yARX_homemade6 = resolution_ARX_MIMO(A=p_opt['a'], B=p_opt['b'], C=p_opt['c'], vU=df_flux, offsetY=myOffsetY_)
    yARX_homemade7 = GEKKO_solve_sysid(u=df_flux.copy(), y=df_temp.copy(), alpha=p_opt['a'], beta=p_opt['b'], gamma=p_opt['c'])
    yARX_homemade8 = GEKKO_solve_sysid(u=df_flux.copy(), y=myOffsetY_, alpha=p_opt['a'], beta=p_opt['b'], gamma=p_opt['c'])


    rtol = 1e-5
    print('test 1', np.allclose(yARX_opt, yARX_homemade1, rtol=rtol))
    print('test 2', np.allclose(yARX_opt, yARX_homemade2, rtol=rtol))
    print('test 3', np.allclose(yARX_opt, yARX_homemade3, rtol=rtol))
    print('test 4', np.allclose(yARX_opt, yARX_homemade4, rtol=rtol))
    print('test 5', np.allclose(yARX_opt, yARX_homemade5, rtol=rtol))
    print('test 6', np.allclose(yARX_opt, yARX_homemade6, rtol=rtol))
    print('test 7', np.allclose(yARX_opt, yARX_homemade7, rtol=rtol))
    print('test 8', np.allclose(yARX_opt, yARX_homemade8, rtol=rtol))


    # -------------------------------------------- Create GEKKO model
    m = GEKKO(remote=False)
    # Build GEKKO ARX model
    y, u = m.arx(p=p_opt)
    '''
    m.y = m.Array(m.CV, dim=1)
    m.u = m.Array(m.MV, dim=1)
    y, u = m.arx(p, m.y, m.u)
    '''
    '''
    # 🔹 **Initialize Past Values (Backsteps)**
    y[0].value = df_temp[:na].tolist()  # First `na` known outputs
    u[0].value = df_flux[:nb].tolist()  # First `nb` known inputs
    '''
    u[0].value = df_flux.flatten().tolist()
    y[0].value = df_temp.flatten()[0].tolist() * 0 + df_temp[0, :]
    m.time = df_time.flatten().tolist() # np.linspace(start=0, stop=tf, num=int(tf / dt) + 1)
    m.options.IMODE     = 4     # 4
    m.options.NODES     = 2     # 2
    m.options.SOLVER    = 2     # 2
    ''' 
    ------- IMODE
    1	Steady-State Simulation	Solves equations assuming steady-state (derivatives = 0).
    2	Dynamic Simulation	Simulates over time with given inputs.
    3	Parameter Estimation	Fits model parameters to data.
    4	Dynamic Simulation with Measured Inputs	Simulates system response to measured inputs (like your ARX model).
    5	Moving Horizon Estimation (MHE)	Online parameter estimation.
    6	Nonlinear Control (NLC)	Optimizes inputs for tracking a desired trajectory.
    7	Sequential Dynamic Simulation	Runs a dynamic simulation step-by-step.
    ------- NODES
    NODES = 1 → Uses only one collocation point per time step (default).
    NODES = 2 → Uses two collocation points for more accuracy. (ARX)
    Higher NODES (3-5) → Increases accuracy but may slow down solving.
    '''
    m.solve(disp=False, debug=False)    #m.solve(disp=True, debug=True)
    yARX_homemade9 = np.array(y, dtype=float).T
    print('test 9', np.allclose(yARX_opt, yARX_homemade9, rtol=rtol))

    # -------------------------------------------- Autres calculs GEKKO
    n = GEKKO(remote=False)

    y_1, p_1, k_1 = n.sysid(t=df_time.copy(), u=df_flux.copy(), y=df_temp.copy(), na=i_opt, nb=j_opt, pred='meas', shift='calc')# 'meas' for ARX regression form, explicit solution
    print('\n')
    print('ARX GEKKO')
    for keys, value in p_1.items():
        print(keys, value)
    print('\n')

    y_2, p_2, k_2 = n.sysid(t=df_time.copy(), u=df_flux.copy(), y=df_temp.copy(), na=i_opt, nb=j_opt, pred='model')             # 'model' for output error regression form, implicit solution
    print('ARX Output Error')
    for keys, value in p_2.items():
        print(keys, value)
    print('\n')

    '''
    y_3, p_3, k_3 = n.sysid(t=df_time.copy(), u=df_flux.copy(), y=df_temp.copy(), na=2, nb=2)                           # pref=None => 'model' for output error regression form, implicit solution
    for keys, value in p_3.items():
        print(keys, value)
    '''

    yARX_homemade_ = np.concatenate((
         yARX_homemade1,
         yARX_homemade2,
         yARX_homemade3,
         yARX_homemade4,
         yARX_homemade5,
         yARX_homemade6,
         yARX_homemade7,
         yARX_homemade8,
         yARX_homemade9,
         y_1,
         y_2
         ), axis=1)

    # -------------------------------------------- FIGURES
    plt.figure(num="Figure 0: ARX Homemade", figsize=(8, 6), tight_layout=True)
    plt.plot(df_time, df_temp, label='df_temp', color='black', ls='-', marker='x', lw=1, markersize=2)
    for i in range(yARX_homemade_.shape[1]):
        plt.plot(df_time, yARX_homemade_[:, i], label='yARX_' + str(i), ls='-', marker='x', lw=1, markersize=1)

    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.ylabel('Température [°C]')
    plt.xlabel('Time (sec)')
    plt.legend()
    ax0 = plt.gca().twinx()  # Set a second y-axis
    ax0.plot(df_time, df_flux, label='Flux', color='red', ls='--', lw=1, marker='x', markersize=1)
    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.legend()
    plt.ylabel('Flux [W]')

    plt.figure(num="Figure 1: ARX Homemade", figsize=(8, 6), tight_layout=True)
    plt.plot(df_time, df_temp, label='df_temp', color='black', ls='-', marker='x', lw=1, markersize=2)
    for i in range(yARX_homemade_.shape[1]):
        if i != 8:
            plt.plot(df_time, yARX_homemade_[:, i], label='yARX_' + str(i), ls='-', marker='x', lw=1, markersize=1)

    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.ylabel('Température [°C]')
    plt.xlabel('Time (sec)')
    plt.legend()
    ax0 = plt.gca().twinx()  # Set a second y-axis
    ax0.plot(df_time, df_flux, label='Flux', color='red', ls='--', lw=1, marker='x', markersize=1)
    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.legend()
    plt.ylabel('Flux [W]')

    plt.figure(num="Figure 2: ARX Homemade", figsize=(8, 6), tight_layout=True)
    plt.plot(df_time, df_temp, label='df_temp', color='black', ls='-', marker='x', lw=1, markersize=2)
    myList = (0, 2, 4, 8)
    for i in range(yARX_homemade_.shape[1]):
        if i not in myList:
            plt.plot(df_time, yARX_homemade_[:, i], label='yARX_' + str(i), ls='-', marker='x', lw=1, markersize=1)

    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.ylabel('Température [°C]')
    plt.xlabel('Time (sec)')
    plt.legend()
    ax0 = plt.gca().twinx()  # Set a second y-axis
    ax0.plot(df_time, df_flux, label='Flux', color='red', ls='--', lw=1, marker='x', markersize=1)
    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.legend()
    plt.ylabel('Flux [W]')

    plt.figure(num="Figure 3: ARX Homemade", figsize=(8, 6), tight_layout=True)
    plt.subplot(2, 1, 1)
    plt.plot(df_time, df_temp, label='df_temp', color='black', ls='-', marker='x', lw=1, markersize=2)
    for i in range(yARX_homemade_.shape[1]):
        if i not in myList:
            plt.plot(df_time, yARX_homemade_[:, i], label='yARX_' + str(i), ls='-', marker='x', lw=1, markersize=1)
    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.ylabel('Température [°C]')
    plt.xlabel('Time (sec)')
    plt.legend()
    ax0 = plt.gca().twinx()  # Set a second y-axis
    ax0.plot(df_time, df_flux, label='Flux', color='red', ls='--', lw=1, marker='x', markersize=1)
    plt.grid(True, lw=0.2, ls='--', color='gray')
    plt.legend()
    plt.ylabel('Flux [W]')
    plt.subplot(2, 1, 2)
    for i in range(yARX_homemade_.shape[1]):
        if i not in myList:
            plt.plot(df_time, yARX_homemade_[:, i] - df_temp[:, 0], label='yARX_' + str(i) + '- y_{Reference}', ls='-', marker='x', lw=1, markersize=1)
    plt.ylabel('Delta Température [°C]')
    plt.xlabel('Time (sec)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)

    plt.figure(num="Figure 4: ARX Homemade", figsize=(8, 10), tight_layout=True)
    kk = 0
    for i in range(yARX_homemade_.shape[1]):
        if i not in myList:
            kk += 1
    jj = 0
    for i in range(yARX_homemade_.shape[1]):
        if i not in myList:
            jj += 1
            plt.subplot(kk, 1, jj)
            plt.plot(df_time, yARX_homemade_[:, i] - df_temp[:, 0], label='yARX_' + str(i) + '- y_{Reference}', ls='-', marker='x', lw=1, markersize=1)
            plt.ylabel('DTemp [°C]')
            #plt.xlabel('Time (sec)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
            print(i, '\t', yARX_homemade_[-1, i])

    plt.show()

if __name__ == "__main__":
    calculation()