# see https://apmonitor.com/wiki/index.php/Apps/ARXTimeSeries
from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and parse into columns
url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)
t = data['Time']
u = data['H1']
y = data['T1']

m = GEKKO()

# system identification
na = 2 # output coefficients
nb = 2 # input coefficients
yp, p, K = m.sysid(t, u, y, na, nb, pred='meas')

for keys, value in p.items():
    print(keys, value)

plt.figure(figsize=(7, 4))
plt.subplot(2, 1, 1)
plt.plot(t, u, label=r'$Heater_1$')
plt.legend()
plt.ylabel('Heater')
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.plot(t, yp)
plt.legend([r'$T_{meas}$', r'$T_{pred}$'])
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (sec)')
plt.tight_layout()
plt.savefig('test.png', dpi=300)
plt.show()
