import numpy as np
import apm_id as arx

######################################################
# Configuration
######################################################
# number of terms
ny = 3 # output coefficients
nu = 3 # input coefficients
# number of inputs
ni = 2
# number of outputs
no = 2
# load data and parse into columns
data = np.loadtxt('data_no_headers.csv',delimiter=',')
######################################################

# generate time-series model
arx.apm_id(data,ni,nu,ny)
