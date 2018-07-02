# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 


import ti_opt_general 
import ti_opt_latpar_min
import ti_opt_elastconst
import ti_opt_constraints_variation
import ti_opt_vary_params
import ti_opt_bandwidth_norm

######################################################################################################################
######################################################################################################################




npass = 0

#A = 100 
#B = 5
#a = 1.2 ## 1.5
#b = 0.6 ## 0.7
#[21.1645847   1.16374756  1.02409773  0.629057    1.        ]


#A = 21.215
#a = 1.205 
#B = 1.06546811
#b = 0.5995
#scl_const =0.95
#par_arr = np.array([ A*scl_const, a, B*scl_const, b, 1.0]) # np.array( [21.28898933,  1.16721662 , 1.08835262 , 0.61034674,  1.        ])

#bond_int = 1.0 ; bond_int_temp=0.0
#evtol = 0.003
#pR0 = 8.18
#qR0 = 2.77
#alat_ideal = 5.57678969
#spanjdec=qR0/alat_ideal
#ddcoeffs = np.array([3.075984, 2.075984, 0.575984, spanjdec]) # np.array( [ 5.72701056e+00,  7.46293801e-01 , 4.14389303e-02 , 6.79489556e-01] )

A = 24.85
a = 1.247525
B = 0.886571
b = 0.565  #np.array( [ 24.85, 1.247525, 0.886571, 0.565  ] )
scl_const =1. 
par_arr = np.array([ A*scl_const, a, B*scl_const, b, 1.0]) # np.array( [21.28898933,  1.16721662 , 1.08835262 , 0.61034674,  1.        ])

bond_int = 1.0 ; bond_int_temp=0.0
evtol = 0.002
pR0 = 8.18
qR0 = 2.77
alat_ideal = 5.57678969
spanjdec=qR0/alat_ideal
ddcoeffs = np.array([3.075984, 2.075984, 0.575984, spanjdec]) # np.array( [ 5.72701056e+00,  7.46293801e-01 , 4.14389303e-02 , 6.79489556e-01] )
dddeltanorm =  0.19542694091796875 #0.208098266125

A = 21.215
a = 1.2
B = 1.075
b = 0.6  #np.array( [ 24.85, 1.247525, 0.886571, 0.565  ] )
scl =1.0
#par_arr = np.array([21.0825141 ,  1.17983585,  0.99597137,  0.6188312 ,  1. ])


#A = 25.0
#a = 1.2
#B = 1.2
#b = 0.5
scl = 1.#0.5
par_arr = np.array([ A*scl, a, B*scl, b, 1.0]) 
ddcoeffs = np.array([6., 4., 1., spanjdec])

#par_arr[0] = par_arr[0] * dddeltanorm/0.19542694091796875
#par_arr[2] = par_arr[2] * dddeltanorm/0.19542694091796875

pair_pot_bank= []
ddcoeff_bank = []
total_error=[0.0001]
dftwidth = False
ext = 'ti'
symmpt = 0
filename = 'out'
maxit = 2000
for i in range(maxit):
    npass, par_arr, ddcoeffs, pair_pot_bank, ddcoeff_bank, total_error = vary_params(maxit, npass, par_arr, ddcoeffs, pair_pot_bank, ddcoeff_bank, 
                                                                                                            total_error, dddeltanorm, pR0, qR0)
