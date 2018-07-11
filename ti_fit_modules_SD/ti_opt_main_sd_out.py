# usr/bin/env/ python 	
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 


import ti_opt_general_sd as g
import ti_opt_latpar_min_sd as lm
import ti_opt_elastconst_sd as ec
import ti_opt_constraints_variation_sd as cv
import ti_opt_vary_params_sd as vp 
import ti_opt_bandwidth_norm_sd as b
import ti_opt_output_sd as outp

######################################################################################################################
######################################################################################################################
#########################################     Initial Arguments    ###################################################


ext     = '.ti' 
LMarg   = ' tbe --mxq'
args    = ' -vfp=0 -vrfile=0 -vppmodti=10 -vB1TTSDpp=0 -vB2TTSDpp=0 '


#########################################################################################
#################    Pair potential and bond integral coefficients    ###################

pR0 = 8.18
qR0 = 2.77


alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
coa_ideal  = (8./3.)**(0.5)       

spanjdec    = qR0/alat_ideal
spanjdecpp  = pR0/alat_ideal

pair_pot    = np.array( [ 118.485, spanjdecpp, 0, 0  ] )
ppnames     = [ 'A1TTSDpp', 'C1TTSDpp', 'A2TTSDpp', 'C2TTSDpp' ]

ddcoeffs    = np.array( [ 6., 4., 1., spanjdec] )
ddnames     = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD', 'spanjdec', 'spanjddd']



#########################################################################################
#############################     Ideal lattice parameters     ##########################

names_lp  = ['alatTi'  , 'coa']
ideals_lp = [alat_ideal, coa_ideal]


#########################################################################################
########################    Extra Energies to Calculate      ############################

## Senkov, Chakoumas, Effect of temperature and hydrogen concentration on the lattice parameter of beta titanium, 2001
alat_bcc_ideal = 6.254960504932251  
alat_fcc_ideal = 8.201368154503314


n_energies  = 4
energy_args = (   
                    ' -vnbas=3 -vomegabcc=1 -vubccom=1 -valatTi='   + str(alat_ideal)       + ' ',  ##  bcc
                    ' -vnbas=1 -vbccs=1 -valatTi='                  + str(alat_bcc_ideal)   + ' ',  ##  bcc2
                    ' -vnbas=3 -vomegabcc=1 -vubccom=0 -valatTi='   + str(alat_ideal)       + ' ',  ##  omega
                    ' -vnbas=1 -vfccs=1 -valatTi='                  + str(alat_fcc_ideal)   + ' '   ##  fcc
                ) 





##########################################################################################
##########     Extrapolated Elastic Constants from Fisher and Renken 1964     ############


C11_FR = 1.1103616820304443
C33_FR = 1.1963098653973396
C44_FR = 0.3210116099936396
C66_FR = 0.2867984040386178
C12_FR = 0.5368771097739773
C13_FR = 0.4241634577392404 

###   Bulk modulus = 1./9. * ( 2*C11 + 2*C12 +  C33 + 4*C13 )
K_FR = (1./9.) * ( 2 *  C11_FR +  C33_FR + 2 * C12_FR  + 4 * C13_FR )
R_FR = (1./3.) * ( 0.5 * C11_FR +  C33_FR + 0.5 * C12_FR  - 2 * C13_FR )
H_FR = (1./3.) * ( 1.2 * C11_FR +  0.25 * C33_FR -  C12_FR  - 0.5 * C13_FR )

ec_exp_arr = np.array([     C11_FR,
                            C33_FR,
                            C44_FR,
                            C66_FR,
                            C12_FR,
                            C13_FR, 
                            K_FR, 
                            R_FR,
                            H_FR     ])


##  Ideal number of neighbours in unit cell.
##  nn_ideal = 12, as there are 12 nearest neighours in ideal coa = (8/3)**0.5 Titanium
nn_ideal    = 12

##  Number of lattice parameters
n_lp        = 2
##  Number of linspace values within ranges to find lattice parameters
n_grid      = [10, 10]
##  Number of reduced range grid searches to find lattice parameters
n_iter      = 2
##  Limits for the lattice parameters in form [ a_u, a_l ( (optional) , True_if_Fixed_Upper_Limit , True_if_Fixed_Lower_Limit ) ]
limits_lp   = [  [ 5.2, 6.2 ], [ 1.5, np.sqrt(8./3.), False, True] ]

##  Limits for the bandwidth normalisation and tolerance in eV.
ddnorm_lim  = (5.0, 0.0, 0.02)

npass       = 0



##  Number of passes, for use in optimisation. 
##  Arguments used in the calculation
##  Mininum lattice parameters
##  Difference of lattice parameters from ideal
##  Volume of structure when at minimum lattice parameters
##  Energies calculated for use in optimisation.


(npass,                                                 
args,                                                   
min_alat,  min_coa,                                     
alat_diff, coa_diff,                                    
min_vol,                                                
etot_hcp, etot_bcc, etot_bcc2, etot_omega, etot_fcc,    
e_consts, e_consts_diff) = outp.output_script(              npass, 
                                                            ext,
                                                            LMarg, 
                                                            args, 
                                                            pair_pot, ppnames, 
                                                            ddcoeffs, ddnames, ddnorm_lim,
                                                            ec_exp_arr, 
                                                            nn_ideal,
                                                            n_lp, n_grid, n_iter,
                                                            names_lp, limits_lp, ideals_lp,
                                                            n_energies, energy_args)
                                                        

################################################################################################################################
######################     Trialled Parameters that work reasonably, with bad elastic constants.     ###########################

##  This gets c and a with a very good agreement. The elastic constants are 
#par_arr = np.array([ 24.85, 1.247525, 0.886571, 0.565, 1.  ] )
#ddcoeffs = np.array([6., 4., 1., spanjdec])
#dddeltanorm =  0.19542694091796875


##  One with the lowest error so far
#ddcoeffs = np.array([10.20968613,  6.35171468,  1.70161435,  0.44714869])
#par_arr = np.array( [26.28518485,  1.29113683,  1.04802635,  0.64556841,  1.        ])

#################################################################################################################################

