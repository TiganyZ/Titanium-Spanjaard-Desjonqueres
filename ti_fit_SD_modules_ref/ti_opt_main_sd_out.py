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
args    = ' -vfp=0 -vrfile=0 -vppmodti=10 '


#########################################################################################
#################    Pair potential and bond integral coefficients    ###################

pR0 = 8.18
qR0 = 2.77

spanjdec    = qR0/alat_ideal
spanjdecpp  = pR0/alat_ideal

pair_pot    = np.array( [ 118.485, spanjdecpp, 0, 0, 0  ] )
ppnames     = [ 'A1TTSDpp', 'B1TTSDpp', 'C1TTSDpp' \
                'A2TTSDpp', 'B2TTSDpp', 'C2TTSDpp' ]

ddcoeffs    = np.array( [ 6., 4., 1., spanjdec] )
ddnames     = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD', 'spanjdec']



#########################################################################################
#############################     Ideal lattice parameters     ##########################


alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
coa_ideal  = (8./3.)**(0.5)       

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



npass,                                                  \   ##  Number of passes, for use in optimisation. 
args,                                                   \   ##  Arguments used in the calculation
min_alat,  min_coa,                                     \   ##  Mininum lattice parameters
alat_diff, coa_diff,                                    \   ##  Difference of lattice parameters from ideal
min_vol,                                                \   ##  Volume of structure when at minimum lattice parameters
etot_hcp, etot_bcc, etot_bcc2, etot_omega, etot_fcc,    \   ##  Energies calculated for use in optimisation.
e_consts, e_consts_diff = outp.output_script(               npass, 
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

                                                                                           total_error, dddeltanorm, pR0, qR0)


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

def get_elastic_constants(LMarg, args, alphal, cell_vol):

    print('\n get_elastic_constants Routine \n')
    
        
    C11_FR = 1.1103616820304443
    C33_FR = 1.1963098653973396
    C44_FR = 0.3210116099936396
    C66_FR = 0.2867984040386178
    C12_FR = 0.5368771097739773
    C13_FR = 0.4241634577392404 
    Kexp = 110.



    alphalist = alphal

    strain44 =          ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0.5' +\
                            ' -vexy=0 ' 

    C44, rowC44 = ec.ec_alpha_poly(LMarg, args, strain44, True, alphal, cell_vol) #x-z = [2 -1 -1 q**(-1)]
    #print('C44 = %s' %(C44))


    strain0001 = ' -vexx=-0.5' +\
                            ' -veyy=-0.5' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature0001, row0001 = ec.ec_alpha_poly(LMarg, args, strain0001, True, alphal, cell_vol)  #(0.5*C11 + C33 + 0.5*C12 - 2*C13)

    
    #x = [2 -1 -1 0]
    strainx = ' -vexx=1' +\
                            ' -veyy=-0.5' +\
                            ' -vezz=-0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 ' 
    curvaturex, xrow = ec.ec_alpha_poly(LMarg, args,strainx, True, alphal, cell_vol)  #(1.2*C11 + 0.25*C33 -  C12 - 0.5*C13)
    """
    #y = [1 -1 0 0]
    strainy = ' -vexx=-0.5' +\
                            ' -veyy=1' +\
                            ' -vezz=-0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 ' 
    curvaturey, yrow = ec.ec_alpha_poly(LMarg, args, strainy, True, alphal, cell_vol)  #(1.2*C11 + 0.25*C33 -  C12 - 0.5*C13)

    #x + y + z = [3 -2 -1 0]
    strainxyz = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0.5' +\
                            ' -vexz=0.5' +\
                            ' -vexy=0.5 ' 
    curvaturexyz, xyzrow = ec.ec_alpha_poly(LMarg, args, strainxyz, True, alphal, cell_vol)  #(0.5*(C11-C12) + 2*C44)

    xyzrow = [0.5, 0.0, -0.5, 0 ]  #[a11, a33, a12, a13]
    curvmnc44xyz=curvaturexyz-2*C44

    #print 'TRIAL'
    strainxyz2 = ' -vexx=0.5' +\
                            ' -veyy=0.5' +\
                            ' -vezz=0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 ' 
    curvaturexyz2, xyzrow2 = ec.ec_alpha_poly(LMarg, args, strainxyz2, True, alphal, cell_vol)  #(0.5*(C11-C12) + 2*C44)
    #Another Strain matrix
    """
    strainex1 = ' -vexx=1' +\
                            ' -veyy=0' +\
                            ' -vezz=-1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 ' 
    curvatureex1, ex1row = ec.ec_alpha_poly(LMarg, args, strainex1, True, alphal, cell_vol)  #(C11 + C33 - 2*C13)


    #Another Strain matrix
    strain66 = ' -vexx=-1' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 ' 
    curvature66, row66 = ec.ec_alpha_poly(LMarg, args, strain66, True, alphal, cell_vol)  #4( 0.5*(C11 - C12) )
    #print('curvature 66 is 0.25 * C66')
    #print(' C66 = ', 0.25*curvature66 )
    C66 =  0.25*curvature66




    strain9K = ' -vexx=1' +\
                            ' -veyy=1' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature9K, row9K = ec.ec_alpha_poly(LMarg, args, strain9K, True, alphal, cell_vol)
    #print(' 9K= %s' %(curvature9K) ) 
    K = 1/9. * curvature9K 
    print ('Bulk Modulus = %s' %(1/9. * curvature9K  ))




    C11 = 2 * curvature0001 - 8 * curvaturex + 4.5 * curvature66
    C12 = 2 * curvature0001 - 8 * curvaturex + 4 * curvature66
    C33 = 0.5 * ( curvature9K - 10 * curvature0001 + 48 * curvaturex - 51./2. *  curvature66 ) 
    C13 = 1./8. * curvature9K - 0.75 * curvature0001 + 2 * curvaturex - 17./16. *  curvature66 

    Kcalc =1./9. * ( 2*C11 + 2*C12 +  C33 + 4*C13 )

    evpamin3 = 160.21766208
    print('Elastic Constants: \n C11 = %s,    C11exp = %s eV/A**3,\
                                \n C33 = %s,    C33exp = %s eV/A**3,\
                                \n C12 = %s,    C12exp = %s eV/A**3,\
                                \n C13 = %s,    C13exp = %s eV/A**3,\
                                \n C44 = %s,    C44exp = %s eV/A**3,\
                                \n  C66 = %s,    C66exp = %s eV/A**3\
                                ' %(
                                    C11,      C11_FR,
                                    C33,      C33_FR,
                                    C12,      C12_FR,
                                    C13,      C13_FR,
                                    C44,      C44_FR,
                                    C66,      C66_FR ))

    

    return C11, C33, C12, C13, C44, C66, K     


def fast_elastic_consts(LMarg, args, alphal, cell_vol):
    alphalist = alphal
    
    C11_FR = 1.1103616820304443
    C33_FR = 1.1963098653973396
    C44_FR = 0.3210116099936396
    C66_FR = 0.2867984040386178
    C12_FR = 0.5368771097739773
    C13_FR = 0.4241634577392404 
    
    curvature = []
    strain11 = ' -vexx=1.0' +\
                            ' -veyy=1.0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature11, row11 = ec.ec_alpha_poly(LMarg, args, strain11, True, alphal, cell_vol)  
    
    print(' Curvature 11 = %s' %(curvature11) )
    curvature.append(curvature11)                                                     #1

    strain112 = ' -vexx=1.0' +\
                            ' -veyy=-1.0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature112, row112 = ec.ec_alpha_poly(LMarg, args, strain112, True, alphal, cell_vol) 
    print(' Curvature 12 = %s' %(curvature112) )
    curvature.append(curvature112)

    c11 = 0.25 * (curvature11 + curvature112) 
    c12 = 0.25 * (curvature11 - curvature112)            

    print('C11 = 0.25 * ( curvature110 +  curvature1-10 ) = %s' %(c11 ) )        
    print('C12 = 0.25 * ( curvature110 -  curvature1-10 ) = %s' %(c12 ) )                                     #2

    strain33 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature33, row33 = ec.ec_alpha_poly(LMarg, args, strain33, True, alphal, cell_vol) 
    print(' C33 = %s' %(curvature33) ) 
    curvature.append(curvature33)  
    c33 = curvature33                                                   #3
  
    strain44 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=1.0' +\
                            ' -vexy=0 '

    curvature44, row44 = ec.ec_alpha_poly(LMarg, args, strain44, True, alphal, cell_vol) 
    print(' C44 = 0.25 curvature44 = %s' %(0.25 * curvature44) ) 
    curvature.append(curvature44)       
    c44 = 0.25 * curvature44                                              #4
                                                    #5
  
    strainbulkmod = ' -vexx=1.0' +\
                            ' -veyy=1.0' +\
                            ' -vezz=1.0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvaturebulkmod, rowbulkmod = ec.ec_alpha_poly(LMarg, args, strainbulkmod, True, alphal, cell_vol)  
    print(' 2C11 + 2C12 +  C33 + 4*C13 = %s' %(curvaturebulkmod) ) 
    curvature.append(curvaturebulkmod)   
    c13 = 0.25 * ( curvaturebulkmod - 2*c11 - 2*c12 - c33 )        
    print('C13 = 0.25 * ( curvaturebulkmod - 2C11 - 2C12 -C33 ) = %s' %(c13))     

    c66 = 0.5 * (c11 - c12) 
    print('C66 = 0.5 * (C11 - C12) = %s ' %(c66))                                           #6
  


    print( 'c11 = %s,   c11exp = %s' %(c11, C11_FR))
    print( 'c33 = %s,   c33exp = %s' %(c33, C33_FR))
    print( 'c12 = %s,   c12exp = %s' %(c12, C12_FR))
    print( 'c13 = %s,   c13exp = %s' %(c13, C13_FR))
    print( 'c44 = %s,   c44exp = %s' %(c44, C44_FR))
    print( 'c66 = %s,   c66exp = %s' %(c66, C66_FR))

    return  [c11, c33, c12, c13, c44, c66] 

#def

def bulk_mod_EV(LMarg, args, alat_min, coa_min, ns):

    vol =  (  3**(0.5) / 2. )  *  alat_min**3  *  coa_min

    ##  for a 5% difference either side we have alat vary between a / (1.05)**(1./3.)  and   a / (0.95)**(1./3.)

    al = np.linspace( min_alat / (0.99)**(1./3.),  min_alat / (1.01)**(1./3.), ns+1)
    t_vol = al**3 * coa_min
    el = []
    for i in range(len(al)):
        xx_args = args + g.construct_cmd_arg('alatTi', al[i]) + g.construct_cmd_arg('coa', min_coa)
        #print('/n xx_args = \n %s'%(xx_args))
        etot = g.find_energy(LMarg, xx_args, 'bulkmodtest')
        el.append(etot)
        #print('Energy = %s, vol = %s, alat = %s, coa = %s' %(etot, t_vol[i], al[i], min_coa))

    ##  Central difference scheme to find the bulk modulus,
    ##  d2  =  ( t_ip1 - 2*t_i + t_im1 ) / dx
    dx = abs( (t_vol[0] - t_vol[-1])/float(ns) )
    c_ind = np.argmin(el)
    print('ns, cind', ns, c_ind)
    cds = (  el[ (c_ind + 1)%len(al)] - 2 * el[c_ind] + el[c_ind - 1]  ) / dx**2

    alatn = ( ( 2. / ( 3**(0.5) * min_coa ) ) * t_vol[c_ind] )**(1./3.)

    print('min_E = %s, alat = %s, coa = %s' %(el[c_ind], al[c_ind], min_coa ))

    K = vol * cds

    ##  1 eV/Angstrom3 = 160.21766208 GPa
    ev_ang3_to_gpa = 160.21766208
    ##  1 bohr = x angstrom
    bohr_to_angstrom = 0.529177208 
    ##  Rydberg to electron volts
    ryd_to_eV = 13.606

    Kgpa = K * (ryd_to_eV / (bohr_to_angstrom**3 ) ) * ev_ang3_to_gpa
    print('Bulk Modulus = %s(ryd/bohr^3) \n Bulk Mod (GPa) = %sGPa, Expected = %s' %(K, Kgpa, 110))
    #"""
    #g.plot_function(    1, 
    #                        t_vol, 
    #                        el, 
    #                        'r-', 
    #                        'Energy-Volume curve for a = %.3f and c/a = %.3f'%(min_alat, min_coa), 
    #                        r'Volume ($ \AA ^ {3} $)', 
    #                        'Energy (Ryd)')
    #"""

    return K

#################################################################################################################################
################################     Testing Scripts for the Elastic constants and c/a     ######################################


#"""

LMarg = 'tbe --mxq ctrl.ti '
args = ' -vfp=0 -vrfile=0 '

#args += ' -vspanjdec=' + str(ddcoeffs[-1]) + ' '


dddeltanorm = 0.1954345703125
ddnames = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD', 'spanjdec']
d_norm = ' -vspanjddd=' + str(dddeltanorm) + ' '
#d_norm, E_F = b.band_width_normalise( LMarg, args, symmpt, ext, ddnames, ddcoeffs[:-1], bond_int, bond_int_temp, evtol)

dargs = g.construct_extra_args('', ddnames[:-1], ddcoeffs[:-1]) + d_norm #+ E_F
args += dargs 


par_arr = np.array([ 135, spanjdecpp, 0, 0 , 0  ] )
ppargs =  g.get_pp_args(par_arr)

alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
coa_ideal = clat_ideal/alat_ideal




tol_a = 0.001
tol_coa = 0.001
a_u = 6.2; a_l = 5.4; coa_u = (8./3.)**0.5; coa_l = 1.5

#"""
p0 = np.linspace(135, 110, 5)
p2 = np.linspace(0.1, 1.2, 5)
adli = []
coadli = []
kl = []
p0 = [p0[0]]
p2 = [p2[0]]
"""
for i in range(len(p0)):
    for j in range(len(p2)):
        print('\n Testing iter = %s \n '%(i))
        par_arr[0] = p0[i]* (1.)
        par_arr[2] = 0.#p2[j]
        par_arr[3] = spanjdecpp/2.
        ppargs =  g.get_pp_args(par_arr)
        print('pp = %s' %(par_arr))
        min_alat, alat_diff, min_coa, coa_diff, cell_vol = lm.opt_latpars_grid(LMarg, args + ppargs, par_arr, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal)
        adli.append(alat_diff)
        coadli.append(coa_diff * min_alat)
        K = bulk_mod_EV(LMarg, args + ppargs, min_alat, min_coa, 30)
        kl.append(K)
        print('adli = %s' %(adli))
        print('coadli = %s' %(coadli))
        print('kl = %s' %(kl))
"""
print('#########################################\n adli = %s\n coadli = %s' %(adli, coadli))

min_alat = 5.574222222222223
min_coa    = 1.632993161855452
cell_vol    = 244.9445792941632

args += ppargs
#K = bulk_mod_EV(LMarg, args, min_alat, min_coa, 30)

tdl = np.asarray(adli) + np.asarray(coadli)
print('minimum'  )




##########################################################
######    Parameters to keep    ##########################
##########################################################
##  To get the right ideal c/a and the right alat
##  par_arr = np.array([ 135, spanjdecpp, 0, 0, 0  ] )

##  The best Bulk Modulus with closest alat: (alat = 5.464, c/a = 1.632993161855452) K = 109.735
##  par_arr = [121.71717172,   1.4667937,    0.,           0.,   0.        ]

##  The best Bulk Modulus: (alat = 5.4391, c/a = 1.632993161855452) K = 110.16 GPa 
##  par_arr = [118.48484848   1.4667937    0.           0.73339685   0.        ]









#"""
n_a = 10
n_coa = 10
#min_alat, min_coa =  line_search_min_coa_alat(LMarg, args, par_arr, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, tol_a, tol_coa)
#brute_force_min_coa_and_alat(LMarg, args, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, n_a, n_coa)

 
args +=' -vcoa=' + str(min_coa)  +  ' -valatTi=' + str(min_alat)  


alphal = np.linspace(-0.01, 0.01, 11)

#eco = get_elastic_constants(LMarg, args, alphal, cell_vol)
#cell_vol = 238.
#"""
#"""
curvature = []
strain11 = ' -vexx=0.5' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature11, row11 = ec.ec_alpha_cds(LMarg, args, strain11, True, alphal, cell_vol)  
    
print(' C11 = %s' %(curvature11) )
curvature.append(curvature11)                                                     #1

strain112 = ' -vexx=0.0' +\
                            ' -veyy=0.5' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature112, row112 = ec.ec_alpha_cds(LMarg, args, strain112, True, alphal, cell_vol) 
print(' C12 = %s' %(curvature112) )
curvature.append(curvature112)                                                     #2

strain33 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature33, row33 = ec.ec_alpha_cds(LMarg, args, strain33, True, alphal, cell_vol) 
print(' C33 = %s' %(curvature33) ) 
curvature.append(curvature33)                                                     #3

strain2C112C22 = ' -vexx=0.5.0' +\
                            ' -veyy=0.5' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature2C112C12, row2C112C22 = ec.ec_alpha_cds(LMarg, args, strain2C112C22, True, alphal, cell_vol) 
print(' 2*C11 + 2*C12 = %s' %(curvature2C112C12) ) 
curvature.append(curvature2C112C12)                                                     #4
  
strain5o4C11C12 = ' -vexx=0.25' +\
                            ' -veyy=0.5' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature5o4C11C12, row5o4C11C12 = ec.ec_alpha_cds(LMarg, args, strain5o4C11C12, True, alphal, cell_vol)  
print(' 5/4 * C11 + C12 = %s' %(curvature5o4C11C12) ) 
curvature.append(curvature5o4C11C12)                                                     #5
 
strainC11C332C13 = ' -vexx=0.5' +\
                            ' -veyy=0' +\
                            ' -vezz=0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvatureC11C332C13, rowC11C332C13 = ec.ec_alpha_cds(LMarg, args, strainC11C332C13, True, alphal, cell_vol)  
print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C13) ) 
curvature.append(curvatureC11C332C13)                                                           #6
  
strainC11C332C132 = ' -vexx=0' +\
                            ' -veyy=0.5' +\
                            ' -vezz=0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvatureC11C332C132, rowC11C332C132 = ec.ec_alpha_cds(LMarg, args, strainC11C332C132, True, alphal, cell_vol)  
print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C132) ) 
curvature.append(curvatureC11C332C132)                                                            #7
#"""
"""  
    strain2C112C12C334C13 = ' -vexx=1' +\
                            ' -veyy=1' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature2C112C12C334C13, row2C112C12C334C13 = ec_alpha_poly(LMarg, args, strain2C112C12C334C13, False, alphal, cell_vol)  
    print(' 2*C11 + 2*C12 + C33 + 4*C13 = %s' %(curvature2C112C12C334C13) ) 
    curvature.append(curvature2C112C12C334C13)                                                             #8
"""
#"""
curvature.append(0)


strain4C44 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0.5' +\
                            ' -vexy=0 '

curvature4C44, row4C44 = ec.ec_alpha_cds(LMarg, args, strain4C44, True, alphal, cell_vol)  
print(' 4*C44 1 = %s' %(curvature4C44) ) 
curvature.append(curvature4C44  * 4.)                                                           #9
  
strain4C442 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0.5' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature4C442, row4C442 = ec.ec_alpha_cds(LMarg, args, strain4C442, True, alphal, cell_vol)  
print(' 4*C44 2= %s' %(curvature4C442) ) 
curvature.append(curvature4C442 * 4.)                                                               #10
  
strain8C44 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0.5' +\
                            ' -vexz=0.5' +\
                            ' -vexy=0 '

curvature8C44, row8C44 = ec.ec_alpha_cds(LMarg, args, strain8C44, True, alphal, cell_vol)  
print(' 8*C44 + 2C11 - 2C12 (8*C44 + 4*C66)= %s' %(curvature8C44) ) 
curvature.append(curvature8C44 * 4.)                                                            #11
  
strain4C66 = ' -vexx=1' +\
                            ' -veyy=-1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature4C66, row4C66 = ec.ec_alpha_cds(LMarg, args, strain4C66, True, alphal, cell_vol)  
print(' 4*C66 1 = %s' %(curvature4C66) ) 
curvature.append(curvature4C66 * 4.)                                                                #12
  
strain4C662 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=1 '

curvature4C662, row4C662 = ec.ec_alpha_cds(LMarg, args, strain4C662, True, alphal, cell_vol) 
print(' 4*C66 2 = %s' %(curvature4C662) ) 
#print ('row 4c662', row4C662)
curvature.append(curvature4C662 * 4.)    
#curvature.append(curvature8C44)       

#"""
