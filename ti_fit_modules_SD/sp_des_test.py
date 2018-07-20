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
import bayes_opt_ti_g_process as gpr

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
pqr = pR0/qR0


alat_ideal  = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
clat_ideal  = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
coa_ideal   = 1.587       #(8./3.)**(0.5)       

spanjdec    = qR0/alat_ideal
spanjdecpp  = pR0/alat_ideal

pair_pot    = np.array( [ 125., spanjdecpp, 0.001, spanjdecpp/2.  ] ) #np.array( [ 118.485, spanjdecpp, 0, 0  ] )
ppnames     = [ 'A1TTSDpp', 'C1TTSDpp', 'A2TTSDpp', 'C2TTSDpp' ]

ddcoeffs    = np.array( [ 6., 4., 1., spanjdec] )
ddnames     = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD', 'spanjdec', 'spanjddd']



#########################################################################################
#############################     Ideal lattice parameters     ##########################

names_lp  = ['alatTi'  , 'coa']
ideals_lp = [alat_ideal, coa_ideal]

rmx_name  = 'rmaxh'

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
limits_lp   = [  [ 5.2, 6.2, False, False ], [ 1.5, np.sqrt(8./3.) + 0.3, False, False] ]

##  Limits for the bandwidth normalisation and tolerance in eV.
ddnorm_lim  = (10.0, 0.0, 0.02)

##  Plots the Elastic constants and the bulk modulus curves if True. 
plotECandB = False

npass       = 0



##  npass:  		Number of passes, for use in optimisation. 
##  args:   		Arguments used in the calculation
##  a (, b/a, c/a):  	Mininum lattice parameters
##  a_d(, b_d, c_d):	Difference of lattice parameters from ideal
##  cell_vol:		Volume of structure when at minimum lattice parameters
##  etot(, etot2, etc)	Energies calculated for use in optimisation.
##  e_c, e_c_d		Elastic constants calculated and their difference from expected values.


##  Notes on parameter searches

##  bulk mod deccreases with addition of second exponential
##  a lat increases with addition of second exponential
##  coa is still constant

A = np.linspace(400, 1400 , 5)
# 1200 2 0.2

A = np.linspace(0.8,2000, 2000)*262.4
B = np.linspace(1., 2.5, 2000)* spanjdecpp
C = np.linspace(0., 5., 2000)
D = np.linspace(0, 0.9, 2000  ) * B * spanjdecpp 
#D = np.linspace(0.01)

t_      = np.array( [ ] )
x_      = np.array( [ ] )
t_coa   = np.array( [ ] )
x_coa   = np.array( [ ] )
t_a     = np.array( [ ] )
x_a     = np.array( [ ] )

x_A     = np.array( [ ] )
x_B     = np.array( [ ] )
x_C     = np.array( [ ] )
x_D     = np.array( [ ] )

t_A     = np.array( [ ] )
t_B     = np.array( [ ] )
t_C     = np.array( [ ] )
t_D     = np.array( [ ] )

K_A     = np.array( [ ] )
K_B     = np.array( [ ] )
K_C     = np.array( [ ] )
K_D     = np.array( [ ] )

K       = []
K_coa   = []
K_a     = []
inddef = False
C_      = []
C_inv   = [] 
beta    = 0.0001

n = 200
for knt  in range(2000):
    if knt < 1:
        ind_A = np.random.choice( range( len(A) ) )
        ind_B = np.random.choice( range( len(B) ) )
        ind_C = np.random.choice( range( len(C) ) )
        ind_D = np.random.choice( range( len(D) ) )

    print('Gaussian process regression:\n   Bulk Modulus and A pp\n    Iteration  %s' %(knt))

    #Pair potential  = [5.95219592e+05 3.66698426e+00 1.34564103e-01 7.33396851e-01]
    #Pair potential  = [5.95219592e+05 3.66698426e+00 2.69128205e-01 7.33396851e-01]
    ##
    ## Right c/a wrong a
    ##  Pair potential  = [5.95219592e+05, 3.66698426e+00, 9.41948718e-01, 7.33396851e-01]



    #pair_pot = np.array([ 24.85, 1.247525, 0.886571, 0.565, 1.  ] )
    pair_pot  = np.array( [ A[ind_A], B[ind_B] , C[ind_C] , D[ind_D]   ] )

    ddcoeffs    = np.array( [ 6., 4., 1., B[ind_B] / pqr] )
    
    #pair_pot  = np.array( [ 0.8*262.4, spanjdecpp, 0.005*262.4, spanjdecpp/2.   ] )
    #Okay    
    #pair_pot  = np.array( [ 1.0*262.4, spanjdecpp, 0.005*262.4, spanjdecpp/2.   ] )
    # mainly +ve ec, a = 6.2 c/a ideal

    theta     = np.array( [ 200., 0.1, 0., 0. ] )
    theta_a   = np.array( [ 40,   0.1, 0., 0. ] )
    theta_coa = np.array( [ 40,   0.1, 0., 0. ] )

    plotECandB = False

    outs = outp.output_script(      npass, 
                                    ext,
                                    LMarg, 
                                    args, 
                                    pair_pot, ppnames, 
                                    ddcoeffs, ddnames, ddnorm_lim,
                                    ec_exp_arr, 
                                    rmx_name, nn_ideal,
                                    n_lp, n_grid, n_iter,
                                    names_lp, limits_lp, ideals_lp,
                                    n_energies, energy_args,
                                    plotECandB   )
    (npass,                                                 
    itargs,                                                   
    min_alat,  min_coa,                                     
    alat_diff, coa_diff,                                    
    min_vol, Bk,                                                
    etot_hcp, etot_bcc, etot_bcc2, etot_omega, etot_fcc,    
    e_consts, e_consts_diff)                                =   outs



    targets = outs[2] + outs[3] + outs[7] 

    error_lms = alat_diff**2 + coa_diff**2 + np.sum(e_consts_diff**2) + (Bk - 110.)**2

    print('   alat_diff = %s\n   coa_diff = %s\n   e_consts_diff = %s\n   Bk = %s\n   error_lms = %s' %(alat_diff, coa_diff, e_consts_diff, Bk, error_lms) )
    print('Gaussian process regression: Iteration  %s' %(knt))



    x_A  = np.append( x_A, A[ind_A]  )
    x_B  = np.append( x_B, B[ind_B]  )
    x_C  = np.append( x_C, C[ind_C]  )
    x_D  = np.append( x_D, D[ind_D]  )

    t_  = np.append( t_, error_lms )

    if npass < 2:
        update = False
    else:
        update = True

    theta     = np.array( [ 1., 4, 0., 0. ] )

    m_A, var_A, K_A, C_N_A, C_N_inv_A = gpr.gaussian_process_regression( x_A, A[ind_A], t_, K_A, beta, theta, update )
    m_B, var_B, K_B, C_N_B, C_N_inv_B = gpr.gaussian_process_regression( x_B, B[ind_B], t_, K_B, beta, theta, update )
    m_C, var_C, K_C, C_N_C, C_N_inv_C = gpr.gaussian_process_regression( x_C, C[ind_C], t_, K_C, beta, theta, update )
    m_D, var_D, K_D, C_N_D, C_N_inv_D = gpr.gaussian_process_regression( x_D, D[ind_D], t_, K_D, beta, theta, update )



    E_A    = np.array([]); E_B = np.array([]); E_C = np.array([]); E_D = np.array([])
    y_EI_A = np.array([])
    y_EI_B = np.array([])
    y_EI_C = np.array([])
    y_EI_D = np.array([])

    for j in range( len(A) ):
        E = np.linspace(0, 20, len(A))
        m_A    =  gpr.m_pred_xnp1_sum(      theta, x_A, A[j], C_N_inv_A, t_       )
        k_     =  gpr.get_next_k_(          theta, x_A, A[j]                      )
        var_A  =  gpr.var_pred_next_target( theta,      A[j], C_N_inv_A, k_, beta )

        y_EI_A = np.append( y_EI_A, gpr.EI_point(m_A, var_A, E[j], t_, n) )
        E_A  = np.append(E_A,  m_A)

    for j in range( len(B) ):
        E = np.linspace(0, 20, len(B))
        m_B    =  gpr.m_pred_xnp1_sum(      theta, x_B, B[j], C_N_inv_B, t_       )
        k_     =  gpr.get_next_k_(          theta, x_B, B[j]                      )
        var_B  =  gpr.var_pred_next_target( theta,      B[j], C_N_inv_B, k_, beta )

        y_EI_B = np.append( y_EI_B, gpr.EI_point(m_B, var_B, E[j], t_, n) )
        E_B  = np.append(E_B,  m_B)

    for j in range( len(C) ):
        E = np.linspace(0, 20, len(C))
        m_C    =  gpr.m_pred_xnp1_sum(      theta, x_C, C[j], C_N_inv_C, t_       )
        k_     =  gpr.get_next_k_(          theta, x_C, C[j]                      )
        var_C  =  gpr.var_pred_next_target( theta,      C[j], C_N_inv_C, k_, beta )

        y_EI_C = np.append( y_EI_C, gpr.EI_point(m_C, var_C, E[j], t_, n) )
        E_C  = np.append(E_C,  m_C)

    for j in range( len(D) ):
        E = np.linspace(0, 20, len(D))
        m_D    =  gpr.m_pred_xnp1_sum(      theta, x_D, D[j], C_N_inv_D, t_       )
        k_     =  gpr.get_next_k_(          theta, x_D, D[j]                      )
        var_D  =  gpr.var_pred_next_target( theta,      D[j], C_N_inv_D, k_, beta )

        y_EI_D = np.append( y_EI_D, gpr.EI_point(m_D, var_D, E[j], t_, n) )
        E_D  = np.append(E_D,  m_D)


    grad_A = ( E[ind_A + 1] - E[ind_A] ) / ( A[ind_A + 1] - A[ind_A] )
    grad_B = ( E[ind_B + 1] - E[ind_B] ) / ( B[ind_B + 1] - B[ind_B] )
    grad_C = ( E[ind_C + 1] - E[ind_C] ) / ( C[ind_C + 1] - C[ind_C] )
    grad_D = ( E[ind_D + 1] - E[ind_D] ) / ( D[ind_D + 1] - D[ind_D] )

    inddef = False

    if npass % 10 == 0:
        ind_A = np.argmax(y_EI_A)
        ind_B = np.argmax(y_EI_B)
        ind_C = np.argmax(y_EI_C)
        ind_D = np.argmax(y_EI_D)

        print('EI indices:\n    ind = %s, A = %s,\n    ind = %s, B = %s,\n    ind = %s, C = %s,\n    ind = %s, D = %s, ' %(ind_A, A[ind_A],
                                    ind_B, B[ind_B], ind_C, C[ind_C], ind_D, D[ind_D]))

    #elif inddef:
    #    pass
    else:
        ind_A = np.random.choice( range( len(A) ) )
        ind_B = np.random.choice( range( len(B) ) )
        ind_C = np.random.choice( range( len(C) ) )
        ind_D = np.random.choice( range( len(D) ) )

        print('Random indices:\n    ind = %s, A = %s,\n    ind = %s, B = %s,\n    ind = %s, C = %s,\n    ind = %s, D = %s, ' %(ind_A, A[ind_A],
                                    ind_B, B[ind_B], ind_C, C[ind_C], ind_D, D[ind_D]) )

    plotgpr = False

    if plotgpr: 
        fig = plt.figure()
        ax1 = plt.subplot(221); ax2 = plt.subplot(222); ax3 = plt.subplot(223); ax4 = plt.subplot(224)

        ax1.plot( x_A, t_,  'b^') ; ax1.plot( A,   y_EI_A, 'c-'); 
        ax1.plot( A,   E_A, 'k-') 
        ax2.plot( x_B, t_,  'b^') ; ax2.plot( B,   y_EI_B, 'c-'); 
        ax2.plot( B,   E_B, 'k-') 
        ax3.plot( x_C, t_,  'b^') ; ax3.plot( C,   y_EI_C, 'c-'); 
        ax3.plot( C,   E_C, 'k-') 
        ax4.plot( x_D, t_,  'b^') ; ax4.plot( D,   y_EI_D, 'c-'); 
        ax4.plot( D,   E_D, 'k-') 

        ax1.set_title( 'GPR: Error vs Pair potential A coeff.' )
        ax2.set_title( 'GPR: Error vs Pair potential B coeff.' )
        ax3.set_title( 'GPR: Error vs Pair potential C coeff.' )
        ax4.set_title( 'GPR: Error vs Pair potential D coeff.' )

        ax1.set_xlabel( 'Pair potential A coeff.' )
        ax2.set_xlabel( 'Pair potential B coeff.' )
        ax3.set_xlabel( 'Pair potential C coeff.' )
        ax4.set_xlabel( 'Pair potential D coeff.' )

        ax1.set_ylabel( 'Squared Error' )
        ax2.set_ylabel( 'Squared Error' )
        ax3.set_ylabel( 'Squared Error' )
        ax4.set_ylabel( 'Squared Error' )

        plt.show()

    """
    x_  = np.append( x_, A[knt]  )
    t_  = np.append( t_, B )


    x_a  = np.append( x_a, A[knt]  )
    t_a  = np.append( t_a, min_alat )
    x_coa  = np.append( x_coa, A[knt]  )
    t_coa  = np.append( t_coa, min_coa )

    if knt == 1:
        update = False
    else:
        update = True


    m_p_tnp1,     var_p_tnp1,     K,     C_N,     C_N_inv     = gpr.gaussian_process_regression(x_,    A[knt], t_,    K,     beta, theta,  update)
    m_p_tnp1_coa, var_p_tnp1_a,   K_a,   C_N_a,   C_N_inv_a   = gpr.gaussian_process_regression(x_a,   A[knt], t_a,   K_a,   beta, theta_a,  update)
    m_p_tnp1_coa, var_p_tnp1_coa, K_coa, C_N_coa, C_N_inv_coa = gpr.gaussian_process_regression(x_coa, A[knt], t_coa, K_coa, beta, theta_coa,  update)


    ybayes  = np.array([]); ybayes2 = np.array([]); ybayes3 = np.array([])
    ybayes4 = np.array([]); ybayes5 = np.array([]); ybayes6 = np.array([])

    ybayes_a  = np.array([]); ybayes2_a = np.array([]); ybayes3_a = np.array([])
    ybayes4_a = np.array([]); ybayes5_a = np.array([]); ybayes6_a = np.array([])

    ybayes_coa  = np.array([]); ybayes2_coa = np.array([]); ybayes3_coa = np.array([])
    ybayes4_coa = np.array([]); ybayes5_coa = np.array([]); ybayes6_coa = np.array([])

    As = np.linspace(120, 160, 200)
    for j in range( len(As)):

        m_p_tnp1     = gpr.m_pred_xnp1_sum(theta, x_,    As[j], C_N_inv,     t_)
        m_p_tnp1_a   = gpr.m_pred_xnp1_sum(theta_a, x_a,   As[j], C_N_inv_a,   t_a)
        m_p_tnp1_coa = gpr.m_pred_xnp1_sum(theta_coa, x_coa, As[j], C_N_inv_coa, t_coa)

        t_np1        = np.random.normal(m_p_tnp1, var_p_tnp1, 6)
        t_np1_a      = np.random.normal(m_p_tnp1_a, var_p_tnp1_a, 6)
        t_np1_coa    = np.random.normal(m_p_tnp1_coa, var_p_tnp1_coa, 6)

        ybayes  = np.append(ybayes,  m_p_tnp1); ybayes2 = np.append(ybayes2, t_np1[1]); ybayes3 = np.append(ybayes3, t_np1[2])
        ybayes4 = np.append(ybayes4, t_np1[3]); ybayes5 = np.append(ybayes5, t_np1[4]); ybayes6 = np.append(ybayes6, t_np1[5])     

        ybayes_a  = np.append(ybayes_a,  m_p_tnp1_a); ybayes2_a = np.append(ybayes2_a, t_np1_a[1]); ybayes3_a = np.append(ybayes3_a, t_np1_a[2])
        ybayes4_a = np.append(ybayes4_a, t_np1_a[3]); ybayes5_a = np.append(ybayes5_a, t_np1_a[4]); ybayes6_a = np.append(ybayes6_a, t_np1_a[5])  

        ybayes_coa  = np.append(ybayes_coa,  m_p_tnp1_coa); ybayes2_coa = np.append(ybayes2_coa, t_np1_coa[1]); ybayes3_coa = np.append(ybayes3_coa, t_np1_coa[2])
        ybayes4_coa = np.append(ybayes4_coa, t_np1_coa[3]); ybayes5_coa = np.append(ybayes5_coa, t_np1_coa[4]); ybayes6_coa = np.append(ybayes6_coa, t_np1_coa[5])      
            
    xp =     [x_, As,      As,       As,       As,       As,       As      ]
    yp =     [t_, ybayes, ybayes2, ybayes3, ybayes4, ybayes5, ybayes6]

    xp_a =     [x_a, As,      As,       As,       As,       As,       As      ]
    yp_a =     [t_a, ybayes_a, ybayes2_a, ybayes3_a, ybayes4_a, ybayes5_a, ybayes6_a]

    xp_coa =     [x_coa, As,      As,       As,       As,       As,       As      ]
    yp_coa =     [t_coa, ybayes_coa, ybayes2_coa, ybayes3_coa, ybayes4_coa, ybayes5_coa, ybayes6_coa]

    colour = [ 'b^', 'k-', 'y-', 'y-', 'y-', 'y-', 'y-']

    print(len(ybayes), len(As), len(ybayes_a), len(x_coa), len(t_coa))
    if knt  == len(A) - 1:
        g.plot_function(2, xp, yp, colour, 'GPR: Bulk Modulus vs A.', 
                            'A', 'Bulk Modulus (GPa)')

        g.plot_function(2, xp_a, yp_a, colour, 'GPR: a lattice parameter vs A.', 
                            'A', 'alat (Bohr)')

        g.plot_function(2, xp_coa, yp_coa, colour, 'GPR: c/a lattice parameter vs A.', 
                            'A', 'c/a ratio')
    """

################################################################################################################################
################################################################################################################################
##############################     Results with good bulk modulus and IDEAL c/a ratio     ######################################

##  Arguments 
""" 
-vfp=0 -vrfile=0 -vppmodti=10 -vB1TTSDpp=0 -vB2TTSDpp=0  
-vA1TTSDpp=150.0  -vC1TTSDpp=1.4667937029556515  -vA2TTSDpp=0.0  -vC2TTSDpp=0.0  
-vddsigTTSD=6.0  -vddpiTTSD=4.0  -vdddelTTSD=1.0  -vspanjdec=0.4967015351084541  -vspanjddd=0.21484375  
-vef=-0.04858 -ef=-0.04858  -valatTi=5.582222222222222  -vcoa=1.632993161855452 

 Arguments 
  -vfp=0 -vrfile=0 -vppmodti=10 -vB1TTSDpp=0 -vB2TTSDpp=0 
  -vA1TTSDpp=150.44444444444446  -vC1TTSDpp=1.4667937029556515  -vA2TTSDpp=0.0  -vC2TTSDpp=0.0 
  -vddsigTTSD=6.0  -vddpiTTSD=4.0  -vdddelTTSD=1.0  -vspanjdec=0.4967015351084541  -vspanjddd=0.21484375 
  -vef=-0.04858 -ef=-0.04858  -valatTi=5.582222222222222  -vcoa=1.632993161855452 

   C11 = 0.378,   C11_FR = 1.110
 C33 = 1.513,   C33_FR = 1.196
 C44 = -0.272,   C44_FR = 0.321
 C44 = -0.137,   C44_FR = 0.321
 C66 = -0.539,   C66_FR = 0.287
 C12 = 1.602,   C12_FR = 0.537
 C13 = 0.695,   C13_FR = 0.424
 K = 0.917,   K_FR = 0.687
 R = 0.371,   R_FR = 0.391
 H = -0.373,   H_FR = 0.294 
 


"""

    #pair_pot    = np.array( [ 21.215, 1.2, 1.075, 0.6 ] )
    # with cnaonical band ratios
    ## Bulk Mod (GPa) = 65.43670576206576
                                                    

################################################################################################################################
######################     Trialled Parameters that work reasonably, with bad elastic constants.     ###########################

##  This gets c and a with a very good agreement. The elastic constants are 
#pair_pot = np.array([ 24.85, 1.247525, 0.886571, 0.565, 1.  ] )
#ddcoeffs = np.array([6., 4., 1., spanjdec])
#dddeltanorm =  0.19542694091796875


##  One with the lowest error so far
#ddcoeffs = np.array( [ 10.20968613,  6.35171468,  1.70161435,  0.44714869])
#par_arr =  np.array( [ 26.28518485,  1.29113683,  1.04802635,  0.64556841,  1.        ])

#################################################################################################################################

