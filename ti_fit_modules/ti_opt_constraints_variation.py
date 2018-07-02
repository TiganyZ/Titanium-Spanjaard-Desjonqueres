# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 



#######################################################################################################
##########################     Error Penalties and Constraints    #####################################

def apply_error_penalties(latpar_diff, min_latpar, latpar_u, latpar_l, tol_p, latpar):
    print('Error Penalties: ')



    if abs(latpar_diff) < tol_p:  
        latpar_diff = latpar_diff/8.
        print(' Lattice parameter %s is close to ideal value, tol_p = %s \n' %(latpar, tol_p))
    elif abs(min_latpar - latpar_u) < 1. * tol_p or abs(min_latpar - latpar_l) < 1. * tol_p:
        latpar_diff = 2 * latpar_diff
        print(' Lattice parameter %s is far from ideal value, tol_p = %s \n' %(latpar, tol_p))
    else:
        latpar_diff = latpar_diff/2.
        print(' Lattice parameter %s is within binary search bounds, tol_p = %s \n' %(latpar, tol_p))

    return latpar_diff


def apply_constraints(params, name_of_params):

    if name_of_params == 'bondint':
        ## Constrain the bond integrals. 
        ## params == bond_coeffs
        if params[0] > 10:
            print('Changing ddsigma')
            params[0] = 6.
        if params[1] > 8:
            print('Changing ddpi')
            params[1] = 4.
        if params[2] > 6:
            print('Changing dddelta')
            params[2] = 1.



        if params[1] < params[2]:
            params[1] = 4. * params[2]
   
        if params[0] < params[1]:
            params[0] = 6. * params[2]


    elif name_of_params == 'pairpot':
        ## Constrain the pair potential parameters. 
        ## params == pair_params
        if params[0] * params[-1] > 200 * 0.23:
            print('Changing A')
            params[0] = 100 * 0.21215
        if params[2] * params[-1] > 10 * 0.23:
            print('Changing B')
            params[2] = 5 * 0.21215
        if params[3] <  params[1]:
            params[3] = 0.5 * params[1]

    return np.abs(params)



###############################################################################################################
##########################     Variation of Parameters and Constraints    #####################################

def pp_scale_const(LMarg, par_arr_p, ppwg, ppert, xargs, npass, maxit):
    """
    Scales the pair potential by a constant, it being the last element of the array. 
    A weighted mean is used such that there is not much deviation from the c/a ratio, 
    as fitting to a fixed c/a is better
    """
    if npass == 1:
        ppwg = np.array( [ 24.85, 1.247525, 0.886571, 0.565, 1.  ] ) #np.array( [21.215,       1.2 ,        1.06546811 , 0.629057 ,   1.   ] )
        varwg = np.array([2., 0.1, 0.1, 0.1, 0.1])

    elif npass < 3:
        varwg = np.array([2., 0.1, 0.1, 0.1, 0.1])

    else:
        ##  Depending on how much the error is between the last two iterations, the variance will increase or decrease accordingly
        varwg = np.array([2., 0.1, 0.1, 0.1, 0.1]) * ( maxit - npass % maxit  ) /  maxit * np.abs(  (ppert[-1] / ppert[-2])  )



    par_arr_p, alat_diff, clat_diff, coa_diff, min_coa, min_alat  = pp_gaussian_variation(LMarg, xargs, par_arr_p, 
                                                                                            ppwg, varwg, npass )


    #[21.34704905,  1.2102055,   1.13163312,  0.63892678,  1.05225734  ] 
    #np.array([21.0825141 ,  1.17983585,  0.99597137,  0.6188312 ,  1. ]), np.array([0.1, 0.05, 0.05, 0.05, 0.1]) )
        #np.array([21.215, 1.2, 1.068, 0.6, 1.]), np.array([0.1, 0.05, 0.05, 0.05, 0.1]) )
    # [20.950028193676797 , 1.1596717048798084 , 0.9239427333755603 , 0.6376623958184923, 1] #6,4,1  # This has a c/a of 1.60, with a small alat error 
    par_arr_p[-1] = 1.

    return np.abs(par_arr_p), alat_diff, clat_diff, coa_diff, min_coa, min_alat 







def pp_gaussian_variation(LMarg, args, par_arr_p, m_v, sig_v, npass):#, pair_pot_bank, pair_pot_bank_err):

    print('\n     Pair potential Gaussian Variation\n')
    ## Constrain the pair potential parameters. 
    ## est the parameters continuously until pair potential has been found
    alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
    clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
    coa_ideal = clat_ideal/alat_ideal

    tol_a = 0.001
    tol_coa = 0.001
    a_u = 6.2; a_l = 5.4; coa_u = (8./3.)**0.5; coa_l = 1.5
    #min_alat, min_coa, diff_ratioa, diff_ratioc, alat_diff, coa_diff = obtain_min_alat_coa(LMarg, args, par_arr_p,
    #                                                                    a_u, a_l, tol_a, coa_u, coa_l, tol_coa )

    ppargs = get_pp_args(par_arr_p)

    min_alat, alat_diff, min_coa, coa_diff = get_min_coa_and_alat(LMarg, args + ppargs, par_arr_p, 
                                a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal)
    #min_alat, alat_diff = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, a_u, a_l, tol_a, 'alatTi', alat_ideal)
    #min_coa, coa_diff = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)
    
    clat_diff = min_coa * min_alat - coa_ideal * alat_ideal

    coa_diff = apply_error_penalties(coa_diff, min_coa, coa_u, coa_l, 0.01, 'coa')
    alat_diff = apply_error_penalties(alat_diff, min_alat, a_u, a_l, 0.05, 'alatTi')

    if npass > 1:

        pp_err = np.sqrt(alat_diff**2 + coa_diff**2) #This is alpha


        pair_pot_bank = np.array([par_arr_p])#.reshape( (1, par_arr_p.shape[0]))
        pp_err1_tot = np.array([pp_err])

        j = 0
        rej = 0

        #"""
        for k in range(1):#len(m_v)-1):
            if k > 0:
                pp_err2 = pp_err1
            else:
                pp_err2 = pp_err

            print('Pair potential minimiser, initialisation: k = %s' %(k) )
            print('par-arr before = %s' %(par_arr_p ))
            i = random.choice(range(len(m_v) -1))
            par_arr_before = par_arr_p[i] 
            par_arr_p[i] = np.abs( np.random.normal(m_v[i], sig_v[i]) )
            print('par-arr after = %s \n' %(par_arr_p ))


            min_alat1, alat_diff1, min_coa1, coa_diff1 = get_min_coa_and_alat(LMarg, args, par_arr_p, 
                                a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal)
            #min_alat1, alat_diff1 = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, a_u, a_l, tol_a, 'alatTi', alat_ideal)

            #min_coa1, coa_diff1 = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)

            #min_alat1, min_coa1, diff_ratioa1, diff_ratioc1, alat_diff1, coa_diff1  = obtain_min_alat_coa(LMarg, args, par_arr_p,
                                                                             #a_u, a_l, tol_a, coa_u , coa_l, tol_coa )

            clat_diff1 = min_coa1 * min_alat1 - coa_ideal * alat_ideal
            pp_err1 = np.sqrt(alat_diff1**2 + coa_diff1**2)

            coa_diff1 = apply_error_penalties(coa_diff1, min_coa1, coa_u, coa_l, 0.005, 'coa')
            alat_diff1 = apply_error_penalties(alat_diff1, min_alat1, a_u, a_l, 0.05, 'alatTi')

            if pp_err1 > pp_err2:
                ##  Reject the change to the pair potential
                rej += 1
                print('\n Reject change to the pair potential, n_rejects = %s ' %(rej))
                print(' pp_err = %s, pp_err_1 = %s, pp_err2 = %s \n' %(pp_err, pp_err1, pp_err2))
                par_arr_p[i] = par_arr_before 
                #pp_err1 = 2 * pp_err1
            print('Minimum alat = %s, Minimum coa = %s' %(min_alat1, min_coa1))
            pair_pot_bank = np.append(pair_pot_bank, par_arr_p).reshape( 
                                        ( pair_pot_bank.shape[0] + 1, pair_pot_bank.shape[1] ) )
            pp_err1_tot = np.append(pp_err1_tot, pp_err1)
        #"""

        S0 = np.diag(sig_v**2)
        cond = True #pp_err1 > pp_err
        #print('Pre-while loop:  pp_err = %s, pp_err_1 = %s \n' %(pp_err, pp_err1))
        itmax = 7
        while cond:
            ## This pair potential is worse than initial calculation
            if j > 0:
                pp_err2 = pp_err1
            else:
                pp_err2 = pp_err

            j +=1

            print('\n Within pair potential gaussian variation, iteration = %s \n' %( j) ) 
            print('Pair potential before = %s' %(par_arr_p ))
            par_arr_before = par_arr_p
            pp_wgtd_mean_err = np.sqrt( 1. / np.sum((1./pp_err1_tot**2) ))
            pp_wgtd_mean = np.sum( pair_pot_bank * (1./pp_err1_tot**2)[:,np.newaxis], axis = 0 ) * pp_wgtd_mean_err**2 

            pp_cov = sample_mean_covarance(pair_pot_bank, (1./pp_err1_tot**2) , pp_wgtd_mean )
            
            

            par_arr_p = ( np.random.multivariate_normal(mean=pp_wgtd_mean, cov=(pp_cov * j  + S0 * (itmax - j ) )/itmax) + par_arr_p ) / 2.
            print('Pair potential after = %s' %(par_arr_p ))

            print('\n Weighted means: \n pp_wgtd_mean = %s \n pp_wgtd_mean_err = %s \n ' %( pp_wgtd_mean, pp_wgtd_mean_err))
            tol_a = 0.01
            tol_coa = 0.001

            min_alat1, alat_diff1, min_coa1, coa_diff1 = get_min_coa_and_alat(LMarg, args, par_arr_p, 
                                a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal)

            #min_alat1, alat_diff1 = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, a_u, a_l, tol_a, 'alatTi', alat_ideal)

            #min_coa1, coa_diff1 = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)

            ##  Giving more weight if a parameter is close to the true value
            ##  Could generalise this and use an actual function.

            coa_diff2 = apply_error_penalties(coa_diff1, min_coa1, coa_u, coa_l, 0.005, 'coa')
            alat_diff2 = apply_error_penalties(alat_diff1, min_alat1, a_u, a_l, 0.05, 'alatTi')


            
            pp_err1 = np.sqrt(alat_diff2**2 + coa_diff2**2)
            print('Minimum alat = %s, Minimum coa = %s' %(min_alat1, min_coa1))

            pair_pot_bank = np.append(pair_pot_bank, par_arr_p).reshape( 
                                            ( pair_pot_bank.shape[0] + 1, pair_pot_bank.shape[1] ) )
            if pp_err1 > pp_err2:
                ##  Reject the change to the pair potential
                rej += 1
                print('\n Reject change to the pair potential, n_rejects = %s ' %(rej))
                print(' pp_err = %s, pp_err_1 = %s, pp_err2 = %s \n' %(pp_err, pp_err1, pp_err2))
                par_arr_p = par_arr_before 
            pp_err1_tot = np.append(pp_err1_tot, pp_err1)


            if j % 4 == 0:
                ##  Every second iteration after the second pass, discard the worst result. 
                ind_w = np.argmax(pp_err1_tot)
                pplen = len(par_arr_p)
                print('\n pp_finder:  Removing\n    pp = %s,\n    with pp error = %s \n ' %(pair_pot_bank[ind_w], pp_err1_tot[ind_w]))
                
                pair_pot_bank = np.delete(pair_pot_bank.flatten(), range( ind_w*pplen, (ind_w + 1)*pplen))
                #print('removed pair_pot_bank = %s' %(pair_pot_bank))
                pair_pot_bank = pair_pot_bank.reshape( (len(pair_pot_bank)//pplen, pplen  ))
                pp_err1_tot = np.delete(pp_err1_tot, ind_w)

            if coa_diff2 < coa_diff1:
                ##  coa ratio is within the bounds, if j is low, then try to get closer to ideal
                if j > int(itmax/2.):
                    cond = False
                else:
                    cond = True
            if alat_diff2 < alat_diff1:
                ##  alat is within the bounds, if j is low, then try to get closer to ideal
                if j > int(itmax/2.):
                    cond = False
                else:
                    cond = True
            if j > itmax:
                cond = False
            else:
                cond = True

            print('Pair potential bank = %s \n Pair potential Errors = %s' %(pair_pot_bank, pp_err1_tot))

        par_arr_p = pair_pot_bank[ np.argmin(pp_err1_tot) ]
        pp_err = np.min(pp_err1_tot)
        print('Best pair potential is: pp = %s,\n   with error = %s' %(par_arr_p, pp_err) )
        pp_wgtd_mean_err = np.sqrt( 1. / np.sum((1./pp_err1_tot**2) ))
        pp_wgtd_mean = np.sum( pair_pot_bank * (1./pp_err1_tot**2)[:,np.newaxis], axis = 0 ) * pp_wgtd_mean_err**2 
        par_arr_p = pp_wgtd_mean
        print(' Choose weighted average as pair potential: pp = %s, err_avg = %s \n  ' %(par_arr_p, pp_wgtd_mean_err))
    else:
        alat_diff1 = alat_diff
        coa_diff1 = coa_diff
        min_coa1 = min_coa
        min_alat1 = min_alat
        clat_diff1 = clat_diff



    return par_arr_p, alat_diff1, clat_diff1, coa_diff1, min_coa1, min_alat1








def check_latpar_differences(diff, pair_params, bond_coeffs, pp_chnge, dd_chnge, noise):
    tolerance = 0.0001
    print('\n check_latpar_differences \n')
    print('diff',diff)

    ##  Assuming distribution among the ddcoefficients such that they have Gaussian noise

    if abs(diff) > tolerance:
        ##  The pair potential and the lattice constants must be changed
        r_ind = random.choice(range(len(bond_coeffs)))
        if np.random.uniform() > 0.5:
            bond_coeffs[r_ind] =  bond_coeffs[r_ind]  -  ( np.abs(diff)  * (dd_chnge[r_ind] + noise[r_ind] )/2. )
        else:
            bond_coeffs[r_ind] =  bond_coeffs[r_ind]  +  ( np.abs(diff)  * (dd_chnge[r_ind] + noise[r_ind] )/2. )
    else:
        ## Pair potential and bond integrals seem okay for this difference in parameters
        None

    bond_coeffs = apply_constraints(bond_coeffs, 'bondint')
    pair_params = apply_constraints(pair_params, 'pairpot')

    return pair_params, np.abs(bond_coeffs)


 





