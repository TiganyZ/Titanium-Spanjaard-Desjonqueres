# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 

############################################################################################
##########################     Minimum lattice parameters     ##############################




def lmu_latpar_energies(LMarg, args, latpar, l, u, m):
    xargs = args + ' '
    xx_args = xargs + construct_cmd_arg(latpar, m); etot_m = find_energy(LMarg, xx_args, 'pptest')
    xx_args = xargs + construct_cmd_arg(latpar, l); etot_l = find_energy(LMarg, xx_args, 'pptest')
    xx_args = xargs + construct_cmd_arg(latpar, u); etot_u = find_energy(LMarg, xx_args, 'pptest')
    return etot_l, etot_m, etot_u

def binary_search_latpar(LMarg, args, latpar, l, u, m):
    ##  1-D minimisation of lattice parameter, with the other one fixed. 

    etot_l, etot_m, etot_u = lmu_latpar_energies(LMarg, args, latpar, l, u, m)
    if (etot_m > etot_l) and (etot_m < etot_u) :
        ##  Check the range between m and l
        u = m 
        m = (u + l)/2.
        etot_l, etot_m, etot_u = lmu_latpar_energies(LMarg, args, latpar, l, u, m)
    elif  (etot_m < etot_l) and (etot_m > etot_u):
        ##  Check the range between m and u
        l = m 
        m = (u + l)/2.
        etot_l, etot_m, etot_u = lmu_latpar_energies(LMarg, args, latpar, l, u, m)
    elif etot_m < etot_l and etot_m < etot_u: 
        ##  Shrink the bounds
        l2 = (m + l)/2. ; u2 = m ; m2 = (l2 + u2)/2.
        etot_l2, etot_m2, etot_u2 = lmu_latpar_energies(LMarg, args, latpar, l2, u2, m2)
        l3 = m ; u3 = (m + u)/2. ; m3 = (l3 + u3)/2.
        etot_l3, etot_m3, etot_u3 = lmu_latpar_energies(LMarg, args, latpar, l3, u3, m3)
        
        ##  Boundary array
        bound_arr = np.array([l,m,u,l2,m2,u2,l3,m3,u3])
        ##  ...with corresponding energies
        etot_arr = np.array([etot_l,etot_m,etot_u,etot_l2,etot_m2,etot_u2,
                            etot_l3,etot_m3,etot_u3])
        ##  Indices with smallest energies
        l2enidx = np.argsort(etot_arr)[:2]
        etot_l = etot_arr[l2enidx[0]] ; etot_u = etot_arr[l2enidx[1]]
        l = bound_arr[l2enidx[0]] ; u = bound_arr[l2enidx[1]]
        m = (l+u)/2.

    return etot_l, etot_m, etot_u, l, m, u

def minimum_energy_lattice_parameter( LMarg, args, lattice, latpar, a_u, a_l, tol):
    """  Finds the lattice parameter (latpar) of a lattice which has the lowest energy between the 
        upper and lower bounds, a_u and a_l respectively, within a given tolerance, tol.  """
    
    a_m = (a_u + a_l)/2.
    minimum_found = False
    min_count =  15
    it = 0 
    while minimum_found == False:
        it += 1
        etot_l, etot_m, etot_u, a_l, a_m, a_u = binary_search_latpar(LMarg, args, latpar, a_l, a_u, a_m)
        min_count = min_count -  1
        if min_count < 0:
            ##  Too long to find a minimum. Break. 
            print("No minimum " + latpar + "---Breaking...")
            break
        elif abs(a_u-a_l) < tol:
            print("Found minimum " + latpar + " = %s, iter = %s, tolerance = %s" %(a_m, it, tol))
            minimum_found = True
            break
    print( '%s:   Total Energy = %s. %s_min = %s, tol = %s'%(lattice, etot_m, latpar, a_m, tol))

    return etot_m, a_m


def get_min_coa_and_alat(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal):
    ##  An abhorrent and inelegant method to get minimum coa and alat...

    print( 'Get minimum coa and alat')
    min_tol = 0.01

    ##  Start with coarse-grained binary search and then constrain with smaller tolerances.
    tol_a = 0.01; tol_coa = 0.01
    min_alat, alat_diff = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, a_u, a_l, tol_a, 'alatTi', alat_ideal)
    min_coa, coa_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -valatTi=' + str(min_alat) + ' ', 
                                                            par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)
    #alat_min, coa_min, min_found = check_latpar_out_of_bounds(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, 
    #                                   coa_u, coa_l, coa_ideal, tol_a, tol_coa, np.array([min_alat, min_coa]))
    min_found = False                                        
    counter = 1
    while min_found == False:
        ##  Another iteration with a smaller tolerance
        ##  Hopefully this will run to self consistency

        print('Within alat and coa minimiser, iteration = %s' %(counter))
        print('alat = %s, coa = %s ' %(min_alat, min_coa) )
        counter += 1
        if counter < 3:
            tol_a = 0.001; tol_coa = 0.001
        else:
            tol_a = 0.001; tol_coa = 0.001

        if counter > 4:
            min_found = True
            break

        min_alat1, alat_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -vcoa=' + str(min_coa) + ' ', 
                                                    par_arr_p, a_u, a_l,  tol_a, 'alatTi', alat_ideal)

        min_alat1, min_coa1, min_found = check_latpar_out_of_bounds(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, 
                        coa_u, coa_l, coa_ideal, tol_a, tol_coa, np.array([min_alat1, min_coa]))

        if min_found == True:
            break


        min_coa1, coa_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -valatTi=' + str(min_alat) + ' ', 
                                                    par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)

        min_alat1, min_coa1, min_found = check_latpar_out_of_bounds(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, 
                        coa_u, coa_l, coa_ideal, tol_a, tol_coa, np.array([min_alat1, min_coa1]))

        if min_found == True:
            break


        if abs(min_alat1 - min_alat) < min_tol and  abs(min_coa1 - min_coa) < min_tol:
            min_found = True

        min_alat = min_alat1
        min_coa = min_coa1

    if counter == 1: 
        min_alat1 = min_alat
        min_coa1 = min_coa1
        alat_diff1 = alat_diff
        coa_diff1 = coa_diff

    return min_alat1, alat_diff, min_coa1, coa_diff

def get_minimum_lat_par_diff(LMarg, args, var_name, var_ideal, ub, lb, tol):
    ###############     Get minimum  lattice parameter    ##########
    print("\nObtaining minimum " + var_name + " lattice parameter" )


    e_alat_eq, min_alat = minimum_energy_lattice_parameter( LMarg, args, 'hcp', var_name, ub, lb, tol)
    if abs(min_alat - ub) < 1.5 * tol:
        min_alat = ub
    elif abs(min_alat - lb) < 1.5 * tol:
        min_alat = lb


    #filename='pptest'
    #cmd = LMarg + ' ' + args 
    #cmd_write_to_file( cmd, filename)
    alat_diff = min_alat - var_ideal
    print(var_name + ' diff = %s' %(alat_diff))
    return min_alat, alat_diff



def check_latpar_out_of_bounds(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, tol_a, tol_coa, x_np1):

    if abs(x_np1[0] - a_l) < tol_a or x_np1[0] < a_l: 
        ##  alat is at the lower boundary
        print('Lattice parameter alatTi is below lower bound.\n Fixing parameter at boundary while doing 1-D minimisation of other lattice parameter')
        alat_min = a_l
        coa_min, coa_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -valatTi=' + str(alat_min) + ' ',
                                             par_arr_p, coa_u , coa_l, 0.001, 'coa', coa_ideal)
        min_found = True

    elif abs(x_np1[0] - a_u) < tol_a or x_np1[0] > a_u: 
        ##  alat is at the upper boundary
        print('Lattice parameter alatTi is above upper bound.\n Fixing parameter at boundary while doing 1-D minimisation of other lattice parameter')
        alat_min = a_u
        coa_min, coa_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -valatTi=' + str(alat_min) + ' ',
                                             par_arr_p, coa_u , coa_l, 0.001, 'coa', coa_ideal)
        min_found = True



    elif abs(x_np1[1] - coa_l) < tol_coa or x_np1[1] < coa_l:
        print('Lattice parameter coa is below lower bound.\n Fixing parameter at boundary while doing 1-D minimisation of other lattice parameter')
        coa_min = coa_l
        alat_min, alat_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -vcoa=' + str(coa_min) + ' ',
                                             par_arr_p, a_u , a_l, 0.001, 'alatTi',  alat_ideal)
        min_found = True
    elif abs(x_np1[1] - coa_u) < tol_coa or x_np1[1] > coa_u:
        print('Lattice parameter coa is above upper bound.\n Fixing parameter at boundary while doing 1-D minimisation of other lattice parameter')
        coa_min = coa_u
        alat_min, alat_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -vcoa=' + str(coa_min) + ' ',
                                             par_arr_p, a_u , a_l, 0.001, 'alatTi',  alat_ideal)
        min_found = True
    else:
        min_found = False
        coa_min = x_np1[1] 
        alat_min = x_np1[0] 

    return alat_min, coa_min, min_found



def obtain_min_lattice_parameter_pp(LMarg, xargs, par_arr_p, a_u, a_l, tol_a, latpar, latpar_ideal):

    A1TTSDpp = par_arr_p[0] * par_arr_p[-1]
    B1TTSDpp = 0 
    C1TTSDpp = par_arr_p[1]
    A2TTSDpp = par_arr_p[2] * par_arr_p[-1]
    B2TTSDpp = 0
    C2TTSDpp = par_arr_p[3]

    ppargs = ' -vA1TTSDpp=' + str(A1TTSDpp) + \
                    ' -vB1TTSDpp=' + str(B1TTSDpp) + \
                    ' -vC1TTSDpp=' + str(C1TTSDpp) + \
                    ' -vA2TTSDpp=' + str(A2TTSDpp) + \
                    ' -vB2TTSDpp=' + str(B2TTSDpp) + \
                    ' -vC2TTSDpp=' + str(C2TTSDpp) + ' ' 

    args = xargs + ppargs

    min_latpar, latpar_diff = get_minimum_lat_par_diff(LMarg, args, latpar, latpar_ideal, a_u, a_l, tol_a)

    return min_latpar, latpar_diff