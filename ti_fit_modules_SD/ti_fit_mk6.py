# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 




#######################################################################################
###########################     General routines      #################################

def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result,err = proc.communicate() 
    result = result.decode("utf-8")
    return result

def cmd_write_to_file( cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.run(cmd, shell=True, stdout = output_file)
    output_file.close()


def plot_function(n_plots, x, y, colour, title, x_name, y_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        if n_plots > 1:
            for i in range(n_plots):
                ax.plot(x[i], y[i], colour[i])
        else:
            ax.plot(x, y, colour)
        plt.show() 


def remove_bad_syntax(values):
    new_values=[]
    for i in values:
        if '-' in i[1:]:
            temp = i[1:]
            temp = temp.replace("-", " -")
            i =  (i[0]+ temp).split()
            for j in i:
                new_values.append(float(j))
        else:
            new_values.append(float(i))
    return new_values


def construct_cmd_arg(arg_name, value):
    """arg_name is a string corresponding to the variable name with a value."""
    return ' -v' + arg_name + '=' + str(value) + ' ' 

def construct_extra_args( xargs, arg_names, arg_values):
    """Method to construct the extra arguments where arg_names is  a list of strings."""
    for i in range(len(arg_names)):
        arg = construct_cmd_arg(arg_names[i], arg_values[i])
        xargs +=  arg
    return xargs



#######################################################################################
###########################     Energy Routine      ###################################


def find_energy( LMarg, args, filename):
    cmd =  LMarg + ' ' + args 
    cmd_write_to_file(cmd, filename)
    if 'lmf' in LMarg:
        cmd = "grep 'ehk' " + filename + " | tail -2 | grep 'From last iter' | awk '{print $5}'"
    elif 'tbe' in LMarg:
        cmd = "grep 'total energy' " + filename + " | tail -1 | awk '{print $4}'"
    etot = cmd_result(cmd)
    try:
        etot = float(etot[0:-1])
    except ValueError:
        print( 'Error: Cannot obtain energy from file ' + filename + '. Check ' + filename + ' for error. \n Exiting...' )
    
    return etot


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

##############################################################################################################
########################################   These are obsolete routines  ######################################
##############################################################################################################




def brute_force_min_lattice_parameter(LMarg, args):
    latpar = 'coa'
    coa_u, coa_l = 1.6  , 1.57
    coa_r = np.linspace(coa_l, coa_u, 50)
    coa_m = (coa_u + coa_l)/2.
    etot_l, etot_m, etot_u = lmu_latpar_energies(args, 'coa', coa_l, coa_u, coa_m)
    etot_list = []
    for val in coa_r:
        xx_args = args + ' ' + construct_cmd_arg('coa', val)
        etot_list.append( find_energy(LMarg, xx_args, 'pptest') )
    etot_list = np.asarray(etot_list)
    min_ind = np.argmin(np.asarray(etot_list))
    max_ind = np.argmax(etot_list)
    print( 'Minimum Energy  = %s, Minimum coa = %s' %(etot_list[min_ind],  coa_r[min_ind]))
    print( 'Maximum Energy  = %s, Maxmium coa = %s' %(etot_list[max_ind],  coa_r[max_ind]))

def spanjaard_des_model(x, pR0, qR0):
    r0 = 5.574662685664614  # This is in Bohr
    return np.exp(-(pR0/r0)*(x-r0) )/ ( (pR0/qR0) - 1)  + np.exp(-(qR0/r0)*(x-r0) )/ ( (qR0/pR0) - 1) 

def spanjaard_des_model2(x, pR0, qR0, Z, F, A):
    r0 = 5.574662685664614  # This is in Bohr
    return np.exp(-(pR0/r0)*x ) * ( Z * A / 2.) - np.exp(-(qR0/r0)*x ) * Z**(0.5) * F

def bond_int(x, coeffs, qR0):
    r0 = 5.574662685664614  # This is in Bohr
    delt = 0.208098266125
    c = np.mean(coeffs)
    return c* delt *  np.exp(-x*qR0/r0)

def change_bond_int(coeffs):
    coeff = np.random.choice(coeffs)
    return

## 3 < p/q < 5 for spanjaard desjonqueres model

def obtain_min_alat_coa(LMarg, xargs, par_arr_p, a_u, a_l, tol_a, coa_u, coa_l, tol_coa):
    alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
    clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
    coa_ideal = clat_ideal/alat_ideal

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
    
    min_alat, alat_diff, min_coa, coa_diff = get_min_coa_and_alat(LMarg, args, par_arr_p, 
                                a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal)

    #min_alat, alat_diff = get_minimum_lat_par_diff(LMarg, args, 'alatTi', alat_ideal,  a_u, a_l,tol)
    #min_coa, coa_diff = get_minimum_lat_par_diff(LMarg, args, 'coa', coa_ideal, coa_u, coa_l, tol)
    #clat_diff = min_coa * min_alat - coa_ideal * alat_ideal


    differencea = alat_diff
    differencec = coa_diff
    diff_ratioa = np.abs(alat_diff**2/alat_ideal) #abs(coa_diff)
    diff_ratioc = np.abs(coa_diff**2/coa_ideal)
    difference = differencea + differencec
    return min_alat, min_coa, diff_ratioa, diff_ratioc, differencea, differencec








def brute_force_min_coa_and_alat(LMarg, args, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, n_a, n_coa):
    ##  Such a disgusting method for obtaining minimum coa and alat, just to check that the minimisers work. 
    e_list = []
    al = np.linspace(a_l, a_u, n_a)
    coal = np.linspace(coa_l, coa_u, n_coa)
    e_arr = np.zeros(n_a*n_coa)
    for i in range(n_a):
        for j in range(n_coa):
            e = find_energy(LMarg, 
                                    args + ' -valatTi=' + str(al[i]) + ' ' + ' -vcoa=' +  str(coal[j]) + ' ',
                                                                                                    filename)
            e_arr[i * j + j ] = e
            if i + j % 10 == 0:
                print('brute force latpar min:\n    alat = %s, coa = %s, i = %s, j = %s' %(al[i], coal[j], i, j))
    min_ind = np.argmin(e_arr)
    a_ind = min_ind // n_a
    coa_ind = min_ind % n_coa

    return al[a_ind], coal[coa_ind]


def get_line_step_coa_alat(LMarg, args, xn, xn1, sigma): 
    ##  Find energy of what minimum is meant to be, at the ideal values, and energy at a random point. 
    e00 = find_energy(LMarg, args + ' -valatTi=' + str(xn[0]) + ' ' + ' -vcoa=' +  str(xn[1]) + ' ', 'emin')
    e11 = find_energy(LMarg, args + ' -valatTi=' + str(xn1[0]) + ' ' + ' -vcoa=' +  str(xn1[1]) + ' ', 'emin')
    e11_s = find_energy(LMarg, args + ' -valatTi=' + str(xn[0] + sigma * xn1[0]) + ' ' +\
                                         ' -vcoa=' +  str(xn[1] + sigma * xn1[0] ) + ' ', 'emin')


    ##  partial derivatives in each direction, use small step of sigma in direction d_ to find Hessian
    gn = np.array(  [  ( e11 - e00 ) / ( xn1[0] - xn[0] ) , ( e11 - e00 ) / ( xn1[1] - xn[1]) ] )
    gn_s = np.array(  [( e11_s - e00 ) / ( xn1[0] - xn[0] ) , ( e11_s - e00 ) / ( xn1[1] - xn[1])] )


    ##  Using Gradient descent with dn = -gn...
    dn = -gn
    ##  Find step size 
    step = - ( sigma * gn.dot( dn ) ) / ( gn_s.dot( dn ) - gn.dot( dn ) )
    print('step = %s'%(step))

    x_np1 = xn - step * gn

    return xn1, x_np1, step, gn, gn_s, dn, e11

def line_search_min_coa_alat(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, tol_a, tol_coa):

    print('\n    Line Search for minimum coa and alat:')
    min_found = False
    al = np.linspace(a_l, a_u, int(1./tol_a) + 1)
    coal = np.linspace(coa_l, coa_u, int(1./tol_coa) + 1)
    ##  Picking random points to start from 
    a_rind = np.random.choice(range(int(1./tol_a) + 1))
    coa_rind = np.random.choice(range(int(1./tol_coa) + 1))
    xn1 = np.array( [ al[a_rind], coal[coa_rind] ] )
    a_rind = np.random.choice(range(int(1./tol_a) + 1))
    coa_rind = np.random.choice(range(int(1./tol_coa) + 1))
    xn = np.array( [ al[a_rind], coal[coa_rind] ] )
    sigma = 0.05
    #xn = np.array( [al[0], coal[0]]) #np.array( [ alat_ideal, coa_ideal ] )
    
    print('x_n = %s, x_n1 = %s' %(xn, xn1))

    ##  Doing the line search for next parameter
    xn1, x_np1, step, gn, gn_s, dn, e11 = get_line_step_coa_alat(LMarg, args, xn, xn1, np.zeros(2),  sigma)
    print('x_np1 = %s' %(x_np1))

    ##  Checking if the parameter found is out of the bounds
    alat_min, coa_min, min_found = check_latpar_out_of_bounds(LMarg, args, par_arr_p, a_u, a_l, alat_ideal,
                                                             coa_u, coa_l, coa_ideal, tol_a, tol_coa, x_np1)

    #e_x_np1 = find_energy(LMarg, args + ' -valatTi=' + str(x_np1[0]) + ' ' + ' -vcoa=' +  str(x_np1[1]) + ' ', 'emin')

    while min_found == False:

        print('x_n = %s, x_np1 = %s' %(xn, x_np1))
        xn, x_np1, step, gn, gn_s, d_n, e11 = get_line_step_coa_alat(LMarg, args, xn, x_np1, -gn, sigma)
        print('x_np1 = %s' %(x_np1))

        alat_min, coa_min, min_found = check_latpar_out_of_bounds(LMarg, args, par_arr_p, a_u, a_l, alat_ideal,
                                                             coa_u, coa_l, coa_ideal, tol_a, tol_coa, x_np1)
        if np.linalg.norm(xn1 - x_np1) < 0.001:
            min_found = True

    return alat_min, coa_min

def conjugate_gradient(A, x_k, b, tol):
    ##  This solves the equation Ax = b
    r_k = A.dot(x_k) - b
    p_k = -r_k
    while r_k > tol:
        a_k = ( r_k.dot(p_k) )/( p_k.dot( A.dot( p_k ) ) )
        x_k += a_k * p_k
        r_kp1 = r_k + a_k * A.dot( p_k )
        beta_k = r_kp1.dot(r_kp1)/r_k.dot(r_k)
        p_k = -r_kp1 + beta_k*p_k 
        r_k = r_kp1
    return x_k


def fit_pair_potential_binary_search(LMarg, args, par_arr, min_coa, coa_diff, coa_u, coa_l, min_alat, alat_diff, a_u, a_l):
    ##  Routine to find the best pair potential for the nomalised bond integrals given. 
    ##  Know that increasing A and b, increases the height of the pair potential, while increasing a and B decreases the height of the pair potential
    ##  If a particular lattice parameter is near a bound of the binary search initial starting bounds, 
    ##  then we can increase (if the lattice parameter is close to the lower bound), or decrease (if the lattice parameter is close to the upper bound.)

    alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
    clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
    coa_ideal = clat_ideal/alat_ideal

    ##  Determine if the pair potential needs to be increased or decreased.  ##
    ###########################################################################
    
    diffs = np.array([coa_diff, alat_diff])
    diffs0 =  np.array([coa_diff, alat_diff])
    ideal = np.array([coa_ideal, alat_ideal])

    tol_p = [0.005, 0.05]
    flags = np.array([0,0,0,0])
    
    while np.all(diffs > diffs0):
        counter = 0
        for i in range(2):

            if i == 0:  
                latpar = 'coa'
            else:
                latpar = 'alatTi'

            if abs(diffs[i]) < tol_p[i]:  
                print('Lattice parameter is close to ideal value, dont need to change pp for this parameter i = %s, tol_p = %s' %(latpar, tol_p[i]))
            elif abs(min_latpar - latpar_u) < 2 * tol_p or abs(min_latpar - latpar_l) < 2 * tol_p:
                #latpar_diff = 2 * latpar_diff
                counter += 1
                print('Lattice parameter is far from ideal value, need to change pp for this parameter i = %s, tol_p = %s' %(latpar, tol_p[i]))


        changes = np.zeros(len(flags))
        for l in range(counter):

            indx = random.choice(range( len(par_arr) -1 )) #  Choose a random index from pair potential parameter array

            for i in range(2):
                if diffs[i] < 0:
                    ## Then the pair potential must increase 
                    if indx == 0:
                        ##  Increase the A parameter
                        par_arr[indx] = par_arr[indx] * ( 1. + (diffs[i]/ideal[i]) )
                        flags[0] = 1
                    elif indx == 3:
                        ## Increase the b parameter
                        par_arr[indx] = par_arr[indx] * ( 1. +  ( diffs[i]/ideal[i] ) )
                        flags[3] = 1
                    elif indx == 2:
                        ##  Decrease the B parameter
                        par_arr[indx] = par_arr[indx] * ( 1. - (diffs[i]/ideal[i]) )
                        flags[2] = -1
                    elif indx == 1:
                        ## Decrease the b parameter
                        par_arr[indx] = par_arr[indx] * ( 1. - ( diffs[i]/ideal[i] ) )
                        flags[1] = -1

                else:
                    ##  The pair potential must decrease. 
                    if indx == 0:
                        ##  Decrease the A parameter
                        par_arr[indx] = par_arr[indx] * ( 1. - (diffs[i]/ideal[i]) )
                        flags[0] = -1
                    elif indx == 3:
                        ## Decrease the b parameter
                        par_arr[indx] = par_arr[indx] * ( 1. - ( diffs[i]/ideal[i] ) )
                        flags[3] = -1
                    elif indx == 2:
                        ##  Increase the B parameter
                        par_arr[indx] = par_arr[indx] * ( 1. + (diffs[i]/ideal[i]) )
                        flags[2] = 1
                    elif indx == 1:
                        ## Increase the b parameter
                        par_arr[indx] = par_arr[indx] * ( 1. + ( diffs[i]/ideal[i] ) )
                        flags[1] = 1

                changes += flags * diffs[i] / ideal[i]
            print('changes = %s, i = %s, diff = %s' %(changes, i, diffs[i]) )
            tol_a = 0.01
            tol_coa = 0.001
            min_alat1, min_coa1, diff_ratioa1, diff_ratioc1, alat_diff1, coa_diff1 = obtain_min_alat_coa(LMarg, args, par_arr,
                                                                         a_u, a_l, tol_a, coa_u , coa_l, tol_coa )

            min_alat1, alat_diff1 = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, a_u, a_l, tol_a, 'alatTi', alat_ideal)

            min_coa1, coa_diff1 = obtain_min_lattice_parameter_pp(LMarg, args, par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)

            diffs = np.array([coa_diff1, alat_diff1 ])
            if np.all(diffs < diffs0):
                print('Found better pair potential parameters')
                break


    return

##############################################################################################################
##############################################################################################################





##############################################################################################################
###################     Elastic Constant Routines     ########################################################


def elastic_constant_shear_info( LMarg, args, alpha, strain):
        """ Obtain information of the strain applied to the lattice, with a given small deformation, alpha.  
            xijc == coefficients of a given elastic constant in the strain 
            ijv  == the indices of the elastic constants in the strain
            dil  == dilatation  """

        cmd =  LMarg + ' ' + args  + ' -valpha=' + str(alpha) + ' '  + strain + ' ' 
        cmd_write_to_file(cmd, 'ecout')

        cmd = "grep -A 4 'SHEAR: distortion' ecout"; cmd_write_to_file(cmd, 'shearinfo')

        if alpha != 0.0:
            cmd = "sed -n '5p' shearinfo"; xijc = cmd_result(cmd)[0:-1]
            cmd = "sed -n '4p' shearinfo"; ijv = cmd_result(cmd)  
            cmd = "sed -n '1p' shearinfo"; line_dil_alpha = cmd_result(cmd).split()  
            dil = float(line_dil_alpha[-1].replace("=",""))
        else:
                xijc='0' 
                ijv ='0'
                dil='0'

        cmd = "grep 'total energy' ecout | tail -1 | awk '{print $4}'"; etot = float(cmd_result(cmd)[0:-1])

        return [etot, alpha, xijc, ijv, dil]




def fifth_ord_poly(x,a,b,c,d,e,f):
        return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5


def check_ij_coeff( ec_ind, ijns, xijc):
    if ec_ind in ijns:
        xij = xijc.split()[ ijns.split().index( ec_ind ) ]
    else:
        xij = 0
    return float(xij)


def make_ec_arr_row( ijns, xijc):
    """ Routine to obtain the coefficients of the elastic constants 
        and make rows of a matrix for which they could be solved, 
        as it is a linear system of equations"""
    a11 = check_ij_coeff('11', ijns, xijc)
    a22 = check_ij_coeff('22', ijns, xijc)
    a33 = check_ij_coeff('33', ijns, xijc)

    a12 = check_ij_coeff('12', ijns, xijc)
    a21 = check_ij_coeff('21', ijns, xijc)

    a13 = check_ij_coeff('13', ijns, xijc)
    a23 = check_ij_coeff('23', ijns, xijc)
    a32 = check_ij_coeff('32', ijns, xijc)
    a31 = check_ij_coeff('31', ijns, xijc)

    a66 = check_ij_coeff('66', ijns, xijc)
    #   In HCP a13 == a23, a11 == a22
    a13 = a13 + a23 + a32 + a31
    a12 = a12 + a21
    a11 = a11 + a22
    row = [a11, a33, a12, a13]# a66]
    return row
    

def ec_alpha_poly(LMarg, args,  strain, plotcurve, alphalist, cell_vol):
    """ Calculates the elastic constants by fitting a fifth order polynomial to the energies obtained
        from each lattice strained with a small deformation alpha.
        Curvature at alpha = 0 is proportional to the elastic constants. (Energy per unit volume expansion)"""
    etot_list=[]

    for i in alphalist:
        etot, alpha, xijc, ijns, dil= elastic_constant_shear_info(LMarg, args, i, strain)
        etot_list.append(etot)
    
    ##  1 bohr = x angstrom
    bohr_to_angstrom = 0.529177208 
    ##  Rydberg to electron volts
    ryd = 13.606 
    etot_list = np.asarray(etot_list)/cell_vol
    ##  Converting to eV/AA**3
    etot_list = etot_list * (ryd / bohr_to_angstrom**3)

    ##  popt[0] * x**5 + popt[1] * x**4 + popt[2] * x**3 + popt[3] * x**2 + popt[4] * x + popt[5]
    popt = np.polyfit(alphalist, etot_list, 5) 
    ec_arr_row = make_ec_arr_row(ijns, xijc) 
    ##  Curvature at alpha = 0
    curvature =  2 * popt[3] #+ 3*2*popt[2] + 4*3*popt[1] * 0 +  5*4*popt[0] * 0 

    if plotcurve == True:
        afunclist=np.linspace(alphalist[0], alphalist[-1], 100)
        fitcurveE=[]

        for i in afunclist:
            fitcurveE.append(fifth_ord_poly(i,popt[5],popt[4],
                                    popt[3],popt[2],popt[1],popt[0]))
        x = [alphalist, afunclist]
        y = [etot_list, fitcurveE]
        colour = ['r*', 'g--']
        plot_function(2, x, y, colour, 'Total Energy vs Alpha', 
                            'Alpha (Deformation parameter)', 'Energy (ev)')

    return curvature, ec_arr_row 


##########################################################################################
########################     Elastic Constant Strain Routines      #######################

def Girshick_Elast(LMarg, args, alphal, cell_vol):
    
    print('\n Girshick_Elast Routine \n')

    ##  Note: FATAL error (   Exit -1 FERMI: NOS ( 0.00000 0.00000 ) does not encompass Q = 8   )  
    ##  if the elastic constants are calculated at minimum alat and coa, they must be done at ideal values. 
    ##  But this makes curvatures off somewhat due to the real lower energy structure being at a lower energy,
    ##  such that if one shears in a direction where the energy of the structure is lower the curvature obtained if off. 
    ##  Hope it just converges regardless... 


    alphalist = alphal
    

    ####################     Extrapolated Elastic Constants from Fisher and Renken 1964     ###########################
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


    """
    ####################     Extrapolated from Ogi, Kai and Ledbetter et al, 2004     #################################

    ##  Extrapolated Elastic Constants from BELOW Debye Temperature, Ogi and Kai: 
    C11_OKL = 1.137944141944627,
    C33_OKL = 1.2174239872668087,
    C44_OKL = 0.36965564052749383,
    C66_OKL = 0.2639877492337949,
    C12_OKL = 0.609968643477037,
    C13_OKL = 0.35290924243899613 


    ##  Extrapolated Elastic Constants from ABOVE Debye Temperature, Ogi and Kai: 
    C11_OKL2 = 1.1603429485408543,
    C33_OKL2 = 1.19734567400176,
    C44_OKL2 = 0.36001378920477395,
    C66_OKL2 = 0.27089740679561325,
    C12_OKL2 = 0.618548134949628,
    C13_OKL2 = 0.378197149603258 


    ####################     Girshick's Fit, Simmons and Wang 1971     #################################################
    c11exp = 1.099
    c33exp = 1.189
    c44exp = 0.317
    c66exp = 0.281
    c12exp = 0.542
    c13exp = 0.426


    kkexp  = 0.687
    rrexp  = 0.386
    hhexp  = 0.305
    """


    ec_exp_arr = np.array([     C11_FR,
                                C33_FR,
                                C44_FR,
                                C66_FR,
                                C12_FR,
                                C13_FR ])
    
    curvature = []
    strain11 =              ' -vexx=1' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature11, row11 = ec_alpha_poly(LMarg, args, strain11, False, alphal, cell_vol)  
    
    print(' C11 = %s' %(curvature11) )
    curvature.append(curvature11)                                                     #1

    strain112 =             ' -vexx=0' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature112, row112 = ec_alpha_poly(LMarg, args, strain112, False, alphal, cell_vol) 
    print(' C12 = %s' %(curvature112) )
    curvature.append(curvature112)                                                     #2

    strain33 =              ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature33, row33 = ec_alpha_poly(LMarg, args, strain33, False, alphal, cell_vol) 
    print(' C33 = %s' %(curvature33) ) 
    curvature.append(curvature33)                                                     #3
  
    strain2C112C22 =        ' -vexx=1.0' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature2C112C12, row2C112C22 = ec_alpha_poly(LMarg, args, strain2C112C22, False, alphal, cell_vol) 
    print(' 2*C11 + 2*C12 = %s' %(curvature2C112C12) ) 
    curvature.append(curvature2C112C12)                                                     #4
  
    strain5o4C11C12 =       ' -vexx=0.5' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature5o4C11C12, row5o4C11C12 = ec_alpha_poly(LMarg, args, strain5o4C11C12, False, alphal, cell_vol)  
    print(' 5/4 * C11 + C12 = %s' %(curvature5o4C11C12) ) 
    curvature.append(curvature5o4C11C12)                                                     #5
  
    strainC11C332C13 =      ' -vexx=1' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvatureC11C332C13, rowC11C332C13 = ec_alpha_poly(LMarg, args, strainC11C332C13, False, alphal, cell_vol)  
    print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C13) ) 
    curvature.append(curvatureC11C332C13)                                                           #6
  
    strainC11C332C132 =     ' -vexx=0' +\
                            ' -veyy=1' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvatureC11C332C132, rowC11C332C132 = ec_alpha_poly(LMarg, args, strainC11C332C132, False, alphal, cell_vol)  
    print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C132) ) 
    curvature.append(curvatureC11C332C132)                                                            #7
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
    curvature.append(0)


    strain4C44 =            ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=1' +\
                            ' -vexy=0 '

    curvature4C44, row4C44 = ec_alpha_poly(LMarg, args, strain4C44, False, alphal, cell_vol)  
    print(' 4*C44 1 = %s' %(curvature4C44) ) 
    curvature.append(curvature4C44)                                                           #9
  
    strain4C442 =           ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=1' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature4C442, row4C442 = ec_alpha_poly(LMarg, args, strain4C442, False, alphal, cell_vol)  
    print(' 4*C44 2= %s' %(curvature4C442) ) 
    curvature.append(curvature4C442)                                                              #10
  
    strain8C44 =            ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=1' +\
                            ' -vexz=1' +\
                            ' -vexy=1 '

    curvature8C44, row8C44 = ec_alpha_poly(LMarg, args, strain8C44, False, alphal, cell_vol)  
    ##  For some reason, in tbe this is not simply 8C44, but 8*C44 + 4*C66... ? 
    print(' 8*C44,+ 2C11 - 2C12 = %s' %(curvature8C44) ) 
    curvature.append(curvature8C44)                                                           #11
  
    strain4C66 =            ' -vexx=1' +\
                            ' -veyy=-1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature4C66, row4C66 = ec_alpha_poly(LMarg, args, strain4C66, False, alphal, cell_vol)  
    print(' 4*C66 1 = %s' %(curvature4C66) ) 
    curvature.append(curvature4C66)                                                               #12
  
    strain4C662 =           ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=1 '

    curvature4C662, row4C662 = ec_alpha_poly(LMarg, args, strain4C662, False, alphal, cell_vol) 
    print(' 4*C66 2 = %s' %(curvature4C662) ) 
    #print ('row 4c662', row4C662)
    curvature.append(curvature4C662)                                                           #13
  
    """  
    strain3R = ' -vexx=-0.5' +\
                            ' -veyy=-0.5' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature3R, row3R = ec_alpha_poly(LMarg, args, strain3R, False, alphal, cell_vol)
    print(' 3*R = %s' %(curvature3R) ) 
    #print('0.5*C11 + C33 + 0.5*C12 - 2*C13')
    curvature.append(curvature3R)                                                           #14
  
    strain3H = ' -vexx=1' +\
                            ' -veyy=-0.5' +\
                            ' -vezz=-0.5' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature3H, row3H = ec_alpha_poly(LMarg, args, strain3H, False, alphal, cell_vol)
    print(' 3*H = %s' %(curvature3H) ) 
    #print('1.2*C11 + 0.25*C33 - C12 - 0.5*C13')
    curvature.append(curvature3H)                                                                #15
  
    """


    c11 = 0.5   * ( curvature[1-1 ] + curvature[2-1 ] ) #/ (10**(9)) #Correct
    c33 = 1.0   *   curvature[3-1] / (10**(9)) #Correct
    c12 = 1./3.  * ( curvature[4-1 ] + curvature[5-1 ] - 3.25*c11 ) #/ (10**(9)) #Correct
    c13 = 0.25 * ( curvature[6-1 ] + curvature[7-1 ] - 2*c11 - 2*c33 ) #/ (10**(9)) #Correct
    c66 = 0.125 * ( curvature[12-1 ] + curvature[13-1 ] ) #/ (10**(9))
    c442 = 0.0625* ( curvature[9-1 ] + curvature[10-1 ] + curvature[11-1 ] - 4*c66 ) # / (10**(9)) 
    c44 = 0.0625* ( curvature[9-1 ] + curvature[10-1 ] + curvature[11-1 ]  )# / (10**(9)) 

    kk = (1./9.) * ( 2 *  c11 +  c33 + 2 * c12  + 4 * c13 )

    rr = (1./3.) * ( 0.5 * c11 +  c33 + 0.5 * c12  - 2 * c13 )

    hh = (1./3.) * ( 1.2 * c11 +  0.25 * c33 -  c12  - 0.5 * c13 )


    #kk  = curvature[  8 -1 ] / 9. / (10**(9))
    #rr  = curvature[ 14 -1 ] / 3. / (10**(9))
    #hh  = curvature[ 15 -1 ] / 3.  / (10**(9))

    print( '\n Elastic Constants: Girshick Routine \n' )
    print('\n C11 = %.3f,   C11_FR = %.3f' %(c11, C11_FR))
    print(  ' C33 = %.3f,   C33_FR = %.3f' %(c33, C33_FR))
    print(  ' C44 = %.3f,   C44_FR = %.3f' %(c44, C44_FR))
    print(  ' C44 = %.3f,   C44_FR = %.3f' %(c442, C44_FR))
    print(  ' C66 = %.3f,   C66_FR = %.3f' %(c66, C66_FR))
    print(  ' C12 = %.3f,   C12_FR = %.3f' %(c12, C12_FR))
    print(  ' C13 = %.3f,   C13_FR = %.3f' %(c13, C13_FR))

    print( ' K = %.3f,   K_FR = %.3f' %(kk, K_FR))
    print( ' R = %.3f,   R_FR = %.3f' %(rr, R_FR))
    print( ' H = %.3f,   H_FR = %.3f \n ' %(hh, H_FR))
    print(  'C66 - 0.5(C11 - C12) = %.3f,   C66_FR = %.3f' %(c66 - 0.5 * (c11 - c12), C66_FR) )
    
    return  np.array([c11, c33, c44, c66, c12, c13]) - np.asarray(ec_exp_arr)







def weighted_mean(x, w):
    """This routine returns the mean of x and the variance of it from the weighed mean
       It assumes lists are one dimensional arrays of the x_i values and the weights w_i"""
    w=np.asarray(w)
    x=np.asarray(x)
    x_bar = np.sum(w*x)/np.sum(w)    
    var_xbar= x_bar/np.sum(w*x) #This is true if the weights are the variances. 

    return x_bar, var_xbar 

def sample_mean_covarance(X, w , mu ):
    """Returns to covariance of a matrix of input values X, which have weights w, with a mean vector mu/"""
    w = w / np.sum(w)
    q = np.zeros((len(mu), len(mu)))
    sumoveri = 0
    for j in range(len(mu)):
        for k in range(len(mu)):
            for i in range(len(X)):
                sumoveri +=  w[i] * ( X[i][j] -  mu[j]) * ( X[i][k] - mu[k] )
            q[j][k] = 1. / ( 1. - np.sum(w**2) ) * sumoveri 
            sumoveri= 0
    print('Covariance Matrix', q, '\n')
    return q



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

            #clat_diff1 = min_coa1 * min_alat1 - coa_ideal * alat_ideal
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

def get_pp_args(par_arr_p):

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
    return ppargs
 
def vary_params(maxit, npass, par_arr_p, ddcoeffs_p, pair_pot_bank, ddcoeff_bank, total_error_p, dddelta_norm, pR0, qR0):
    """
    c11exp = 1.099
    c12exp = 0.542
    c13exp = 0.426
    c33exp = 1.189
    c44exp = 0.317
    c66exp = 0.281
    kkexp  = 0.687
    rrexp  = 0.386
    hhexp  = 0.305
    """
    #maxit = 2000
    n_max = 20

    C11_FR = 1.1103616820304443
    C33_FR = 1.1963098653973396
    C44_FR = 0.3210116099936396
    C66_FR = 0.2867984040386178
    C12_FR = 0.5368771097739773
    C13_FR = 0.4241634577392404 



    ec_exp_arr = np.array([     C11_FR,
                                C33_FR,
                                C44_FR,
                                C66_FR,
                                C12_FR,
                                C13_FR ])
    
    alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
    clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 

    alat_bcc_ideal = 6.254960504932251  ## Senkov, Chakoumas, Effect of temperature and hydrogen concentration on the lattice parameter of beta titanium, 2001
    alat_fcc_ideal = 8.201368154503314
    coa_ideal = clat_ideal/alat_ideal

    npass = npass + 1

    print('\n    Vary Parameters Routine     npass = %s \n' %(npass))

    print('par_arr_p = %s' %(par_arr_p) )
    print('ddcoeffs_p = %s' %(ddcoeffs_p) )
    print('total_error_p = %s' %(total_error_p) )


    ###################################################################
    ################     Initial Args     #############################

    ctrlf = 'ctrl.ti'   
    ext = 'ti'
    symfile='syml'
    LMarg= 'tbe --mxq ' + ctrlf

    symmpt = 0
    bond_int=5.0         #0.208098266125 #0.30511275
    bond_int_temp=0.0   #0.208098266 #0.304249125
    evtol = 0.002

    alphal = np.linspace(-0.01, 0.01, 11)


    xargs = ' -vfp=0 -vrfile=0 -vppmodti=10 -vSDTqR0=' + str(qR0)\
                                  + ' ' + '-vSDTpR0=' + str(pR0) + ' '#\
                                  #+ ' -vspanjddd=' + str(dddeltanorm) + ' ' #0.208098266125 '

    ###############################################################################
    ##################      Normalize dddelta coefficients      ###################

    ##  Do not need to have pair potential in place to normalise the ddd coefficients. 
    xargs += ' -vspanjdec=' + str(ddcoeffs_p[-1]) + ' '

    d_norm, E_F = band_width_normalise( LMarg, xargs, symmpt, ext, ddcoeffs_p[:-1], bond_int, bond_int_temp, evtol)

    ddnames = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD']
    dargs = construct_extra_args('', ddnames, ddcoeffs_p[:-1]) + d_norm + E_F

    ###############################################################################
    ##################      Input Pair Potential Parameters      ##################

    if npass == 1:
        pp_wgtd_mean = par_arr_p
        dd_wgtd_mean = ddcoeffs_p
        pp_wgtd_mean_err = 1.

    else:
        pp_wgtd_mean_err = np.sqrt( 1. / np.sum((1./total_error_p**2) ))
        pp_wgtd_mean = np.sum( pair_pot_bank * (1./total_error_p**2)[:,np.newaxis], axis = 0 ) * pp_wgtd_mean_err**2 


    par_arr_p, alat_diff, clat_diff, coa_diff, min_coa, min_alat = pp_scale_const(LMarg, par_arr_p, pp_wgtd_mean, total_error_p,
                                                                                  xargs + dargs, npass, maxit)
    ppargs = get_pp_args(par_arr_p)

    ################################################################################################
    ###################     Energies of different structures    ####################################
    
    etot_bcc = find_energy( LMarg, xargs + ppargs + dargs + ' -vnbas=3 -vomegabcc=1 -vubccom=1 -valatTi=' + str(alat_ideal) + ' ', 'ebcc')
    etot_bcc2 = find_energy( LMarg, xargs + ppargs + dargs + ' -vnbas=1 -vbccs=1 -valatTi=' + str(alat_bcc_ideal) + ' ', 'ebcc2')
    etot_omega = find_energy( LMarg, xargs + ppargs + dargs + ' -vnbas=3 -vomegabcc=1 -vubccom=0 -valatTi=' + str(alat_ideal) + ' ', 'eomega')
    etot_fcc = find_energy( LMarg, xargs + ppargs + dargs + ' -vnbas=1 -vfccs=1 -valatTi=' + str(alat_fcc_ideal) + ' ', 'efcc')
    

    coaarg = ' -vcoa=' + str(coa_ideal) + ' '
    alatarg = ' -valatTi=' + str(alat_ideal) + ' '
    args = xargs + ppargs + dargs + alatarg + coaarg

    etot_hcp = find_energy( LMarg, args, 'ehcp')

    print ('\n Arguments \n %s' %(args) )


    ###################################################################
    ###########     Initial Test      ################################

    
    filename='pptest'
    cmd = LMarg + ' ' + args + ' ' 
    cmd_write_to_file( cmd, filename)

    ###############     Cell volume at equilibrium c_lat and a_lat     ##########
    print(' Obtaining Cell Volume at ideal c and a \n' )
    filename='equibtest'
    cmd = LMarg + ' ' + args #+ ' ' + xargs 
    cmd_write_to_file( cmd, filename)
    cell_vol = float(cmd_result( "grep 'Cell vol' " + filename + " | awk '{print $7}'" ))
    print('cell_vol = %s' %(cell_vol) )

    ################     Get Elastic Constants     #####################
    print(' Obtaining elastic constants at ideal c and a \n' )
    e_consts_diff = Girshick_Elast(LMarg, args + xargs , alphal, cell_vol)
    print('Elastic Constants diff = %s' %(e_consts_diff))


    ################     Obtain Total Error and Evaluate Changes    ##############

    alat_err = (alat_diff)/alat_ideal
    clat_err =  (clat_diff)/clat_ideal
    coa_err = coa_diff/coa_ideal
    ec_err = np.sum(abs(e_consts_diff))/np.sum(abs(ec_exp_arr)) 

    print('Errors:\n  alat_err = %s \n coa_err = %s \n clat_err = %s \n EC_err = %s' %( alat_err, coa_err, clat_err, ec_err ))

    if ec_err > coa_err + alat_err:
        ec_err_dd = ( np.random.uniform() * 0.3)
    else:
        ec_err_dd= ec_err

    total_error = (alat_diff**2 + coa_err**2 + np.sum(e_consts_diff**2) )**(0.5)   

    if npass == 1:
        pair_pot_bank = np.zeros(par_arr_p.shape).reshape(  (1, par_arr_p.shape[0]))
        ddcoeff_bank = np.zeros(ddcoeffs_p.shape).reshape( (1, ddcoeffs_p.shape[0]))
        pp_wgtd_mean = par_arr_p
        dd_wgtd_mean = ddcoeffs_p

    else:
        pp_wgtd_mean_err = np.sqrt( 1. / np.sum((1./total_error_p**2) ))
        pp_wgtd_mean = np.sum( pair_pot_bank * (1./total_error_p**2)[:,np.newaxis], axis = 0 ) * pp_wgtd_mean_err**2 
        dd_wgtd_mean   = np.sum( ddcoeff_bank * (1./total_error_p**2)[:,np.newaxis], axis = 0 ) * pp_wgtd_mean_err**2 
        dd_wgtd_mean_err = pp_wgtd_mean_err 

        print('\n Weighted means: \n pp_wgtd_mean = %s \n pp_wgtd_mean_err = %s \n \n dd_wgtd_mean = %s \n  dd_wgtd_mean_err = %s \n' %( pp_wgtd_mean, pp_wgtd_mean_err, dd_wgtd_mean, dd_wgtd_mean_err ))

        print(' pair_pot ',  par_arr_p )
        print(' dd coeffs \n', ddcoeffs_p )


    dd_mn = dd_wgtd_mean  #np.array([6., 4., 1., 0.4967])
    dd_cov_0 = np.diag([1.0, 0.8, 0.2, 0.25])**2

    if npass < 3:
        dd_cov = np.zeros( ( len(dd_mn), len(dd_mn) ) )
    else:
        dd_cov = sample_mean_covarance(ddcoeff_bank, (1./total_error_p**2) , dd_wgtd_mean )

    dd_cov = (   ( dd_cov  + dd_cov_0  ) / 2.  ) * ( maxit - npass % maxit  ) /  maxit 
    dd_noise = np.random.multivariate_normal(mean=dd_wgtd_mean, cov=dd_cov )




    if total_error_p[-1] < total_error and npass > 1: 
        # Change of the old iteration is better than the new one:
        print('Change of the old iteration is better than the new one:\n')
        par_arr, ddcoeffs = check_latpar_differences(alat_err, pair_pot_bank[-1], ddcoeff_bank[-1], pp_wgtd_mean, dd_wgtd_mean, dd_noise)
        print('new dd coeffs: alatdiff = %s' %( ddcoeffs ) )
        par_arr, ddcoeffs = check_latpar_differences(clat_err, np.abs(par_arr), np.abs(ddcoeffs), pp_wgtd_mean, dd_wgtd_mean, dd_noise)
        print('new dd coeffs: clatdiff = %s' %( ddcoeffs ) )       
        par_arr, ddcoeffs = check_latpar_differences(ec_err_dd,   np.abs(par_arr), np.abs(ddcoeffs), pp_wgtd_mean, dd_wgtd_mean, dd_noise)
        print('new dd coeffs: ECdiff = %s \n' %( ddcoeffs ) )
    else:
        # Change of the new iteration is better than the old one:
        print('Change of the new iteration is better than the old one:\n')
        par_arr, ddcoeffs = check_latpar_differences(alat_err, par_arr_p,  ddcoeffs_p,    pp_wgtd_mean, dd_wgtd_mean, dd_noise)
        print('new dd coeffs: alatdiff = %s' %( ddcoeffs ) )
        par_arr, ddcoeffs = check_latpar_differences(clat_err, par_arr,    ddcoeffs,      pp_wgtd_mean, dd_wgtd_mean, dd_noise)
        print('new dd coeffs: clatdiff = %s' %( ddcoeffs ) )
        par_arr, ddcoeffs = check_latpar_differences(ec_err_dd,   par_arr,    ddcoeffs,      pp_wgtd_mean, dd_wgtd_mean, dd_noise)
        print('new dd coeffs: ECdiff = %s \n'%( ddcoeffs ) )
 




    ###################################################################################################################################
    #######################     Changing the total error based on the energy ordering of the structures     ###########################

    print('Energies of different structures:\n  ehcp = %s \n   ebcc = %s \n   ebcc2 = %s \n eomega = %s \n efcc = %s ' %(etot_hcp, etot_bcc, etot_bcc2, etot_omega, etot_fcc))

    energy_list = np.array([ etot_hcp, etot_bcc, etot_omega, etot_fcc ])
    e_name_list = ['ehcp', 'ebcc', 'eomega', 'efcc']
    e_arg_sort = np.argsort(energy_list)

    if e_name_list[ e_arg_sort[0] ] != 'ehcp':
        print('Energy ordering is very wrong!!! ')
        if e_name_list[ e_arg_sort[0] ] == 'efcc':
            print('FCC has the least energy, penalty:   8 * error ')
            ##  Give the greatest penalty 
            total_error_p = 8 * total_error_p
        if e_name_list[ e_arg_sort[0] ] == 'eomega':
            print('Omega phase has the least energy,  penalty:  6 * error')
            ##  Give a lesser penalty 
            total_error_p =  6* total_error_p
        if e_name_list[ e_arg_sort[0] ] == 'ebcc':
            print('BCC has the least energy, penalty:  4 * error')
            ##  Give the smallest penalty 
            total_error_p = 4 * total_error_p
    else:
        print('HCP has the least energy')
        ##  Give no penalty 
        if e_name_list[ e_arg_sort[1] ] == 'ebcc':
            print('BCC has the second lowest energy, penalty: 2 * error ')
            total_error_p =  3 *  total_error_p
        elif e_name_list[ e_arg_sort[1] ] == 'eomega':
            print('FCC has the second lowest energy, penalty: 3 * error ')
            total_error_p = 2 * total_error_p
        else:
            print('FCC has the second lowest energy: No penalties')
        
        
    ##################################################################################################################################
    #######################    Appending pair potentials and dd coeff banks    #######################################################

    if npass == 1:
        pair_pot_bank = np.array([par_arr_p])#.reshape( (1, par_arr_p.shape[0]))
        ddcoeff_bank = np.array([ddcoeffs_p])
        total_error = np.array([total_error])
    else:
        pair_pot_bank = np.append(pair_pot_bank, par_arr_p).reshape( 
                                        ( pair_pot_bank.shape[0] + 1, pair_pot_bank.shape[1] ) )
        ddcoeff_bank = np.append(ddcoeff_bank, ddcoeffs_p).reshape( 
                                        ( ddcoeff_bank.shape[0] + 1, ddcoeff_bank.shape[1] ) )
        total_error = np.append(total_error_p, total_error)

    ###########################     Removing worst result of the bunch     ###############################
    if npass > 2:
        if npass % 3 == 0 or len(pair_pot_bank) > n_max:
            ##  Every second iteration after the second pass, discard the worst result. 
            ind_w = np.argmax(total_error)

            print('\n Removing\n    pp = %s,\n    dd = %s,\n    with total error = %s \n ' %(pair_pot_bank[ind_w], ddcoeff_bank[ind_w], total_error[ind_w]))
            
            pplen = len(par_arr_p)
            ddlen = len(ddcoeffs_p)

            pair_pot_bank = np.delete( pair_pot_bank.flatten(), range( ind_w * pplen, (ind_w + 1) * pplen ) ) 
            pair_pot_bank = pair_pot_bank.reshape( ( len(pair_pot_bank)//pplen, pplen ) )

            ddcoeff_bank = np.delete( ddcoeff_bank.flatten(), range( ind_w * ddlen, (ind_w + 1) * ddlen  ) )
            ddcoeff_bank = ddcoeff_bank.reshape( ( len(ddcoeff_bank)//ddlen, ddlen ) )

            total_error = np.delete(total_error, ind_w)

    print('pair_pot ',  par_arr_p )
    print('new dd coeffs ', ddcoeffs )
    print('pair_pot_bank', pair_pot_bank)
    print('dddcoeff_bank', ddcoeff_bank)
    print('total_error \n', total_error)

 

    return npass, np.abs(par_arr), np.abs(ddcoeffs), pair_pot_bank, ddcoeff_bank, total_error


#############################################################################################################
##############################    Bandwidth Normalisation Routines    #######################################

def width_dft(dft_bands, symmpt):
    ##  Uses the colour weights of the dft bands that show how much character a particular band has at the symmetry point. 
    ##  The bands that have the most d character are used for the dft bandwidth
    dft_band_energies, colwgts= get_band_energies(dft_bands, True, symmpt) ## DFT == True
    dbands, wgttol = determine_wgt_fit(dft_band_energies, colwgts[0])
    width = (np.max(dbands)-np.min(dbands)) * 13.606
    return width, dft_band_energies

def determine_wgt_fit( band_energies, colwgt):
    """This is a routine that looks at the amount of d character there is in the dft results and then obtains the 10 eigenvalues 
    that have the most d character. From this we can compare with tbe, look at bandwidth and change normalisation accordingly. """
    bnde = np.asarray(band_energies)
    wgt = np.asarray(colwgt)
    a = 1.0
    al = 0.0
    alh=0.5
    dbands= bnde[wgt>alh]
    while len(dbands) != 10: 
        if len(dbands) < 10:
            a = alh
        elif len(dbands) > 10:
            al = alh
        alh = (a + al)/2.
        dbands= bnde[wgt>alh]
    return dbands, alh

def get_bandwidth(LMarg, args, symmpt, filename, ext):
    E_F = fermi_energy(LMarg, args)
    dbw = band_width(LMarg, args, E_F, filename, symmpt, ext)
    return dbw, E_F


def fermi_energy(LMarg, args):
    ##  Get Fermi energy of system 
    cmd = LMarg + ' ' + args 
    cmd_write_to_file(cmd, 'out')
    cmd = "grep 'Fermi energy:' out | tail -1 | awk '{print $4}' | sed 's/.$//'"
    E_F = float( cmd_result(cmd)[:-2].strip() )
    return E_F

def band_width(LMarg, args, E_F, filename, symmpt, ext):
    band_calc(LMarg, args, E_F, filename)
    chk=check_bandcalc_resolves(filename)
    while len(chk) > 2:
        band_calc(LMarg, args, E_F, filename)
        chk=check_bandcalc_resolves(filename)
    bndfile = open('bnds.' + ext, mode='r')
    d_width = width_symm_pt(bndfile, symmpt, False, [])
    bndfile.close()
    return d_width

def check_bandcalc_resolves( file):
    cmd = "grep 'Exit -1' " + file + " | tail -1"
    check = cmd_result(cmd)[0:-1]
    return check


def band_calc(LMarg, args, E_F, filename):
    cmd = LMarg + ' ' + args  +  ' -vef=' + str(E_F) +  ' -ef=' + str(E_F) + ' '  + '--band~fn=syml'
    cmd_write_to_file(cmd, filename)
    return 

def width_symm_pt( bandfile, symmpt, dft, dftbe):
    ##  symmpt is a number specifying the Zeroth (Gammma), first, second, third, etc symmpoint where the width is to be taken
    nit=str(80)

    band_energies_float=[]
    if dft:
        mx = np.max(dftbe)
        mn = np.min(dftbe)
        print ('mx, mn , max min', mx, mn,  dftbe[-1] , dftbe[0])
        d_width1 = ( dftbe[-1] - dftbe[0] ) * 13.606
        d_width = ( mx - mn ) * 13.606
        print('**DFT** width of %s point' %(symmpt), d_width, dwidth1)
    else: 
        lines = [' ']  
        for i in range(0, 4 + symmpt * ( 2 * int(nit) + 1 ) ):
            if i == 4 + symmpt*(2*int(nit) + 1) - 1: 
                lines[0] = bandfile.readline()[0:-1]
            else:
                bandfile.readline()[0:-1]
        band_energies = lines[0].split()
        for i in band_energies:
            if '-' in i[1:]:
                temp = i[1:]
                temp = temp.replace("-", " -")
                i =  (i[0]+ temp).split()
                for j in i:
                    band_energies_float.append(float(j))
            else: 
                band_energies_float.append(float(i))
        d_width = ( -band_energies_float[0] + band_energies_float[-1] ) * 13.606
    return d_width

def band_width_normalise( LMarg, xargs, symmpt, ext, ddcoeffs, bond_int, bond_int_temp, evtol):
    ##  Bandwidth normalised with regards to the band width at a symmetry point defined by symmpt

    print( "\n Bandwidth Normalisation routine \n")

    dftfile = 'dftticol2'
    filename = 'out'
    ddnames = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD']
    dargs = construct_extra_args('', ddnames, ddcoeffs)
    d_norm =  ' -vspanjddd=' + str(bond_int) + ' ' 
    b_width, E_F =  get_bandwidth(LMarg, (xargs + dargs + d_norm), symmpt, filename, ext) 
    b_width = abs(b_width)

    dft_bands=open('bnds.' + dftfile, mode='r')
    dftwidth, dftbe = width_dft(dft_bands, symmpt)
    dft_bands.close()
    #DFTwidthG=9.2 #This is the bandwidth of the Gamma point taken from Jafari 2012

    Fitting=False
    its=0
    bond_int1 = (bond_int + bond_int_temp)/2.
    while Fitting==False:
        print( "\n Normalisation iteration = %s" %(its))
        print('Bond integrals: upper = %s, lower = %s' %(bond_int, bond_int_temp))

        if abs( abs(dftwidth) - abs(b_width) ) < evtol:
            bond_int1 = (bond_int + bond_int_temp)/2.
            print('\n Found Bond integral Normalisation coefficient: %s \n ' %(bond_int1) )
            Fitting == True
            break
        else:  
            ##  Binary search with upper and lower limits as bond_int and bond_int_temp 
            bond_int1 = (bond_int + bond_int_temp)/2.
            d_norm1 =  ' -vspanjddd=' + str(bond_int1) + ' ' 
            b_width1, E_F =  get_bandwidth(LMarg, (xargs + dargs + d_norm1) , symmpt, filename, ext) 
            b_width1 = abs(b_width1)
            print('TBE Bandwidth = %s,  DFT Bandwidth = %s' %(b_width1, dftwidth))
            if b_width1 - dftwidth >  0: 
                ##  Width is greater than required so new upper limit is set
                bond_int=bond_int1

            if b_width1 - dftwidth <  0:  
                ##  Width is lower than required so new lower limit is set
                bond_int_temp=bond_int1

            its += 1
            b_width = abs( b_width1 )

    print("Finished Binary search for bond integral")
    bond_int1 = (bond_int + bond_int_temp)/2.
    d_norm =  ' -vspanjddd=' + str(bond_int1) + ' ' 
    E_F = ' -vef=' + str(E_F) +  ' -ef=' + str(E_F) + ' '
    return d_norm, E_F

def get_band_energies( bandfile, dft, symmpt):

    nbands =  bandfile.readline()[0:-1].split()
    nbands, ef , ncol = int(nbands[0]), float(nbands[1]), int(nbands[2])
    nit=str(80)
    lines = [' ' for i in range(2 + 2*ncol)]
    colwgts = []
    col_wgts_1 = []
    col_wgts_2 = []
    print('nbands = %s, ef = %s, ncol = %s' % (nbands, ef, ncol ) )

    if dft != True:
        ##  Get TBE band energies from bandfile
        for i in range(0, 4 + symmpt*(2*int(nit) + 1)):

            if i == 4 + symmpt*(2*int(nit) + 1) - 2: 
                ##  This obtains the line of eigenvalues at the symmetry point defined. 
                lines[0] = bandfile.readline()[0:-1] 
            else:
                ##  Read next line, but don't record it
                bandfile.readline()[0:-1]
    else:
        ##  Get DFT Band energies
        for i in range(0, 4 + symmpt*( ( 3*(ncol+1) ) * int(nit) + 1) ):

            if i == 4 + symmpt*((3*(ncol+1))*int(nit) + 1) - 2: 
                ##  This obtains the line of eigenvalues at the symmetry point defined. 
                ##  Two lines of eigenvalues, 0 and 1. 
                lines[0] = bandfile.readline()[0:-1] 
                lines[1] =  bandfile.readline()[0:-1]
                if ncol > 0:
                    for i in range(ncol):
                        bandfile.readline()[0:-1] #ignore initial 'k point' line
                        lines[2 + 2*i] = bandfile.readline()[0:-1]
                        lines[3 + 2*i] = bandfile.readline()[0:-1]
                        if i == 0:
                            col_wgts_1 = (lines[2 + 2*i] + ' ' +  lines[3 + 2*i]).split() 
                            col_wgts_1 = remove_bad_syntax( col_wgts_1 ) 
                            colwgts.append( col_wgts_1 )
                        elif i == 1:
                            col_wgts_2 = (lines[2 + 2*i] + ' ' +  lines[3 + 2*i]).split() 
                            col_wgts_2 = remove_bad_syntax( col_wgts_2 ) 
                            colwgts.append( col_wgts_2 )
            else:
                bandfile.readline()[0:-1]
        

    band_energies=(lines[0]  + ' ' + lines[1]).split()
    band_energies = remove_bad_syntax(band_energies)
    
    if (symmpt==7) and (dft):
        band_energies=band_energies[:12]
        band_energies.pop(6)
        band_energies.pop(6)

    return band_energies, colwgts




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

"""

LMarg = 'tbe --mxq ctrl.ti '
args = ' -vfp=0 -vrfile=0 '

#args += ' -vspanjdec=' + str(ddcoeffs[-1]) + ' '




d_norm = ' -vspanjddd=' + str(dddeltanorm) + ' '
#d_norm, E_F = band_width_normalise( LMarg, args, symmpt, ext, ddcoeffs[:-1], bond_int, bond_int_temp, evtol)

ddnames = ['ddsigTTSD', 'ddpiTTSD', 'dddelTTSD']
dargs = construct_extra_args('', ddnames, ddcoeffs[:-1]) + d_norm #+ E_F
args += dargs

ppargs =  get_pp_args(par_arr)

alat_ideal = 5.57678969  ## 2.951111 Angstrom R.M. Wood 1962 
clat_ideal = 8.85210082  ## 2.951111 Angstrom R.M. Wood 1962 
coa_ideal = clat_ideal/alat_ideal

args +=  ppargs

tol_a = 0.001
tol_coa = 0.001
a_u = 6.2; a_l = 5.4; coa_u = (8./3.)**0.5; coa_l = 1.4
min_alat, alat_diff,  min_coa, coa_diff = get_min_coa_and_alat(LMarg, args, par_arr, 
                                a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal)

n_a = 10
n_coa = 10
#min_alat, min_coa =  line_search_min_coa_alat(LMarg, args, par_arr, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, tol_a, tol_coa)
#brute_force_min_coa_and_alat(LMarg, args, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal, n_a, n_coa)

 
args += ' -valatTi=' + str(alat_ideal) + ' -vcoa=' + str(coa_ideal)


alphal = np.linspace(-0.01, 0.01, 11)

cell_vol = 238.
#"""
"""
curvature = []
strain11 = ' -vexx=1.0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature11, row11 = ec_alpha_poly(LMarg, args, strain11, True, alphal, cell_vol)  
    
print(' C11 = %s' %(curvature11) )
curvature.append(curvature11)                                                     #1

strain112 = ' -vexx=0.0' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature112, row112 = ec_alpha_poly(LMarg, args, strain112, True, alphal, cell_vol) 
print(' C12 = %s' %(curvature112) )
curvature.append(curvature112)                                                     #2

strain33 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature33, row33 = ec_alpha_poly(LMarg, args, strain33, True, alphal, cell_vol) 
print(' C33 = %s' %(curvature33) ) 
curvature.append(curvature33)                                                     #3

strain2C112C22 = ' -vexx=1.0' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature2C112C12, row2C112C22 = ec_alpha_poly(LMarg, args, strain2C112C22, True, alphal, cell_vol) 
print(' 2*C11 + 2*C12 = %s' %(curvature2C112C12) ) 
curvature.append(curvature2C112C12)                                                     #4
  
strain5o4C11C12 = ' -vexx=0.5' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature5o4C11C12, row5o4C11C12 = ec_alpha_poly(LMarg, args, strain5o4C11C12, True, alphal, cell_vol)  
print(' 5/4 * C11 + C12 = %s' %(curvature5o4C11C12) ) 
curvature.append(curvature5o4C11C12)                                                     #5
 
strainC11C332C13 = ' -vexx=1' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvatureC11C332C13, rowC11C332C13 = ec_alpha_poly(LMarg, args, strainC11C332C13, True, alphal, cell_vol)  
print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C13) ) 
curvature.append(curvatureC11C332C13)                                                           #6
  
strainC11C332C132 = ' -vexx=0' +\
                            ' -veyy=1' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvatureC11C332C132, rowC11C332C132 = ec_alpha_poly(LMarg, args, strainC11C332C132, True, alphal, cell_vol)  
print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C132) ) 
curvature.append(curvatureC11C332C132)                                                            #7
"""
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
"""
curvature.append(0)


strain4C44 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=1' +\
                            ' -vexy=0 '

curvature4C44, row4C44 = ec_alpha_poly(LMarg, args, strain4C44, True, alphal, cell_vol)  
print(' 4*C44 1 = %s' %(curvature4C44) ) 
curvature.append(curvature4C44)                                                           #9
  
strain4C442 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=1' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature4C442, row4C442 = ec_alpha_poly(LMarg, args, strain4C442, True, alphal, cell_vol)  
print(' 4*C44 2= %s' %(curvature4C442) ) 
curvature.append(curvature4C442)                                                              #10
  
strain8C44 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=1' +\
                            ' -vexz=1' +\
                            ' -vexy=0 '

curvature8C44, row8C44 = ec_alpha_poly(LMarg, args, strain8C44, True, alphal, cell_vol)  
print(' 8*C44 + 2C11 - 2C12 (8*C44 + 4*C66)= %s' %(curvature8C44) ) 
curvature.append(curvature8C44)                                                           #11
  
strain4C66 = ' -vexx=1' +\
                            ' -veyy=-1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

curvature4C66, row4C66 = ec_alpha_poly(LMarg, args, strain4C66, True, alphal, cell_vol)  
print(' 4*C66 1 = %s' %(curvature4C66) ) 
curvature.append(curvature4C66)                                                               #12
  
strain4C662 = ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=1 '

curvature4C662, row4C662 = ec_alpha_poly(LMarg, args, strain4C662, True, alphal, cell_vol) 
print(' 4*C66 2 = %s' %(curvature4C662) ) 
#print ('row 4c662', row4C662)
curvature.append(curvature4C662)    
#curvature.append(curvature8C44)       

"""
"""
for i in range(1000):
    npass, par_arr, ddcoeffs, pair_pot_bank, ddcoeff_bank, total_error = vary_params(npass, par_arr, ddcoeffs, pair_pot_bank, ddcoeff_bank, 
                                                                                                            total_error, dddeltanorm, pR0, qR0)
"""
"""

A = par_arr[0]
a = par_arr[1]
B = par_arr[2]
b = par_arr[3]

coeffs = np.array([-3.4499440477, 1.55341756799, -1.93263756948])#np.array([-6.,4.,-1.])

x = np.linspace(3.4, 14, 100)
y = [ znam_pair_potential(xi, 'exp')[0] for xi in x ]
y2 = [ two_exp(xi, A, a, B, b) +  bond_int(xi, coeffs, qR0) for xi in x]
y3 = [ spanjaard_des_model(xi, pR0, qR0) for xi in x]
y4 = [ spanjaard_des_model2(xi, pR0, qR0, 12., 1, 1) for xi in x]
y = [y, y2]#, y3]#, y4]
x = [x for i in range(len(y)) ]
colours = ['g--', 'r-', 'k--']#, 'b>']
plot_function(len(x), x, y, colours, 'Znam pair potential---Rydbergs', 'Bohr', 'Energy (Ryd)')

"""
