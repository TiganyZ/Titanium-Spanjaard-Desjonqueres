# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 
import ti_opt_general_sd as g

############################################################################################
##########################     Minimum lattice parameters     ##############################

def find_latpars_grid(LMarg, args, n_lp, names_lp, limits_lp, n_grid):
    names_limits = '    Find Latpars Grid:\n    n_grid = %s\n'%(n_grid)  
    al_l = [] 
    sz = ()
    for i in range(n_lp):
        name_limits = '     %s_l---%s_u = %s---%s\n'%(names_lp[i], names_lp[i], limits_lp[i][0], limits_lp[i][1] )
        al = np.linspace(limits_lp[i], limits_lp[i+1], n_grid[i])
        al_l.append(al)
        sz += (n_grid[i],)
    print( name_limits )
    
    etot_a = np.zeros( sz )

    if n_lp == 1:
        for i in range(n_grid):
            xx_args = args + g.construct_cmd_arg(names_lp, al_l[0][i])
            etot = g.find_energy(LMarg, xx_args, 'pptest')
            if etot is not str:
                etot_a[i] = etot

    elif n_lp == 2:
        for i in range(n_grid[0]):
            for j in range(n_grid[1]):
                xx_args = args + g.construct_cmd_arg(names_lp[0], al_l[0][i]) \
                               + g.construct_cmd_arg(names_lp[1], al_l[1][j])
                etot = g.find_energy(LMarg, xx_args, 'pptest')
                if etot is not str:
                    etot_a[i][j] = etot
    else:
        for i in range(n_grid[0]):
            for j in range(n_grid[1]):
                for k in range(n_grid[2]):
                    xx_args = args + g.construct_cmd_arg(names_lp[0], al_l[0][i])  \
                                   + g.construct_cmd_arg(names_lp[1], al_l[1][j])  \
                                   + g.construct_cmd_arg(names_lp[2], al_l[2][k])
                    etot = g.find_energy(LMarg, xx_args, 'pptest')
                    if etot is not str:
                        etot_a[i][j][k] = etot

    min_ind = np.unravel_index( np.argmin(etot_a), sz )
    print('\n Minimum lattice parameters')

    if n_lp == 1:
        al_min = al_l[0][min_ind]
        print('     %s = %s' %(names_lp[0], al_min))
        min_vol = ( 3**(0.5) / 2. ) * ( al_min**3 )
        ret = (al_min,)
    if n_lp > 1:
        al_min  = al_l[0][ min_ind[0] ]
        al2_min = al_l[1][ min_ind[1] ]
        print('     %s = %s' %(names_lp[0], al_min))
        print('     %s = %s' %(names_lp[1], al2_min))
        min_vol = ( 3**(0.5) / 2. ) * ( al_min**3 ) * al2_min
        ret = (al_min, al2_min)
    if n_lp > 2:
        al3_min = al_l[2][ min_ind[2] ]
        print('     %s = %s' %(names_lp[2], al3_min))
        min_vol = ( 3**(0.5) / 2. ) * ( al_min**3 ) * al2_min * al3_min
        ret += (al3_min,)

    print('     vol    = %s\n' %(min_vol))
    ret += (min_vol,)
    return ret


def opt_latpars_grid(LMarg, args, n_lp, names_lp, limits_lp, ideals_lp, n_grid, n_iter):
    ##  This routine makes a grid of lattice parameters of c and a such that the ideal one can be sought. 
    min_lps = find_latpars_grid(LMarg, args, n_lp, names_lp, limits_lp, n_grid)
    limits_lp = np.asarray(limits_lp)
    tol = [0 for i in range(n_lp)]
    for i in range(n_iter):
        
        ##  Recalculate the limits 
        for j in range(n_lp):
            if not limits_lp[j][2]:
                ##  Can change the upper limit
                limits_lp[j][0] += tol[j]
            if not limits_lp[j][3]:
                ## Cab change the lower limit
                limits_lp[j][1] -= tol[j]

        min_lps  = find_latpars_grid(LMarg, args, 
                                                   n_lp,
                                                   names_lp, 
                                                   limits_lp,
                                                   n_grid)
        
        tol = 4 * np.abs(limits_lp[,:2][1] - limits_lp[,:2][0])   /  np.asarray(n_grid)
    lps = np.array(min_lps[:n_lp])
    lp_diff = lps - ideals_lp
    ret = (lps, lp_diff) + (min_lps[-1],)


    return ret

def latpar_energy_range(LMarg, args, latpars, latpar):
	## range of the energy values of the lattice parameter
	print('Obtaining energies while varying %s in range %s -- %s' %(latpar, latpars[0], latpars[-1]))
	etot_list = []
	xargs = args + ' '
	for lp in latpars:
		xx_args = xargs + g.construct_cmd_arg(latpar, lp)
		#print(xx_args)
		etot = g.find_energy(LMarg, xx_args, 'pptest')
		if etot is not str:
			etot_list.append(etot)
	return np.asarray(etot_list)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def opt_latpars(LMarg, args, par_arr_p, a_u, a_l, alat_ideal, coa_u, coa_l, coa_ideal):
	##  Optimise the lattice parameters for the pair potential obtained
	print('\n    Optimising Lattice parameters \n  ')
	n = 50
	args += g.get_pp_args(par_arr_p) + ' '
	## Initially we have coa at the ideal value
	alatpars = np.linspace(a_l, a_u, n)
	alat_etots = latpar_energy_range(LMarg, args + ' -vcoa=' + str(coa_ideal) + ' ' , alatpars, 'alatTi')
	alat_vol = ( 3**(0.5) / 2. ) * ( alatpars**3 ) * coa_ideal
	alat_min = alatpars[ np.argmin(alat_etots) ]

	coalatpars = np.linspace(coa_l, coa_u, n)
	coalat_etots = latpar_energy_range(LMarg, args + ' -valatTi=' + str(alat_min) + ' ', coalatpars, 'coa')
	coalat_vol = ( 3**(0.5) / 2. ) * ( alat_min**3 ) * coalatpars

	##  Getting a list of alat parameters in the smaller range of the coa
	alatpars_t = (( coalat_vol / coa_ideal ) * ( 2. / (3**(0.5)) )   )**(1./3.)
	alat_etots_t = latpar_energy_range(LMarg, args + ' -vcoa=' + str(coa_ideal) + ' ' , alatpars_t, 'alatTi')

	#indxs = np.where( np.logical_and(alat_vol > coalat_vol[0], alat_vol <= coalat_vol[-1] ) )[0]
	#print(print(indxs), len(indxs), len(coalat_etots))
	#if len(indxs) < len(coalat_etots):
	#	indxs = np.append(indxs, indxs[-1] + 1)

	#print(len(indxs), len(coalat_etots))
	#alat_etots_t = alat_etots[indxs]  #and not np.all( alat_vol >= coalat_vol[-1])]#
	#, np.where(alat_vol > coalat_vol[-1] )[0] )	#alat_etots_t = np.delete(alat_etots_t, np.where(alat_vol > coalat_vol[-1] )[0] )

	#print(len(alat_vol_t), len(alat_etots_t), len(coalat_etots))

	#while len(alat_etots_t) != len(coalat_etots):
	#		print('Inalatwhle loop')
	#	idx = (np.abs(alat_vol - coalat_vol[-1])).argmin()
	#	if len(alat_etots_t) < len(coalat_etots):
	#		idx +=1
	#		alat_etots_t = np.append(alat_etots_t, alat_etots[idx])

	e_sub = alat_etots_t - coalat_etots

	##  Interpolate the points using a second order polynomial. Higher orders will not fit well. 
	##  acoeff[0] * x**2   +   acoeff[1] * x   +   acoeff[2]

	pcod = np.polyfit(coalat_vol, e_sub, 2) 
	
	min_vol =  np.sqrt( ( pcod[1]**2 ) - 4 * pcod[0] * pcod[2] )

	min_vol =  ( - pcod[1] + min_vol ) / ( 2. * pcod[0] )
	min_idx = (np.abs(coalat_vol - min_vol)).argmin()
	coa_min = coalatpars[min_idx]
	alat_min = alatpars_t[min_idx]

	print('\n Minimum lattice parameters')
	print('     alatTi = %s' %(alat_min))
	print('     coa    = %s' %(coa_min))
	print('     vol    = %s\n' %(min_vol))

	alat_diff = alat_min - alat_ideal
	print('alat difference = %s' %(alat_diff))
	coa_diff = coa_min - coa_ideal
	print('coa difference = %s' %(coa_diff))


	plotc = False
	if plotc == True:
		acoeff = np.polyfit(coalat_vol, alat_etots_t, 2) 
		coacoeff = np.polyfit(coalat_vol, coalat_etots, 2) 
		pcod_vol = np.linspace( 215 , 325, 100 )
		print('Coeffs:\n acoff = %s,\n coacoeff = %s,\n pcod = %s' %(acoeff, coacoeff, pcod))
		pcod_etots = pcod[0] * pcod_vol**2  +  pcod[1] * pcod_vol[1]  +  pcod[2]
		acoeff_etots = acoeff[0] * alat_vol**2  +  acoeff[1] * alat_vol[1]  +  acoeff[2]
		coacoeff_etots = coacoeff[0] * coalat_vol**2  +  coacoeff[1] * coalat_vol[1]  +  coacoeff[2]
		g.plot_function(	5, 
							[alat_vol, coalat_vol, pcod_vol, alat_vol, coalat_vol], 
							[alat_etots, coalat_etots, pcod_etots, acoeff_etots, coacoeff_etots], 
							['r-', 'g-', 'k--', 'r:','g:'], 
							'Alat and c/a fitting', 
							r'Volume ($ \AA ^ {3} $)', 
							'Energy (Ryd)')


	return alat_min, alat_diff, coa_min, coa_diff, min_vol


def lmu_latpar_energies(LMarg, args, latpar, l, u, m):
    xargs = args + ' '
    xx_args = xargs + g.construct_cmd_arg(latpar, m); etot_m = g.find_energy(LMarg, xx_args, 'pptest')
    xx_args = xargs + g.construct_cmd_arg(latpar, l); etot_l = g.find_energy(LMarg, xx_args, 'pptest')
    xx_args = xargs + g.construct_cmd_arg(latpar, u); etot_u = g.find_energy(LMarg, xx_args, 'pptest')
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
    min_coa, coa_diff = obtain_min_lattice_parameter_pp(LMarg, args, 
                                                            par_arr_p, coa_u , coa_l, tol_coa, 'coa', coa_ideal)
    min_alat1, alat_diff = obtain_min_lattice_parameter_pp(LMarg, args + ' -vcoa=' + str(min_coa) + ' ', 
                                                    par_arr_p, a_u, a_l,  tol_a, 'alatTi', alat_ideal)
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
