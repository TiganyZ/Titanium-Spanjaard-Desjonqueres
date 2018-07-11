# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 
import ti_opt_general_sd as g
import ti_opt_elastconst_sd as ec
import ti_opt_latpar_min_sd as lm
import ti_opt_constraints_variation_sd as cv
import ti_opt_bandwidth_norm_sd as b

def output_script(  npass, 
                    ext,
                    LMarg, 
                    args, 
                    par_arr_p, ppnames, 
                    ddcoeffs_p, ddnames, ddnorm_lim,
                    ec_exp_arr, 
                    rmx_name, nn_ideal,
                    n_lp, n_grid, n_iter,
                    names_lp, limits_lp, ideals_lp,
                    n_energies, energy_args):

    """
    This script, given a pair potential and bond integrals, gives alat and c/a, the elastic constants, 
    the energies of different structures, such as fcc, bcc, hcp and omega phase. 
    """

    npass += 1

    print('\n    Output Script Routine     npass = %s \n' %(npass))

    print('Pair potential  = %s' %(par_arr_p) )
    print('ddx Bond Integrals = %s' %(ddcoeffs_p) )


    ###################################################################
    ################     Initial Args     #############################
  
    symfile =   'syml'
    LMarg   +=  ' ctrl' + ext + ' '

    ppargs = g.construct_extra_args('', ppnames, par_arr_p)
    symmpt = 0
    bond_int_u, bond_int_l, evtol  =   ddnorm_lim

    alphal = np.linspace(-0.01, 0.01, 11)


    

    ###############################################################################
    ##################      Scale dddelta coefficients      ###################

    ##  Do not need to have pair potential in place to normalise the ddd coefficients. 

    d_norm, E_F = b.band_width_normalise( LMarg, args, 
                                          symmpt, ext, 
                                          ddnames,  ddcoeffs_p, 
                                          bond_int_u, bond_int_l, evtol)

    dargs = g.construct_extra_args('', ddnames[:-1], ddcoeffs_p) + d_norm + E_F

    #################################################################################
    ##################      Obtain optimum lattice parameters      ##################

    min_lps  = lm.opt_latpars_grid( LMarg, 
                                    args + ppargs + dargs, 
                                    n_lp, names_lp,
                                    limits_lp, ideals_lp, 
                                    n_grid, n_iter)
        
    print('minlps', min_lps)
    for i in range(n_lp):
        if i == 0:
            lp_args  = g.construct_cmd_arg( names_lp[i], min_lps[i] )
            lp_diffs = (min_lps[n_lp + i], )  
        else:
            lp_args  += g.construct_cmd_arg( names_lp[i], min_lps[i] )
            lp_diffs += ( min_lps[n_lp + i], )
            
    ################################################################################################
    ###################     Energies of different structures    ####################################
    
    etot = g.find_energy( LMarg, args + ppargs + dargs + lp_args, 'etot')
    etots = (etot,)
    for i in range(n_energies):
        etot_ex = g.find_energy( LMarg, args + ppargs + dargs + energy_args[i], 'ex_etot')
        etots   += (etot_ex,)
        
    args += ppargs + dargs + lp_args


    print ('\n Arguments \n %s' %(args) )


    ###############     Cell volume at equilibrium c_lat and a_lat     ##########
    print(' Obtaining Cell Volume at ideal c and a \n' )

    filename = 'voltest'
    cmd      = LMarg + ' ' + args  
    g.cmd_write_to_file( cmd, filename )

    cell_vol = float(g.cmd_result( "grep 'Cell vol' " + filename + " | awk '{print $7}'" ))
    print('Cell_Vol = %s' %(cell_vol) )

    ################     Get Elastic Constants     #####################
    print(' Obtaining Elastic Constants at optimised lattice parameters \n' )

    e_consts, e_consts_diff = ec.Girshick_Elast(LMarg, args, alphal, cell_vol, ec_exp_arr, rmx_name, nn_ideal)


    print('Elastic Constants: difference = %s' %(e_consts_diff))


    return  (npass, args) + min_lps[:n_lp] + lp_diffs + (cell_vol,) +  etots + (e_consts, e_consts_diff)



