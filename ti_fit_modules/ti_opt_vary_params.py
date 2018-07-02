
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
