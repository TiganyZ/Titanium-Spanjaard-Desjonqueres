# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 
import ti_opt_general_sd as g



##############################################################################################################
###################     Elastic Constant Routines     ########################################################


def elastic_constant_shear_info( LMarg, args, alpha, strain, nnid):
        """ Obtain information of the strain applied to the lattice, with a given small deformation, alpha.  
            xijc == coefficients of a given elastic constant in the strain 
            ijv  == the indices of the elastic constants in the strain
            dil  == dilatation  """

        cmd =  LMarg + ' ' + args  + ' -valpha=' + str(alpha) + ' '  + strain + ' ' 
        g.cmd_write_to_file(cmd, 'ecout')


        ##  Get alat from list 
        nn = g.get_nearest_neighbours(LMarg, args, filename)
        if nn != nnid: 
            if 'rmaxh' in args:
                rmaxh = float( g.find_arg_value('rmaxh', args) )
                print('args1 = %s' %(args))
                args = g.remove_arg('rmaxh', args)
                print('args2 = %s' %(args))

            else:
                alat = g.find_arg_value('alat', args)
                rmaxh = 1.001 * float(alat)

        
        rmaxh = g.check_rmaxh(LMarg, args +  ' -valpha=' + str(alpha) + ' '  + strain + ' ', 
                    'ecout', rmaxh, nnid) 
        ##  12 is for the 12 nearest neighours in ideal titanium

        args += ' -vrmaxh=' + str(rmaxh) + ' '

        cmd = "grep -A 4 'SHEAR: distortion' ecout"; g.cmd_write_to_file(cmd, 'shearinfo')

        if alpha != 0.0:
            cmd = "sed -n '5p' shearinfo"; xijc = g.cmd_result(cmd)[0:-1]
            cmd = "sed -n '4p' shearinfo"; ijv = g.cmd_result(cmd)  
            cmd = "sed -n '1p' shearinfo"; line_dil_alpha = g.cmd_result(cmd).split()  
            dil = float(line_dil_alpha[-1].replace("=",""))
        else:
                xijc='0' 
                ijv ='0'
                dil='0'

        cmd = "grep 'total energy' ecout | tail -1 | awk '{print $4}'"; etot = float(g.cmd_result(cmd)[0:-1])

        return [etot, alpha, xijc, ijv, dil, rmaxh]




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
    

def ec_alpha_poly(LMarg, args,  strain, plotcurve, alphalist, cell_vol, nnid):
    """ Calculates the elastic constants by fitting a fifth order polynomial to the energies obtained
        from each lattice strained with a small deformation alpha.
        Curvature at alpha = 0 is proportional to the elastic constants. (Energy per unit volume expansion)"""
    etot_list=[]

    for i in alphalist:
        etot, alpha, xijc, ijns, dil, rmaxh= elastic_constant_shear_info(LMarg, args, i, strain, nnid)
        etot_list.append(etot)
    
    ##  1 bohr = x angstrom
    bohr_to_angstrom = 0.529177208 
    ##  Rydberg to electron volts
    ryd = 13.606 
    etot_list = np.asarray(etot_list)/cell_vol
    ##  Converting to eV/AA**3
    etot_list = etot_list * (ryd / bohr_to_angstrom**3)

    ##  popt[0] * x**5    +     popt[1] * x**4     +     popt[2] * x**3     +     popt[3] * x**2    +     popt[4] * x     +     popt[5]
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
        g.plot_function(2, x, y, colour, 'Total Energy vs Alpha', 
                            'Alpha (Deformation parameter)', 'Energy (ev)')

    return curvature, ec_arr_row 






def ec_alpha_cds(LMarg, args,  strain, plotcurve, alphalist, cell_vol, nnid):
    """ Calculates the elastic constants by fitting a fifth order polynomial to the energies obtained
    from each lattice strained with a small deformation alpha.
    Curvature at alpha = 0 is proportional to the elastic constants. (Energy per unit volume expansion)
    """

    etot_list=[]
    ##  rmaxh = alatTi*1.0001
    for i in alphalist:
        etot, alpha, xijc, ijns, dil, rmaxh= elastic_constant_shear_info(LMarg, args, i, strain, nnid)
        if 
        args += ' -vrmaxh=' + str(rmaxh) + ' '
        etot_list.append(etot)
    
    ##  1 bohr = x angstrom
    bohr_to_angstrom = 0.529177208 
    ##  Rydberg to electron volts
    ryd = 13.606 
    etot_list = np.asarray(etot_list)/cell_vol
    ##  Converting to eV/AA**3
    etot_list = etot_list * (ryd / bohr_to_angstrom**3)

    ##  Curvature at alpha = 0

    dx = abs(   ( alphalist[-1] - alphalist[0] )  /  float( len(alphalist) )   )
    c_ind = np.argmin(etot_list)
    
    cds = (  etot_list[c_ind + 1]  -  2 * etot_list[c_ind]  +  etot_list[c_ind - 1]  ) / dx**2
    print('dx = %s, cind = %s, curvature = %s' %(dx, c_ind, cds))
    if plotcurve == True:
        afunclist=np.linspace(alphalist[0], alphalist[-1], 100)
  
        x = alphalist
        y = etot_list
        colour = 'r*'
        g.plot_function(1, x, y, colour, 'Total Energy vs Alpha, CDS', 
                            'Alpha (Deformation parameter)', 'Energy (ev)')
    return cds, []


##########################################################################################
########################     Elastic Constant Strain Routines      #######################

def Girshick_Elast(LMarg, args, alphal, cell_vol, ec_exp_arr, nnid):
    
    print('\n Girshick_Elast Routine \n')

    ##  Note: FATAL error (   Exit -1 FERMI: NOS ( 0.00000 0.00000 ) does not encompass Q = 8   )  
    ##  if the elastic constants are calculated at minimum alat and coa, they must be done at ideal values. 
    ##  But this makes curvatures off somewhat due to the real lower energy structure being at a lower energy,
    ##  such that if one shears in a direction where the energy of the structure is lower the curvature obtained if off. 
    ##  Hope it just converges regardless... 


    alphalist = alphal
    


    curvature = []
    strain11 =              ' -vexx=1' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature11, row11 = ec_alpha_poly(LMarg, args, strain11, False, alphal, cell_vol, nnid)  
    
    print(' C11 = %s' %(curvature11) )
    curvature.append(curvature11)                                                     #1

    strain112 =             ' -vexx=0' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature112, row112 = ec_alpha_poly(LMarg, args, strain112, False, alphal, cell_vol, nnid)   
    print(' C12 = %s' %(curvature112) )
    curvature.append(curvature112)                                                     #2

    strain33 =              ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature33, row33 = ec_alpha_poly(LMarg, args, strain33, False, alphal, cell_vol, nnid)   
    print(' C33 = %s' %(curvature33) ) 
    curvature.append(curvature33)                                                     #3
  
    strain2C112C22 =        ' -vexx=1.0' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature2C112C12, row2C112C22 = ec_alpha_poly(LMarg, args, strain2C112C22, False, alphal, cell_vol, nnid)   
    print(' 2*C11 + 2*C12 = %s' %(curvature2C112C12) ) 
    curvature.append(curvature2C112C12)                                                     #4
  
    strain5o4C11C12 =       ' -vexx=0.5' +\
                            ' -veyy=1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature5o4C11C12, row5o4C11C12 = ec_alpha_poly(LMarg, args, strain5o4C11C12, False, alphal, cell_vol, nnid)    
    print(' 5/4 * C11 + C12 = %s' %(curvature5o4C11C12) ) 
    curvature.append(curvature5o4C11C12)                                                     #5
  
    strainC11C332C13 =      ' -vexx=1' +\
                            ' -veyy=0' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvatureC11C332C13, rowC11C332C13 = ec_alpha_poly(LMarg, args, strainC11C332C13, False, alphal, cell_vol, nnid)    
    print(' C11 + C33 + 2*C13 = %s' %(curvatureC11C332C13) ) 
    curvature.append(curvatureC11C332C13)                                                           #6
  
    strainC11C332C132 =     ' -vexx=0' +\
                            ' -veyy=1' +\
                            ' -vezz=1' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvatureC11C332C132, rowC11C332C132 = ec_alpha_poly(LMarg, args, strainC11C332C132, False, alphal, cell_vol, nnid)    
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

    curvature4C44, row4C44 = ec_alpha_poly(LMarg, args, strain4C44, False, alphal, cell_vol, nnid)    
    print(' 4*C44 1 = %s' %(curvature4C44) ) 
    curvature.append(curvature4C44)                                                           #9
  
    strain4C442 =           ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=1' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature4C442, row4C442 = ec_alpha_poly(LMarg, args, strain4C442, False, alphal, cell_vol, nnid)    
    print(' 4*C44 2= %s' %(curvature4C442) ) 
    curvature.append(curvature4C442)                                                              #10
  
    strain8C44 =            ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=1' +\
                            ' -vexz=1' +\
                            ' -vexy=1 '

    curvature8C44, row8C44 = ec_alpha_poly(LMarg, args, strain8C44, False, alphal, cell_vol, nnid)    
    ##  For some reason, in tbe this is not simply 8C44, but 8*C44 + 4*C66... ? 
    print(' 8*C44,+ 2C11 - 2C12 = %s' %(curvature8C44) ) 
    curvature.append(curvature8C44)                                                           #11
  
    strain4C66 =            ' -vexx=1' +\
                            ' -veyy=-1' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=0 '

    curvature4C66, row4C66 = ec_alpha_poly(LMarg, args, strain4C66, False, alphal, cell_vol, nnid)    
    print(' 4*C66 1 = %s' %(curvature4C66) ) 
    curvature.append(curvature4C66)                                                               #12
  
    strain4C662 =           ' -vexx=0' +\
                            ' -veyy=0' +\
                            ' -vezz=0' +\
                            ' -veyz=0' +\
                            ' -vexz=0' +\
                            ' -vexy=1 '

    curvature4C662, row4C662 = ec_alpha_poly(LMarg, args, strain4C662, False, alphal, cell_vol, nnid)   
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
    c33 = 1.0   *   curvature[3-1] #/ (10**(9)) #Correct
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
    print('\n C11 = %.3f,   C11_FR = %.3f' %(c11, ec_exp_arr[0]) )
    print(  ' C33 = %.3f,   C33_FR = %.3f' %(c33, ec_exp_arr[1]) )
    print(  ' C44 = %.3f,   C44_FR = %.3f' %(c44, ec_exp_arr[2]) )
    print(  ' C44 = %.3f,   C44_FR = %.3f' %(c442, ec_exp_arr[3]) )
    print(  ' C66 = %.3f,   C66_FR = %.3f' %(c66, ec_exp_arr[4]) )
    print(  ' C12 = %.3f,   C12_FR = %.3f' %(c12, ec_exp_arr[5]) )
    print(  ' C13 = %.3f,   C13_FR = %.3f' %(c13, ec_exp_arr[6]) )

    print( ' K = %.3f,   K_FR = %.3f' %(kk, ec_exp_arr[7]))
    print( ' R = %.3f,   R_FR = %.3f' %(rr, ec_exp_arr[8]))
    print( ' H = %.3f,   H_FR = %.3f \n ' %(hh, ec_exp_arr[9]))
    print(  'C66 - 0.5(C11 - C12) = %.3f,   C66_FR = %.3f' %(c66 - 0.5 * (c11 - c12), ec_exp_arr[4]) )
    
    return  np.array([c11, c33, c44, c66, c12, c13]), np.array([c11, c33, c44, c66, c12, c13]) - np.asarray(ec_exp_arr[:-3])


