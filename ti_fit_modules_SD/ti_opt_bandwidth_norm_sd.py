# usr/bin/env/ python 
import numpy as np
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 
import ti_opt_general_sd as g
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
    g.cmd_write_to_file(cmd, 'out')
    cmd = "grep 'Fermi energy:' out | tail -1 | awk '{print $4}' | sed 's/.$//'"
    E_F = float( g.cmd_result(cmd)[:-2].strip() )
    return E_F

def band_width(LMarg, args, E_F, filename, symmpt, ext):
    band_calc(LMarg, args, E_F, filename)
    chk=check_bandcalc_resolves(filename)
    while len(chk) > 2:
        band_calc(LMarg, args, E_F, filename)
        chk=check_bandcalc_resolves(filename)
    bndfile = open('bnds' + ext, mode='r')
    d_width = width_symm_pt(bndfile, symmpt, False, [])
    bndfile.close()
    return d_width

def check_bandcalc_resolves( file):
    cmd = "grep 'Exit -1' " + file + " | tail -1"
    check = g.cmd_result(cmd)[0:-1]
    return check


def band_calc(LMarg, args, E_F, filename):
    cmd = LMarg + ' ' + args  +  ' -vef=' + str(E_F) +  ' -ef=' + str(E_F) + ' '  + '--band~fn=syml'
    g.cmd_write_to_file(cmd, filename)
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

def band_width_normalise( LMarg, xargs, symmpt, ext, ddnames, ddcoeffs, bond_int, bond_int_temp, evtol):
    ##  Bandwidth normalised with regards to the band width at a symmetry point defined by symmpt

    print( "\n Bandwidth Normalisation routine \n")

    dftfile = 'dftticol2'
    filename = 'out'
    dargs = g.construct_extra_args('', ddnames[:-1], ddcoeffs)
    d_norm =  g.construct_cmd_arg(ddnames[-1], bond_int)
    print(dargs + d_norm)
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
            d_norm1 =  g.construct_cmd_arg(ddnames[-1], bond_int1) 
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
    d_norm = g.construct_cmd_arg(ddnames[-1], bond_int1) 
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
                            col_wgts_1 = g.remove_bad_syntax( col_wgts_1 ) 
                            colwgts.append( col_wgts_1 )
                        elif i == 1:
                            col_wgts_2 = (lines[2 + 2*i] + ' ' +  lines[3 + 2*i]).split() 
                            col_wgts_2 = g.remove_bad_syntax( col_wgts_2 ) 
                            colwgts.append( col_wgts_2 )
            else:
                bandfile.readline()[0:-1]
        

    band_energies=(lines[0]  + ' ' + lines[1]).split()
    band_energies = g.remove_bad_syntax(band_energies)
    
    if (symmpt==7) and (dft):
        band_energies=band_energies[:12]
        band_energies.pop(6)
        band_energies.pop(6)

    return band_energies, colwgts
