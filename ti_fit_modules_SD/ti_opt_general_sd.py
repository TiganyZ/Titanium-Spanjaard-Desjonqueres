# usr/bin/env/ python   
import numpy as np
#try:
#    # for Python2
#    from Tkinter import *   ## notice capitalized T in Tkinter 
#except ImportError:
#    # for Python3
#    from tkinter import * 
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

def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout = output_file)
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



##########################################################################################
###########################     Argument routines      ###################################

def find_arg_value(arg_to_find, args):
    arg_ind = np.where( [arg_to_find in i for i in args.split()] )[0][0]
    arg = args.split()[arg_ind]
    if '=' in arg:
        arg = arg.replace("=", "= ").split()[1]
    if not arg[0].isdigit():
        arg = arg[1:]
    return arg

def find_arg_ind(arg_to_find, args):
    arg_ind = np.where( [arg_to_find in i for i in args.split()] )[0][0]
    return arg_ind

def remove_arg(arg_to_remove, args):
    rind = find_arg_ind(arg_to_remove, args)
    args = args.split()
    del args[rind]
    args = ' '.join(args)
    return args

def construct_cmd_arg(arg_name, value):
    """arg_name is a string corresponding to the variable name with a value."""
    return ' -v' + arg_name + '=' + str(value) + ' ' 

def construct_extra_args(xargs, arg_names, arg_values):
    """Method to construct the extra arguments where arg_names is  a list of strings."""
    for i in range( len(arg_names) ):
        xargs += construct_cmd_arg(arg_names[i], arg_values[i])
    return xargs

def get_pp_args(par_arr_p):

    A1TTSDpp = par_arr_p[0] 
    B1TTSDpp = 0 
    C1TTSDpp = par_arr_p[1]
    A2TTSDpp = par_arr_p[2] 
    B2TTSDpp = 0
    C2TTSDpp = par_arr_p[3]

    ppargs = ' -vA1TTSDpp=' + str(A1TTSDpp) + \
                        ' -vB1TTSDpp=' + str(B1TTSDpp) + \
                        ' -vC1TTSDpp=' + str(C1TTSDpp) + \
                        ' -vA2TTSDpp=' + str(A2TTSDpp) + \
                        ' -vB2TTSDpp=' + str(B2TTSDpp) + \
                        ' -vC2TTSDpp=' + str(C2TTSDpp) + ' ' 
    return ppargs


#######################################################################################
###########################     Energy Routine      ###################################


def find_energy(LMarg, args, filename):
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
        cmd = "grep 'Exit' " + filename + " "
        error = cmd_result(cmd)
        print( str(error) )
        print( ' Error: \n       ' + str(error) + ' From file ' + filename +  ' \n Exiting...' )
        etot = 'error'
        
    
    return etot

######################################################################################################################
###########################     Checking the neighbour list and setting rmaxh      ###################################


def get_nearest_neighbours(LMarg, args, filename):
    cmd =  LMarg + ' ' + args  + ' ' 
    cmd_write_to_file(cmd, filename)
    cmd = "grep 'pairc,' " + filename + "| tail -1 | awk '{print $4}'"
    res = cmd_result(cmd)  
    return int(res)

def check_rmaxh(LMarg, args, filename, rmx_name,  rmaxh, nmax):

    res = get_nearest_neighbours(LMarg, args + construct_cmd_arg(rmx_name, rmaxh), filename) 
    cond = (int(res) - 1) is nmax

    if cond == False:
        iters = 0
        rmaxh_u = 2 * rmaxh
        rmaxh_l = 0.5 * rmaxh
        resu = get_nearest_neighbours(LMarg, args + construct_cmd_arg(rmx_name, rmaxh_u), filename)
        print('\n Initial Neighbours\n   rmaxh_l = %s, rmaxh_u = %s \n    nn_l = %s,     nn_u = %s' %(rmaxh_l, rmaxh_u, res, resu))
        
    while cond == False:
        iters += 1
        rmaxh_m = ( rmaxh_u + rmaxh_l) / 2.
        resm = get_nearest_neighbours(LMarg, args + construct_cmd_arg(rmx_name, rmaxh_m), filename)
        print('RMAXH binary search:\nrmaxh = %s, iteration = %s, nn_m = %s\n' %(rmaxh_m, iters, int(resm)))
        if int(resm) - 1 < nmax:
            ##  Must increase rmaxh to get the right number of neighbours
            rmaxh_l = rmaxh_m
        if int(resm) - 1 > nmax:
            ##  Must decrease rmaxh to get the right number of neighbours
            rmaxh_u = rmaxh_m

        res = get_nearest_neighbours(LMarg, args + construct_cmd_arg(rmx_name, rmaxh_m), filename)
        cond = (int(res) - 1) is nmax
        rmaxh = rmaxh_m
    return rmaxh


########################################################################################################
###########################     Mean and Covariance of Data set      ###################################


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



