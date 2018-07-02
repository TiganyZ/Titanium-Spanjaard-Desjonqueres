#######################################################################################
###########################     General routines      #################################


def cmd_result(self, cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result,err = proc.communicate() 
    result = result.decode("utf-8")
    return result

def cmd_write_to_file(self, cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.run(cmd, shell=True, stdout = output_file)
    output_file.close()


def plot_function(self, n_plots, x, y, colour, title, x_name, y_name):
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


def remove_bad_syntax(self, values):
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


def construct_cmd_arg(self, arg_name, value):
    """arg_name is a string corresponding to the variable name with a value."""
    return ' -v' + arg_name + '=' + str(value) + ' ' 

def construct_extra_args(self, xargs, arg_names, arg_values):
    """Method to construct the extra arguments where arg_names is  a list of strings."""
    for i in range(len(arg_names)):
        arg = construct_cmd_arg(arg_names[i], arg_values[i])
        xargs +=  arg
    return xargs

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


#######################################################################################
###########################     Energy Routine      ###################################


def find_energy(self, LMarg, args, filename):
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



