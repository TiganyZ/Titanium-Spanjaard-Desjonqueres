import numpy as np 
import ti_opt_general_sd as g
#import ti_opt_main_sd_out as out


########################################################################################################################
###############################     Notes about gaussian_process_fit function     ######################################
    
"""
This routine is to find values for the pair potential and bond integrals that reproduce the target values. 

The input vector is x_n, and this consists of the pair potential and the bond integral coefficients (without the normalisation term.)


        The aim, with regards to TB, will be to find the ideal values for the pair potential and bond integrals. 
        We wish construct functions that describe how each of these parameters affects the target variables. 
        We can use gaussian process regression to find these functions. 
        Once we have found functions, we can then find the best value for that parameter given the target values. 
        
        We can vary one input parameter and then build functions that describe it best. 
        Consequently, once this has been done for all variables, we can find the best value using the minima. 
        

      Variables:

  x  : The input vector. These are the values that are mapped to get the target values. 
         t_n = y_n + eps_n  <--(Gaussian noise)

  X  : This is the array of previous input vectors x

  t: The target values that we want to fit pair potential and bond integrals to.

  T: The array of target values from previous iterations 

  K_: The array of Gram matrices for each target variable. 

  beta: The hyper parameters that are used to specify the confidence in a particular posterior. 
        This is 1./variance of that particular value. This is an array, one beta for each target variable.  
    
  theta: These arguments correspond to theta parameters for the parametric kernel 


     Notes:

  kernel: The kernel used here in the regression is the parametric kernel.  

"""


def gaussian_process_fit( npass, LMarg, args, ext,
                              t,     T,    K_, 
                              x,     X,
                           beta,  theta             ):


    if npass == 0:
        ##  Initialise
        t_      = np.array( [ ] )
        x_      = np.array( [  ] )
        K_      = np.zeros( (shape.t,shape.x,shape.x  )  )
        M       = np.zeros( len(t) )
        update  = False
    else:
        update  = True
    
    ##  Array that contains the means for each of the target variables. 
    M  = np.array([])   
    ##  Array that contains the variances for each of the target variables. 
    V  = np.array([])

    if not np.all(X[-1] == x):
        X = append_vector(X, x)
    
    for i in range( len( K_ ) ):

        m_p_tp1, var_p_tp1, K, C_N_inv = gaussian_process_regression(X[i,:], x[i], T[i,:], K_[i], beta[i], update)
    
        M = np.append( M,  m_p_tp1  )
        V = np.append( V, var_p_tp1 )

        K_[i] = K

    return M, V, K_


def gpr_mean_values(x_vals, x_, C_N_inv, t_):
    ##  From the mean and variance defining the distributions of the Gaussian process we can construct lists of the one-to-one 
    ##  mappings that describe how the function varies.
    m_p_tnpl = []
    for j in range(len(x_vals)):
        m_p_tnpl.append( m_pred_xnp1_sum(theta, x_, x_vals[j], C_N_inv, t_) )
    
    return m_p_tnpl

def append_vector( T, t_vec):
    if len(T) > 1:
        T = np.append(T, t_vec).reshape( (T.shape[0] + 1, T.shape[1]) )
    else:
        T = np.array([t_vec])
    return T  

###############################################################################################
##########################     Kernel Bayesian Regression     #################################


def get_next_k_(theta, x_, x_np1):
    ##  x_np1 is the next input vector for the distribution
    #   x_    is a vector of previous N input vectors. 

    return np.array([ parametric_kernel(theta, x_n, x_np1)  for x_n in x_ ])

def m_pred_next_target( k_, C_N_inv, t_):
    ##  Obtain the mean of the target predictive distribution

    return k_.T.dot(C_N_inv.dot(t_))

def var_pred_next_target(theta, x_np1, C_N_inv, k_, beta):
    ##  Obtain the variance of the target predictive distribution

    c =  parametric_kernel(theta, x_np1, x_np1) + 1./beta
    return c - k_.T.dot( C_N_inv.dot( k_ ))

def Gram_matrix(N, k_args, basis):
    K = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            if basis:
                K[n][m] = basis_kernel_nm(k_args[0], k_args[1], k_args[2]) 
                # k_args = [ phi_n, phi_m, alpha]
            else:
                K[n][m] = parametric_kernel(k_args[0], k_args[1], k_args[2]) 
                # k_args = [theta, x_n, x_m]
    return K

def K_matrix(theta, x, update, Kp):
    ##  Obtain the Gram matrix. 
    ##  If update == True, then the Gram matrix is updated with the last input vector x[-1]

    N = len(x)
    K = np.zeros((N,N))

    if update:
      
        K[:-1,:-1] = Kp
        for n in range(N):
            K[-1, n] = parametric_kernel(theta, x[n], x[-1])
            K[n, -1] = K[-1, n] 
    else:  
        for n in range(N):
                for m in range(N):
                    K[n][m] = parametric_kernel(theta, x[n], x[m])
    return K

def C_matrix( beta, K):
    # kernel (k_nm) defines the Gram Matrix K , giving the covariance matrix C
    # This is the covariance matrix for the marginal distribution over targets p(t_) = N(t_|0_, C)

    return K + 1./beta * np.eye(len(K))

def m_pred_xnp1_sum(theta, x_, x_np1, C_N_inv, t_):
    ##  This obtains the predictive distribution of the mean for a given C_N_inv matrix. 
    ##  This can be used to obtain the fits of the Gaussian regression. 
    Ctn = C_N_inv.dot( t_ )
    kn  = np.array( [ parametric_kernel(theta, xn, x_np1) for xn in x_  ] ) 
    return Ctn.dot(kn) 

def gaussian_process_regression(x_, x_np1, t_, K, beta, theta, update):
    ## Obtaining the mean and variance of the predictive target distribution p( t_{n+1} | t_, x_, x_np1) 

    k_              =  get_next_k_( theta, x_, x_np1    )
    K               =  K_matrix(    theta, x_, update, K)
    C               =  C_matrix(    beta,  K            )
    C_N_inv         =  np.linalg.inv(      C            )

    m_pred_tnp1     =  m_pred_next_target(   k_, C_N_inv, t_  )
    var_pred_tnp1   =  var_pred_next_target( theta, x_np1, C_N_inv, k_, beta)

    return m_pred_tnp1, var_pred_tnp1, K, C_N_inv




########################################################################################
###################  Test for gaussian process regression   ############################

def bayesian_check_process(iters, n, noise):

    x = np.linspace(0,10, n)
    t = np.sin(4*x) + 3 * np.cos(x) + 1.5 * np.cos(10 * x) + np.random.normal(0, noise, n)
    ##  This is the target data
    y = np.sin(4*x) + 3 * np.cos(x) + 1.5 * np.cos(10 * x)
    
    alpha   = 10
    beta    = 1./noise**2

    T       = np.array([])
    Phi     = np.array([])
    xrlist  = []
    yrlist  = []
    ind     = np.random.choice( range( len(y) ) )
    ind2     = np.random.choice( range( len(y) ) )
    xr      = x[ind]
    yr      = y[ind]
    t_      = np.array( [ ] )
    x_      = np.array( [  ] )
    K       = []
    for knt in range(iters):
        print('Gaussian process regression: Iteration  %s' %(knt))
        ind = np.random.choice( range( len(y) ) )
        xn= x[ind]
        yn   = y[ind]
        x_  = np.append( x_, xn  )
        t_  = np.append( t_, t[ind] )
        if knt == 0:
            update = False
        else:
            update = True

        m_p_tnp1, var_p_tnp1, K, C_N_inv = gaussian_process_regression(x_, xn, t_, K, beta, update)
        #t_np1_ = np.random.normal(m_p_tnp1, var_p_tnp1, 6)

        xrlist.append(xr)
        yrlist.append(yr)
        ybayes  = np.array([]); ybayes2 = np.array([]); ybayes3 = np.array([])
        ybayes4 = np.array([]); ybayes5 = np.array([]); ybayes6 = np.array([])
        for j in range( len(x)):
            #m_p_tnp1, var_p_tnp1, Kx = gaussian_process_regression( np.append(x_, x[j]), x[j], np.append(t_,t[j]), K, beta, update)
            m_p_tnp1 = m_pred_xnp1_sum(theta, x_, x[j], C_N_inv, t_)
            t_np1    = np.random.normal(m_p_tnp1, var_p_tnp1, 6)

            ybayes  = np.append(ybayes,  m_p_tnp1); ybayes2 = np.append(ybayes2, t_np1[1]); ybayes3 = np.append(ybayes3, t_np1[2])
            ybayes4 = np.append(ybayes4, t_np1[3]); ybayes5 = np.append(ybayes5, t_np1[4]); ybayes6 = np.append(ybayes6, t_np1[5])         
            
        print( len(x), len(t) , len(x_), len(t_), len(x), len(ybayes2))
        print( x,t ,x_, t_, ybayes2)
        print( x.shape,t.shape ,x_.shape, t_.shape, ybayes2.shape, K.shape)
        xp =     [x, x, x_, x,      x,       x,       x,       x,       x      ]
        yp =     [t, y, t_, ybayes, ybayes2, ybayes3, ybayes4, ybayes5, ybayes6]
        colour = ['r--', 'g--', 'b^', 'k-', 'b-', 'b-', 'b-', 'b-', 'b-']
        if knt %1 == 0:
            g.plot_function(4, xp, yp, colour, 'Gaussian Process regression.', 
                                'x parameter', 'y')
    return M



################################################################################
##########################     Kernel Methods     ##############################
### The kernel function k(x_n, x_m) expresses how strongly y(x_n) and y(x_m)
### are correlated for similar points of x_n, x_m

def basis_kernel_nm(phi_n, phi_m, alpha):
    return ( 1./alpha ) * phi_n.dot(phi_m)

def parametric_kernel(theta, x_n, x_m):
    if isinstance(x_n, np.ndarray):
        xx = x_n.dot(x_m)
    else:
        xx = x_n * x_m
    return theta[0] * np.exp( -(theta[1]/2.)*np.linalg.norm(x_n - x_m)**2 ) + theta[2] + theta[3] * xx

def dC_dtheta_i(C_N, C_N_min_1, theta_i_N, theta_i_N_min_1):
    return (C_N - C_N_min_1)/(theta_i_N - theta_i_N_min_1)

def log_likelihood_t_g_theta( N, C_N, C_N_inv, theta_i, t_):
    return -0.5 * np.log(np.abs(C_N)) - 0.5 * t_.T.dot( C_N_inv.dot(t_) ) - N/2. * np.log(2*pi)

def d_log_likelihood_t_g_thetai( N, C_N, theta_i, t_):
    dC_dtheta_i = dC_dtheta_i(C_N, theta_i)
    return -0.5 * np.trace( C_N_inv.dot( dC_dtheta_i ) ) + 0.5 * t_.T.dot( C_N_inv.dot( dC_dtheta_i.dot( C_N_inv.dot(t_) ) ) ) 



#############################################################################################
##########################     Stochastic Gradient Descent     ##############################


def stoc_grad_des_mom(deltaw, gamma, eta, Qi_list):
    Q_i = np.random.choice(range(len(Qi_list))) #This is the change in the cost function at each step 
    Q_i = Qi_list[Q_i]
    newdeltaw = gamma * deltaw - eta * Q_i 
    deltaw = newdeltaw
    return newdeltaw



def fpoly_reg(b, deg, reg):
    ##  Have the equation Ax = b that needs to be solved in a polynomial basis 
    ##  This constructs the Vandermond matrix with regularisation specified by reg 
    ##  deg is the degree of the polynomial for the fitting. 
    ##  With regularisation (e.g. Weight decay/ridge/quadratic) this becomes
    ##  x = np.inv( (A.T.dot(A) + G.T.dot(G)) ).dot( A.dot(b) )
    ##  Where G is the Tikhonov matrix which for quadratic regularisation is alpha * I 
    return 
    



#######################################################################################
##########################     Conjugate Gradient     #################################

def conjugate_gradient(self, A, x_k, b, tol):
    #This solves the equation Ax = b
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




#iters = 200
#n=100
#theta = np.array( [1., 4*4., 0., 0.  ] )
#noise = 0.3
#deg = 12
#bayesian_check_process(iters, n, noise)

#deg = 5
#bayesian_check(iters, n, noise, deg)


