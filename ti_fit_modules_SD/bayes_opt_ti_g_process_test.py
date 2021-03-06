import numpy as np 
import ti_opt_general_sd as g
#import ti_opt_main_sd_out as out






#####################################################################################################
##########################     Linear Basis Bayesian Regression     #################################


def bayes_lin_regress( M, S, S_inv, 
                                    W, t, T, 
                                            phi, Phi, 
                                                    alpha, beta, 
                                                                oneD, reg, update ):
    ##  This function goes through one pass of a bayesian regression algorithm. 
    ##  M is the mean values used in the multivariate gaussian defining the prior over parameters. 
    ##  S is the covariance matrix ---  ""  ---
    ##  W is the matrix of parameters. A row corresponds to one specific target value which is constructed from the design matrix. 
    ##  Phi is the design matrix which contains the information of the basis functions upon which we are doing the regression. 
    ##  alpha and beta define the amount of confidence we have in the prior and posterior distributions. 
    
    S0_inv = S_inv 
    
    Phi = append_vector( Phi, phi )
    T   = append_vector( T  , t   )
    
    M, S, S_inv, W = W_posterior_update(alpha, beta, Phi, T, W, S0_inv, reg)
    ##  This is updates the gaussian distribution (updates to the mean and covariance matrices), and the parameter matrix

    if update:
        alpha, beta = get_alpha_beta_sc(alpha, beta, Phi, M, T, oneD)
        ##  Updating the confidence in the prior and posterior distributions self consistently. 
    
    return M, S, S_inv, W, T, Phi, alpha, beta 
  

def append_vector( T, t_vec):
    if len(T) > 1:
        T = np.append(T, t_vec).reshape( (T.shape[0] + 1, T.shape[1]) )
    else:
        T = np.array([t_vec])
    return T  


def W_posterior_update(alpha, beta, Phi, T, W, S0_inv, wreg):
    ##  S0 is an array of covariance matrices corresponding to each row in the W parameter matrix.  
    ##  If single is true then T = t_ is just one vector and W = w_
    if len(S0_inv.shape) < 3:    
        if len(T.shape) > 1:
            T = T[0]
            
        print(' \n Phi = %s\n\n T = %s\n\n W = %s\n\n S0_inv = %s\n' %(Phi, T, W, S0_inv) )

        S, S_inv = S_N_wgt(Phi, alpha, beta, S0_inv)
        M = m_N_wgt( S, S0_inv, Phi, T, beta, W)
        ##  These parameters define the prior distribution over the w parameters given target values. 
        if wreg:
            W = w_ML_reg(Phi, T, alpha/beta)#beta/alpha)
        else:
            W = np.random.multivariate_normal(mean=M, cov=S)
    else:
        for i in range(len(S0_inv)):

            if not isinstance(alpha, np.ndarray):
                s_N, s_N_inv = S_N_wgt(Phi, alpha, beta, S0_inv[i])
                m_N = m_N_wgt( s_N, S0_inv[i], Phi, T[:,i], beta, m0=W[i,:])
            else:
                s_N, s_N_inv = S_N_wgt(Phi, alpha[i], beta[i], S0_inv[i])
                m_N = m_N_wgt( s_N, S0_inv[i], Phi, T[:,i], beta[i], m0=W[i,:])

            if i == 0:
                M = np.array([m_N])
                S = np.array([s_N])
                S_inv = np.array([s_N_inv])
            else:
                M = np.append( M, m_N ).reshape(M.shape[0] + 1, M.shape[1])
                S = np.append( S, s_N ).reshape(S.shape[0] + 1, S.shape[1], S.shape[2])
                S_inv = np.append( S_inv, s_N_inv ).reshape(S_inv.shape[0] + 1, S_inv.shape[1], S_inv.shape[2])

            if wreg:
                W[i,:] = w_ML_reg(Phi, T[:,i], 0.5)
            else:
                W[i,:] = np.random.multivariate_normal(mean=m_N, cov=s_N)
        
    return M, S, S_inv, W

###########################################################################################
##############   Mean and covariance for posterior/prior over w parameters    #############

def S_N_wgt(Phi, alpha, beta, S0_inv):  #This is the S matrix for the posterior distribution over w, where the prior is p(w|alpha)=N(w|0, 1/alpha *I)  
    
    Sn_inv = S0_inv  +  beta * Phi.T.dot(Phi)
    Sn = np.linalg.inv(Sn_inv)
    return Sn, Sn_inv

def m_N_wgt( S_N, S0_inv, Phi, t_, beta, m0): #This is the mean for the posterior distribution over w, where the prior is p(w|alpha)=N(w|0, 1/alpha *I)  
    m =  S0_inv.dot(m0) + beta * Phi.T.dot(t_) 
    return S_N.dot( m )



###########################################################################################
##################################   Basis functions    ###################################
def get_poly_phi(a, x, mu, deg):
    return np.sum([ a[i] *(x - mu)**i for i in range(deg + 1) ] )


def gauss_basis(x, mu_i, s):
    return np.exp( -(x - mu_i)**2 / (2 * s**2) )


#####################################################################################################
##############   Self consistent updates to the precision parameters, alpha and beta    #############


def get_alpha_beta_sc(alpha, beta, Phi, M_N, T, oneD):
    #Chosen initial value of alpha and then will update it using the eigenvectors of the Phi matrix
    #M_N is the most likely value of parameters for W
    #T must by a N x K matrix, Phi must be a N x M matrix, M_N must be a M x K matrix
    tol = 0.1
    Phi_e = np.linalg.eigvals( ( Phi.transpose() ).dot( Phi ) ) 
    print('Phi.T.dot( Phi )  eigenvalues')
    print(Phi_e)
    Phi_e = np.real(Phi_e) 
    print('Phi', Phi)
    tdiff = True
    if oneD:
        iters = 30
        while tdiff:
            iters -= 1
            Phi_eig = beta * Phi_e
            gamma = np.sum( [ lmda / (lmda + alpha)  for lmda in Phi_eig ] ) 

            alpha_new   = gamma / ( M_N.dot( M_N ) )
            beta_new    = ( 1./ (len(T) - gamma) ) * np.sum( [ ( T[i] - M_N.T.dot( Phi[i] ) )**2 for i in range( len(T) ) ]  ) 
            beta_new = 1./ beta_new
            a_diff = np.linalg.norm( alpha_new - alpha );  b_diff = np.linalg.norm( beta_new  - beta  )
            tdiff = (b_diff + a_diff) > tol
            alpha = alpha_new ; beta = beta_new
            print('   alpha_new = %s,\n   beta_new = %s' %(alpha_new, beta_new) )
            tdiff = False

            if iters < 0:
                ##  Breaking out of the loop as there has been too much time spent here... 
                print( 'Breaking out of the self-consistency loop for alpha and beta' )
                #alpha = (alpha + alpha_new)/2.
                #beta = (beta + beta_new)/2.
                break 
    else:
        for j in range( len(beta) ):
            tdiff = True
            iters = 20
            while tdiff:
                iters -= 1
                Phi_eig = beta[j] * Phi_e
                gamma = np.sum( [ Phi_eig[i]/(Phi_eig[i] + alpha[j])  for i in range(len(Phi_eig))] ) #vector of gamma 
                print('gamma',gamma)
                alpha_new = gamma/M_N[j,:].dot(M_N[j,:].T)
                
                beta_new =  1. / ( (1./(len(T) - gamma)) * sum([ ( T[i, j] - M_N[j,:].dot(Phi[i,:]) )**2 for i in range(len(T)) ]) )
                a_diff = np.linalg.norm(alpha_new-alpha[j]) ; b_diff = np.linalg.norm(beta_new-beta[j])
                tdiff = b_diff + a_diff > tol

                print('alpha_new=%s, beta_new=%s' %(alpha_new, beta_new) )
                print('adiff, bdiff', a_diff, b_diff)
                alpha[j] = alpha_new ; beta[j] = beta_new
                if iters < 0:
                    ##  Breaking out of the loop as there has been too much time spent here... 
                    print( 'Breaking out of the self-consistency loop for alpha and beta' )
                    alpha[j] = (alpha[j] + alpha_new)/2.
                    beta[j] = (beta[j] + beta_new)/2.
                    break 

    return alpha, beta

def w_ML_reg(Phi, t_, lamb):
    pp = Phi.T.dot( Phi )
    l =  lamb * np.eye(pp.shape[0])
    print('pp', pp)
    print('llllllllllllllll', pp + l)
    try:
        inv = np.linalg.inv( pp + l) 

    except np.linalg.linalg.LinAlgError:
        print('raised exception',Phi, lamb, Phi.T.dot(Phi))
        inv = np.linalg.pinv( lamb + Phi.T.dot( Phi ) )
    return inv.dot( Phi.T.dot(t_) )


def poly(a, x):
    res = 0
    for i in range( len(a) ):
        res += a[i] * x**i
    return res

def ref_func(x):
    return  np.sin(x) #+ 0.3 * np.cos(5 * x)  #oly([1,2,3,4,5,6,7,8],x, 7)#

def ref_func_poly(x):
    return poly([11,-2,2],x)

def bayesian_check(iters, n, noise, deg):

    x = np.linspace(0,10, n)
    t = ref_func_poly(x)
    ##  This is the target data
    y = t + np.random.normal(0, noise, n)
    
    alpha   = 10
    beta    = 1./noise**2
    M       = np.ones(deg + 1)
    W       = np.ones(deg + 1)
    T       = np.array([])
    Phi     = np.array([])
    S       = np.diag( [ 1. / beta for i in range(deg + 1) ] ) 
    S_inv   = np.diag( [  beta for i in range(deg + 1) ] )
    xrlist  = []
    yrlist  = []
    mlist   = []
    oneD    = True
    reg     = False
    update  = False
    s       = 1
    for k in range(iters):
        print(W)
        print('Iter  %s' %(k))
        ind = np.random.choice( range( len(y) ) )
        xr  = x[ind]
        yr  = y[ind]
        tr  = [t[ind]]
        phi = np.array( [ xr**i for i in range(deg + 1) ] )
        M, S, S_inv, W, T, Phi, alpha, beta = bayes_lin_regress( M, S, S_inv, 
                                                                 W, tr, T, 
                                                                 phi, Phi, 
                                                                 alpha, beta, oneD, reg, update )
        W2 = np.random.multivariate_normal(M, S)
        W3 = np.random.multivariate_normal(M, S)
        xrlist.append(xr)
        yrlist.append(yr)
        ybayes = poly(W, x)  #poly(W, x, deg)
        ybayes2 = poly(W2, x)
        ybayes3 = poly(W3, x)
        print('parameter matrix = %s' %(W) )
        xp =     [x, x, xrlist, x,      x,       x      ]
        yp =     [t, y, yrlist, ybayes, ybayes2, ybayes3]
        colour = ['r--', 'g--', 'b^', 'k-', 'r-', 'r-']
        if k %1 == 0:
            g.plot_function(6, xp, yp, colour, 'Bayesian Linear regression.', 
                                'x parameter', 'y')
    return M

def bayesian_check_gauss(iters, n, noise, deg):

    x = np.linspace(0,10, n)
    t = ref_func(x)
    ##  This is the target data
    y = t + np.random.normal(0, noise, n)
    
    alpha   = 10
    beta    = 1./noise**2
    M       = np.arange(deg + 1) 
    Mu      = np.arange(deg) 
    W       = np.ones(deg + 1)
    T       = np.array([])
    Phi     = np.array([])
    S       = np.diag( [ 1. / beta for i in range(deg + 1) ] ) 
    S_inv   = np.diag( [  beta for i in range(deg + 1) ] )
    xrlist  = []
    yrlist  = []
    mlist   = []
    oneD    = True
    reg     = False
    update  = False
    s       = 1
    for k in range(iters):
        print(W)
        print('Iter  %s' %(k))
        ind = np.random.choice( range( len(y) ) )
        xr  = x[ind]
        yr  = y[ind]
        tr  = [t[ind]]
        phi = np.append( np.asarray( [ gauss_basis( xr, Mu[i], s ) for i in range(deg)  ] ), 1. )
        phi = np.roll(phi, 1)

        M, S, S_inv, W, T, Phi, alpha, beta = bayes_lin_regress( M, S, S_inv, 
                                                                 W, tr, T, 
                                                                 phi, Phi, 
                                                                 alpha, beta, 
                                                                 oneD, reg, update )
        W2 = np.random.multivariate_normal(M, S)
        W3 = np.random.multivariate_normal(M, S)
        W4 = np.random.multivariate_normal(M, S)
        W5 = np.random.multivariate_normal(M, S)
        W6 = np.random.multivariate_normal(M, S)
        xrlist.append(xr)
        yrlist.append(yr)
        ybayes = []; ybayes2 = []; ybayes3 = []
        ybayes4 = []; ybayes5 = []; ybayes6 = []
        for j in range( len(x)):
            app =  np.asarray([ gauss_basis( x[j], Mu[i], s ) for i in range(deg) ])
            app =  np.append(app, 1.); app = np.roll(app, 1)
            prod = W.dot(app);   prod2 = W2.dot(app);   prod3 = W3.dot(app)
            ybayes.append(prod); ybayes2.append(prod2); ybayes3.append(prod3)
            prod = W4.dot(app);   prod5 = W2.dot(app);   prod6 = W3.dot(app)
            ybayes4.append(prod); ybayes5.append(prod2); ybayes6.append(prod3)    
        print('parameter matrix = %s' %(W) )
        xp =     [x, x, xrlist, x,      x,       x,       x,       x,       x      ]
        yp =     [t, y, yrlist, ybayes, ybayes2, ybayes3, ybayes4, ybayes5, ybayes6]
        colour = ['r--', 'g--', 'b^', 'b-', 'b-', 'b-', 'b-', 'b-', 'b-']
        if k %2 == 0:
            g.plot_function(9, xp, yp, colour, 'Bayesian Linear regression.', 
                                'x parameter', 'y')
    return M
    

######################################################################################
######################################################################################


def init_bayes(   t_n, res, 
                  x_n,
                  kernel, basis, error_function,
                  alpha, beta):
        
    if   kernel == 'Matern':
        ##  Use the Matern Kernel 
        k = matern_kernel(theta, x_n, x_np1)

    elif kernel == 'parametric':
        k = parametric_kernel(theta, x_n, x_np1) 

    else: 
        ##  kernel == False:
        ##  Use the Gram Matrix construction for the covariance matrix
        Gram_matrix( len(x_n), k_args, basis)
        

def bayes_fit( npass, LMarg, args, ext, 
                      t, T, y, 
                      x, X,
                      kernel, basis, error_function,
                      alpha, beta, 
                      k_args ):
    """
    This routine is to find values for the pair potential and bond integrals that reproduce the target values. 
    
    The input vector is x_n, and this consists of the pair potential and the bond integral coefficients (without the normalisation term.)
    
          Variables:

      x  : The input vector. These are the values that are mapped to get the target values. 
             t_n = y_n + eps_n  <--(Gaussian noise)
             y_n = w_T . phi(z) <--(Basis function of z: the pair potential/bond integral values used)

                                Generally, with regards to TB, this will be the pair potential and bond integrals. 
                                This is a non-linear mapping, but using a linear combination of basis fuctions, 
                                one can reconstruct a function which, at some point, will be able to describe how 
                                variation of each parameter in the pair potential and the bond integrals changes the target variables.
                                

      X  : This is the array of previous input vectors x

      t: The target values that we want to fit pair potential and bond integrals to.

      T: The array of target values from previous iterations

      y: Results of the target quantities from from input pair_pot and bond integrals input. 

      alpha, beta: The hyper parameters that are used to specify the confidence in a particular prior/posterior
    
      kernel: A string such that we can choose the appropriate kernel for Gaussian process regression
              kernel can be Matern, or Gaussian.
              If kernel == False then we use linear bayesian regression with a basis given by basis.

      basis:  The specification of the basis set used for Bayesian regression. This can be identity or a linear combination of 
              basis functions (e.g. Gaussian, polynomial or sigmoid etc.) 
      
      error_function: Specifies the degree of regularisation used in the error function. [ coeff, reg_degree]
                      In general the quadratic regulariser will give the best results as it has a closed form solution. 
                      E_t  =  np.sum( t_n  -  w_T . phi )  + 0.5 * w_T.w_
                      can have the 0.5 factor to be alpha/beta hyperparameters

      k_args: These arguments correspond to theta parameters for the parametric kernel or other 
    """
    return 

#############################################################################################
##########################     Stochastic Gradient Descent     ##############################


def stoc_grad_des_mom(deltaw, gamma, eta, Qi_list):
    Q_i = np.random.choice(range(len(Qi_list))) #This is the change in the cost function at each step 
    Q_i = Qi_list[Q_i]
    newdeltaw = gamma * deltaw - eta * Q_i 
    self.deltaw = newdeltaw
    return newdeltaw



def fpoly_reg(b, deg, reg):
    ##  Have the equation Ax = b that needs to be solved in a polynomial basis 
    ##  This constructs the Vandermond matrix with regularisation specified by reg 
    ##  deg is the degree of the polynomial for the fitting. 
    ##  With regularisation (e.g. Weight decay/ridge/quadratic) this becomes
    ##  x = np.inv( (A.T.dot(A) + G.T.dot(G)) ).dot( A.dot(b) )
    ##  Where G is the Tikhonov matrix which for quadratic regularisation is alpha * I 
    return 
    

     
        
        



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
    dC_dtheta_i = self.dC_dtheta_i(C_N, theta_i)
    return -0.5 * np.trace( C_N_inv.dot( dC_dtheta_i ) ) + 0.5 * t_.T.dot( C_N_inv.dot( dC_dtheta_i.dot( C_N_inv.dot(t_) ) ) ) 



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


###############################################################################################
##########################     Kernel Bayesian Regression     #################################

##  def marginal_y_   
##  This is a function p(y_) = N(y_|0_,K)

##  p(t_N+1) = N (tN+1|0, CN+1)

def get_next_k_(theta, x_, x_np1):
    # x_np1 is the next input vector for the distribution
    # x_ is a vector of previous N input vectors. 
    return np.array([ parametric_kernel(theta, x_n, x_np1)  for x_n in x_ ])

def m_pred_next_target( k_, C_N_inv, t_):
    return k_.T.dot(C_N_inv.dot(t_))

def var_pred_next_target(theta, x_np1, C_N_inv, k_, beta):
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

def K_matrix(theta, x, K):
    N = len(x)
    K = np.zeros((N,N))
    for n in range(N):
            for m in range(N):
                K[n][m] = parametric_kernel(theta, x[n], x[m])
    return K

def C_matrix( beta, K):
    # kernel (k_nm) defines the Gram Matrix K , giving the covariance matrix C
    # This is the covariance matrix for the marginal distribution over targets p(t_) = N(t_|0_, C) 
    return K + 1./beta * np.eye(len(K))

def gaussian_process_regression(x_, x_np1, t_, K, beta):
    ## Obtaining the mean and variance of the predictive distribution 
    ## p(t_{n+1}|t_) 

    k_              =  get_next_k_(theta, x_, x_np1)
    K               =  K_matrix(theta, x_, K)
    C               =  C_matrix(beta, K)
    C_N_inv         =  np.linalg.inv(C)

    m_pred_tnp1     =  m_pred_next_target(k_, C_N_inv, t_)
    var_pred_tnp1   =  var_pred_next_target(theta, x_np1, C_N_inv, k_, beta)

    return m_pred_tnp1, var_pred_tnp1, K

def bayesian_check_process(iters, n, noise):

    x = np.linspace(0,10, n)
    t = np.sin(5*x) + np.random.normal(0, noise, n)
    ##  This is the target data
    y = np.sin(5*x) 
    
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
    t_      = np.array( [ t[ind], t[ind2] ] )
    x_      = np.array( [ x[ind], x[ind2] ] )
    K       = np.zeros((1,1))
    for knt in range(iters):
        print('Gaussian process regression: Iteration  %s' %(knt))
        ind = np.random.choice( range( len(y) ) )
        xn= x[ind]
        yn   = y[ind]
        x_  = np.append( x_, xn  )
        t_  = np.append( t_, t[ind] )
        m_p_tnp1, var_p_tnp1, K = gaussian_process_regression(x_, xn, t_, K, beta)
        t_np1_ = np.random.normal(m_p_tnp1, var_p_tnp1, 6)

        xrlist.append(xr)
        yrlist.append(yr)
        ybayes  = np.array([]); ybayes2 = np.array([]); ybayes3 = np.array([])
        ybayes4 = np.array([]); ybayes5 = np.array([]); ybayes6 = np.array([])
        for j in range( len(x)):
            m_p_tnp1, var_p_tnp1, K = gaussian_process_regression(x_, x[j], t_, K, beta)

            t_np1 = np.random.normal(m_p_tnp1, var_p_tnp1, 6)

            ybayes = np.append(ybayes, t_np1[0]); ybayes2 = np.append(ybayes2, t_np1[1]); ybayes3 = np.append(ybayes3, t_np1[2])
            ybayes4 = np.append(ybayes4, t_np1[3]); ybayes5 = np.append(ybayes5, t_np1[4]); ybayes6 = np.append(ybayes6, t_np1[5])         
            
        print( len(x), len(t) , len(x_), len(t_), len(x), len(ybayes2))
        print( x,t ,x_, t_, ybayes2)
        print( x.shape,t.shape ,x_.shape, t_.shape, ybayes2.shape)
        xp =     [x, x, x_, x,      x,       x,       x,       x,       x      ]
        yp =     [t, y, t_, ybayes, ybayes2, ybayes3, ybayes4, ybayes5, ybayes6]
        colour = ['r--', 'g--', 'b^', 'b-', 'b-', 'b-', 'b-', 'b-', 'b-']
        if knt %2 == 0:
            g.plot_function(9, xp, yp, colour, 'Gaussian Process regression.', 
                                'x parameter', 'y')
    return M

iters = 200
n=100
theta = np.array( [1., 4., 0., 0.  ] )
noise = 0.2
deg = 12
bayesian_check_process(iters, n, noise)

deg = 5
#bayesian_check(iters, n, noise, deg)


