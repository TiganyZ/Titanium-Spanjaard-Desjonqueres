import numpy as np 
import ti_opt_general as g
import ti_opt_main_sd_out as out

def init_bayes(   t_n, res, 
                  x_n,
                  kernel, basis, error_function,
                  alpha, beta):
        
    if   kernel == 'Matern':
        ##  Use the Matern Kernel 
        k = matern_kernel(theta, x_n, x_np1)

    elif kernel == 'parametric':
        k = parametric_kernel(theta, x_n, x_np1) 

    else kernel == False:
        ##  Use the Gram Matrix construction for the covariance matrix
        Gram_matrix( len(x_n), k_args, basis)
        

def bayes_fit( npass, LMarg, args, ext, 
                      t_n, res, 
                      pair_pot, bond_ints,
                      kernel, basis, error_function,
                      alpha, beta, 
                      k_args ):
    """
    This routine is to find values for the pair potential and bond integrals that reproduce the target values. 
    
    The input vector is x_n, and this consists of the pair potential and the bond integral coefficients (without the normalisation term.)
    
         Variables:

      t_n: The target values that we want to fit pair potential and bond integrals to.
      res: Results of the target quantities from from input pair_pot and bond integrals input. 

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

#############################################################################################
##########################     Stochastic Gradient Descent     ##############################


def stoc_grad_des_mom(deltaw, gamma, eta, Qi_list):
    Q_i = np.random.choice(range(len(Qi_list))) #This is the change in the cost function at each step 
    Q_i = Qi_list[Q_i]
    newdeltaw = gamma * deltaw - eta * Q_i 
    self.deltaw = newdeltaw
    return newdeltaw



def get_design_matrix(self, Phi, phi_v):
    #This updates the total design matrix
    if len(Phi) > 1:
        Phi = np.append(Phi, phi_v).reshape( (Phi.shape[0] + 1, Phi.shape[1]) )
    else:
        Phi = np.array([phi_v])
    return Phi
        



################################################################################
##########################     Kernel Methods     ##############################
### The kernel function k(x_n, x_m) expresses how strongly y(x_n) and y(x_m)
### are correlated for similar points of x_n, x_m

def basis_kernel_nm(self, phi_n, phi_m, alpha):
    return ( 1./alpha ) * phi_n.dot(phi_m)

def parametric_kernel(self, theta, x_n, x_m):
    return theta[0] * np.exp( -(theta[1]/2.)*np.norm(x_n - x_m)**2 ) + theta[2] + theta[3] * x_n.dot(x_m)

def dC_dtheta_i(C_N, C_N_min_1, theta_i_N, theta_i_N_min_1):
    return (C_N - C_N_min_1)/(theta_i_N - theta_i_N_min_1)

def log_likelihood_t_g_theta(self, N, C_N, C_N_inv, theta_i, t_):
    return -0.5 * np.log(np.abs(C_N)) - 0.5 * t_.T.dot( C_N_inv.dot(t_) ) - N/2. * np.log(2*pi)

def d_log_likelihood_t_g_thetai(self, N, C_N, theta_i, t_):
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

def get_next_k_(self, x_, x_np1):
    # x_np1 is the next input vector for the distribution
    # x_ is a vector of previous N input vectors. 
    return np.array([ self.parametric_kernel(theta, x_n, x_np1)  for x_n in x_ ])

def m_pred_next_target(self, k_, C_N_inv, t_):
    return k_.T.dot(C_N_inv.dot(t_))

def var_pred_next_target(self, C_N_inv, k_, beta):
    c =  self.parametric_kernel(theta, x_np1, x_np1) + 1./beta
    return c - k_.T.dot( C_N_inv.dot( k_ ))

def Gram_matrix(self, N, k_args, basis):
    K = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            if basis:
                K[n][m] = self.basis_kernel_nm(k_args[0], k_args[1], k_args[2]) 
                # k_args = [ phi_n, phi_m, alpha]
            else:
                K[n][m] = self.parametric_kernel(k_args[0], k_args[1], k_args[2]) 
                # k_args = [theta, x_n, x_m]
    return K



def C_matrix(self, beta, K):
    # kernel (k_nm) defines the Gram Matrix K , giving the covariance matrix C
    # This is the covariance matrix for the marginal distribution over targets p(t_) = N(t_|0_, C) 
    return K + 1./beta * np.eye(len(K))

def gaussian_process_regression(self):



    ## Obtaining the mean and variance of the predictive distribution 
    ## p(t_{n+1}|t_) 
    k_ = self.get_next_k_(x_, x_np1)
    m_pred_tnp1 = self.m_pred_next_target(k_, C_N_inv, t_)
    var_pred_tnp1 = self.var_pred_next_target(C_N_inv, k_, beta)




#####################################################################################################
##########################     Linear Basis Bayesian Regression     #################################


def get_alpha_beta_sc(self, alpha, beta, Phi, M_N, T):
    #Chosen initial value of alpha and then will update it using the eigenvectors of the Phi matrix
    #M_N is the most likely value of parameters for W
    #T must by a N x K matrix, Phi must be a N x M matrix, M_N must be a M x K matrix
    tol = 0.1
    Phi_e = np.linalg.eigvals(Phi.T.dot(Phi))
    print('Phi and Phi_e')
    print(Phi, Phi_e)
    for j in range(len(beta)):
        Phi_eig = beta[j] * Phi_e
        gamma = np.sum( [ Phi_eig[i]/(Phi_eig[i] + alpha[j])  for i in range(len(Phi_eig))] ) #vector of gamma 
        print('gamma',gamma)
        alpha_new = gamma/M_N[j,:].dot(M_N[j,:].T)
        print('alpha_new',alpha_new)
        beta_new =  1. / ( (1./(len(T) - gamma)) * sum([ ( T[i, j] - M_N[j,:].dot(Phi[i,:]) )**2 for i in range(len(T)) ]) )
        a_diff = np.linalg.norm(alpha_new-alpha[j]) ; b_diff = np.linalg.norm(beta_new-beta[j])
        tdiff = b_diff + a_diff
        alpha[j] = alpha_new ; beta[j] = beta_new
        iter = 20
        while tdiff > tol:
            iter -= 1
            Phi_eig = beta[j] * Phi_e
            gamma = np.sum( [ Phi_eig[i]/(Phi_eig[i] + alpha[j])  for i in range(len(Phi_eig))] ) #vector of gamma 
            print('gamma',gamma)
            alpha_new = gamma/M_N[j,:].dot(M_N[j,:].T)
            
            beta_new =  1. / ( (1./(len(T) - gamma)) * sum([ ( T[i, j] - M_N[j,:].dot(Phi[i,:]) )**2 for i in range(len(T)) ]) )
            a_diff = np.linalg.norm(alpha_new-alpha[j]) ; b_diff = np.linalg.norm(beta_new-beta[j])
            tdiff = b_diff + a_diff
            print('alpha_new=%s, beta_new=%s' %(alpha_new, beta_new) )
            print('adiff, bdiff', a_diff, b_diff)
            alpha[j] = alpha_new ; beta[j] = beta_new
            if iter < 0:
                alpha[j] = (alpha[j] + alpha_new)/2.
                beta[j] = (beta[j] + beta_new)/2.
                break #tdiff = tol - 1 #breaking from the loop
                

    return alpha, beta


def Bayesian_bond_int(self, npass):
    """In this scheme, will try to vary the bond integrals. 
        Prior over the bond integrals is produced with a mean and variance around the canonical band ratios initially. 
        These bond integrals are used and then the pair potential has been fitted. 
        After this then the bond integrals are then evaluated to see how well of a fit they give to all of the target parameters. 
        From this, the hyperparameters, b_alpha and b_beta are updated so then the posterior distributions of these parameters can be used.
        Convergence on a set of bond integrals from a starting point may not be unique. 


        With scaling of the bond integral we effectively are changing the lattice constant, such that if you were ot increase or decrease the lattice 
        size, then you would be able to find a model that is able to describe Titanium, given an appropriate pair potential."""

    if npass == 0:
        #Initially set the bond integrals to the canonical values 
        self.b_vars = ['ddsigTTSD','ddpiTTSD','dddelTTSD','spanjddd']
        W_b = np.array([6.,4.,1.])
        dddelta = 0.208098266125
        b_args = self.construct_extra_args(xargs='', coeffargs=self.b_vars, coefflist=np.append(W_b, dddelta) )
        b_alpha = np.array([1/3.,1/3.,1/3.]) #These values are 1/var of the parameter W_b 
        b_S0_inv = np.diag(b_alpha)
        b_beta = b_alpha
        A = 75. ; b = 8.18/self.equibR0 ; C = 75. ; d = 8.18/self.equibR0
        G_M = np.array([0, A, b, C, d])
        M_N, S_N, S_N_inv, W, T, Phi, alpha, beta, G, Phi_g, g_mu, g_alpha, g_beta, g_s_N, g_s_N_inv = self.Bayesian_Fitting_Gaussian_Basis(
                                                        npass=npass, M_N=[], S_N=[], S_N_inv=[], W=[], T=[], Phi=[], alpha=1/0.001, beta=25, 
                                                        g_alpha=1/0.001, g_beta=25, G=[], Phi_g=[], g_mu = G_M, g_s_N=0, g_s_N_inv=0)
    delta_err = self.delta_err[-1] #This is the errors in the target variables from the cost function from fitting the pair potential





    return npass

def Bayesian_Fitting_Gaussian_Basis(self, npass, 
                                                M_N, S_N, S_N_inv, 
                                                W, T, Phi, 
                                                alpha, beta, 
                                                g_alpha, g_beta, 
                                                G, Phi_g, g_mu, 
                                                g_s_N, g_s_N_inv):
    """   
    In this scheme we are finding the functions that best describe the relationship between a target parameter (a parameter we want to fit to) 
    to that of the pair potential parameters. We use a Bayesian Linear Regression technique. 

    Want to maximise ln( p(w_|t_) ) = -beta/2 * sum_1^N {t_n - w.T * phi(x_n)}**2 - alpha/2 w.T * w
    where phi(x_n) is in a Gaussian Basis: phi_j(x) = exp{ (x-mu_j)**2 / 2s**2}, 
    where these x_n are the inputs for the pair potential. 
    
    This means we put in four values for the pair potential and then can find functions of each of the parameters that we want to fit to.

    This means we now have functions in parameter space that can be subsequently optimized such that we can find a minimum 
    in the M dimensional parameter space, where M is the number of target parameters that we want to fit. 
    
    This is regularised meaning that there won't be a matrix that is not singular. 
    alpha corresponds to the precision of the prior distribution of the parameters, while beta and analagous to the posterior. 

    To find a minimum in the ten dimensional parameter space we can use something like Stochastic Gradient descent with momentum. 

    Bayesian model needs prior and posterior precisions over the parameter spaces. 

    (0)---Have prior distribution over the input vector x_n, and a different prior distribution over the parameters 
          to describe the relation of calculated parameters and targets. 
    (1)---From input parameters, x_n = [A, b, C, d], (we have a randomly varying uniform distribution for the x values 
          in the exponent), for the pair potential, look at Elastic Constants,  Bulk Modulus, Energy Minimum for c/a? obtained, 
          these are t_n target values for this level.  
    (2)---Find parameters, w, which act in linear combination in a Gaussian basis, to see how each of the parameters for the 
          pair potential affects a target value. 
    (3)---Obtain the change in the cost function from the iteration---this is how different the constructed function values are
          from the target values. 
    (4)---Use this change in a stochastic gradient descent with momentum to change the values of the w_ parameters to describe the targets. 
    (5)---Use stochastic gradient descent with momentum again, using something proportional to the change in cost function, 
          to change the values of the input vectors, thus finding more optimum parameters. 


    

    
    This should find the optimal model. """

    npass = npass + 1
    print('Bayesian fitting routine. Iteration = %s' %(npass))
    
    """Initially tried to make phi(x) just one function, but I could make it a polynomial of powers of a certain coefficient---say up to the third power
     Would like the w parameters to have a good bearing on each pair potential parameter, powers are probably too 'strong'
     For each term in the pair potential, it can be described in the polynomial basis by
        w_j * (w'_i * x^i), 
     where w'_i is coefficient, and then have general coefficients w_j which describe the proportion used in the fitting. 

     This means that each w_j = w_j(x)

     For each point obtained, w_j, I can use a simple Bayesian scheme for the fitting of a polynomial with Gaussian noise, as in Bishop, to find the actual dependence.""" 
    #alpha is the range of the distribution from whic input parameters are given, it is the precision of the prior 
    #Basis for the regression of the polynomial for each coefficient. Have multiple targets, hence 2-D matrix

    S0inv=S_N_inv
    g_S0_inv=g_s_N_inv

    """ 
    ###############################################################################################################
    ##########################     Polynomial for the iniital input vectors     ###################################
    ###############################################################################################################

    S0_inv_poly = S_poly

    poly_d = 3 #This is the highest order term in the polynomial 
    poly_basis = self.poly_basis(x=x, d=poly_d) #have w0 parameter for the bias set to 1, so vector of length 4
    Phi_poly = self.get_design_matrix(Phi_poly, poly_basis)

    if npass == 1:
    #    # Construct the parameter matrix of w'_i, where we have each row corresponding to the coefficients of the polynomial describing a pair potential parameter
        # Make a uniform prior distribution of polynomial values
        poly_l, poly_u = 0, 100
        uni_var = (1./12.) * (poly_l - poly_u)**2
        alpha_poly = np.array([poly_u for i in range(poly_d + 1)])        #np.array([1./uni_var for i in range(poly_d + 1)]) 
        W_poly = np.random.uniform(low=poly_l, high=poly_u, size=(len(g_mu),len(poly_basis)))
        beta_poly = alpha_poly
        S0_inv_poly = np.array( [ alpha_poly[i] * np.eye(len(W_poly)) for i in len(alpha_poly) ])
    else:
        alpha_poly, beta_poly = self.get_alpha_beta_sc(alpha_poly, beta_poly, Phi_poly, M_N_poly, T_poly)

    
    T_poly = self.get_T_matrix(T_poly, t_poly)

    M_poly, S_poly, S_inv_poly, W_poly = self.W_posterior_update(alpha_poly, beta_poly, Phi_poly, T_poly, W_poly, S0_inv_poly)
    #Using the polynomial guessed, to find a value for the input for the pair potential

    """

    #################################################################################################################################################
    #####################################################       Prior over the values for the pair potential     #####################################################


    if npass == 1:
        # Make a uniform prior distribution of pair potential parameter values
        g_alpha = np.array([1, 150., 1.5, 150., 1.5])        #np.array([1./uni_var for i in range(mu_d + 1)])
        g_mu = self.get_g_mu(g_alpha, np.diag(g_alpha))
        #for i in range(g_alpha):
        #    g_mu.append(np.random.normal(g_alpha[i], 1./g_alpha[i]**(0.5) ))

        g_S0_inv = np.diag(1./g_alpha) #np.array( [ g_alpha[i] * np.eye(len()) for i in len(g_alpha) ])
        g_s_N = g_S0_inv
        Phi_g = []
        G = []
        g_alpha = np.linalg.norm(1./g_alpha)
        g_beta = g_alpha
    else:
        print(g_mu)
        print(g_S0_inv)
        g_mu = self.get_g_mu(g_mu, g_s_N)
        print('g_mu = %s iteration = %s' %( g_mu, npass) )

    # #########################################################################################################
     ################################################################################################################


    #################################################################################################################################################
    #####################################################       Constructing the matrices of the basis     #####################################################


    x = np.random.uniform(0.0, 10.0) #This is the input for the pair potential
    sep = np.linalg.norm(g_S0_inv) #Having the spatial extent of each of the gaussians proportional to the average covariances of all of them

    phi_g = g_mu #This is the identity basis for g_mu
    Phi_g = self.get_design_matrix(Phi_g, phi_g)

    phi_v = self.get_gauss_phi(x, g_mu, sep) #self.get_poly_phi(x, g_mu, sep)# #this is a 1 x 4 vector of the basis set, each element corresponds to a particular g_mu
    phi_v[0] = 1 #Setting the basis for bias to 1
    Phi = self.get_design_matrix(Phi, phi_v) #building design matrix
    
    if npass > 1:
        S0inv=S_N_inv
        g_S0_inv=g_s_N_inv



    #################################################################################################################################################
    #####################################################       Obtaining Target vector from input of x_n vectors     #####################################################


    t_v = self.get_target_vector(g_mu)#phi_v) #This is the vector of all of the resultant parameters that are calculated subtracted by the actual values 
    #This target vector is, itself, a measure of the cost.

    WGTS = np.array([        1.0  #11
                            ,1.0  #33
                            ,1.0  #12
                            ,1.0  #13
                            ,1.0  #44
                            ,1.0  #66
                            ,1.0  #kk
                            ,1.0  #rr
                            ,1.0  #hh
                            ,1.0  #Bulk Modulus
                            ,10.0  #Ideal c/a ratio hcp
                            ,10.0] )  #Ideal minimum energy at equilibrium
    t_v = WGTS * t_v
    t_dist = np.sum(np.abs(t_v))  #weighted distance from targets



    print('distance_from_exp = %s, %s' %(np.abs(t_v), np.sum(np.abs(t_v))))
    #This means that the parameters should have a gaussian centred at a mean of zero, at these targets, in principle should be the same value
    if npass == 1:
        alpha = np.array([0.001 for i in range(len(t_v))])  #np.array([0.01,0.01,0.01,0.01,0.01])
        print(alpha)
        W = np.random.normal(0, 1/0.001, size=(len(t_v),len(g_mu)))
        # This is an array of covariant matrices corresponding to each vector of parameters for a given target
        S0inv =  np.array( [ (alpha[i])*np.eye(len(g_mu)) for i in range(len(t_v)) ] ) 
        beta = alpha
        #print('in npass, alpha, beta', alpha, beta)
        #g_s_N, g_s_N_inv = self.S_N_wgt(g_mu, g_alpha, g_beta, g_S0_inv)
        #g_mu = self.get_g_mu(g_mu, g_s_N)


        #    g_mu = np.array([0, A, b, C, d]) #These are the averages used in the Gaussian basis, the first value is the bias
        #This is a prior which initially is quite flat. p(w|alpha) = N(w| 0, I*1/alpha). Alpha is very small initially 
        #This gives a posterior distribution with m_n = beta*S_N * Phi.T * t_ >>    
     

    #Finding new W matrix and covariance 


    T = self.get_T_matrix(T, t_v) # This is an M x K matrix

    M_N, S_N, S_N_inv, W = self.W_posterior_update(alpha, beta, Phi, T, W, S0inv)

    alpha, beta = self.get_alpha_beta_sc(alpha, beta, Phi, M_N, T)
    #print('alpha, beta sc', alpha, beta)

    """
    alpha = self.alpha_hyp(T, Phi, W, len(W[0]))
    beta = self.beta_hyp(W, npass)
    g_alpha = self.g_alpha_hyp(G, g_mu, len(g_mu))
    g_beta = self.beta_hyp(g_mu, npass)
    """


    #################################################################################################################################################
    #####################################################       Stochastic Gradient Descent     #####################################################

    delta_err, Q_i_n = self.delta_reg_error_func(alpha, beta, t_v, T, W, Phi) #Finding the change in the error function---regularised least squares
    if npass==1:
        self.Qi_list = np.array([Q_i_n])
        self.deltaw = Q_i_n
        self.delta_err = np.array([delta_err])
    else:
        self.Qi_list = np.append(self.Qi_list, Q_i_n).reshape( (self.Qi_list.shape[0] + 1, self.Qi_list.shape[1], self.Qi_list.shape[2]) )
        self.delta_err = np.append(self.delta_err, delta_err).reshape((self.delta_err.shape[0] + 1, self.delta_err.shape[1]))
    
    self.deltaw = self.StocGradDes_Momentum( gamma=0.1, eta=0.9 , Qi_list=self.Qi_list) #Finding the new change in the parameters necessary
    #print(self.deltaw)
    W += self.deltaw # Updating W. This is an M x K matrix where K is the number of parameters 

    delta_coeff= np.mean(self.deltaw, axis=0)/np.mean(W, axis=0) #This is the proportional change in the coefficients of W

    #delta_coeff = self.delta_err

    #################################################################################################################################################
    #################################################################################################################################################



    #g_mu = self.get_g_mu(g_mu_N, g_s_N)
    #g_mu_t_var_p = self.var_pred_tgt_ab( phi_g, g_s_N.reshape((1,)+g_s_N.shape), g_beta.reshape((1,)+g_beta.shape))[0]
    #g_mu_t_mean_p = self.mean_pred_tgt_ab( g_mu_N, phi_g)[0]


    g_mu_N =self.get_g_mu(g_mu, np.diag(delta_coeff**2)) #Using the change in the coefficients as a form of error so then we can choose a more apporopriate value for the pair potential coefficients

    G = self.get_T_matrix(G, g_mu_N) #Building the G matrix, equivalent of T matrix for the input pair potential parameters
    print(g_mu_N, g_S0_inv, g_alpha, g_beta)
    g_mu_N, g_s_N, g_s_N_inv, W_g = self.W_posterior_update(g_alpha.reshape((1,)+g_alpha.shape), g_beta.reshape((1,)+g_beta.shape), Phi_g, G, g_mu_N.reshape((1,)+g_mu_N.shape), g_S0_inv.reshape((1,)+g_S0_inv.shape))
    
    g_alpha, g_beta = self.get_alpha_beta_sc(g_alpha.reshape((1,)+g_alpha.shape), g_beta.reshape((1,)+g_beta.shape), Phi_g, g_mu_N, G)
    g_mu_N, g_s_N, g_s_N_inv, g_alpha, g_beta = g_mu_N[0], g_s_N[0], g_s_N_inv[0], g_alpha[0], g_beta[0] 
    #print('delta_coeff=%s'%(delta_coeff))
    #g_mu = np.random.multivariate_normal(mean=g_mu, cov=g_s_N)
    #g_mu = self.get_g_mu(g_mu, g_s_N) #* delta_coeff #* t_dist
    #g_s_N, g_s_N_inv = self.S_N_wgt(g_mu, g_alpha, g_beta, g_S0_inv)
    #g_m_N = self.m_N_wgt(g_s_N, g_S0_inv, np.ones((1,len(g_mu))), np.ones(1), g_beta, g_mu)
    #print('G', G)
    print('Delta coeff=%s'%(delta_coeff))
    print('New g_mu=%s'%(g_mu_N))
    print('alpha = %s, beta = %s, g_alpha = %s, g_beta = %s' %(alpha, beta, g_alpha, g_beta))
    print('g_s_N', g_s_N)
    

    #print('npass = %s, M_N = %s, S_N = %s, W = %s, T = %s, Phi = %s, alpha = %s, beta = %s, g_mu=%s' %(npass, M_N, S_N, W, T, Phi, alpha, beta, g_mu) )
    return M_N, S_N, S_N_inv, W, T, Phi, alpha, beta, G, Phi_g, g_mu_N, g_alpha, g_beta, g_s_N, g_s_N_inv

def W_posterior_update(self, alpha, beta, Phi, T, W, S0_inv):
    #print('alpha, beta',alpha, beta)
    for i in range(len(S0_inv)):
        s_N, s_N_inv = self.S_N_wgt(Phi, alpha[i], beta[i], S0_inv[i])
        m_N = self.m_N_wgt( s_N, S0_inv[i], Phi, T[:,i], beta[i], m0=W[i,:])
        #print('m_N = %s, i = %s' %( m_N, i))
        #print('s_N', s_N)
        if i == 0:
            M = np.array([m_N])
            #print('M shape i = 0 ', M.shape)
            S = np.array([s_N])
            #print('S shape i = 0 ', S.shape)
            S_inv = np.array([s_N_inv])
            #print('S inv shape i = 0 ', S_inv.shape)
        else:
            M = np.append( M, m_N ).reshape(M.shape[0] + 1, M.shape[1])
            S = np.append( S, s_N ).reshape(S.shape[0] + 1, S.shape[1], S.shape[2])
            S_inv = np.append( S_inv, s_N_inv ).reshape(S_inv.shape[0] + 1, S_inv.shape[1], S_inv.shape[2])
        W[i,:] = np.random.multivariate_normal(mean=m_N, cov=s_N)#, size=(len(W),1))
        
    return M, S, S_inv, W

def W_posterior(self, alpha, beta, Phi, T, W, S0_inv):
    S_N = self.S_N_wgt(Phi, alpha, beta, S0_inv)
    M_N = self.m_N_wgt( S_N, S0_inv, Phi, T, beta, m0=W)
    print('M_N', M_N)
    for i in range(len(W)):
        print(M_N[i,:])
        W[i,:] = np.random.multivariate_normal(mean=M_N[i,:], cov=S_N)#, size=(len(W),1))
    return M_N, S_N, W

def poly_basis(self, x, d):
    return np.array( [x**i for i in range(0,d+1)] ) #bias is set to 1

def S_N_wgt(self, Phi, alpha, beta, S0_inv):  #This is the S matrix for the posterior distribution over w, where the prior is p(w|alpha)=N(w|0, 1/alpha *I)  
    
    #if len(Phi) > 1:
    sn_inv = beta * Phi.T.dot( Phi)
    sn_inv = S0_inv + sn_inv
    #else:
    #    sn_inv = beta * np.outer(Phi, Phi) + S0_inv#.reshape((2,2))
    self.sn_inv = sn_inv
    self.S_N = np.linalg.inv(sn_inv)
    return self.S_N, self.sn_inv

def m_N_wgt(self, S_N, S0_inv, Phi, t_, beta, m0): #This is the mean for the posterior distribution over w, where the prior is p(w|alpha)=N(w|0, 1/alpha *I)  

    m = S_N.dot( S0_inv.dot(m0) + beta * Phi.T.dot(t_) )

    return m

def cmd_result(self,cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result,err = proc.communicate() 
    result = result.decode("utf-8")[0:-1]
    return result


def get_T_matrix(self, T, t_vec):
    if len(T)>1:
        T = np.append(T, t_vec).reshape( (T.shape[0] + 1, T.shape[1]) )
    else:
        T = np.array([t_vec])
    return T


def delta_reg_error_func(self, alpha, beta, t, T, W, Phi):
    mulphi=np.asarray([ Phi[-1] for i in range(len(t))])
    #delta_err = np.zeros(W.shape)
    #for i in range(len(W)):
    print('beta shape', beta.shape)
    print('alpha shape', alpha.shape)
    Q_i_n = np.zeros(W.shape)
    delta_err = np.zeros(T.shape)
    de_x = np.zeros(W.shape[0])
    for i in range(len(W)):
        de_x[i] = alpha[i] * W[i,:].dot(W[i,:].T) 
        Q_i_n[i, :] = beta[i] *  ( t[i] - W[i,:].dot(Phi[-1]) ) * Phi[-1]  + alpha[i] * W[i,:] #np.array([t - W.dot(Phi[-1])]).T * mulphi
    for j in range(len(T)):
        delta_err[j,:] =   beta *  ( T[j] - W.dot(Phi[j].T) )**2  
    delta_err = 0.5 * ( np.sum(delta_err, axis=1) + de_x )
    print('delta_err = %s' %(delta_err))
    return delta_err, Q_i_n
    
def input_arg(name, value):
    return ' -v' + name + '=' + value + ' ' 

def get_target_vector(self, g_mu):
    ppargs = self.construct_ppargs(g_mu)
    print('ppargs',ppargs)

    #Getting hcp coa min
    print("Obtaining coa minimum")
    etot_min, coa_min = self.minimum_energy_lattice_parameter(lattice='hcp', tol=0.01)


    alphal=[-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015]
    print("Obtaining Elastic Constants")
    econsts = self.Girshick_Elast(alphal)
    #print(ppargs)
    print("Obtaining Energy")
    energy = self.find_energy(ppargs)
    ppenergy = float(self.cmd_result("grep 'pair potential energy' out | tail -1 | awk '{print $5}'"))
    bulk_modl = self.bulk_modulus(econsts)
    econsts.append(bulk_modl)
    econsts.append(coa_min)
    if energy < 1.4:
        #put an exponential error on it 
        energy = energy**2
    econsts.append(energy)
    res_target = np.asarray( econsts ) 
    #self.vals_to_fit = np.asarray(self.vals_to_fit)
    print(res_target, self.vals_to_fit)

    difference = self.vals_to_fit - res_target
    return difference 

def alpha_hyp(self, T, Phi, W, M):  #These are only valid for large N 
    return float(M)/ (np.linalg.norm(  sum([T[:,i] - Phi.dot(W[i,:]) for i in range(len(W)) ])/float(len(W)) )**2)

def beta_hyp(self, W, N):
    return float(N)/np.linalg.norm( W.T.dot( W ) )

def g_alpha_hyp(self, G, gm, M):  #These are only valid for large N 
    return float(M)/ (np.linalg.norm(  sum([sum(G[:,i]) - gm.dot(gm) for i in range(len(G)) ])/float(len(G)) )**2)

def g_beta_hyp(self, gm, N):
    return float(N)/np.linalg.norm( gm.T.dot( gm) )

def get_design_matrix(self, Phi, phi_vec):
    #This constructs the total design matrix
    #phi_t = np.ones(phi_v.shape[0] + 1)
    #phi_t[1:] = phi_vec
    if len(Phi) > 1:
        Phi = np.append(Phi, phi_vec).reshape( (Phi.shape[0] + 1, Phi.shape[1]) )
    else:
        Phi = np.array([phi_vec])
    return Phi
        

def get_poly_phi(self, x, mu_i, s):
    return 1 + (x-mu_i)/s + ((x-mu_i)/s)**2 + ((x-mu_i)/s)**3


def get_gauss_phi(self, x, mu_i, s):
    return np.exp( -(x - mu_i)**2 / (2 * s**2) )


def StocGradDes_Momentum(self, gamma, eta, Qi_list):
    Q_i = np.random.choice(range(len(Qi_list))) #This is the change in the cost function at each step 
    Q_i = Qi_list[Q_i]
    newdeltaw = gamma * self.deltaw - eta * Q_i 
    self.deltaw = newdeltaw
    return newdeltaw

def var_pred_tgt_ab(self, phi, S_N, beta):
    #phi = Phi[-1]
    var_i = np.ones(len(S_N))
    for i in range(len(S_N)):
        var_i[i] = 1./beta[i] + phi.T.dot(S_N[i]).dot(phi)
    return var_i

def mean_pred_tgt_ab(self, M_N, phi):
    m_n_i = np.ones(len(M_N))
    for i in range(len(M_N)):
        m_n_i[i] = M_N[i].dot(phi)
    return m_n_i

def get_g_mu_2(self, mu, cov):
    mu_new = np.random.multivariate_normal(mean=mu, cov=cov)


    coord = 1
    cond_exp_min = False#mu_new[1]/mu_new[2] < np.exp(coord*mu_new[2])/mu_new[3] + np.exp(coord*(mu_new[2]-mu_new[4]))
    cond_exp_min = cond_exp_min or mu_new[1] < mu_new[3]
    val = np.any(mu_new[2::2] > 5)
    rootx = np.log( mu_new[1]/(mu_new[3])/(mu_new[2]-mu_new[4]))
    minx = np.log( mu_new[2]*mu_new[1]/(mu_new[3]*mu_new[4]) )/(mu_new[2]-mu_new[4])
    print('minx', minx)
    cond_exp_min = val  or cond_exp_min #or  minx < 4 or minx > 7
    while cond_exp_min:
        mu_new[1:]= np.abs(np.random.multivariate_normal(mean=mu[1:], cov=cov[1:,1:]) )
        print('cond min mu_new = %s' %(mu_new))
        cond_exp_min = False#mu_new[1]/mu_new[2] < np.exp(coord*mu_new[2])/mu_new[3] + np.exp(coord*(mu_new[2]-mu_new[4]))
        val = mu_new[2] < mu_new[4] or mu_new[1] < mu_new[3]
        rootx = np.log( mu_new[1]/(mu_new[3])/(mu_new[2]-mu_new[4]))
        minx = np.log( mu_new[2]*mu_new[1]/(mu_new[3]*mu_new[4]) )/(mu_new[2]-mu_new[4])
        print('minx = %s, rootx = %s'%( minx, rootx) )
        cond_exp_min = val  or cond_exp_min #or  minx < 4 or minx > 7

    return mu_new


    

def get_g_mu(self, mu, cov):  #This is essentially the prior/posterior of the distribution 
    mu_new = np.random.multivariate_normal(mean=mu, cov=cov)
    val = (((mu_new[2] < mu_new[4]) or (mu_new[1] < mu_new[3])) or np.any(mu_new[1:] < 0)) == True or mu_new[4] > 10 or mu_new[2] > 10
    minx = np.log( mu_new[2]*mu_new[1]/(mu_new[3]*mu_new[4]) )/(mu_new[2]-mu_new[4])
    coord = 1
    #val2 =  mu_new[1]*np.exp(-mu_new[2] * coord) - mu_new[3]*np.exp(-mu_new[4] * coord)
    val = val or minx < 4 or minx > 7# or val2 < 1
    while val: 
        print('mu_new = %s' %(mu_new))
        mu_new = np.random.multivariate_normal(mean=mu, cov=cov)
        mu_new[1:] = np.abs(mu_new[1:]) 
        val = (((mu_new[2] < mu_new[4]) or (mu_new[1] < mu_new[3])) or np.any(mu_new[1:] < 0)) == True or mu_new[4] > 10 or mu_new[2] > 10
        minx = np.log( mu_new[2]*mu_new[1]/(mu_new[3]*mu_new[4]) )/(mu_new[2]-mu_new[4])
        #val2 =  mu_new[1]*np.exp(-mu_new[2] * coord) - mu_new[3]*np.exp(-mu_new[4] * coord)
        val = val or minx < 4 or minx > 7 #or val2 < 1
    print('g_mu values =%s, ' %(mu_new))

    ########################################################
    ####### Maybe try and try and find one exponential and then vary with a gaussian error of say 2.5*value
    ####### Can possibly do a similar thing with the coefficients, that a must be arooung 
    ####### Could probably use a bayesian approach such that this can be automatic
    return mu_new#np.array([0, w0, w1, w2, w3])

